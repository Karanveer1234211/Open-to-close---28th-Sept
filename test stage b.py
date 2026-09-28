#!/usr/bin/env python3
"""
tests/test_stage_b.py - stage_b_screen v1.1. Prints VERIFIED on success.

Run from the open_close folder with the main folder on PYTHONPATH:
    python tests\\test_stage_b.py

 0. The modules loaded are this folder's (no shadowing by the main folder).
 1. Statistics against independent code: within-day average ranks (ties),
    daily Spearman IC (scipy), Newey-West t (explicit double sum), BH q-values.
 2. A planted world, screened end to end:
      D_pos / D_neg / D_pos_dup / D_nanpos / D_disc   planted signal -> PASS, right sign
      D_noise_00..39                                  pure noise -> (almost) nothing passes
      D_unstable   strong in folds 1-3, mildly reversed in 4-5 -> significant but FAILS stability
      D_lockonly   predicts only UNBUYABLE rows       -> must not pass (eligibility)
      D_late       predicts only after 2025-03-05     -> must not pass (lockbox never read)
      D_early      predicts only before fold 1        -> must not pass (test windows only)
      D_leak       = the target + noise               -> flagged as a possible leak
      MKT_regime   same value for every stock, drives the day's average -> market-level table
      GOLD_close, D_close_proxy                       raw level / price-level proxy -> excluded
      label_* and OHLCV                               never screened
    Exact reproduction of IC and quintile money by independent code; target
    distributions (N, mean, median, SD, percentiles, positive share by fold)
    reproduced by independent code on the same rows; report order (targets ->
    screen -> cells) and wording (quintile economics are not a trading result;
    marginal not incremental; families descriptive); families; determinism;
    ledger; re-run guard.
 3. Power checks (the planted effect is found when the guard is removed):
    eligibility off -> D_lockonly detected; lockbox read -> D_late detected.
 4. Preconditions: no/failed/stale audit, changed manifest, old panel_build,
    a label smuggled into the feature list.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import research_common as RC   # noqa: E402
import panel_build as PB       # noqa: E402
import stage_b_screen as SB    # noqa: E402

FAILS = []
TARGETS = ["label_oc_1", "label_oc_2", "label_oc_3", "label_oc_5"]


def check(cond, msg):
    print(("  ok  " if cond else "  FAIL") + f"  {msg}")
    if not cond:
        FAILS.append(msg)


# ----------------------------------------------------------------------
def test_module_origin():
    print("0. the new modules are the ones loaded (no shadowing by the main folder)")
    here = HERE.parent.resolve()
    for m in (SB, RC, PB):
        check(Path(m.__file__).resolve().parent == here, f"{m.__name__} from {Path(m.__file__).resolve().parent}")


# ----------------------------------------------------------------------
def test_statistics():
    print("1. statistics against independent code")
    from scipy.stats import spearmanr
    rng = np.random.default_rng(3)
    nD, per = 40, 30
    d = np.repeat(np.arange(nD), per)
    x = np.round(rng.normal(size=nD * per), 1)          # ties on purpose
    y = 0.3 * x + rng.normal(size=nD * per)
    r = SB.group_rank(d, x)
    ref = pd.Series(x).groupby(d).rank(method="average").to_numpy()
    check(np.allclose(r, ref, atol=0, rtol=0), "within-day average ranks = pandas rank(method='average'), ties included")
    ic, cnt = SB.daily_ic(d, SB.group_rank(d, x), SB.group_rank(d, y), nD, 15)
    ref_ic = np.array([spearmanr(x[d == k], y[d == k]).statistic for k in range(nD)])
    check(np.max(np.abs(ic - ref_ic)) < 1e-12, f"daily IC = scipy spearmanr on every day (max diff {np.max(np.abs(ic - ref_ic)):.1e})")
    ic2, _ = SB.daily_ic(d[:per * 3 + 10], SB.group_rank(d[:per * 3 + 10], x[:per * 3 + 10]),
                         SB.group_rank(d[:per * 3 + 10], y[:per * 3 + 10]), nD, 15)
    check(np.isnan(ic2[3]) and np.isfinite(ic2[2]), "a day with fewer than 15 names gets no IC")

    s = np.zeros(300)
    e = rng.normal(size=300)
    for i in range(1, 300):
        s[i] = 0.7 * s[i - 1] + e[i]
    s = s + 0.2
    L = 20
    n = len(s)
    ee = s - s.mean()
    S = 0.0
    for i in range(n):
        for j in range(n):
            k = abs(i - j)
            if k <= L:
                S += (1 - k / (L + 1)) * ee[i] * ee[j]
    t_ref = s.mean() / math.sqrt(S / n / n)
    t = SB.hac_t(s, L)
    check(abs(t - t_ref) < 1e-9, f"Newey-West t = explicit double sum ({t:.6f} vs {t_ref:.6f})")
    t_naive = s.mean() / (s.std() / math.sqrt(n))
    check(abs(t) < abs(t_naive) / 1.5, f"autocorrelated series: HAC t {t:.2f} well below naive {t_naive:.2f}")

    p = rng.uniform(size=200) ** 3
    q = SB.bh_qvalues(p)
    o = np.argsort(p)
    ps = p[o]
    qref = np.array([min(ps[j] * len(p) / (j + 1) for j in range(i, len(p))) for i in range(len(p))])
    qref = np.minimum(qref, 1)
    qq = np.empty(len(p))
    qq[o] = qref
    check(np.allclose(q, qq), "BH q-values = definition min_{j>=i} p_(j) n / j")
    check(abs(SB.p_two_sided(1.959963984540054) - 0.05) < 1e-9, "two-sided p of t=1.96 is 0.05")


# ----------------------------------------------------------------------
def make_world(root: Path, seed: int = 1, n_sym: int = 60) -> Path:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-06-01", "2025-09-30")
    T, N = len(dates), n_sym
    dv = dates.values.astype("datetime64[ns]")
    research_end, lockbox = SB.research_window(dv)
    rs = dv[dv <= research_end]
    folds = SB.fold_windows(rs)
    fold_of = np.zeros(T, int)
    for f in folds:
        a, b = np.datetime64(pd.Timestamp(f["test_start"]), "ns"), np.datetime64(pd.Timestamp(f["test_end"]), "ns")
        fold_of[(dv >= a) & (dv <= b)] = int(f["fold"])
    first_test = np.datetime64(pd.Timestamp(folds[0]["test_start"]), "ns")

    di = np.repeat(np.arange(T), N)
    si = np.tile(np.arange(N), T)
    keep = rng.random(T * N) >= 0.03
    di, si = di[keep], si[keep]
    R = len(di)
    z = lambda: rng.normal(size=R)
    pos, neg, unst, lock, late, early = z(), z(), z(), z(), z(), z()
    mkt_t = rng.normal(size=T)
    buy = (rng.random(R) >= 0.10).astype(float)
    fo = fold_of[di]
    u = np.where(np.isin(fo, (1, 2, 3)), 0.004, np.where(np.isin(fo, (4, 5)), -0.001, 0.0))
    late_eff = np.where(dv[di] >= np.datetime64(SB.LOCKBOX_START_PINNED, "ns"), 0.006, 0.0)
    early_eff = np.where(dv[di] < first_test, 0.006, 0.0)
    y1 = (0.004 * pos - 0.004 * neg + u * unst + late_eff * late + early_eff * early
          + 0.004 * mkt_t[di] + 0.012 * z())
    y1 = y1 + np.where(buy == 0, 0.05 * lock, 0.0)
    y2 = y1 + 0.012 * z()
    y3 = y2 + 0.012 * z()
    y5 = y3 + 0.017 * z()

    lvl = np.exp(rng.uniform(np.log(10), np.log(5000), size=N))
    walk = np.exp(np.cumsum(0.01 * rng.normal(size=(T, N)), axis=0))
    close = lvl[si] * walk[di, si]
    gold = 1000 + np.cumsum(rng.normal(size=T))
    df = pd.DataFrame({
        "timestamp": dv[di], "symbol": np.array([f"S{i:02d}" for i in range(N)])[si],
        "open": close * (1 + 0.001 * z()), "high": close * 1.01, "low": close * 0.99, "close": close,
        "volume": rng.integers(10_000, 1_000_000, size=R).astype(float),
        "D_pos": pos, "D_neg": neg, "D_pos_dup": pos + 0.05 * z(),
        "D_nanpos": np.where(rng.random(R) < 0.30, np.nan, pos),
        "D_disc": np.round(pos, 1), "D_unstable": unst, "D_lockonly": lock, "D_late": late,
        "D_early": early, "D_leak": y1 + 0.002 * z(), "MKT_regime": mkt_t[di], "GOLD_close": gold[di],
        "D_close_proxy": close * 1.001,
        "label_oc_1": y1, "label_oc_2": y2, "label_oc_3": y3, "label_oc_5": y5,
        "label_buyable_o1": buy, "label_exit_ret": z(), "label_fwd_ret_5d": z(),
    })
    for i in range(40):
        df[f"D_noise_{i:02d}"] = z()
    pdir = root / "panel_oc"
    pdir.mkdir(parents=True)
    df.to_parquet(pdir / "panel.parquet", index=False)
    built = pd.Timestamp.now() - pd.Timedelta(hours=1)
    (pdir / "panel_meta.json").write_text(json.dumps({"panel_build_version": PB.PANEL_BUILD_VERSION,
                                                      "etf_rule": RC.ETF_RULE_VERSION,
                                                      "built_at": built.isoformat()}), encoding="utf-8")
    _, sha = SB.load_manifest()
    RC.ledger_append(pdir / "panel.parquet", {"kind": "panel_audit", "tool": "label_audit v2", "passed": True,
                                              "panel_build_version": PB.PANEL_BUILD_VERSION,
                                              "experiment_family": "oc_v1", "manifest_sha": sha})
    return pdir


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **k)
    return out, buf.getvalue()


def cell(C, f, tg="label_oc_1"):
    r = C[(C["feature"] == f) & (C["target"] == tg)]
    assert len(r) == 1, (f, tg, len(r))
    return r.iloc[0]


def independent_ic_and_money(pdir: Path, feature: str, tg: str, res: dict):
    """Plain pandas, day by day, from the parquet - no code shared with the tool."""
    P = pd.read_parquet(pdir / "panel.parquet", columns=["timestamp", "symbol", feature, tg, "label_buyable_o1"])
    re_ = pd.Timestamp(res["research_end"])
    t0 = pd.Timestamp(res["folds"][0]["test_start"])
    P = P[(P["timestamp"] >= t0) & (P["timestamp"] <= re_) & (P["label_buyable_o1"] == 1)]
    mk = P.dropna(subset=[tg]).groupby("timestamp")[tg].mean()
    P = P.dropna(subset=[feature, tg])
    ics, spreads, longs, mkts = [], [], [], []
    for day, g in P.groupby("timestamp"):
        if len(g) < 15:
            continue
        rx = g[feature].rank(method="average")
        ry = g[tg].rank(method="average")
        ics.append(np.corrcoef(rx, ry)[0, 1])
        pct = (rx - 0.5) / len(g)
        q = np.minimum(np.floor(pct * 5).astype(int), 4)
        m = g[tg].groupby(q.to_numpy()).mean()
        spreads.append(m.get(4) - m.get(0))
        longs.append(m)
        mkts.append(mk.loc[day])
    mean_ic = float(np.mean(ics))
    lq = 4 if mean_ic >= 0 else 0
    lg = np.array([m.get(lq) for m in longs])
    return mean_ic, float(np.mean(spreads) * 1e4), float(np.mean(lg) * 1e4), float(np.mean(lg - np.array(mkts)) * 1e4)


def test_target_distribution(pdir: Path, J: dict):
    """Independent pandas on the same rows: research, scored windows, eligible."""
    P = pd.read_parquet(pdir / "panel.parquet", columns=["timestamp", "label_buyable_o1"] + TARGETS)
    P = P[(P["timestamp"] >= pd.Timestamp(J["folds"][0]["test_start"]))
          & (P["timestamp"] <= pd.Timestamp(J["research_end"])) & (P["label_buyable_o1"] == 1)]
    worst = 0.0
    for tg in TARGETS:
        y = P[tg].dropna()
        td = J["target_distribution"][tg]
        ref = {"mean_bp": y.mean() * 1e4, "median_bp": y.median() * 1e4, "sd_bp": y.std() * 1e4,
               "positive_pct": (y > 0).mean() * 100, "above_cost_pct": (y > 0.0035).mean() * 100,
               **{f"p{p}_bp": y.quantile(p / 100) * 1e4 for p in (1, 5, 25, 50, 75, 95, 99)}}
        worst = max([worst] + [abs(ref[k] - td[k]) for k in ref])
        check(td["n"] == len(y), f"{tg}: N = {len(y):,} rows, same as independent count")
        mk = P.dropna(subset=[tg]).groupby("timestamp")[tg].mean().mean() * 1e4
        check(abs(mk - td["market_mean_bp"]) < 1e-9 and abs(td["market_net_after_cost_bp"] - (mk - 35)) < 1e-9,
              f"{tg}: market mean {mk:+.2f} bp and net after 35 bp reproduced")
        for f in J["folds"]:
            w = y[(P.loc[y.index, "timestamp"] >= pd.Timestamp(f["test_start"]))
                  & (P.loc[y.index, "timestamp"] <= pd.Timestamp(f["test_end"]))]
            if abs((w > 0).mean() * 100 - td["positive_pct_by_fold"][f"fold{f['fold']}"]) > 1e-9:
                check(False, f"{tg}: positive share in fold {f['fold']} reproduced")
                break
        else:
            check(True, f"{tg}: positive share reproduced in all 5 folds")
    check(worst < 1e-8, f"mean, median, SD, p1..p99, positive %, above-cost % reproduced (max diff {worst:.1e})")


def test_world():
    print("2. a planted world, screened end to end")
    tmp = Path(tempfile.mkdtemp())
    try:
        pdir = make_world(tmp)
        rc, log = quiet(SB.main, ["--panel", str(pdir)])
        check(rc == 0 and "STAGE B COMPLETE" in log, "main() completes and prints its verdict line")
        runs = sorted((pdir / "stage_b").iterdir())
        check(len(runs) == 1, f"one run folder written ({runs[0].name if runs else '-'})")
        out = runs[0]
        for fn in ("screen.csv", "market_level.csv", "ic_daily.parquet", "stage_b.json", "stage_b_report.md"):
            check((out / fn).exists(), f"output {fn}")
        C = pd.read_csv(out / "screen.csv")
        J = json.loads((out / "stage_b.json").read_text(encoding="utf-8"))

        feats = set(C["feature"])
        check(not any(f.startswith("label_") for f in feats), "no label_* column was screened")
        check(not (feats & {"open", "high", "low", "close", "volume"}), "OHLCV never screened")
        check("GOLD_close" in J["excluded"]["raw_levels_of_outside_series"] and "GOLD_close" not in feats,
              "raw level of an outside series excluded (GOLD_close)")
        check("D_close_proxy" in J["excluded"]["price_level_proxies"] and "D_close_proxy" not in feats,
              "stock price-level proxy excluded (D_close_proxy)")
        check("MKT_regime" in J["market_level"] and "MKT_regime" not in feats,
              "market-level column detected and kept out of the ranked screen (MKT_regime)")
        M = pd.read_csv(out / "market_level.csv")
        mr = M[M["feature"] == "MKT_regime"].iloc[0]
        check(mr["label_oc_1_ts_corr"] > 0.2 and mr["label_oc_1_ts_t_hac"] > 3,
              f"MKT_regime's link to the day's average is visible, descriptively (corr "
              f"{mr['label_oc_1_ts_corr']:+.3f}, t {mr['label_oc_1_ts_t_hac']:+.1f})")

        check(J["max_date_read"] <= J["research_end"] < J["lockbox_start"] == "2025-03-05",
              f"rows read end {J['max_date_read']}, research end {J['research_end']}, lockbox {J['lockbox_start']}")
        check(J["folds"][0]["test_start"] > "2021-06-01" and len(J["folds"]) == 5, "5 folds; first 35% not scored")

        for f in ("D_pos", "D_pos_dup", "D_nanpos", "D_disc"):
            ok = all(bool(cell(C, f, tg)["passes"]) and cell(C, f, tg)["mean_ic"] > 0 for tg in TARGETS)
            check(ok, f"{f}: passes on all 4 targets with positive IC (oc_1 IC {cell(C, f)['mean_ic']:+.4f})")
        ok = all(bool(cell(C, "D_neg", tg)["passes"]) and cell(C, "D_neg", tg)["mean_ic"] < 0 for tg in TARGETS)
        check(ok and cell(C, "D_neg")["long_side"] == "Q1",
              f"D_neg: passes with negative IC; the long side is the BOTTOM fifth (Q1)")
        check(cell(C, "D_pos")["long_side"] == "Q5" and cell(C, "D_pos")["quintile_excess_bp"] > 0
              and cell(C, "D_neg")["quintile_excess_bp"] > 0, "favoured quintile beats buying everything for both planted signs")
        check(not any(c.startswith("long_") and c != "long_side" for c in C.columns)
              and {"quintile_gross_bp", "quintile_excess_bp", "quintile_net_after_cost_bp"} <= set(C.columns),
              "money columns named quintile_* (no 'long_net' that reads like a strategy result)")

        noise = C[C["feature"].str.startswith("D_noise_")]
        check(len(noise) == 160 and int(noise["passes"].sum()) <= 2,
              f"pure noise: {int(noise['passes'].sum())} of 160 cells pass (q <= 0.10 and 4/5 folds)")
        u = cell(C, "D_unstable")
        check(u["q_value"] <= 0.10 and int(u["sign_agree_folds"]) == 3 and not bool(u["passes"]),
              f"D_unstable: significant (q {u['q_value']:.1e}) but sign holds in 3/5 folds -> does NOT pass")
        for f, why in (("D_lockonly", "eligibility: only unbuyable rows carry it"),
                       ("D_late", "lockbox: only post-2025-03-05 rows carry it"),
                       ("D_early", "test windows only: it lives before fold 1")):
            r = cell(C, f)
            fmax = max(abs(r[f"fold{i}_ic"]) for i in range(1, 6))
            check(not bool(r["passes"]) and abs(r["mean_ic"]) < 0.02 and fmax < 0.05,
                  f"{f}: does not pass, IC {r['mean_ic']:+.4f} ({why})")
        lk = cell(C, "D_leak")
        check(bool(lk["leak_suspect"]) and "D_leak" in J["summary"]["label_oc_1"]["leak_suspects"],
              f"D_leak flagged as a possible leak (IC {lk['mean_ic']:+.3f})")
        check("possible leak" in log.lower() and "D_leak" in log, "leak warning printed on the console")

        for f in ("D_nanpos", "D_disc"):
            ic, spr, lg, ex = independent_ic_and_money(pdir, f, "label_oc_1", J)
            r = cell(C, f)
            check(abs(ic - r["mean_ic"]) < 1e-10 and abs(spr - r["spread_q5_q1_bp"]) < 1e-7
                  and abs(lg - r["quintile_gross_bp"]) < 1e-7 and abs(ex - r["quintile_excess_bp"]) < 1e-7,
                  f"{f}: IC, spread, quintile gross and quintile excess reproduced by independent day-by-day code "
                  f"(IC {ic:+.6f} vs {r['mean_ic']:+.6f})")
        ic, _, _, _ = independent_ic_and_money(pdir, "D_neg", "label_oc_5", J)
        check(abs(ic - cell(C, "D_neg", "label_oc_5")["mean_ic"]) < 1e-10, "D_neg on oc_5 reproduced too")

        ICD = pd.read_parquet(out / "ic_daily.parquet")
        r = cell(C, "D_pos")
        check(abs(float(ICD["D_pos|label_oc_1"].mean()) - r["mean_ic"]) < 1e-6
              and ICD.index.max() <= pd.Timestamp(J["research_end"]), "daily IC file matches the summary")
        fam = cell(C, "D_pos_dup")["family"]
        check(fam == cell(C, "D_nanpos")["family"] == cell(C, "D_pos")["family"] != cell(C, "D_neg")["family"],
              f"families: D_pos, D_pos_dup, D_nanpos share one leader ({fam}); D_neg is its own")
        rep = (out / "stage_b_report.md").read_text(encoding="utf-8")
        i1, i2, i3 = rep.find("## 1. Target distribution"), rep.find("## 2. Feature screen"), rep.find("## 3. Strongest")
        check(0 < i1 < i2 < i3, "report order: target distribution -> feature screen -> strongest cells")
        low = rep.lower()
        for phrase in ("hole every long trade starts in", "descriptive quintile economics - not a trading result",
                       "it is marginal evidence", "families are descriptive", "does not say anything is tradable",
                       "positive-return share by fold"):
            check(phrase in low, f"report says: '{phrase}'")
        check("TARGET DISTRIBUTION" in log and log.find("TARGET DISTRIBUTION") < log.find("FEATURE SCREEN"),
              "console prints target distributions before the screen counts")
        test_target_distribution(pdir, J)

        led = [e for e in RC.ledger_entries(pdir / "panel.parquet") if e.get("kind") == "stage_b_screen"]
        check(len(led) == 1 and led[0]["experiment_family"] == "oc_v1" and led[0]["run_id"] == out.name,
              "run logged in the research ledger")
        try:
            quiet(SB.main, ["--panel", str(pdir)])
            check(False, "second run without --rerun-reason refused")
        except SystemExit as e:
            check("already run" in str(e), "second run without --rerun-reason refused")
        rc2, _ = quiet(SB.main, ["--panel", str(pdir), "--rerun-reason", "determinism test"])
        runs = sorted((pdir / "stage_b").iterdir())
        C2 = pd.read_csv(runs[-1] / "screen.csv")
        check(rc2 == 0 and len(runs) == 2 and C2.equals(C), "re-run with a logged reason: identical screen.csv")
        led = [e for e in RC.ledger_entries(pdir / "panel.parquet") if e.get("kind") == "stage_b_screen"]
        check(led[-1].get("rerun_reason") == "determinism test", "the re-run reason is in the ledger")
        return pdir, tmp
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def test_power(pdir: Path):
    print("3. power: the planted effects ARE found when a guard is removed")
    orig_e, orig_w = SB.eligible_mask, SB.research_window
    try:
        SB.eligible_mask = lambda b: np.ones(len(b), bool)
        res, _ = quiet(SB.run_screen, pdir / "panel.parquet", TARGETS, verbose=False)
        r = cell(res["cells"], "D_lockonly")
        check(abs(r["t_hac"]) > 5, f"eligibility off -> D_lockonly detected (t {r['t_hac']:+.1f}); so the "
                                   f"guard, not a weak plant, keeps it out")
    finally:
        SB.eligible_mask = orig_e
    try:
        SB.research_window = lambda s, horizon=5: (np.sort(s)[-1], np.sort(s)[-1])
        res, _ = quiet(SB.run_screen, pdir / "panel.parquet", TARGETS, verbose=False)
        r = cell(res["cells"], "D_late")
        check(r["fold5_ic"] > 0.2, f"lockbox read -> D_late visible in the last fold (IC {r['fold5_ic']:+.3f}); so "
                                   f"the pinned boundary, not a weak plant, keeps it out")
    finally:
        SB.research_window = orig_w


def test_preconditions(pdir: Path):
    print("4. preconditions")
    _, sha = SB.load_manifest()

    def fresh(mutate):
        t = Path(tempfile.mkdtemp())
        q = t / "panel_oc"
        q.mkdir()
        shutil.copy(pdir / "panel.parquet", q / "panel.parquet")
        shutil.copy(pdir / "panel_meta.json", q / "panel_meta.json")
        mutate(q)
        return t, q

    cases = []

    def no_audit(q):
        pass
    cases.append((no_audit, "no label audit on record"))

    def failed(q):
        RC.ledger_append(q / "panel.parquet", {"kind": "panel_audit", "passed": False, "manifest_sha": sha})
    cases.append((failed, "FAILED"))

    def stale(q):
        RC.ledger_append(q / "panel.parquet", {"kind": "panel_audit", "passed": True, "manifest_sha": sha})
        m = json.loads((q / "panel_meta.json").read_text())
        m["built_at"] = (pd.Timestamp.now() + pd.Timedelta(days=1)).isoformat()
        (q / "panel_meta.json").write_text(json.dumps(m))
    cases.append((stale, "rebuilt"))

    def other_sha(q):
        RC.ledger_append(q / "panel.parquet", {"kind": "panel_audit", "passed": True, "manifest_sha": "0" * 16})
    cases.append((other_sha, "changed since the audit"))

    def old_build(q):
        RC.ledger_append(q / "panel.parquet", {"kind": "panel_audit", "passed": True, "manifest_sha": sha})
        m = json.loads((q / "panel_meta.json").read_text())
        m["panel_build_version"] = "panel_build v31"
        (q / "panel_meta.json").write_text(json.dumps(m))
    cases.append((old_build, "v32+"))

    for mut, expect in cases:
        t, q = fresh(mut)
        try:
            quiet(SB.main, ["--panel", str(q)])
            check(False, f"refuses: {mut.__name__}")
        except SystemExit as e:
            check(expect in str(e) and not (q / "stage_b").exists(),
                  f"refuses: {mut.__name__} ({str(e)[:70]}...)")
        finally:
            shutil.rmtree(t, ignore_errors=True)

    import signal_engine as SE
    orig = SE.clean_features
    try:
        SE.clean_features = lambda p, **k: (orig(p)[0] + ["label_oc_1"], orig(p)[1])
        try:
            quiet(SB.run_screen, pdir / "panel.parquet", TARGETS, verbose=False)
            check(False, "a label smuggled into the feature list is refused")
        except RC.LabelLeakError:
            check(True, "a label smuggled into the feature list is refused (LabelLeakError)")
    finally:
        SE.clean_features = orig


def main():
    test_module_origin()
    test_statistics()
    pdir, tmp = test_world()
    try:
        test_power(pdir)
        test_preconditions(pdir)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if FAILS:
        print(f"FAILED  {len(FAILS)} check(s):")
        for f in FAILS:
            print("   - " + f)
        return 1
    print(f"VERIFIED  {SB.CODE_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
