#!/usr/bin/env python3
"""
tests/test_stage_d.py - stage_d_gate v1. Prints VERIFIED on success.

Run from the open_close folder with the main folder on PYTHONPATH:
    python tests\\test_stage_d.py

 0. The modules loaded are this folder's (no shadowing by the main folder).
 1. Pieces against independent code: the day-demeaned clipped label; the block
    bootstrap (at 95% it must equal research_common's exactly); daily top-n
    with ties broken by symbol and missing outcomes counted, not skipped.
 2. The declaration: the tool refuses when its settings differ.
 3. A planted world where each target has a different truth:
      oc_1  pure noise                                -> FAIL
      oc_2  real selection skill, but the market loses 200 bp a trade
            (excess passes, net does not)             -> FAIL on net, not excess
      oc_3  strong skill that stops at 2024           -> FAIL on the 2024+ rule only
      oc_5  strong persistent skill                   -> PASS
    plus: UNBUYABLE rows that a feature can spot and whose labels are fictional
    big wins (like upper-band locks), missing outcomes on the strongest
    oc_5 rows, an unresolved tail. Checks: verdicts and reasons; every daily
    top-3 and excess re-derived from preds.parquet and the panel by pandas;
    no pick is ever unbuyable; embargo gaps; labels clipped from training rows
    only; ledger order (declared before results); re-run guard.
 4. Power and sabotage: eligibility off -> the model buys unbuyable rows;
    embargo 0 -> EmbargoError; a label in the feature list -> LabelLeakError;
    stage B not run -> refused.
"""

from __future__ import annotations

import contextlib
import io
import json
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
import stage_d_gate as SD      # noqa: E402

FAILS = []
TARGETS = ["label_oc_1", "label_oc_2", "label_oc_3", "label_oc_5"]


def check(cond, msg):
    print(("  ok  " if cond else "  FAIL") + f"  {msg}")
    if not cond:
        FAILS.append(msg)


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **k)
    return out, buf.getvalue()


# ----------------------------------------------------------------------
def test_module_origin():
    print("0. the new modules are the ones loaded (no shadowing by the main folder)")
    here = HERE.parent.resolve()
    for m in (SD, SB, RC, PB):
        check(Path(m.__file__).resolve().parent == here, f"{m.__name__} from {Path(m.__file__).resolve().parent}")


def test_pieces():
    print("1. pieces against independent code")
    y = np.array([0.01, 0.03, -0.02, 0.10, 0.00, 0.02])
    d = np.array([0, 0, 0, 1, 1, 1])
    lab, lo, hi = SD.make_label(y, d, (0, 100))
    ref = np.array([0.01 - 0.02 / 3, 0.03 - 0.02 / 3, -0.02 - 0.02 / 3, 0.10 - 0.04, -0.04, 0.02 - 0.04])
    check(np.allclose(lab, ref), "label = target minus its day's mean (hand example)")
    rng = np.random.default_rng(1)
    yy = rng.standard_t(3, size=5000)
    dd = np.repeat(np.arange(50), 100)
    lab, lo, hi = SD.make_label(yy, dd, (1, 99))
    dm = yy - pd.Series(yy).groupby(dd).transform("mean").to_numpy()
    check(np.allclose(lab, np.clip(dm, np.percentile(dm, 1), np.percentile(dm, 99)))
          and abs(lo - np.percentile(dm, 1)) < 1e-15, "label clipped at the 1st/99th percentiles of the rows given")

    x = rng.normal(0.001, 0.02, size=900)
    a = SD.block_bootstrap(x, 0.95, 2000, 10, 0)
    b = RC.block_bootstrap_mean(x, B=2000, block=10, seed=0)
    check(a["lo"] == b["lo"] and a["hi"] == b["hi"] and a["mean"] == b["mean"],
          "bootstrap at 95% = research_common.block_bootstrap_mean exactly (same draws)")
    w = SD.block_bootstrap(x, 0.9875, 10000, 10, 0)
    n = SD.block_bootstrap(x, 0.95, 10000, 10, 0)
    check(w["lo"] < n["lo"] and w["hi"] > n["hi"], f"98.75% interval wider than 95% ({w['lo']:.5f} < {n['lo']:.5f})")

    day = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])
    sym = np.array(["B", "A", "C", "D", "E", "A", "B", "C", "D"])
    sc = np.array([0.5, 0.5, 0.9, 0.1, 0.5, 0.3, 0.2, 0.9, 0.8])
    yv = np.array([0.01, 0.02, 0.03, 0.04, 0.05, np.nan, 0.02, np.nan, 0.06])
    D = SD.daily_top_n(day, sym, sc, yv, (1, 3))
    # day 0: C 0.9, then ties at 0.5 broken by symbol: A, B, (E) -> top-3 = C, A, B
    check(abs(D["top3"][0] - np.mean([0.03, 0.02, 0.01])) < 1e-15 and bool(D["tie_at_3"][0]),
          "ties broken by symbol (top-3 = C, A, B; E is 4th) and the tie is reported")
    # day 1: C 0.9 (no outcome), D 0.8, A 0.3 (no outcome) -> top-3 mean over known = D only
    check(abs(D["top3"][1] - 0.06) < 1e-15 and D["top3_missing"][1] == 2 and np.isnan(D["top1"][1])
          and D["top1_missing"][1] == 1,
          "picks with no outcome are picked, counted as missing, and the day uses the known ones")
    check(abs(D["market"][0] - 0.03) < 1e-15 and abs(D["market"][1] - 0.04) < 1e-15,
          "market = mean of known outcomes that day")


def test_declaration():
    print("2. the declaration")
    decl, sha = SD.load_declaration()
    check(decl["config"] == json.loads(json.dumps(SD.CFG)) and len(sha) == 16, f"settings match the declaration (sha {sha})")
    orig = SD.CFG
    try:
        SD.CFG = dict(orig, fit_cap=1_000_000)
        try:
            SD.load_declaration()
            check(False, "a setting that differs from the declaration is refused")
        except SystemExit as e:
            check("fit_cap" in str(e), "a setting that differs from the declaration is refused (names fit_cap)")
    finally:
        SD.CFG = orig


# ----------------------------------------------------------------------
def make_world(root: Path, seed: int = 5, n_sym: int = 40) -> Path:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", "2026-06-30")
    T, N = len(dates), n_sym
    dv = dates.values.astype("datetime64[ns]")
    di = np.repeat(np.arange(T), N)
    si = np.tile(np.arange(N), T)
    keep = rng.random(T * N) >= 0.02
    di, si = di[keep], si[keep]
    R = len(di)
    z = lambda: rng.normal(size=R)
    E2, E3, E5 = z(), z(), z()
    buy = (rng.random(R) >= 0.10).astype(float)
    # like a stock locked at the upper band at T: visible in a feature, and its label is a fictional big win
    LOCK = (buy == 0) * 1.0 + 0.3 * z()
    m2 = (-0.02 + 0.005 * rng.normal(size=T))[di]
    pre24 = (dv[di] < np.datetime64("2024-01-01", "ns")).astype(float)
    lock = np.where(buy == 0, 0.10, 0.0)
    y1 = 0.02 * z() + lock
    y2 = 0.006 * E2 + 0.02 * z() + m2 + lock
    y3 = 0.03 * E3 * pre24 + 0.02 * z() + lock
    y5 = 0.03 * E5 + 0.02 * z() + 0.002 + lock
    y5 = np.where((E5 > 2.3) & (buy == 1) & (rng.random(R) < 0.5), np.nan, y5)
    tail = di >= T - 5
    for y in (y1, y2, y3, y5):
        y[tail] = np.nan
    lvl = np.exp(rng.uniform(np.log(20), np.log(3000), size=N))
    close = lvl[si] * np.exp(np.cumsum(0.01 * rng.normal(size=(T, N)), axis=0))[di, si]
    df = pd.DataFrame({
        "timestamp": dv[di], "symbol": np.array([f"S{i:02d}" for i in range(N)])[si],
        "open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1e5,
        "D_e2": E2, "D_e3": E3, "D_e5": E5, "D_lock": LOCK, "MKT_level": rng.normal(size=T)[di],
        "D_atr_pct": np.abs(rng.normal(3, 1, size=R)), "X_turnover_med": np.exp(rng.normal(17, 1, size=R)),
        "label_oc_1": y1, "label_oc_2": y2, "label_oc_3": y3, "label_oc_5": y5, "label_buyable_o1": buy,
        "label_exit_ret": z(),
    })
    for i in range(6):
        df[f"D_noise_{i}"] = z()
    pdir = root / "panel_oc"
    pdir.mkdir(parents=True)
    df.to_parquet(pdir / "panel.parquet", index=False)
    (pdir / "panel_meta.json").write_text(json.dumps({
        "panel_build_version": PB.PANEL_BUILD_VERSION, "etf_rule": RC.ETF_RULE_VERSION,
        "built_at": (pd.Timestamp.now() - pd.Timedelta(hours=2)).isoformat()}), encoding="utf-8")
    _, sha = SB.load_manifest()
    RC.ledger_append(pdir / "panel.parquet", {"kind": "panel_audit", "passed": True, "manifest_sha": sha,
                                              "tool": "label_audit v2"})
    return pdir


def mark_stage_b(pdir: Path):
    RC.ledger_append(pdir / "panel.parquet", {"kind": "stage_b_screen", "experiment_family": "oc_v1",
                                              "run_id": "STAGEB_TEST"})


def test_world():
    print("3. a planted world with a different truth per target")
    tmp = Path(tempfile.mkdtemp())
    pdir = make_world(tmp)
    pp = pdir / "panel.parquet"
    try:
        quiet(SD.main, ["--panel", str(pdir)])
        check(False, "refused before stage B has run")
    except SystemExit as e:
        check("stage B has not run" in str(e) and not (pdir / "stage_d").exists(), "refused before stage B has run")
    mark_stage_b(pdir)

    rc, log = quiet(SD.main, ["--panel", str(pdir)])
    out = sorted((pdir / "stage_d").iterdir())[-1]
    for fn in ("stage_d.json", "stage_d_report.md", "daily.csv", "picks.csv", "preds.parquet"):
        check((out / fn).exists(), f"output {fn}")
    J = json.loads((out / "stage_d.json").read_text(encoding="utf-8"))
    R = J["results"]
    v = {tg: R[tg]["verdict"] for tg in TARGETS}
    check(not v["label_oc_1"]["pass"] and not v["label_oc_1"]["net_lower_bound_above_zero"],
          f"oc_1 (noise): FAIL (net {R['label_oc_1']['top3_net']['mean'] * 1e4:+.1f} bp)")
    check(not v["label_oc_2"]["pass"] and v["label_oc_2"]["excess_lower_bound_above_zero"]
          and not v["label_oc_2"]["net_lower_bound_above_zero"],
          f"oc_2 (skill in a losing market): excess passes ({R['label_oc_2']['top3_excess']['lo'] * 1e4:+.1f} bp lower "
          f"bound) but net fails ({R['label_oc_2']['top3_net']['mean'] * 1e4:+.1f} bp) -> FAIL")
    check(not v["label_oc_3"]["pass"] and v["label_oc_3"]["net_lower_bound_above_zero"]
          and v["label_oc_3"]["excess_lower_bound_above_zero"] and not v["label_oc_3"]["recent_mean_above_zero"],
          f"oc_3 (skill that stops at 2024): net and excess intervals pass, 2024+ mean "
          f"{R['label_oc_3']['recent_top3_net']['mean'] * 1e4:+.1f} bp -> FAIL on recency alone")
    check(v["label_oc_5"]["pass"], f"oc_5 (persistent skill): PASS (net {R['label_oc_5']['top3_net']['mean'] * 1e4:+.1f} "
                                   f"bp, lower bound {R['label_oc_5']['top3_net']['lo'] * 1e4:+.1f})")
    check("label_oc_5 PASS" in log and "forward paper-test arm" in log, "console names the passing target and its limits")
    check(R["label_oc_5"]["top3_missing_outcomes"] > 0,
          f"oc_5: picks with no outcome are counted ({R['label_oc_5']['top3_missing_outcomes']}), not silently skipped")

    P = pd.read_parquet(pp, columns=["timestamp", "symbol", "label_buyable_o1"] + TARGETS)
    daily = pd.read_csv(out / "daily.csv", parse_dates=["timestamp"])
    preds = pd.read_parquet(out / "preds.parquet")
    picks = pd.read_csv(out / "picks.csv", parse_dates=["timestamp"])
    worst = 0.0
    for tg in TARGETS:
        pr = preds[preds["target"] == tg].sort_values(["timestamp", "score", "symbol"], ascending=[True, False, True])
        top3 = pr.groupby("timestamp").head(3)
        ind = top3.groupby("timestamp")["outcome"].mean() - 0.0035
        E = P[P["label_buyable_o1"] == 1].dropna(subset=[tg])
        mk = E.groupby("timestamp")[tg].mean().reindex(ind.index)
        dd = daily[daily["target"] == tg].set_index("timestamp")
        worst = max(worst, float(np.nanmax(np.abs(ind.to_numpy() - dd["top3_net"].to_numpy()))),
                    float(np.nanmax(np.abs((ind + 0.0035 - mk).to_numpy() - dd["top3_excess"].to_numpy()))))
        check(abs(float(ind.mean()) - R[tg]["top3_net"]["mean"]) < 1e-6,
              f"{tg}: mean top-3 net re-derived by pandas from preds.parquet ({ind.mean() * 1e4:+.2f} bp)")
    check(worst < 1e-6, f"every daily top-3 net and excess (vs a market recomputed from the panel) matches "
                        f"(max diff {worst:.1e})")
    elig = P.loc[P["label_buyable_o1"] == 1, ["timestamp", "symbol"]]
    j = picks.merge(P[["timestamp", "symbol", "label_buyable_o1"]], on=["timestamp", "symbol"], how="left")
    check(len(j) == len(picks) and (j["label_buyable_o1"] == 1).all(), f"all {len(picks):,} picks are buyable rows")
    n_test = preds.groupby("target").size()
    check(set(n_test.index) == set(TARGETS) and preds["timestamp"].max() <= pd.Timestamp(J["sessions"]["last_resolved"]),
          "predictions only for test windows up to the last resolved session")
    check(len(preds.merge(elig, on=["timestamp", "symbol"])) == len(preds), "every prediction is an eligible row")

    F = pd.DataFrame(J["folds"])
    gaps_ok = all(r["embargo_gap_sessions"] > SD.horizon_of(r["target"]) for _, r in F.iterrows())
    check(gaps_ok and len(F) == 20, f"20 fits; every last training label ends before its test window "
                                    f"(min gap {F['embargo_gap_sessions'].min()} sessions)")
    f1 = F[(F["target"] == "label_oc_5") & (F["fold"] == 1)].iloc[0]
    E = P[(P["label_buyable_o1"] == 1) & (P["timestamp"] <= pd.Timestamp(f1["train_end"]))].dropna(subset=["label_oc_5"])
    dm = E["label_oc_5"] - E.groupby("timestamp")["label_oc_5"].transform("mean")
    check(abs(np.percentile(dm, 1) * 1e4 - f1["label_clip_bp"][0]) < 1e-9
          and abs(np.percentile(dm, 99) * 1e4 - f1["label_clip_bp"][1]) < 1e-9,
          "fold-1 label clip bounds recomputed from its training rows only")

    led = RC.ledger_entries(pp)
    kinds = [e.get("kind") for e in led]
    i_dec, i_res = kinds.index("stage_d_declared"), kinds.index("stage_d_gate")
    check(i_dec < i_res and led[i_dec]["declaration_sha"] == J["declaration_sha"] == led[i_res]["declaration_sha"],
          "ledger: declaration hash logged BEFORE the results, same hash on both")
    check(led[i_res]["passed"] == ["label_oc_5"], "ledger records which targets passed")
    rep = (out / "stage_d_report.md").read_text(encoding="utf-8")
    check("**FAIL** (2024+ mean)" in rep and "**FAIL** (net interval, 2024+ mean)" in rep and "**PASS**" in rep,
          "report states each verdict with its reason")
    try:
        quiet(SD.main, ["--panel", str(pdir)])
        check(False, "a second run without --rerun-reason is refused")
    except SystemExit as e:
        check("already run" in str(e), "a second run without --rerun-reason is refused")
    return tmp, pdir, J, daily


def test_power_and_sabotage(pdir: Path, J: dict, daily: pd.DataFrame):
    print("4. power and sabotage")
    pp = pdir / "panel.parquet"
    res, _ = quiet(SD.run_gate, pp, ["label_oc_5"], verbose=False)
    d5 = daily[daily["target"] == "label_oc_5"].reset_index(drop=True)
    check(np.allclose(res["daily"]["top3_net"].to_numpy(), d5["top3_net"].to_numpy(), equal_nan=True),
          "deterministic: re-running oc_5 gives identical daily top-3")
    orig = SD.eligible_mask
    try:
        SD.eligible_mask = lambda b: np.ones(len(b), bool)
        res, _ = quiet(SD.run_gate, pp, ["label_oc_1"], verbose=False)
        P = pd.read_parquet(pp, columns=["timestamp", "symbol", "label_buyable_o1"])
        pk = res["picks"][res["picks"]["rank"] <= 3].merge(P, on=["timestamp", "symbol"])
        share = float((pk["label_buyable_o1"] == 0).mean())
        check(share > 0.5, f"eligibility off -> {share:.0%} of top-3 picks are unbuyable (planted lock effect); so the "
                           f"guard is what keeps them out")
    finally:
        SD.eligible_mask = orig
    try:
        quiet(SD.run_gate, pp, ["label_oc_1"], verbose=False, cfg=dict(SD.CFG, embargo=0))
        check(False, "embargo 0 -> EmbargoError")
    except SD.EmbargoError:
        check(True, "embargo 0 -> EmbargoError before any fit")
    import signal_engine as SE
    o = SE.clean_features
    try:
        SE.clean_features = lambda p, **k: (o(p)[0] + ["label_oc_5"], o(p)[1])
        try:
            quiet(SD.run_gate, pp, ["label_oc_1"], verbose=False)
            check(False, "a label smuggled into the features is refused")
        except RC.LabelLeakError:
            check(True, "a label smuggled into the features is refused (LabelLeakError)")
    finally:
        SE.clean_features = o
    o2 = SD.run_gate
    try:
        SD.run_gate = lambda pp_, tg, verbose=True, cfg=None: None
        try:
            quiet(SD.main, ["--panel", str(pdir), "--rerun-reason", "test"])
        except Exception:
            pass
        led = [e for e in RC.ledger_entries(pp) if e.get("kind") == "stage_d_declared"]
        check(led[-1].get("rerun_reason") == "test", "a re-run with a reason proceeds and logs the reason")
    finally:
        SD.run_gate = o2


def main():
    test_module_origin()
    test_pieces()
    test_declaration()
    tmp, pdir, J, daily = test_world()
    try:
        test_power_and_sabotage(pdir, J, daily)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if FAILS:
        print(f"FAILED  {len(FAILS)} check(s):")
        for f in FAILS:
            print("   - " + f)
        return 1
    print(f"VERIFIED  {SD.CODE_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
