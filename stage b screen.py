#!/usr/bin/env python3
"""
stage_b_screen.py - stage B of family oc_v1: which EXISTING features rank
stocks for the four open-to-close targets?

    python stage_b_screen.py --panel %CACHE_DAILY_ROOT%\\panel_oc
    python stage_b_screen.py --panel %CACHE_DAILY_ROOT%\\panel_oc --rerun-reason "fixing <bug>"

WHAT IT ANSWERS - AND WHAT IT DOES NOT
======================================
For every clean feature and every target (label_oc_1/2/3/5: buy at the OPEN
of T+1, sell at the CLOSE of T+h), does ranking stocks by the feature at the
close of T line up with how they then perform? It trains no model and picks
no trades. It is a screen: it says which features carry a stable, ranking
signal and how big that signal is in money. Stage D decides tradability.

THE RULE (declared in EXPERIMENT_OC_FAMILY.json before any result existed)
-------------------------------------------------------------------------
A feature x target cell PASSES only if BOTH hold:
    1. Benjamini-Hochberg q <= 0.10 across every tested cell, and
    2. the IC has the pooled sign in at least 4 of the 5 walk-forward folds.
The highest IC alone never selects anything. This file checks that the
manifest still says exactly that, and that the manifest's hash is the one the
passing label audit recorded - an edited manifest cannot run.

THE MEASURE: PER-DATE RANK IC
-----------------------------
Each day, rank the eligible stocks by the feature and by the target and
correlate the two ranks (Spearman). One number per day; the day is the unit,
because the system decides once a day. Ranking within a day also cancels the
market: if every stock rose 2% the ranks do not change. So an IC is pure
stock SELECTION skill. The money side is reported separately, as
DESCRIPTIVE QUINTILE ECONOMICS - NOT A TRADING RESULT:
    spread_q5_q1_bp             mean(top fifth) - mean(bottom fifth), per day
    long_side                   the fifth the IC's sign favours (Q5 if IC > 0, else Q1)
    quintile_gross_bp           that fifth's mean return per trade
    quintile_excess_bp          that fifth minus buying every eligible stock that day
                                (the manifest's "spread excess over the market": a
                                Q5-Q1 spread is already market-neutral)
    quintile_net_after_cost_bp  that fifth minus the 35 bp round trip. A fifth is
                                ~150 stocks a day, far milder than a top-3 pick.

MARGINAL, NOT INCREMENTAL
-------------------------
Stage B asks whether each feature ON ITS OWN ranks future returns (marginal
evidence). Whether it adds anything beyond the other features (incremental
evidence) is a stage-D question. Families (|rank correlation| >= 0.80 among
passing features) are descriptive: nothing is removed because of them.

REPORT ORDER
------------
target distributions (N, mean, median, SD, percentiles, positive share, by
fold; buying everything net of cost) -> feature-screen counts -> possible
leaks -> strongest passing cells -> market-level columns -> caveats.

HONEST STATISTICS
-----------------
* t-statistic: Newey-West (lag 20 sessions = 4x the longest target). Labels
  of 2-5 sessions overlap, and slow features (52-week ranges, volatility)
  keep today's IC close to yesterday's; a plain t would count those days as
  independent and overstate the evidence.
* q-values: Benjamini-Hochberg over every tested cell (~1,000). Out of all
  cells with q <= 0.10, about 10% are expected to be flukes; that is the
  price of the screen, and why stage D must still pass its own gate.
* Stability: 4 of 5 folds. A signal that lived in 2020 and died after fails.
* Market-level columns (the same value for every stock on a day: NIFTY
  returns, VIX, crude ...) cannot rank stocks, so they have no IC. They are
  listed separately with a DESCRIPTIVE time-series correlation to the day's
  average target - outside the family, no pass/fail, no multiple-testing
  correction; slow series inflate its t.
* Any |IC| >= 0.15 is flagged as a possible leak. The old atlas never saw
  more than ~0.04; a daily cross-sectional IC that large is more likely a
  feature that knows tomorrow than a discovery. Investigate before stage D.

WHICH ROWS
----------
* Eligible rows only: label_buyable_o1 == 1 (the next open was buyable).
* Research rows only: the spent lockbox began 2025-03-05 (pinned - the 15%
  rule would drift as the panel grows) and rows whose 5-session target
  reaches into it are cut too. Rows from 2025-03-05 on are never read, so
  stage D's recency check keeps a period no feature was selected on.
* Test windows only: the 5 folds of feasibility_test.splits (the same
  windows a walk-forward model is judged on). The first 35% of research
  sessions, a walk-forward model's first training window, is not scored.
* Features: signal_engine.clean_features - the panel's model-safe columns
  minus OHLCV, raw levels of outside series and stock price-level proxies.
  The label firewall is asserted on the final list.

WHAT STAGE D MUST NOT DO WITH THIS
----------------------------------
Use this pass list as the feature set of a walk-forward over these same
windows. The list was chosen by looking at those windows, so a model built on
it would be graded partly on its own answer key. Stage D reselects inside
each fold's training window (nested); this run says whether there is signal
at all, where, and how large.

OUTPUT: <panel>\\stage_b\\STAGEB_YYYYMMDD_NNN\\
    screen.csv         every feature x target cell, every recorded statistic
    market_level.csv   market-level columns, descriptive only
    ic_daily.parquet   the daily IC series of every cell (for later checks)
    stage_b.json       config, data window, folds, per-target summary
    stage_b_report.md  the readable report
and one entry in <panel>\\research_ledger.jsonl.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import research_common as RC     # first: from a subfolder, this folder's copy must win
import panel_build as PB         # noqa: E402

CODE_VERSION = "stage_b_screen v1.1"   # v1.1: target distributions first; quintile_* names (never run as v1)
FAMILY_FILE = "EXPERIMENT_OC_FAMILY.json"
FAMILY_ID = "oc_v1"
ELIG_COL = "label_buyable_o1"
MIN_PANEL_BUILD = 32
LOCKBOX_START_PINNED = "2025-03-05"   # the spent lockbox's first session; never moves

CFG = {
    "horizon_max": 5,            # research rows end this many sessions before the lockbox (+1)
    "n_splits": 5, "min_train_frac": 0.35, "embargo": 5,   # feasibility_test.splits
    "min_names": 15,             # eligible stocks needed on a day for that day's IC
    "min_dates": 60,             # days needed for a cell to be tested at all
    "hac_lag": 20,               # Newey-West lag, sessions
    "q_max": 0.10,               # declared in oc_v1
    "min_sign_folds": 4,         # declared in oc_v1
    "quantiles": 5,
    "market_level_share": 0.95,  # constant across stocks on >= 95% of days -> market-level
    "leak_ic": 0.15,
    "family_corr": 0.80, "family_sample_dates": 60,
    "cost_bps": 35.0,
    "seed": 0,
}


def _log(msg: str) -> None:
    print(f"{dt.datetime.now():%H:%M:%S}  {msg}", flush=True)


def _naive(s) -> pd.Series:
    t = pd.to_datetime(pd.Series(s))
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_localize(None)
    return t


# ----------------------------------------------------------------------
# preconditions: the declared rule, an audited panel
# ----------------------------------------------------------------------
class PreconditionError(SystemExit):
    pass


def load_manifest() -> Tuple[dict, str]:
    mf = HERE / FAMILY_FILE
    if not mf.exists():
        raise PreconditionError(f"{FAMILY_FILE} not found next to {Path(__file__).name}")
    raw = mf.read_bytes()
    man = json.loads(raw.decode("utf-8"))
    sha = hashlib.sha256(raw).hexdigest()[:16]
    if man.get("family_id") != FAMILY_ID:
        raise PreconditionError(f"manifest family is {man.get('family_id')!r}, this tool screens {FAMILY_ID}")
    sb = man.get("stage_B_feature_screen", {})
    txt = json.dumps(sb)
    if "q <= 0.10" not in txt or "at least 4 of 5" not in txt:
        raise PreconditionError("the manifest's stage-B rule is not 'q <= 0.10 AND sign in at least 4 of 5 "
                                "folds' - this tool implements only that rule")
    return man, sha


def check_panel(pp: Path, manifest_sha: str) -> dict:
    meta_p = pp.parent / "panel_meta.json"
    if not pp.exists() or not meta_p.exists():
        raise PreconditionError(f"no panel at {pp.parent} (need panel.parquet and panel_meta.json)")
    meta = json.loads(meta_p.read_text(encoding="utf-8"))
    ver = str(meta.get("panel_build_version", ""))
    m = re.search(r"v(\d+)", ver)
    if not m or int(m.group(1)) < MIN_PANEL_BUILD:
        raise PreconditionError(f"panel built by {ver or 'an unversioned panel_build'}; stage B needs "
                                f"panel_build v{MIN_PANEL_BUILD}+ (open-to-close targets, ETFs out)")
    audits = [e for e in RC.ledger_entries(pp) if e.get("kind") == "panel_audit"]
    if not audits:
        raise PreconditionError("this panel has no label audit on record - run label_audit.py --panel first")
    last = audits[-1]
    if not last.get("passed"):
        raise PreconditionError("the latest label audit of this panel FAILED - stage B runs only on an audited panel")
    built = pd.Timestamp(meta.get("built_at", "1970-01-01"))
    if pd.Timestamp(last.get("at", "1970-01-01")) < built:
        raise PreconditionError(f"the panel was rebuilt ({built}) after its last audit ({last.get('at')}) - "
                                f"audit it again first")
    if last.get("manifest_sha") != manifest_sha:
        raise PreconditionError(f"{FAMILY_FILE} changed since the audit recorded it "
                                f"(audit sha {last.get('manifest_sha')}, now {manifest_sha}). The declared "
                                f"rules cannot change after declaration; a change is a new family.")
    return {"panel_build_version": ver, "built_at": str(built), "audit_at": last.get("at"),
            "audit_tool": last.get("tool"), "etf_rule": meta.get("etf_rule")}


def check_rerun(pp: Path, reason: Optional[str]) -> List[dict]:
    prior = [e for e in RC.ledger_entries(pp)
             if e.get("kind") == "stage_b_screen" and e.get("experiment_family") == FAMILY_ID]
    if prior and not reason:
        raise PreconditionError(
            f"stage B has already run on this panel ({prior[-1].get('run_id')}). Re-running is allowed only "
            f"to fix a verified bug: pass --rerun-reason \"...\" - it goes into the ledger. Never re-run to "
            f"get a different answer.")
    return prior


# ----------------------------------------------------------------------
# which rows
# ----------------------------------------------------------------------
def research_window(sessions: np.ndarray, horizon: int = CFG["horizon_max"]):
    """(research_end, lockbox_start) from the PINNED lockbox start."""
    s = np.sort(np.asarray(sessions, dtype="datetime64[ns]"))
    lb = np.datetime64(LOCKBOX_START_PINNED, "ns")
    i = int(np.searchsorted(s, lb))
    if i >= len(s):
        raise PreconditionError(f"the panel ends before the pinned lockbox start {LOCKBOX_START_PINNED}")
    if i - horizon - 1 < 0:
        raise PreconditionError("no research sessions before the lockbox")
    return s[i - horizon - 1], s[i]


def eligible_mask(buyable: np.ndarray) -> np.ndarray:
    return buyable == 1


def fold_windows(research_sessions: np.ndarray) -> List[dict]:
    import feasibility_test as FT
    return FT.splits(research_sessions, CFG["n_splits"], CFG["embargo"], CFG["min_train_frac"])


# ----------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------
def group_rank(d: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Average ranks (1..n, ties share the mean rank) of v within each value of d."""
    n = len(v)
    if n == 0:
        return np.empty(0)
    o = np.lexsort((v, d))
    vs, ds = v[o], d[o]
    gb = np.empty(n, bool)
    gb[0] = True
    gb[1:] = ds[1:] != ds[:-1]
    rb = gb.copy()
    rb[1:] |= vs[1:] != vs[:-1]
    run_start = np.flatnonzero(rb)
    run_end = np.append(run_start[1:], n) - 1
    run_id = np.cumsum(rb) - 1
    g_start = np.flatnonzero(gb)[np.cumsum(gb) - 1]
    r = np.empty(n)
    r[o] = (run_start[run_id] + run_end[run_id]) / 2.0 - g_start + 1.0
    return r


def daily_ic(d: np.ndarray, rx: np.ndarray, ry: np.ndarray, n_dates: int, min_names: int):
    """Per-date Pearson correlation of within-date ranks = per-date Spearman IC."""
    cnt = np.bincount(d, minlength=n_dates).astype(float)
    mid = (cnt[d] + 1.0) / 2.0          # mean of average ranks within a date, exactly
    cx, cy = rx - mid, ry - mid
    sxy = np.bincount(d, cx * cy, minlength=n_dates)
    sxx = np.bincount(d, cx * cx, minlength=n_dates)
    syy = np.bincount(d, cy * cy, minlength=n_dates)
    ok = (cnt >= min_names) & (sxx > 0) & (syy > 0)
    ic = np.full(n_dates, np.nan)
    ic[ok] = sxy[ok] / np.sqrt(sxx[ok] * syy[ok])
    return ic, cnt


def hac_t(x: np.ndarray, lag: int) -> float:
    """Newey-West (Bartlett) t-statistic of the mean of a date-ordered series."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    e = x - x.mean()
    g0 = float(e @ e) / n
    v = g0
    for l in range(1, min(lag, n - 1) + 1):
        v += 2.0 * (1.0 - l / (lag + 1.0)) * float(e[l:] @ e[:-l]) / n
    if not v > 0:
        v = g0
    return float(x.mean() / math.sqrt(v / n)) if v > 0 else float("nan")


def p_two_sided(t: float) -> float:
    return float(math.erfc(abs(t) / math.sqrt(2.0))) if np.isfinite(t) else float("nan")


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    if n == 0:
        return p
    o = np.argsort(p, kind="mergesort")
    ranked = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1.0)
    return out


# ----------------------------------------------------------------------
# the screen
# ----------------------------------------------------------------------
def clean_feature_list(pp: Path) -> Tuple[List[str], dict]:
    import signal_engine as SE      # after RC / PB: SE puts the main folder first on sys.path
    feats, excluded = SE.clean_features(pp)
    RC.assert_no_label_leak(feats, "stage_b_screen")
    return list(feats), excluded


def _read_col(pp: Path, col: str) -> np.ndarray:
    import pyarrow.parquet as pq
    s = pq.read_table(pp, columns=[col]).column(0).to_pandas()
    return pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64")


def run_screen(pp: Path, targets: List[str], verbose: bool = True) -> dict:
    import pyarrow.parquet as pq
    t0 = time.perf_counter()
    cfg = dict(CFG)
    schema = pq.ParquetFile(pp).schema_arrow.names
    miss = [c for c in targets + [ELIG_COL, "timestamp", "symbol"] if c not in schema]
    if miss:
        raise PreconditionError(f"panel is missing {miss}")

    feats, excluded = clean_feature_list(pp)
    base = pd.read_parquet(pp, columns=["timestamp", "symbol", ELIG_COL])
    ts = _naive(base["timestamp"]).to_numpy(dtype="datetime64[ns]")
    sym = base["symbol"].astype(str).to_numpy()
    buy = pd.to_numeric(base[ELIG_COL], errors="coerce").to_numpy(dtype=float)
    del base
    sessions = np.unique(ts)
    research_end, lockbox_start = research_window(sessions)
    rs = sessions[sessions <= research_end]
    folds = fold_windows(rs)
    test_start = np.datetime64(pd.Timestamp(folds[0]["test_start"]), "ns")
    test_sessions = rs[rs >= test_start]
    fold_of = np.zeros(len(test_sessions), dtype=int)
    for f in folds:
        a = np.datetime64(pd.Timestamp(f["test_start"]), "ns")
        b = np.datetime64(pd.Timestamp(f["test_end"]), "ns")
        fold_of[(test_sessions >= a) & (test_sessions <= b)] = int(f["fold"])

    in_test = (ts >= test_start) & (ts <= research_end)
    keep = in_test & eligible_mask(buy)
    order = np.lexsort((sym, ts))
    sel = order[keep[order]]
    d = np.searchsorted(test_sessions, ts[sel])
    nD = len(test_sessions)
    rows_total, rows_test = int(len(ts)), int(in_test.sum())
    rows_ineligible_test = int((in_test & ~eligible_mask(buy)).sum())
    del ts, sym, buy, keep, order
    if verbose:
        _log(f"research rows end {pd.Timestamp(research_end).date()} (lockbox from "
             f"{pd.Timestamp(lockbox_start).date()} never read) | scored windows "
             f"{pd.Timestamp(test_sessions[0]).date()}..{pd.Timestamp(test_sessions[-1]).date()} "
             f"({nD:,} sessions) | {len(sel):,} eligible rows | {len(feats)} clean features")

    Y, YM, RY, MKT = {}, {}, {}, {}
    for tg in targets:
        y = _read_col(pp, tg)[sel]
        m = np.isfinite(y)
        Y[tg], YM[tg] = y, m
        RY[tg] = group_rank(d[m], y[m])
        c = np.bincount(d[m], minlength=nD).astype(float)
        s = np.bincount(d[m], y[m], minlength=nD)
        with np.errstate(invalid="ignore", divide="ignore"):
            MKT[tg] = np.where(c > 0, s / np.maximum(c, 1), np.nan)

    target_dist = target_distributions(Y, YM, MKT, d, fold_of, targets, cfg)

    rng = np.random.default_rng(cfg["seed"])
    samp_dates = np.sort(rng.choice(nD, min(cfg["family_sample_dates"], nD), replace=False))
    samp = np.isin(d, samp_dates)
    samp_vals: Dict[str, np.ndarray] = {}

    Q = cfg["quantiles"]
    cells, market_rows, ic_cols = [], [], {}
    for k, f in enumerate(feats, 1):
        x = _read_col(pp, f)[sel]
        fx = np.isfinite(x)
        cover = float(fx.mean()) if len(fx) else 0.0
        # market-level? constant across stocks on (almost) every day
        dv = d[fx]
        is_mkt, const_share = False, float("nan")
        if fx.sum() > 0:
            g = pd.DataFrame({"d": dv, "x": x[fx]}).groupby("d")["x"].agg(["min", "max", "size"])
            g = g[g["size"] >= 2]
            if len(g):
                const_share = float((g["max"] == g["min"]).mean())
                is_mkt = const_share >= cfg["market_level_share"]
        if is_mkt:
            per_day = pd.Series(x[fx]).groupby(dv).mean()
            v = np.full(nD, np.nan)
            v[per_day.index.to_numpy()] = per_day.to_numpy()
            row = {"feature": f, "coverage": cover, "constant_share": const_share}
            for tg in targets:
                a, b = v, MKT[tg]
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() >= cfg["min_dates"]:
                    za = (a[ok] - a[ok].mean()) / (a[ok].std() or np.nan)
                    zb = (b[ok] - b[ok].mean()) / (b[ok].std() or np.nan)
                    z = za * zb
                    row[f"{tg}_ts_corr"] = float(np.nanmean(z))
                    row[f"{tg}_ts_t_hac"] = hac_t(z, cfg["hac_lag"])
                    row[f"{tg}_ts_days"] = int(ok.sum())
            market_rows.append(row)
            if verbose and k % 25 == 0:
                _log(f"  {k}/{len(feats)} features")
            continue

        samp_vals[f] = x[samp].astype("float32")
        for tg in targets:
            m = fx & YM[tg]
            rec = {"feature": f, "target": tg, "coverage": cover, "n_obs": int(m.sum())}
            if m.sum() == 0:
                rec.update({"n_dates": 0, "testable": False})
                cells.append(rec)
                continue
            dm = d[m]
            rx = group_rank(dm, x[m])
            ry = RY[tg] if m.sum() == YM[tg].sum() else group_rank(dm, Y[tg][m])
            ic, cnt = daily_ic(dm, rx, ry, nD, cfg["min_names"])
            okd = np.isfinite(ic)
            ics = ic[okd]
            n_dates = int(okd.sum())
            mean_ic = float(ics.mean()) if n_dates else float("nan")
            t = hac_t(ics, cfg["hac_lag"]) if n_dates >= 3 else float("nan")
            fold_ic = []
            for fo in range(1, cfg["n_splits"] + 1):
                w = okd & (fold_of == fo)
                fold_ic.append(float(ic[w].mean()) if w.any() else float("nan"))
            sg = np.sign(mean_ic) if np.isfinite(mean_ic) else 0.0
            agree = int(sum(1 for v in fold_ic if np.isfinite(v) and sg != 0 and np.sign(v) == sg))
            pos = int(sum(1 for v in fold_ic if np.isfinite(v) and v > 0))
            # quintile money
            pct = (rx - 0.5) / cnt[dm]
            qi = np.minimum((pct * Q).astype(int), Q - 1)
            key = dm * Q + qi
            qc = np.bincount(key, minlength=nD * Q).reshape(nD, Q)
            qs = np.bincount(key, Y[tg][m], minlength=nD * Q).reshape(nD, Q)
            with np.errstate(invalid="ignore", divide="ignore"):
                qm = np.where(qc > 0, qs / np.maximum(qc, 1), np.nan)
            both = okd & (qc[:, 0] > 0) & (qc[:, Q - 1] > 0)
            long_q = Q - 1 if (np.isfinite(mean_ic) and mean_ic >= 0) else 0
            spread = qm[both, Q - 1] - qm[both, 0]
            longv = qm[both, long_q]
            mk = MKT[tg][both]
            rec.update({
                "n_dates": n_dates, "testable": n_dates >= cfg["min_dates"] and np.isfinite(t),
                "names_per_day_median": float(np.median(cnt[okd])) if n_dates else float("nan"),
                "mean_ic": mean_ic, "ic_sd": float(ics.std(ddof=1)) if n_dates > 1 else float("nan"),
                "t_hac": t, "p_value": p_two_sided(t),
                **{f"fold{i + 1}_ic": v for i, v in enumerate(fold_ic)},
                "positive_folds": pos, "sign_agree_folds": agree,
                "spread_q5_q1_bp": float(spread.mean() * 1e4) if len(spread) else float("nan"),
                "long_side": f"Q{long_q + 1}",
                "quintile_gross_bp": float(longv.mean() * 1e4) if len(longv) else float("nan"),
                "market_bp": float(mk.mean() * 1e4) if len(mk) else float("nan"),
                "quintile_excess_bp": float((longv - mk).mean() * 1e4) if len(longv) else float("nan"),
                "quintile_net_after_cost_bp": (float(longv.mean() * 1e4 - cfg["cost_bps"]) if len(longv)
                                               else float("nan")),
                "leak_suspect": bool(np.isfinite(mean_ic) and abs(mean_ic) >= cfg["leak_ic"]),
            })
            cells.append(rec)
            ic_cols[f"{f}|{tg}"] = ic.astype("float32")
        if verbose and k % 25 == 0:
            _log(f"  {k}/{len(feats)} features ({time.perf_counter() - t0:.0f}s)")

    C = pd.DataFrame(cells)
    for col in ("testable", "leak_suspect"):
        if col not in C:
            C[col] = False
        C[col] = C[col].fillna(False).astype(bool)
    C["q_value"] = np.nan
    tm = C["testable"].to_numpy()
    C.loc[tm, "q_value"] = bh_qvalues(C.loc[tm, "p_value"].to_numpy(float))
    C["q_pass"] = tm & (C["q_value"] <= cfg["q_max"])
    C["stable"] = tm & (C.get("sign_agree_folds", 0) >= cfg["min_sign_folds"])
    C["passes"] = C["q_pass"] & C["stable"]
    C["family"] = ""
    families = {}
    for tg in targets:
        P = C[(C["target"] == tg) & C["passes"]].copy()
        P = P.reindex(P["t_hac"].abs().sort_values(ascending=False).index)
        fam = assign_families(P["feature"].tolist(), samp_vals, d[samp], cfg["family_corr"])
        for f, lead in fam.items():
            C.loc[(C["target"] == tg) & (C["feature"] == f), "family"] = lead
        families[tg] = sorted(set(fam.values()), key=lambda z: -abs(
            float(P.loc[P["feature"] == z, "t_hac"].iloc[0])))

    M = pd.DataFrame(market_rows)
    ICD = pd.DataFrame(ic_cols, index=pd.DatetimeIndex(test_sessions, name="timestamp"))
    market_base = {}
    for tg in targets:
        per = {}
        for fo in range(1, cfg["n_splits"] + 1):
            v = MKT[tg][fold_of == fo]
            per[f"fold{fo}"] = float(np.nanmean(v) * 1e4)
        per["pooled"] = float(np.nanmean(MKT[tg]) * 1e4)
        per["pooled_net_of_cost"] = per["pooled"] - cfg["cost_bps"]
        market_base[tg] = per
    return {
        "cells": C, "market": M, "ic_daily": ICD, "families": families,
        "features": feats, "excluded": excluded, "market_level": M["feature"].tolist() if len(M) else [],
        "research_end": str(pd.Timestamp(research_end).date()),
        "lockbox_start": str(pd.Timestamp(lockbox_start).date()),
        "folds": [{"fold": int(f["fold"]), "train_end": str(pd.Timestamp(f["train_end"]).date()),
                   "test_start": str(pd.Timestamp(f["test_start"]).date()),
                   "test_end": str(pd.Timestamp(f["test_end"]).date()),
                   "sessions": int((fold_of == int(f["fold"])).sum())} for f in folds],
        "rows": {"panel": rows_total, "scored_windows": rows_test, "eligible_scored": int(len(sel)),
                 "ineligible_skipped": rows_ineligible_test},
        "max_date_read": str(pd.Timestamp(test_sessions[-1]).date()),
        "market_baseline_bp": market_base, "target_distribution": target_dist, "seconds": round(time.perf_counter() - t0, 1),
        "config": cfg,
    }


PCTS = (1, 5, 25, 50, 75, 95, 99)


def target_distributions(Y, YM, MKT, d, fold_of, targets, cfg) -> dict:
    """Descriptive: each target over exactly the rows the screen scores (research, test windows, eligible)."""
    out = {}
    for tg in targets:
        y = Y[tg][YM[tg]]
        fo = fold_of[d[YM[tg]]]
        pc = np.percentile(y, PCTS) * 1e4 if len(y) else [float("nan")] * len(PCTS)
        rec = {"n": int(len(y)), "mean_bp": float(y.mean() * 1e4), "median_bp": float(np.median(y) * 1e4),
               "sd_bp": float(y.std(ddof=1) * 1e4),
               **{f"p{p}_bp": float(v) for p, v in zip(PCTS, pc)},
               "positive_pct": float((y > 0).mean() * 100),
               "above_cost_pct": float((y > cfg["cost_bps"] / 1e4).mean() * 100),
               "market_mean_bp": float(np.nanmean(MKT[tg]) * 1e4)}
        rec["market_net_after_cost_bp"] = rec["market_mean_bp"] - cfg["cost_bps"]
        rec["positive_pct_by_fold"] = {f"fold{k}": float((y[fo == k] > 0).mean() * 100) if (fo == k).any()
                                       else float("nan") for k in range(1, cfg["n_splits"] + 1)}
        rec["n_by_fold"] = {f"fold{k}": int((fo == k).sum()) for k in range(1, cfg["n_splits"] + 1)}
        out[tg] = rec
    return out


def assign_families(ordered: List[str], samp_vals: Dict[str, np.ndarray], samp_d: np.ndarray,
                    thr: float) -> Dict[str, str]:
    """Greedy: in the given order (strongest first), join the first leader with |rho| >= thr."""
    if not ordered:
        return {}
    R = pd.DataFrame({f: samp_vals[f] for f in ordered})
    R = R.groupby(samp_d).rank(pct=True)
    rho = R.corr(min_periods=50).abs()
    leaders: List[str] = []
    out = {}
    for f in ordered:
        home = next((L for L in leaders if np.isfinite(rho.at[f, L]) and rho.at[f, L] >= thr), None)
        if home is None:
            leaders.append(f)
            home = f
        out[f] = home
    return out


# ----------------------------------------------------------------------
# output
# ----------------------------------------------------------------------
def _next_run_id(out_root: Path) -> str:
    day = dt.date.today().strftime("%Y%m%d")
    n = 1
    while (out_root / f"STAGEB_{day}_{n:03d}").exists():
        n += 1
    return f"STAGEB_{day}_{n:03d}"


def _fmt(v, nd=4, sign=True):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "nan"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def _q(v) -> str:
    return "<1e-300" if (np.isfinite(v) and v == 0) else f"{v:.2g}"


def summarize(res: dict, targets: List[str]) -> dict:
    C = res["cells"]
    out = {}
    for tg in targets:
        T = C[C["target"] == tg]
        out[tg] = {"cells": int(len(T)), "tested": int(T["testable"].sum()),
                   "q_pass": int(T["q_pass"].sum()), "stable": int(T["stable"].sum()),
                   "passes": int(T["passes"].sum()), "families": len(res["families"].get(tg, [])),
                   "passes_positive_ic": int((T["passes"] & (T["mean_ic"] > 0)).sum()),
                   "passes_negative_ic": int((T["passes"] & (T["mean_ic"] < 0)).sum()),
                   "leak_suspects": T.loc[T["leak_suspect"], "feature"].tolist()}
    return out


def write_report(res: dict, summ: dict, targets: List[str], run_id: str, path: Path,
                 manifest_sha: str, top: int = 12) -> None:
    C = res["cells"]
    L = [f"# Stage B feature screen - {run_id}", "",
         f"Family {FAMILY_ID} (manifest sha {manifest_sha}) | {CODE_VERSION} | built "
         f"{dt.datetime.now():%Y-%m-%d %H:%M}", "",
         "**Rule (declared before this run):** a cell passes only with Benjamini-Hochberg q <= 0.10 across "
         "every tested cell AND the IC sign equal to the pooled sign in at least 4 of 5 folds.", "",
         "**What an IC is:** each day, stocks ranked by the feature vs ranked by the target, correlated. "
         "Ranking within a day cancels the market, so it measures selection only. It is MARGINAL evidence "
         "(each feature on its own), not incremental (beyond other features) - that is stage D's question.", "",
         f"Rows: research only, eligible only (next open buyable). Research ends "
         f"{res['research_end']}; the lockbox from {res['lockbox_start']} is never read. Scored windows: "
         + "; ".join(f"fold {f['fold']} {f['test_start']}..{f['test_end']} ({f['sessions']} sessions)"
                     for f in res["folds"]) + ".", "",
         f"Eligible rows scored: {res['rows']['eligible_scored']:,} (unbuyable rows skipped in these windows: "
         f"{res['rows']['ineligible_skipped']:,}). Clean features: {len(res['features'])} "
         f"({len(res['market_level'])} market-level, listed separately).", ""]
    TD = res["target_distribution"]
    L += ["## 1. Target distribution (descriptive, the rows the screen scores)", "",
          "Per trade, basis points, pooled over every eligible stock-day in the five windows. "
          "Market = the equal-weight average of all eligible stocks each day, averaged over days.", "",
          "| statistic | " + " | ".join(targets) + " |", "|---|" + "---|" * len(targets)]
    rows = [("N", "n", "{:,}"), ("mean", "mean_bp", "{:+.1f}"), ("median", "median_bp", "{:+.1f}"),
            ("SD", "sd_bp", "{:.1f}")] + [(f"p{p}", f"p{p}_bp", "{:+.1f}") for p in PCTS if p != 50] + [
           ("positive %", "positive_pct", "{:.1f}"), ("above 35 bp cost %", "above_cost_pct", "{:.1f}"),
           ("market mean", "market_mean_bp", "{:+.1f}"),
           ("market net after 35 bp", "market_net_after_cost_bp", "{:+.1f}")]
    for lab, key, fm in rows:
        L.append(f"| {lab} | " + " | ".join(fm.format(TD[tg][key]) for tg in targets) + " |")
    L += ["", "Positive-return share by fold (%):", "",
          "| target | " + " | ".join(f"fold {i}" for i in range(1, 6)) + " |", "|---|" + "---|" * 5]
    for tg in targets:
        L.append(f"| {tg} | " + " | ".join(f"{TD[tg]['positive_pct_by_fold'][f'fold{i}']:.1f}"
                                          for i in range(1, 6)) + " |")
    L += ["", "Buying every eligible stock at the next open, gross, per trade (bp) - the hole every long "
          "trade starts in:", "",
          "| target | " + " | ".join(f"fold {i}" for i in range(1, 6)) + " | pooled | pooled net of 35 bp |",
          "|---|" + "---|" * 7]
    for tg in targets:
        b = res["market_baseline_bp"][tg]
        L.append(f"| {tg} | " + " | ".join(f"{b[f'fold{i}']:+.1f}" for i in range(1, 6))
                 + f" | {b['pooled']:+.1f} | {b['pooled_net_of_cost']:+.1f} |")
    L += ["", "## 2. Feature screen (declared rule)", "",
          "| target | tested | q <= 0.10 | stable 4/5 | PASS | +IC | -IC | families |",
          "|---|---|---|---|---|---|---|---|"]
    for tg in targets:
        s = summ[tg]
        L.append(f"| {tg} | {s['tested']} | {s['q_pass']} | {s['stable']} | **{s['passes']}** | "
                 f"{s['passes_positive_ic']} | {s['passes_negative_ic']} | {s['families']} |")
    sus = C[C["leak_suspect"]]
    if len(sus):
        L += ["", "## WARNING - possible leaks (|IC| >= 0.15)", "",
              "Flagged, not removed. Investigate these before stage D; an IC this large usually means the feature "
              "knows the future.", ""]
        for _, r in sus.iterrows():
            L.append(f"- {r['feature']} on {r['target']}: IC {r['mean_ic']:+.4f}")
    L += ["", "## 3. Strongest passing cells, per target", "",
          "**Descriptive quintile economics - not a trading result.** The money columns are for the fifth of "
          "stocks the IC favours (~150 names a day), per trade, in bp; a top-3 pick is far more extreme and is "
          "judged only in stage D. One row per family leader (families are descriptive; every passing cell stays "
          "in screen.csv, near-misses included)."]
    for tg in targets:
        T = C[(C["target"] == tg) & C["passes"]].copy()
        L += ["", f"### {tg}", ""]
        if T.empty:
            L.append("No cell passes.")
            continue
        T["abs_t"] = T["t_hac"].abs()
        lead = T[T["feature"] == T["family"]].sort_values("abs_t", ascending=False)
        L += ["| feature | IC | t | q | folds (1..5) | spread Q5-Q1 | quintile | quintile gross | "
              "quintile excess | quintile net after cost | family size |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in lead.head(top).iterrows():
            size = int((T["family"] == r["feature"]).sum())
            folds = " ".join(_fmt(r[f"fold{i}_ic"], 3) for i in range(1, 6))
            L.append(f"| {r['feature']} | {r['mean_ic']:+.4f} | {r['t_hac']:+.1f} | {_q(r['q_value'])} | {folds} | "
                     f"{r['spread_q5_q1_bp']:+.1f} | {r['long_side']} | {r['quintile_gross_bp']:+.1f} | "
                     f"{r['quintile_excess_bp']:+.1f} | {r['quintile_net_after_cost_bp']:+.1f} | {size} |")
    L += ["", "## 4. Market-level columns (descriptive only: outside the family, no correction, slow series "
          "inflate t)", ""]
    M = res["market"]
    if len(M):
        cols = [c for c in M.columns if c.endswith("_ts_t_hac")]
        L += ["| feature | " + " | ".join(c.replace("_ts_t_hac", " corr (t)") for c in cols) + " |",
              "|---|" + "---|" * len(cols)]
        for _, r in M.iterrows():
            cells = []
            for c in cols:
                tg = c.replace("_ts_t_hac", "")
                cells.append(f"{r.get(tg + '_ts_corr', float('nan')):+.3f} ({r.get(c, float('nan')):+.1f})")
            L.append(f"| {r['feature']} | " + " | ".join(cells) + " |")
    else:
        L.append("None.")
    L += ["", "## 5. What this does not say", "",
          "- It does not say anything is tradable. Stage D's gate decides that.",
          "- It is marginal evidence: each feature on its own. Whether a feature adds anything beyond the "
          "others is stage D's question.",
          "- Families are descriptive. No feature is dropped for sharing a family.",
          "- The pass list was chosen on these windows, so stage D must reselect inside each fold's "
          "training window, not reuse this list over the same dates.",
          "- About 10% of passing cells are expected to be false discoveries (that is what q <= 0.10 means).",
          ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage B feature screen for family oc_v1")
    ap.add_argument("--panel", required=True, help="panel folder, e.g. %%CACHE_DAILY_ROOT%%\\panel_oc")
    ap.add_argument("--rerun-reason", default=None,
                    help="required if stage B already ran on this panel; logged in the ledger")
    a = ap.parse_args(argv)
    pp = Path(a.panel) / "panel.parquet"

    man, sha = load_manifest()
    targets = list(man["targets"].keys())
    info = check_panel(pp, sha)
    prior = check_rerun(pp, a.rerun_reason)
    print("=" * 72)
    print(f"{CODE_VERSION} | family {FAMILY_ID} (sha {sha}) | panel {pp}")
    print(f"panel: {info['panel_build_version']} built {info['built_at']}, audited {info['audit_at']}")
    print("RULE (declared before this run): a cell passes only with BH q <= 0.10 across every tested cell")
    print("      AND the IC sign equal to the pooled sign in at least 4 of 5 folds.")
    print("=" * 72)

    res = run_screen(pp, targets)
    summ = summarize(res, targets)
    out_root = pp.parent / "stage_b"
    run_id = _next_run_id(out_root)
    out = out_root / run_id
    out.mkdir(parents=True, exist_ok=False)
    res["cells"].to_csv(out / "screen.csv", index=False)
    res["market"].to_csv(out / "market_level.csv", index=False)
    res["ic_daily"].to_parquet(out / "ic_daily.parquet")
    js = {k: v for k, v in res.items() if k not in ("cells", "market", "ic_daily")}
    js.update({"run_id": run_id, "tool": CODE_VERSION, "experiment_family": FAMILY_ID,
               "manifest_sha": sha, "panel": str(pp), "panel_info": info, "summary": summ,
               "rerun_reason": a.rerun_reason, "prior_runs": [p.get("run_id") for p in prior]})
    (out / "stage_b.json").write_text(json.dumps(js, indent=2, default=str), encoding="utf-8")
    write_report(res, summ, targets, run_id, out / "stage_b_report.md", sha)
    n = RC.ledger_append(pp, {"kind": "stage_b_screen", "tool": CODE_VERSION, "run_id": run_id,
                              "experiment_family": FAMILY_ID, "manifest_sha": sha,
                              "research_end": res["research_end"], "cells_tested":
                                  int(res["cells"]["testable"].sum()),
                              "passes": {tg: summ[tg]["passes"] for tg in targets},
                              "families": {tg: summ[tg]["families"] for tg in targets},
                              "leak_suspects": sum(len(summ[tg]["leak_suspects"]) for tg in targets),
                              "rerun_reason": a.rerun_reason})

    TD = res["target_distribution"]
    print()
    print("TARGET DISTRIBUTION (descriptive; bp per trade; the rows the screen scores)")
    print(f"{'target':<12}{'N':>11}{'mean':>8}{'median':>8}{'SD':>8}{'p5':>8}{'p95':>8}{'pos %':>7}"
          f"{'>cost %':>8}{'mkt net':>9}")
    for tg in targets:
        t_ = TD[tg]
        print(f"{tg:<12}{t_['n']:>11,}{t_['mean_bp']:>+8.1f}{t_['median_bp']:>+8.1f}{t_['sd_bp']:>8.1f}"
              f"{t_['p5_bp']:>+8.1f}{t_['p95_bp']:>+8.1f}{t_['positive_pct']:>7.1f}{t_['above_cost_pct']:>8.1f}"
              f"{t_['market_net_after_cost_bp']:>+9.1f}")
    print()
    print("FEATURE SCREEN (declared rule)")
    print(f"{'target':<12}{'tested':>8}{'q<=0.10':>9}{'stable':>8}{'PASS':>6}{'+IC':>5}{'-IC':>5}"
          f"{'families':>10}   buy-everything gross / net (bp)")
    for tg in targets:
        s, b = summ[tg], res["market_baseline_bp"][tg]
        print(f"{tg:<12}{s['tested']:>8}{s['q_pass']:>9}{s['stable']:>8}{s['passes']:>6}"
              f"{s['passes_positive_ic']:>5}{s['passes_negative_ic']:>5}{s['families']:>10}   "
              f"{b['pooled']:+.1f} / {b['pooled_net_of_cost']:+.1f}")
    C = res["cells"]
    for tg in targets:
        T = C[(C["target"] == tg) & C["passes"] & (C["feature"] == C["family"])].copy()
        if T.empty:
            continue
        T = T.reindex(T["t_hac"].abs().sort_values(ascending=False).index).head(5)
        print(f"\n  {tg}: strongest family leaders (quintile economics: descriptive, not a trading result)")
        for _, r in T.iterrows():
            print(f"    {r['feature']:<32} IC {r['mean_ic']:+.4f}  t {r['t_hac']:+6.1f}  q {_q(r['q_value'])}  "
                  f"folds {r['sign_agree_folds']}/5  {r['long_side']} excess {r['quintile_excess_bp']:+.1f} bp"
                  f"  net after cost {r['quintile_net_after_cost_bp']:+.1f} bp")
    sus = C[C["leak_suspect"]]
    if len(sus):
        print(f"\n  WARNING: {len(sus)} cells with |IC| >= {CFG['leak_ic']} - possible leaks, investigate "
              f"before stage D: " + ", ".join(sorted(set(sus['feature'])))[:400])
    print(f"\n  market-level columns (not screened, descriptive table in the report): {len(res['market_level'])}")
    print(f"  outputs: {out}")
    print(f"  research ledger: entry {n} (stage_b_screen {run_id})")
    total = int(C["passes"].sum())
    print("=" * 72)
    print(f"STAGE B COMPLETE - {total} feature x target cells pass the declared rule "
          f"({res['seconds']:.0f}s). This is a screen, not a trading result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
