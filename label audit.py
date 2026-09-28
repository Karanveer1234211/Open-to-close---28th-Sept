#!/usr/bin/env python3
"""
label_audit.py - prove the target is what the contract says it is.

    python label_audit.py --root %CACHE_DAILY_ROOT%
    python label_audit.py --root %CACHE_DAILY_ROOT% --symbols 120
    python label_audit.py --root %CACHE_DAILY_ROOT% --panel %CACHE_DAILY_ROOT%\\panel_oc

Cheap and standalone. Run it after every panel rebuild, BEFORE any research.

INDEPENDENCE
============
Checking a label with the code that produced it proves nothing. This file
imports nothing from panel_build's label code: it re-derives every sampled
label from the raw cache OHLCV with a plain loop written from the contract,
then compares with what the panel stored. panel_build uses vectorised running
maxima; this uses explicit session-by-session checks. Agreement between two
unrelated implementations is evidence; agreement with itself is not.

THE CONTRACT (primary variant)
------------------------------
    ATR14      SIMPLE 14-day mean of true range, / close at T
               (NOT D_atr14 - that feature is Wilder-smoothed)
    TP level   +1.5 x ATR14      SL level  -1.0 x ATR14
    path       sessions T+1..T+5: high vs TP, low vs SL, from close[T]
    tie        both crossed in one session -> SL
    exit_ret   +TP | -SL | close[T+5]/close[T]-1
    pending    last 5 sessions of each symbol -> NaN, never negative

OPEN-TO-CLOSE TARGETS (panel_build v32, section 5)
--------------------------------------------------
    label_oc_h        close[T+h] / open[T+1] - 1,  h = 1, 2, 3, 5
    label_buyable_o1  0 if T+1 opens AT its high and at a 2/5/10/20% upper
                      band vs close[T]; 1 otherwise; NaN if T+1 is missing
  Re-derived by a plain loop; entry/exit convention proven on EVERY row by the
  identity (1 + fwd_ret_5d) = (1 + gap1) x (1 + oc_5); pending tails per
  horizon; the firewall (no label_* column can be a feature); ETFs absent;
  exclusions reported by year, symbol and market day. --compare-to <old
  panel.parquet> proves the pre-existing labels did not change.

Two checks go beyond restating the contract:
  * a 'neither' outcome must lie STRICTLY between -SL and +TP. If the close
    at T+5 were outside, the high or low on that session would have crossed
    a barrier. A violation is an impossible outcome.
  * label_tp_before_sl, label_first_touch and label_exit_ret must tell the
    same story on every row of the full panel.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LABEL_AUDIT_VERSION = "label_audit v2"
TP_MULT, SL_MULT, H, ATR_N = 1.5, 1.0, 5, 14
OC_H = (1, 2, 3, 5)
OC_LABELS = [f"label_oc_{h}" for h in OC_H] + ["label_buyable_o1"]
BANDS, BAND_TOL_PP, TICK = (0.02, 0.05, 0.10, 0.20), 0.0015, 0.05     # restated, not imported
MANIFEST = "EXPERIMENT_OC_FAMILY.json"
TOL = 1e-9
LABELS = ["label_tp_before_sl", "label_first_touch", "label_exit_ret",
          "label_same_day_ambiguous", "label_fwd_ret_5d"]


class Report:
    def __init__(self):
        self.fail, self.lines = [], []

    def ok(self, msg):
        self.lines.append(("ok  ", msg)); print(f"  [ok  ] {msg}", flush=True)

    def warn(self, msg):
        self.lines.append(("WARN", msg)); print(f"  [WARN] {msg}", flush=True)

    def bad(self, msg):
        self.fail.append(msg); self.lines.append(("FAIL", msg))
        print(f"  [FAIL] {msg}", flush=True)


def _dates(ts: pd.Series) -> pd.Series:
    t = pd.to_datetime(ts)
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return t.dt.normalize()


# ----------------------------------------------------------------------
# independent reference implementation - written from the contract
# ----------------------------------------------------------------------
def reference_labels(o_h_l_c: pd.DataFrame) -> pd.DataFrame:
    h = o_h_l_c["high"].to_numpy(float)
    l = o_h_l_c["low"].to_numpy(float)
    c = o_h_l_c["close"].to_numpy(float)
    n = len(c)
    tr = np.full(n, np.nan)
    for k in range(1, n):
        tr[k] = max(h[k] - l[k], abs(h[k] - c[k - 1]), abs(l[k] - c[k - 1]))
    atr = np.full(n, np.nan)
    for k in range(ATR_N, n):
        w = tr[k - ATR_N + 1:k + 1]
        if np.isfinite(w).all():
            atr[k] = w.mean()
    first = np.full(n, np.nan); exit_r = np.full(n, np.nan)
    tie = np.full(n, np.nan); tp_l = np.full(n, np.nan); sl_l = np.full(n, np.nan)
    for t in range(0, n - H):
        if not (np.isfinite(atr[t]) and np.isfinite(c[t]) and c[t] > 0):
            continue
        a = atr[t] / c[t]
        tp, sl = TP_MULT * a, SL_MULT * a
        tp_l[t], sl_l[t] = tp, sl
        res, tt = 0.0, 0.0
        for k in range(1, H + 1):
            up = h[t + k] / c[t] - 1 >= tp
            dn = l[t + k] / c[t] - 1 <= -sl
            if up and dn:
                res, tt = -1.0, 1.0; break
            if up:
                res = 1.0; break
            if dn:
                res = -1.0; break
        first[t], tie[t] = res, tt
        exit_r[t] = tp if res == 1 else (-sl if res == -1 else c[t + H] / c[t] - 1)
    return pd.DataFrame({"date": _dates(o_h_l_c["timestamp"]).to_numpy(),
                         "r_first": first, "r_exit": exit_r, "r_tie": tie,
                         "r_tp": tp_l, "r_sl": sl_l})


# ----------------------------------------------------------------------
def reference_open_labels(raw: pd.DataFrame) -> pd.DataFrame:
    """
    NEXT-OPEN ENTRY, written independently as a plain loop from the contract:
    enter at the open of T+1; TP/SL sized at the signal (ATR14/close x 1.5 / 1.0)
    and applied to the entry price; session T+1 is intraday only (tie = stop);
    from T+2 an open beyond a barrier fills at that open; else exit at close T+5.
    """
    o = raw["open"].to_numpy(float); h = raw["high"].to_numpy(float)
    l = raw["low"].to_numpy(float); c = raw["close"].to_numpy(float)
    n = len(c)
    tr = np.full(n, np.nan)
    for k in range(1, n):
        tr[k] = max(h[k] - l[k], abs(h[k] - c[k - 1]), abs(l[k] - c[k - 1]))
    atr = np.full(n, np.nan)
    for k in range(ATR_N, n):
        w = tr[k - ATR_N + 1:k + 1]
        if np.isfinite(w).all():
            atr[k] = w.mean()
    first = np.full(n, np.nan); xr = np.full(n, np.nan); mfe = np.full(n, np.nan)
    for t in range(0, n - H):
        if not (np.isfinite(atr[t]) and c[t] > 0 and np.isfinite(o[t + 1]) and o[t + 1] > 0):
            continue
        a = atr[t] / c[t]; tp, sl = TP_MULT * a, SL_MULT * a; e = o[t + 1]
        res, r = 0.0, None
        for k in range(1, H + 1):
            b = t + k
            up, dn = h[b] / e - 1 >= tp, l[b] / e - 1 <= -sl
            if k >= 2:
                g = o[b] / e - 1
                if g <= -sl:
                    res, r = -1.0, g; break
                if g >= tp:
                    res, r = 1.0, g; break
            if dn:
                res, r = -1.0, -sl; break
            if up:
                res, r = 1.0, tp; break
        first[t] = res
        xr[t] = r if r is not None else c[t + H] / e - 1
        mfe[t] = max(h[t + 1:t + H + 1]) / e - 1
    return pd.DataFrame({"date": _dates(raw["timestamp"]).to_numpy(), "o_first": first, "o_xr": xr, "o_mfe": mfe})


def reference_oc_labels(raw: pd.DataFrame) -> pd.DataFrame:
    """
    OPEN-TO-CLOSE, written independently as a plain loop from the contract:
    entry = open of the NEXT bar, exit = close h bars ahead; unbuyable when that
    next bar opens at its high AND within tolerance of close[T] x (1 + band).
    """
    o = raw["open"].to_numpy(float); h = raw["high"].to_numpy(float)
    c = raw["close"].to_numpy(float)
    n = len(c)
    ref = {f"r_oc_{k}": np.full(n, np.nan) for k in OC_H}
    buy = np.full(n, np.nan)
    for t in range(n - 1):
        e = o[t + 1]
        if not (np.isfinite(e) and e > 0):
            continue
        for k in OC_H:
            if t + k < n and np.isfinite(c[t + k]) and c[t + k] > 0:
                ref[f"r_oc_{k}"][t] = c[t + k] / e - 1
        if np.isfinite(c[t]) and c[t] > 0 and np.isfinite(h[t + 1]):
            locked = False
            if abs(e - h[t + 1]) <= 1e-6 * max(abs(e), abs(h[t + 1])):
                g = e / c[t] - 1
                tol = max(BAND_TOL_PP, 1.02 * TICK / c[t])
                locked = any(abs(g - b) <= tol for b in BANDS)
            buy[t] = 0.0 if locked else 1.0
    return pd.DataFrame({"date": _dates(raw["timestamp"]).to_numpy(), **ref, "r_buy": buy})


def _num(s: pd.Series) -> np.ndarray:
    """Any numeric / nullable column -> float64 with NaN."""
    return pd.to_numeric(s, errors="coerce").astype("float64").to_numpy()


def compare_to_old(new_pp: Path, old_pp: Path, R: "Report", recent_sessions: int = 20) -> dict:
    """Pre-existing labels must be unchanged on the rows both panels share.

    Rows in the old panel's last `recent_sessions` sessions are skipped: the
    cache refetches its recent tail, so those bars can legitimately be revised.
    Mismatches on older rows: a handful of symbols = data revisions (corporate
    actions) -> WARN with the list; more than 1% of rows = the label CODE
    changed -> FAIL."""
    import pyarrow.parquet as pq
    old_cols = set(pq.ParquetFile(old_pp).schema_arrow.names)
    new_cols = set(pq.ParquetFile(new_pp).schema_arrow.names)
    labs = sorted(c for c in old_cols & new_cols if c.startswith("label_") and c not in OC_LABELS)
    A = pd.read_parquet(old_pp, columns=["timestamp", "symbol"])
    B = pd.read_parquet(new_pp, columns=["timestamp", "symbol"])
    A["date"], B["date"] = _dates(A["timestamp"]), _dates(B["timestamp"])
    days = np.sort(A["date"].unique())
    cutoff = days[max(0, len(days) - recent_sessions)]
    A["ia"], B["ib"] = np.arange(len(A)), np.arange(len(B))
    M = A.loc[A["date"] < cutoff, ["date", "symbol", "ia"]].merge(B[["date", "symbol", "ib"]],
                                                                 on=["date", "symbol"], how="inner")
    ia, ib = M["ia"].to_numpy(), M["ib"].to_numpy()
    out = {"rows_compared": int(len(M)), "columns": len(labs), "cutoff": str(pd.Timestamp(cutoff).date()),
           "mismatch_rows": {}, "symbols": []}
    bad_rows = np.zeros(len(M), dtype=bool)
    for c in labs:
        a = _num(pd.read_parquet(old_pp, columns=[c])[c])[ia]
        b = _num(pd.read_parquet(new_pp, columns=[c])[c])[ib]
        diff = ~((np.isnan(a) & np.isnan(b)) | (np.abs(a - b) <= 1e-12))
        if diff.any():
            out["mismatch_rows"][c] = int(diff.sum())
            bad_rows |= diff
    worst = max(out["mismatch_rows"].values()) if out["mismatch_rows"] else 0
    out["symbols"] = sorted(M.loc[bad_rows, "symbol"].unique().tolist())
    msg = (f"pre-existing labels vs {old_pp.name}: {len(labs)} columns x {len(M):,} shared rows "
           f"(before {out['cutoff']})")
    if not worst:
        R.ok(msg + ": identical")
    elif worst / max(len(M), 1) > 0.01:
        R.bad(msg + f": {worst:,} rows differ in the worst column - the label code changed")
    else:
        R.warn(msg + f": {int(bad_rows.sum()):,} rows differ in {len(out['symbols'])} symbols "
               f"(data revisions?): {', '.join(out['symbols'][:10])}")
    return out


def audit(root: Path, n_symbols: int = 60, seed: int = 0, compare_to=None, panel_dir=None) -> Report:
    import pyarrow.parquet as pq
    import research_common as RC
    from data_quality import _paths
    pp = (Path(panel_dir) if panel_dir else root / "panel") / "panel.parquet"
    R = Report()
    print(f"  panel: {pp}")

    print("\n=== 1. PANEL STRUCTURE ===")
    schema = pq.ParquetFile(pp).schema_arrow.names
    meta_p = pp.parent / "panel_meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    lc = meta.get("label_contract")
    if not lc:
        R.bad("panel_meta.json has no label_contract - panel built with an old "
              "panel_build.py; rebuild")
    elif lc.get("primary_variant") != "atr1p5_1p0":
        R.bad(f"primary variant is {lc.get('primary_variant')}, expected atr1p5_1p0")
    else:
        R.ok(f"label contract: {lc['take_profit']} TP / {lc['stop_loss']} SL, "
             f"{lc['horizon_sessions']} sessions, ties -> SL")
    if "label_tp" in meta or "label_sl" in meta:
        R.bad("stale label_tp/label_sl still in panel_meta.json")
    missing = [c for c in LABELS if c not in schema]
    if missing:
        R.bad(f"label columns absent: {missing} - rebuild the panel")
        return R
    fwd = [c for c in schema if c in RC.FORWARD_COLUMNS]
    (R.bad if fwd else R.ok)(f"forward-looking columns stored as non-labels: {fwd or 'none'}")

    P = pd.read_parquet(pp, columns=["timestamp", "symbol"] + LABELS)
    P["date"] = _dates(P["timestamp"])
    import panel_build as PB
    n_feat = len(PB.panel_feature_columns(pd.DataFrame(columns=schema)))
    R.ok(f"{len(P):,} rows | {P['symbol'].nunique():,} symbols | "
         f"{P['date'].min().date()} -> {P['date'].max().date()} | {n_feat} feature columns")
    dup = int(P.duplicated(["date", "symbol"]).sum())
    (R.bad if dup else R.ok)(f"duplicate (date, symbol) rows: {dup}")
    for c in LABELS:
        v = pd.to_numeric(P[c], errors="coerce").to_numpy(float)
        ninf = int(np.isinf(v).sum())
        if ninf:
            R.bad(f"{c}: {ninf} infinite values")
    R.ok("no infinite values in any label column") if not any(
        "infinite" in f for f in R.fail) else None

    print("\n=== 2. INTERNAL CONSISTENCY, FULL PANEL ===")
    ft = pd.to_numeric(P["label_first_touch"], errors="coerce").to_numpy(float)
    tb = pd.to_numeric(P["label_tp_before_sl"], errors="coerce").to_numpy(float)
    er = pd.to_numeric(P["label_exit_ret"], errors="coerce").to_numpy(float)
    amb = pd.to_numeric(P["label_same_day_ambiguous"], errors="coerce").to_numpy(float)
    f5 = pd.to_numeric(P["label_fwd_ret_5d"], errors="coerce").to_numpy(float)
    res = np.isfinite(ft)
    c1 = int(((tb == 1) != (ft == 1))[res].sum())
    (R.bad if c1 else R.ok)(f"label_tp_before_sl disagrees with first_touch on {c1:,} rows")
    c2 = int((np.isfinite(ft) != np.isfinite(er)).sum())
    (R.bad if c2 else R.ok)(f"exit_ret missing where outcome is known (or vice versa): {c2:,}")
    c3 = int(((ft == 1) & ~(er > 0)).sum())
    (R.bad if c3 else R.ok)(f"TP-first rows with exit_ret <= 0: {c3:,}")
    c4 = int(((ft == -1) & ~(er < 0)).sum())
    (R.bad if c4 else R.ok)(f"SL-first rows with exit_ret >= 0: {c4:,}")
    nm = (ft == 0) & np.isfinite(f5)
    c5 = int((np.abs(er[nm] - f5[nm]) > TOL).sum())
    (R.bad if c5 else R.ok)(f"'neither' rows where exit_ret != 5-session return: {c5:,}")
    c6 = int(((amb == 1) & (ft != -1)).sum())
    (R.bad if c6 else R.ok)(f"same-day ties NOT counted as SL: {c6:,}")
    n_res = int(res.sum())
    counts = {"TP_first": float((ft == 1).sum() / n_res), "SL_first": float((ft == -1).sum() / n_res),
              "neither": float((ft == 0).sum() / n_res), "tie_as_SL": float((amb == 1).sum() / n_res)}
    R.ok("outcomes: " + " | ".join(f"{k} {v:.1%}" for k, v in counts.items()))
    q = np.nanpercentile(er, [1, 5, 25, 50, 75, 95, 99])
    R.ok("exit_ret percentiles 1/5/25/50/75/95/99: "
         + " ".join(f"{x:+.2%}" for x in q))

    print("\n=== 3. INDEPENDENT RECOMPUTATION FROM THE RAW CACHE ===")
    rng = np.random.default_rng(seed)
    sizes = P.groupby("symbol").size().sort_values()
    systematic = list(sizes.index[:5]) + list(sizes.index[-5:])
    others = [s for s in sizes.index if s not in systematic]
    pick = systematic + list(rng.choice(others, min(n_symbols, len(others)), replace=False))
    tot = {"rows": 0, "outcome": 0, "exit": 0, "tie": 0, "impossible": 0,
           "pending_bad": 0, "no_cache": 0}
    worst = []
    for sym in pick:
        cp, _ = _paths(root, sym)
        if not Path(cp).exists():
            tot["no_cache"] += 1; continue
        raw = pd.read_parquet(cp, columns=["timestamp", "high", "low", "close"])
        raw = raw.sort_values("timestamp").reset_index(drop=True)
        ref = reference_labels(raw)
        pan = P[P["symbol"] == sym][["date", "label_first_touch", "label_exit_ret",
                                     "label_same_day_ambiguous"]]
        m = pan.merge(ref, on="date", how="left")
        pf = pd.to_numeric(m["label_first_touch"], errors="coerce").to_numpy(float)
        pe = pd.to_numeric(m["label_exit_ret"], errors="coerce").to_numpy(float)
        pa = pd.to_numeric(m["label_same_day_ambiguous"], errors="coerce").to_numpy(float)
        rf, re_, rt = m["r_first"].to_numpy(), m["r_exit"].to_numpy(), m["r_tie"].to_numpy()
        both = np.isfinite(pf) & np.isfinite(rf)
        tot["rows"] += int(both.sum())
        tot["outcome"] += int((pf[both] != rf[both]).sum())
        tot["exit"] += int((np.abs(pe[both] - re_[both]) > TOL).sum())
        tot["tie"] += int((pa[both] != rt[both]).sum())
        nei = both & (pf == 0)
        tp_, sl_ = m["r_tp"].to_numpy(), m["r_sl"].to_numpy()
        tot["impossible"] += int((~((pe[nei] > -sl_[nei]) & (pe[nei] < tp_[nei]))).sum())
        # pending: the last H cache sessions must be NaN in the panel
        tail = set(_dates(raw["timestamp"]).iloc[-H:])
        tail_rows = pan[pan["date"].isin(tail)]
        tot["pending_bad"] += int(pd.to_numeric(tail_rows["label_first_touch"],
                                                errors="coerce").notna().sum())
        if both.any():
            d = np.abs(pe[both] - re_[both])
            worst.append((sym, float(np.nanmax(d))))
    checked = len(pick) - tot["no_cache"]
    R.ok(f"{checked} symbols recomputed ({len(systematic)} systematic: fewest/most "
         f"rows; rest random) | {tot['rows']:,} labelled rows compared")
    if tot["no_cache"]:
        R.warn(f"{tot['no_cache']} sampled symbols had no cache file")
    for k, lab in (("outcome", "TP/SL/neither outcome mismatches"),
                   ("exit", f"exit_ret mismatches (> {TOL:g})"),
                   ("tie", "tie-flag mismatches"),
                   ("impossible", "'neither' outcomes outside (-SL, +TP)"),
                   ("pending_bad", "last-5-session rows carrying a label")):
        (R.bad if tot[k] else R.ok)(f"{lab}: {tot[k]:,}")
    if worst:
        s_, w_ = max(worst, key=lambda x: x[1])
        R.ok(f"largest exit_ret difference: {w_:.2e} ({s_})")

    # ---- NEXT-OPEN LABELS, if the panel carries them
    if "label_exit_ret_o1" in schema:
        print("\n=== 4. NEXT-OPEN ENTRY LABELS - independent recomputation ===")
        Po = pd.read_parquet(pp, columns=["timestamp", "symbol", "label_first_touch_o1", "label_exit_ret_o1",
                                          "label_mfe_5d_o1", "label_gap1"])
        Po["date"] = _dates(Po["timestamp"])
        tot_o = {"rows": 0, "outcome": 0, "exit": 0, "mfe": 0}
        for sym in pick:
            cp, _ = _paths(root, sym)
            if not Path(cp).exists():
                continue
            raw = pd.read_parquet(cp, columns=["timestamp", "open", "high", "low", "close"])
            raw = raw.sort_values("timestamp").reset_index(drop=True)
            ref = reference_open_labels(raw)
            m = Po[Po["symbol"] == sym].merge(ref, on="date", how="left")
            pf, rf = m["label_first_touch_o1"].to_numpy(float), m["o_first"].to_numpy(float)
            both = np.isfinite(pf) & np.isfinite(rf)
            tot_o["rows"] += int(both.sum())
            tot_o["outcome"] += int((pf[both] != rf[both]).sum())
            tot_o["exit"] += int((np.abs(m["label_exit_ret_o1"].to_numpy(float)[both] - m["o_xr"].to_numpy(float)[both]) > TOL).sum())
            tot_o["mfe"] += int((np.abs(m["label_mfe_5d_o1"].to_numpy(float)[both] - m["o_mfe"].to_numpy(float)[both]) > TOL).sum())
        R.ok(f"{tot_o['rows']:,} next-open labels compared")
        for k, lab in (("outcome", "next-open outcome mismatches"), ("exit", "next-open return mismatches"),
                       ("mfe", "next-open MFE mismatches")):
            (R.bad if tot_o[k] else R.ok)(f"{lab}: {tot_o[k]:,}")
        ok_ = Po["label_exit_ret_o1"].notna() & Po["label_gap1"].notna()
        rho = float(Po.loc[ok_, "label_exit_ret_o1"].corr(Po.loc[ok_, "label_gap1"], method="spearman"))
        R.ok(f"correlation of next-open return with the overnight gap it excludes: {rho:+.3f} (informational)")

    # ---- OPEN-TO-CLOSE TARGETS (panel_build v32)
    oc_info = {}
    ver = str(meta.get("panel_build_version", ""))
    v32 = ver.startswith("panel_build v") and ver.split("v")[-1].isdigit() and int(ver.split("v")[-1]) >= 32
    print("\n=== 5. OPEN-TO-CLOSE TARGETS, UNIVERSE, FIREWALL ===")
    have_oc = all(c in schema for c in OC_LABELS)
    if not have_oc:
        (R.bad if v32 else R.warn)(f"open-to-close labels absent ({[c for c in OC_LABELS if c not in schema]}) - "
                                   "panel predates panel_build v32" + ("" if not v32 else " but meta says v32"))
    else:
        R.ok(f"panel built by {ver or 'unknown version'} | universe rule {meta.get('etf_rule', '?')} | "
             f"{len(meta.get('etf_excluded', []))} ETFs/funds excluded at build")
        # 5a. firewall
        import research_common as RC
        feats = PB.panel_feature_columns(pd.DataFrame(columns=schema))
        leak = [c for c in feats if c.startswith("label_") or c in OC_LABELS]
        (R.bad if leak else R.ok)(f"label columns in the feature list: {leak or 'none'} "
                                  f"({len(feats)} features, {sum(c.startswith('label_') for c in schema)} labels)")
        unguarded = [c for c in OC_LABELS if not RC.is_forbidden_feature(c)]
        (R.bad if unguarded else R.ok)(f"new targets blocked by the firewall rule: "
                                       f"{'all' if not unguarded else 'NOT ' + str(unguarded)}")
        # 5b. ETFs absent
        extra = RC.load_symbol_list(root / RC.UNIVERSE_EXCLUDE_FILE)
        etfs = sorted(s for s in P["symbol"].unique() if RC.is_etf_symbol(s, extra))
        (R.bad if (etfs and v32) else (R.warn if etfs else R.ok))(
            f"ETF / fund symbols in the panel: {len(etfs)}" + (f" ({', '.join(etfs[:8])})" if etfs else ""))
        # 5c. the entry/exit convention on EVERY row
        Q = pd.read_parquet(pp, columns=["timestamp", "symbol", "label_fwd_ret_5d", "label_gap1"] + OC_LABELS
                            + [c for c in ("MKT_D_ret_1d_pct",) if c in schema])
        Q["date"] = _dates(Q["timestamp"])
        f5, g1, o5 = _num(Q["label_fwd_ret_5d"]), _num(Q["label_gap1"]), _num(Q["label_oc_5"])
        ok_ = np.isfinite(f5) & np.isfinite(g1) & np.isfinite(o5)
        rel = np.abs((1 + f5[ok_]) - (1 + g1[ok_]) * (1 + o5[ok_])) / np.abs(1 + f5[ok_])
        n_id = int((rel > 1e-9).sum())
        (R.bad if n_id else R.ok)(f"entry = OPEN T+1, exit = CLOSE T+5 on every row: identity "
                                  f"(1+fwd_5d) = (1+gap1)(1+oc_5) fails on {n_id:,} of {int(ok_.sum()):,} rows")
        for k in OC_H:
            v = _num(Q[f"label_oc_{k}"])
            if np.isinf(v).any():
                R.bad(f"label_oc_{k}: {int(np.isinf(v).sum())} infinite values")
        bu = _num(Q["label_buyable_o1"])
        badv = int((np.isfinite(bu) & ~np.isin(bu, [0.0, 1.0])).sum())
        (R.bad if badv else R.ok)(f"label_buyable_o1 values outside {{0, 1, NaN}}: {badv}")
        # 5d. independent recomputation on the sampled symbols
        tot5 = {"rows": 0, "buy_rows": 0, "buy_mismatch": 0, "nan_pattern": 0, "pending_bad": 0, "pending_missing": 0}
        for k in OC_H:
            tot5[f"oc_{k}_mismatch"] = 0
        for sym in pick:
            cp, _ = _paths(root, sym)
            if not Path(cp).exists():
                continue
            raw = pd.read_parquet(cp, columns=["timestamp", "open", "high", "low", "close"])
            raw = raw.sort_values("timestamp").reset_index(drop=True)
            ref = reference_oc_labels(raw)
            m = Q.loc[Q["symbol"] == sym].merge(ref, on="date", how="left")
            tot5["rows"] += len(m)
            for k in OC_H:
                pv, rv = _num(m[f"label_oc_{k}"]), m[f"r_oc_{k}"].to_numpy(float)
                tot5["nan_pattern"] += int((np.isfinite(pv) != np.isfinite(rv)).sum())
                both = np.isfinite(pv) & np.isfinite(rv)
                tot5[f"oc_{k}_mismatch"] += int((np.abs(pv[both] - rv[both]) > TOL).sum())
                # pending: the last k bars of the symbol carry no oc_k; the bar k+1 from the end does
                dts = _dates(raw["timestamp"])
                tail = set(dts.iloc[-k:])
                tot5["pending_bad"] += int(np.isfinite(pv[m["date"].isin(tail).to_numpy()]).sum())
                if len(raw) > k + 1:
                    t0 = dts.iloc[-k - 1]
                    row = m.loc[m["date"] == t0]
                    if len(row) and np.isfinite(ref.loc[ref["date"] == t0, f"r_oc_{k}"]).any() \
                            and not np.isfinite(_num(row[f"label_oc_{k}"])).any():
                        tot5["pending_missing"] += 1
            pb, rb = _num(m["label_buyable_o1"]), m["r_buy"].to_numpy(float)
            tot5["nan_pattern"] += int((np.isfinite(pb) != np.isfinite(rb)).sum())
            both = np.isfinite(pb) & np.isfinite(rb)
            tot5["buy_rows"] += int(both.sum())
            tot5["buy_mismatch"] += int((pb[both] != rb[both]).sum())
        R.ok(f"{tot5['rows']:,} open-to-close rows re-derived on the same {len(pick)} sampled symbols")
        for k in OC_H:
            (R.bad if tot5[f"oc_{k}_mismatch"] else R.ok)(f"label_oc_{k} mismatches (> {TOL:g}): {tot5[f'oc_{k}_mismatch']:,}")
        (R.bad if tot5["buy_mismatch"] else R.ok)(f"label_buyable_o1 mismatches: {tot5['buy_mismatch']:,} of {tot5['buy_rows']:,}")
        (R.bad if tot5["nan_pattern"] else R.ok)(f"rows where panel and reference disagree on known vs unknown "
                                                 f"(missing / invalid future bars): {tot5['nan_pattern']:,}")
        (R.bad if tot5["pending_bad"] else R.ok)(f"pending tails (last h bars) carrying an oc_h label: {tot5['pending_bad']:,}")
        (R.bad if tot5["pending_missing"] else R.ok)(f"resolvable rows just before the tail left empty: {tot5['pending_missing']:,}")
        # 5e. what the buyable filter removes - so the exclusion itself can be judged
        has = np.isfinite(bu)
        yr = Q["date"].dt.year.to_numpy()
        share0 = float((bu[has] == 0).mean()) if has.any() else float("nan")
        by_year = {int(y): float((bu[has & (yr == y)] == 0).mean()) for y in np.unique(yr[has])}
        top = Q.loc[has & (bu == 0), "symbol"].value_counts().head(10)
        o1 = _num(Q["label_oc_1"])
        m0, m1 = np.nanmean(o1[has & (bu == 0)]) if (has & (bu == 0)).any() else np.nan, np.nanmean(o1[has & (bu == 1)])
        R.ok(f"unbuyable at the next open: {share0:.2%} of rows | by year: "
             + " ".join(f"{y} {v:.2%}" for y, v in by_year.items()))
        R.ok("most-excluded symbols: " + ", ".join(f"{s} {int(n)}" for s, n in top.items()))
        by_mkt = {}
        if "MKT_D_ret_1d_pct" in Q:
            mk = _num(Q["MKT_D_ret_1d_pct"])
            okm = has & np.isfinite(mk)
            if okm.any():
                cuts = np.nanpercentile(mk[okm], [33.333, 66.667])
                for lab, sel in (("down", mk <= cuts[0]), ("flat", (mk > cuts[0]) & (mk <= cuts[1])), ("up", mk > cuts[1])):
                    s_ = okm & sel
                    by_mkt[lab] = float((bu[s_] == 0).mean()) if s_.any() else float("nan")
                R.ok("unbuyable share by NIFTY day at T (terciles): "
                     + " | ".join(f"{k} {v:.2%}" for k, v in by_mkt.items()))
        R.ok(f"mean open->close T+1 return: unbuyable rows {m0*1e4:+.1f} bp vs buyable {m1*1e4:+.1f} bp "
             f"(informational - the size of what the flag keeps out)")
        # 5f. the declared experiment family
        mf = HERE / MANIFEST
        if mf.exists():
            import hashlib
            man = json.loads(mf.read_text(encoding="utf-8"))
            miss = [t for t in man.get("targets", {}) if t not in schema]
            sha = hashlib.sha256(mf.read_bytes()).hexdigest()[:16]
            (R.bad if miss else R.ok)(f"experiment family {man.get('family_id')} (declared {man.get('declared')}, "
                                      f"sha {sha}): targets present {'all' if not miss else 'MISSING ' + str(miss)}")
            oc_info["manifest"] = {"family_id": man.get("family_id"), "sha": sha}
        else:
            (R.bad if v32 else R.warn)(f"{MANIFEST} not found next to label_audit.py")
        oc_info.update({"recomputation": tot5, "identity_failures": n_id, "unbuyable_share": share0,
                        "unbuyable_by_year": by_year, "unbuyable_by_market": by_mkt,
                        "etfs_in_panel": etfs})
    if compare_to:
        print("\n=== 6. PRE-EXISTING LABELS UNCHANGED ===")
        oc_info["compare_to"] = compare_to_old(pp, Path(compare_to), R)

    out = {"passed": not R.fail, "failures": R.fail, "outcome_mix": counts,
           "exit_ret_pct": dict(zip(["p1", "p5", "p25", "p50", "p75", "p95", "p99"],
                                    map(float, q))),
           "recomputation": tot, "symbols_checked": checked, "version": LABEL_AUDIT_VERSION,
           "open_to_close": oc_info}
    (pp.parent / "label_audit.json").write_text(json.dumps(out, indent=2, default=str),
                                                encoding="utf-8")
    RC.ledger_append(pp, {"kind": "panel_audit", "tool": LABEL_AUDIT_VERSION, "passed": not R.fail,
                          "panel_build_version": meta.get("panel_build_version"),
                          "experiment_family": (oc_info.get("manifest") or {}).get("family_id"),
                          "manifest_sha": (oc_info.get("manifest") or {}).get("sha"),
                          "failures": R.fail[:10]})
    return R


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=None)
    ap.add_argument("--symbols", type=int, default=60)
    ap.add_argument("--panel", default=None, help="panel folder (default <root>/panel)")
    ap.add_argument("--compare-to", default=None,
                    help="an older panel.parquet: prove its labels are unchanged on shared rows")
    a = ap.parse_args()
    root = a.root or os.environ.get("CACHE_DAILY_ROOT")
    if not root:
        raise SystemExit("CACHE_DAILY_ROOT not set and --root not given")
    R = audit(Path(root), a.symbols, compare_to=a.compare_to, panel_dir=a.panel)
    print("\n" + "=" * 64)
    if R.fail:
        print("  LABEL AUDIT FAILED - do not run any research on this panel:")
        for f in R.fail:
            print(f"    - {f}")
        return 1
    print("  LABEL AUDIT PASSED - the target matches its contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
