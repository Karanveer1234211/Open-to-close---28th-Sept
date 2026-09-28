#!/usr/bin/env python3
"""
tests/test_panel_oc.py - panel_build v32 + label_audit v2. Prints VERIFIED on success.

Planted effects, not "the script runs":
 1. The hand example: T+1 open 100, closes 102 / 105 / 103 / . / 110
    -> oc_1 +2%, oc_2 +5%, oc_3 +3%, oc_5 +10%, from panel_build AND from
    label_audit's independent loop.
 2. Upper-band lock at the next open -> label_buyable_o1 = 0 (real bars: ELIN,
    TTML, TANLA, LLOYDSENGG, TVSELECT); RPOWER / EXICOM / PCJEWELLER -> 1;
    missing or non-positive T+1 open -> NaN, never dropped, never 0-filled.
 3. Perturbation: changing close[T], open[T+2..], close[T+h+1] leaves oc_h
    unchanged; changing open[T+1] or close[T+h] changes it (no T-day price, no
    T+h+1 leak, correct entry and exit bar).
 4. Pending tails per horizon.
 5. panel_build vs the independent loop on random paths with planted locks:
    0 mismatches; the lock rule equals tradability_audit's.
 6. Universe rule: ETFs out, equities in, user exclude file honoured.
 7. Firewall: the new labels can never be features.
 8. Pre-existing labels: golden checksums computed with the SHIPPED v31 code
    on deterministic paths - the 44 old label columns must not move.
 0. The modules loaded are this folder's (no shadowing by the parent folder).
 9. End to end: build_panel into <root>/panel_oc on a synthetic cache (gate and calendar stubbed)
    -> ETFs absent, meta/contract correct, label_audit PASSES; then five
    sabotaged panels must each FAIL the audit (oc_5 built close-to-close,
    oc_2 shifted a day, a lock marked buyable, an ETF row injected, old labels
    changed vs --compare-to).
"""

from __future__ import annotations

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

import panel_build as PB  # noqa: E402
import label_audit as LA  # noqa: E402
import research_common as RC  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def frame(o, h, l, c, start="2024-01-01"):
    n = len(c)
    return pd.DataFrame({"timestamp": pd.bdate_range(start, periods=n), "open": np.asarray(o, float),
                         "high": np.asarray(h, float), "low": np.asarray(l, float), "close": np.asarray(c, float)})


def ocs(o, h, c):
    return PB.oc_labels(np.asarray(o, float), np.asarray(h, float), np.asarray(c, float))


# ----------------------------------------------------------------------
def test_hand_example():
    print("1. the hand example")
    #        T      T+1    T+2    T+3    T+4    T+5    T+6
    o = [98.0, 100.0, 104.0, 106.0, 102.0, 108.0, 109.0]
    c = [98.5, 102.0, 105.0, 103.0, 107.0, 110.0, 111.0]
    h = [x * 1.02 for x in np.maximum(o, c)]
    l = [x * 0.98 for x in np.minimum(o, c)]
    r = ocs(o, h, c)
    want = {1: 0.02, 2: 0.05, 3: 0.03, 5: 0.10}
    for k, v in want.items():
        check(math.isclose(r[f"label_oc_{k}"][0], v, abs_tol=1e-12), f"oc_{k} = {v:+.0%} (panel_build)")
    ref = LA.reference_oc_labels(frame(o, h, l, c))
    for k, v in want.items():
        check(math.isclose(ref[f"r_oc_{k}"].iloc[0], v, abs_tol=1e-12), f"oc_{k} = {v:+.0%} (label_audit's own loop)")
    check(r["label_buyable_o1"][0] == 1.0, "no lock -> buyable 1")
    full = PB.compute_labels(frame(o, h, l, c))
    check(all(c_ in full.columns for c_ in PB._OC_LABELS), "compute_labels emits all five new columns")
    check(math.isclose(full["label_oc_5"].iloc[0], 0.10, abs_tol=1e-12), "compute_labels carries the same values")


def test_locks():
    print("2. the next-open lock and missing bars")
    cases = [("ELIN", 177.46, (181.0, 181.0, 181.0, 181.0), 0.0),
             ("TVSELECT", 437.05, (445.75, 445.75, 445.75, 445.75), 0.0),
             ("TTML", 164.0, (172.2, 172.2, 165.1, 172.2), 0.0),
             ("TANLA", 44.55, (46.75, 46.75, 42.35, 46.15), 0.0),
             ("LLOYDSENGG", 15.74, (16.49, 16.49, 15.74, 16.49), 0.0),
             ("RPOWER", 6.65, (7.0, 7.3, 6.65, 7.3), 1.0),
             ("EXICOM", 111.76, (114.0, 117.88, 110.49, 116.48), 1.0),
             ("PCJEWELLER", 7.76, (7.86, 8.14, 7.80, 8.14), 1.0)]
    for name, cT, (o1, h1, l1, c1), want in cases:
        r = ocs([cT, o1], [cT * 1.01, h1], [cT, c1])
        check(r["label_buyable_o1"][0] == want, f"{name}: buyable {int(want)}")
        check(math.isclose(r["label_oc_1"][0], c1 / o1 - 1, abs_tol=1e-12), f"{name}: oc_1 kept ({c1/o1-1:+.2%}) - marked, not deleted")
    r = ocs([100, np.nan, 101], [101, 102, 102], [100, 101, 101.5])
    check(np.isnan(r["label_oc_1"][0]) and np.isnan(r["label_buyable_o1"][0]), "missing T+1 open -> NaN (unknown, not 0)")
    r = ocs([100, 0.0, 101], [101, 102, 102], [100, 101, 101.5])
    check(np.isnan(r["label_oc_1"][0]) and np.isnan(r["label_buyable_o1"][0]), "non-positive T+1 open -> NaN")
    r = ocs([100, 101, 101], [101, 102, 102], [100, 101, np.nan])
    check(np.isnan(r["label_oc_2"][0]) and math.isfinite(r["label_oc_1"][0]), "missing close T+2 -> oc_2 NaN, oc_1 still known")
    r = ocs([100, 105, 106], [101, 105, 107], [np.nan, 104, 106])
    check(np.isnan(r["label_buyable_o1"][0]) and math.isfinite(r["label_oc_1"][0]),
          "no close at T -> lock unknowable -> buyable NaN (oc_1 needs no close T)")


def _path(n, rng, start=100.0):
    o, h, l, c = (np.zeros(n) for _ in range(4))
    p = start
    for i in range(n):
        o[i] = p * (1 + rng.normal(0, 0.005))
        c[i] = o[i] * (1 + rng.normal(0, 0.02))
        h[i] = max(o[i], c[i]) * (1 + rng.uniform(0.001, 0.01))
        l[i] = min(o[i], c[i]) * (1 - rng.uniform(0.001, 0.01))
        p = c[i]
    return [np.round(x, 2) for x in (o, h, l, c)]


def test_perturbation():
    print("3. perturbation: which prices each label may and may not depend on")
    rng = np.random.default_rng(4)
    o, h, l, c = _path(40, rng)
    t = 20
    base = ocs(o, h, c)
    for k in PB.OC_HORIZONS:
        lab = f"label_oc_{k}"
        c2 = c.copy(); c2[t] *= 1.37
        check(ocs(o, h, c2)[lab][t] == base[lab][t], f"oc_{k}: close[T] changed -> unchanged (no T-day price)")
        o2 = o.copy(); o2[t + 2:] *= 0.71
        check(ocs(o2, h, c)[lab][t] == base[lab][t], f"oc_{k}: opens after T+1 changed -> unchanged")
        c3 = c.copy(); c3[t + k + 1:] *= 1.9
        check(ocs(o, h, c3)[lab][t] == base[lab][t], f"oc_{k}: close[T+{k+1}..] changed -> unchanged (no T+h+1 leak)")
        c4 = c.copy(); c4[t + k] *= 1.05
        check(ocs(o, h, c4)[lab][t] != base[lab][t], f"oc_{k}: close[T+{k}] changed -> changes (the exit bar)")
        o3 = o.copy(); o3[t + 1] *= 0.95
        check(ocs(o3, h, c)[lab][t] != base[lab][t], f"oc_{k}: open[T+1] changed -> changes (the entry bar)")
        if k > 1:
            c5 = c.copy(); c5[t + 1:t + k] *= 1.2
            check(ocs(o, h, c5)[lab][t] == base[lab][t], f"oc_{k}: closes between entry and exit changed -> unchanged")


def test_pending():
    print("4. pending tails")
    rng = np.random.default_rng(5)
    o, h, l, c = _path(30, rng)
    r = ocs(o, h, c)
    for k in PB.OC_HORIZONS:
        v = r[f"label_oc_{k}"]
        check(np.isnan(v[-k:]).all() and np.isfinite(v[-k - 1]), f"oc_{k}: last {k} bars NaN, bar {k+1} from the end known")
    check(np.isnan(r["label_buyable_o1"][-1]) and np.isfinite(r["label_buyable_o1"][-2]), "buyable: only the last bar unknown")


def test_vs_reference():
    print("5. panel_build = independent loop; lock rule = tradability_audit's")
    import tradability_audit as TA
    rng = np.random.default_rng(6)
    mism, rows, locks, agree = 0, 0, 0, 0
    for s in range(25):
        o, h, l, c = _path(300, rng, 20 + 10 * s)
        for i in rng.choice(np.arange(5, 290), 12, replace=False):     # planted upper-band opens
            b = rng.choice([0.02, 0.05, 0.10, 0.20])
            lk = math.floor(c[i] * (1 + b) / 0.05 + 1e-9) * 0.05
            o[i + 1] = h[i + 1] = round(lk, 2)
            l[i + 1] = min(l[i + 1], o[i + 1] * 0.99); c[i + 1] = min(c[i + 1], h[i + 1])
        r = ocs(o, h, c)
        ref = LA.reference_oc_labels(frame(o, h, l, c))
        for k in PB.OC_HORIZONS:
            a, b_ = r[f"label_oc_{k}"], ref[f"r_oc_{k}"].to_numpy()
            mism += int((np.isfinite(a) != np.isfinite(b_)).sum()) + int((np.abs(a - b_)[np.isfinite(a)] > 1e-12).sum())
        a, b_ = r["label_buyable_o1"], ref["r_buy"].to_numpy()
        mism += int((np.isfinite(a) != np.isfinite(b_)).sum()) + int((a[np.isfinite(a)] != b_[np.isfinite(a)]).sum())
        rows += len(c)
        locks += int((a == 0).sum())
        f = TA.symbol_flags(o, h, l, c)
        agree += int(((a == 0) == f["open_uc"])[np.isfinite(a)].all())
    check(mism == 0, f"0 mismatches on {rows:,} rows x 5 labels")
    check(locks >= 25 * 12 * 0.95, f"planted locks detected: {locks} (planted {25*12})")
    check(agree == 25, "lock flag identical to tradability_audit v1 on every path")


def test_universe():
    print("6. universe rule")
    d = Path(tempfile.mkdtemp())
    try:
        syms = ["RELIANCE", "KOTAKBANK", "HDFCBANK", "PNBGILTS", "TTML", "MAFANG", "MON100", "NIFTYBEES",
                "GOLDIETF", "SETFNIF50", "CPSEETF", "EBBETF0430", "WEIRDFUND"]
        kept, exc, extra = PB.select_universe(syms, d)
        check(kept == ["RELIANCE", "KOTAKBANK", "HDFCBANK", "PNBGILTS", "TTML", "WEIRDFUND"] and len(exc) == 7,
              f"7 funds out, equities kept ({', '.join(exc)})")
        (d / RC.UNIVERSE_EXCLUDE_FILE).write_text("# mine\nweirdfund\n", encoding="utf-8")
        kept, exc, extra = PB.select_universe(syms, d)
        check("WEIRDFUND" in exc and extra == ["WEIRDFUND"], "universe_exclude.txt adds a name the rule misses")
    finally:
        shutil.rmtree(d)
    s1 = PB._build_signature(["A", "B"], 1e7)
    check(s1 == PB._build_signature(["B", "A"], 1e7) and s1 != PB._build_signature(["A"], 1e7),
          "universe change -> new build signature -> full rebuild")


def test_firewall():
    print("7. firewall")
    cols = ["timestamp", "symbol", "D_rsi14", "X_rank_D_rsi14"] + list(PB._OC_LABELS) + ["label_exit_ret"]
    feats = PB.panel_feature_columns(pd.DataFrame(columns=cols))
    check(not any(c.startswith("label_") for c in feats) and "D_rsi14" in feats, f"features: {feats}")
    check(all(RC.is_forbidden_feature(c) for c in PB._OC_LABELS), "every new label is forbidden by rule")
    try:
        RC.assert_no_label_leak(["D_rsi14", "label_buyable_o1"], "test")
        check(False, "leak not caught")
    except RC.LabelLeakError:
        check(True, "label_buyable_o1 handed to a model -> LabelLeakError")


GOLDEN_OLD_LABELS = {
    '0|label_mfe_5d': (14.027878, 5),
    '0|label_mae_5d': (-13.947016, 5),
    '0|label_fwd_ret_5d': (0.109421, 5),
    '0|label_touch': (120.0, 5),
    '0|label_tp_before_sl_5p3': (120.0, 5),
    '0|label_days_to_tp_5p3': (363.0, 300),
    '0|label_same_day_ambiguous_5p3': (0.0, 5),
    '0|label_touch_5p3': (120.0, 5),
    '0|label_exit_ret_5p3': (2.150737, 5),
    '0|label_tp_before_sl_3p2': (151.0, 5),
    '0|label_days_to_tp_3p2': (338.0, 269),
    '0|label_same_day_ambiguous_3p2': (0.0, 5),
    '0|label_touch_3p2': (151.0, 5),
    '0|label_exit_ret_3p2': (1.20374, 5),
    '0|label_tp_before_sl_atr1p5_1p0': (145.0, 19),
    '0|label_days_to_tp_atr1p5_1p0': (341.0, 275),
    '0|label_same_day_ambiguous_atr1p5_1p0': (0.0, 19),
    '0|label_touch_atr1p5_1p0': (145.0, 19),
    '0|label_exit_ret_atr1p5_1p0': (1.358552, 19),
    '0|label_first_touch': (-31.0, 19),
    '0|label_tp_before_sl': (145.0, 19),
    '0|label_days_to_tp': (341.0, 275),
    '0|label_days_to_sl': (334.0, 244),
    '0|label_same_day_ambiguous': (0.0, 19),
    '0|label_exit_ret': (1.358552, 19),
    '0|label_tp_before_sl_atr2p0_1p0': (128.0, 19),
    '0|label_days_to_tp_atr2p0_1p0': (356.0, 292),
    '0|label_same_day_ambiguous_atr2p0_1p0': (0.0, 19),
    '0|label_touch_atr2p0_1p0': (128.0, 19),
    '0|label_exit_ret_atr2p0_1p0': (2.691232, 19),
    '0|label_tp_before_sl_atr1p0_1p0': (173.0, 19),
    '0|label_days_to_tp_atr1p0_1p0': (330.0, 247),
    '0|label_same_day_ambiguous_atr1p0_1p0': (0.0, 19),
    '0|label_touch_atr1p0_1p0': (173.0, 19),
    '0|label_exit_ret_atr1p0_1p0': (-0.113708, 19),
    '0|label_first_touch_o1': (-35.0, 19),
    '0|label_tp_before_sl_o1': (134.0, 19),
    '0|label_exit_ret_o1': (1.240382, 19),
    '0|label_mfe_5d_o1': (13.798646, 19),
    '0|label_mae_5d_o1': (-13.740767, 19),
    '0|label_days_to_tp_o1': (416.0, 286),
    '0|label_days_to_sl_o1': (464.0, 251),
    '0|label_exit_ret_atr2p0_1p0_o1': (2.778221, 19),
    '0|label_gap1': (-0.167429, 19),
    '1|label_mfe_5d': (15.690274, 5),
    '1|label_mae_5d': (-13.510755, 5),
    '1|label_fwd_ret_5d': (1.869037, 5),
    '1|label_touch': (132.0, 5),
    '1|label_tp_before_sl_5p3': (132.0, 5),
    '1|label_days_to_tp_5p3': (389.0, 288),
    '1|label_same_day_ambiguous_5p3': (0.0, 5),
    '1|label_touch_5p3': (132.0, 5),
    '1|label_exit_ret_5p3': (2.931769, 5),
    '1|label_tp_before_sl_3p2': (162.0, 5),
    '1|label_days_to_tp_3p2': (353.0, 258),
    '1|label_same_day_ambiguous_3p2': (0.0, 5),
    '1|label_touch_3p2': (162.0, 5),
    '1|label_exit_ret_3p2': (1.707924, 5),
    '1|label_tp_before_sl_atr1p5_1p0': (144.0, 19),
    '1|label_days_to_tp_atr1p5_1p0': (331.0, 276),
    '1|label_same_day_ambiguous_atr1p5_1p0': (0.0, 19),
    '1|label_touch_atr1p5_1p0': (144.0, 19),
    '1|label_exit_ret_atr1p5_1p0': (1.569883, 19),
    '1|label_first_touch': (-25.0, 19),
    '1|label_tp_before_sl': (144.0, 19),
    '1|label_days_to_tp': (331.0, 276),
    '1|label_days_to_sl': (319.0, 251),
    '1|label_same_day_ambiguous': (0.0, 19),
    '1|label_exit_ret': (1.569883, 19),
    '1|label_tp_before_sl_atr2p0_1p0': (125.0, 19),
    '1|label_days_to_tp_atr2p0_1p0': (323.0, 295),
    '1|label_same_day_ambiguous_atr2p0_1p0': (0.0, 19),
    '1|label_touch_atr2p0_1p0': (125.0, 19),
    '1|label_exit_ret_atr2p0_1p0': (2.910568, 19),
    '1|label_tp_before_sl_atr1p0_1p0': (176.0, 19),
    '1|label_days_to_tp_atr1p0_1p0': (337.0, 244),
    '1|label_same_day_ambiguous_atr1p0_1p0': (0.0, 19),
    '1|label_touch_atr1p0_1p0': (176.0, 19),
    '1|label_exit_ret_atr1p0_1p0': (0.108127, 19),
    '1|label_first_touch_o1': (-26.0, 19),
    '1|label_tp_before_sl_o1': (134.0, 19),
    '1|label_exit_ret_o1': (1.610566, 19),
    '1|label_mfe_5d_o1': (14.811732, 19),
    '1|label_mae_5d_o1': (-13.665963, 19),
    '1|label_days_to_tp_o1': (406.0, 286),
    '1|label_days_to_sl_o1': (428.0, 260),
    '1|label_exit_ret_atr2p0_1p0_o1': (3.258929, 19),
    '1|label_gap1': (0.041662, 19),
    '2|label_mfe_5d': (16.033881, 5),
    '2|label_mae_5d': (-13.267841, 5),
    '2|label_fwd_ret_5d': (2.325687, 5),
    '2|label_touch': (134.0, 5),
    '2|label_tp_before_sl_5p3': (134.0, 5),
    '2|label_days_to_tp_5p3': (398.0, 286),
    '2|label_same_day_ambiguous_5p3': (0.0, 5),
    '2|label_touch_5p3': (134.0, 5),
    '2|label_exit_ret_5p3': (3.138834, 5),
    '2|label_tp_before_sl_3p2': (169.0, 5),
    '2|label_days_to_tp_3p2': (374.0, 251),
    '2|label_same_day_ambiguous_3p2': (0.0, 5),
    '2|label_touch_3p2': (169.0, 5),
    '2|label_exit_ret_3p2': (1.841022, 5),
    '2|label_tp_before_sl_atr1p5_1p0': (155.0, 19),
    '2|label_days_to_tp_atr1p5_1p0': (353.0, 265),
    '2|label_same_day_ambiguous_atr1p5_1p0': (0.0, 19),
    '2|label_touch_atr1p5_1p0': (155.0, 19),
    '2|label_exit_ret_atr1p5_1p0': (1.944605, 19),
    '2|label_first_touch': (-10.0, 19),
    '2|label_tp_before_sl': (155.0, 19),
    '2|label_days_to_tp': (353.0, 265),
    '2|label_days_to_sl': (309.0, 255),
    '2|label_same_day_ambiguous': (0.0, 19),
    '2|label_exit_ret': (1.944605, 19),
    '2|label_tp_before_sl_atr2p0_1p0': (138.0, 19),
    '2|label_days_to_tp_atr2p0_1p0': (374.0, 282),
    '2|label_same_day_ambiguous_atr2p0_1p0': (0.0, 19),
    '2|label_touch_atr2p0_1p0': (138.0, 19),
    '2|label_exit_ret_atr2p0_1p0': (3.389875, 19),
    '2|label_tp_before_sl_atr1p0_1p0': (182.0, 19),
    '2|label_days_to_tp_atr1p0_1p0': (336.0, 238),
    '2|label_same_day_ambiguous_atr1p0_1p0': (0.0, 19),
    '2|label_touch_atr1p0_1p0': (182.0, 19),
    '2|label_exit_ret_atr1p0_1p0': (0.349608, 19),
    '2|label_first_touch_o1': (-12.0, 19),
    '2|label_tp_before_sl_o1': (145.0, 19),
    '2|label_exit_ret_o1': (2.080423, 19),
    '2|label_mfe_5d_o1': (15.288936, 19),
    '2|label_mae_5d_o1': (-13.224552, 19),
    '2|label_days_to_tp_o1': (444.0, 275),
    '2|label_days_to_sl_o1': (427.0, 263),
    '2|label_exit_ret_atr2p0_1p0_o1': (3.660144, 19),
    '2|label_gap1': (0.247135, 19),
}


def golden_frame(n=420, k=0):
    i = np.arange(n, dtype=float)
    c = 100 * (1 + 0.25 * np.sin(i / 17 + k) + 0.1 * np.cos(i / 5.3 + 2 * k)) + 0.03 * i
    o = c * (1 + 0.006 * np.sin(i / 2.1 + k))
    h = np.maximum(o, c) * (1 + 0.004 + 0.003 * np.abs(np.sin(i / 3.7)))
    l = np.minimum(o, c) * (1 - 0.004 - 0.003 * np.abs(np.cos(i / 4.1)))
    return pd.DataFrame({"timestamp": pd.bdate_range("2020-01-01", periods=n), "open": np.round(o, 2),
                         "high": np.round(h, 2), "low": np.round(l, 2), "close": np.round(c, 2)})


def test_golden():
    print("8. the 44 pre-existing label columns vs the shipped v31 code (golden checksums)")
    bad = []
    for k in range(3):
        a = PB.compute_labels(golden_frame(k=k))
        for key, (s_, nn) in GOLDEN_OLD_LABELS.items():
            kk, col = key.split("|")
            if int(kk) != k:
                continue
            x = pd.to_numeric(a[col]).astype(float).to_numpy()
            if int(np.isnan(x).sum()) != nn or abs(float(np.nansum(x)) - s_) > 1e-6 * max(1.0, abs(s_)):
                bad.append(key)
    check(not bad, f"{len(GOLDEN_OLD_LABELS)} column checksums identical" + (f" - CHANGED: {bad[:5]}" if bad else ""))


# ----------------------------------------------------------------------
class _Cal:
    def __init__(self, sessions):
        self.sessions, self.trusted = sessions, True


def make_cache(root: Path, n_eq=24, n=460, seed=9):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n)
    syms = [f"STK{j:02d}" for j in range(n_eq)] + ["MAFANG", "NIFTYBEES"]
    for j, s in enumerate(syms):
        o, h, l, c = _path(n, rng, 50 + 7 * j)
        for i in rng.choice(np.arange(330, n - 8), 6, replace=False):      # planted upper-band opens
            lk = math.floor(c[i] * 1.05 / 0.05 + 1e-9) * 0.05
            o[i + 1] = h[i + 1] = round(lk, 2)
            l[i + 1] = min(l[i + 1], o[i + 1] * 0.99); c[i + 1] = min(c[i + 1], h[i + 1])
        pd.DataFrame({"timestamp": dates, "open": o, "high": h, "low": l, "close": c,
                      "volume": rng.integers(200_000, 900_000, n).astype(float)}).to_parquet(
            root / f"{s}_daily.parquet", index=False)
        (root / f"{s}_daily.ok.json").write_text(json.dumps({"series_kind": "equity"}), encoding="utf-8")
    return syms, dates


def build(root: Path, syms, dates, folder="panel"):
    orig = (PB.assert_panel_ready, PB.build_calendar)
    PB.assert_panel_ready = lambda r, s, **k: list(s)
    PB.build_calendar = lambda r, s: _Cal(dates)
    try:
        PB.build_panel(root, root / folder, symbols=syms, full=True, min_turnover=0.0, verbose=False)
    finally:
        PB.assert_panel_ready, PB.build_calendar = orig


def run_audit(root, **kw):
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        R = LA.audit(root, n_symbols=30, panel_dir=root / "panel_oc", **kw)
    return R


def test_end_to_end():
    print("9. end to end: build_panel -> label_audit, then five sabotaged panels")
    d = Path(tempfile.mkdtemp())
    try:
        syms, dates = make_cache(d)
        build(d, syms, dates, "panel_oc")                  # the real layout: a separate panel folder
        pp = d / "panel_oc" / "panel.parquet"
        P = pd.read_parquet(pp)
        meta = json.loads((d / "panel_oc" / "panel_meta.json").read_text())
        check(not P["symbol"].isin(["MAFANG", "NIFTYBEES"]).any() and meta["etf_excluded"] == ["MAFANG", "NIFTYBEES"],
              "ETFs absent from the panel and listed in panel_meta")
        check(meta["panel_build_version"] == PB.PANEL_BUILD_VERSION and "open_to_close" in meta["label_contract"],
              f"meta records {meta['panel_build_version']} and the open-to-close contract")
        check(all(c in P.columns for c in PB._OC_LABELS), "all five new labels in the panel")
        check(P["label_oc_1"].dtype == "float64", "labels stay float64 (not downcast with the features)")
        n0 = int((P["label_buyable_o1"] == 0).sum())
        check(n0 >= 24 * 6 * 0.9, f"planted next-open locks marked unbuyable: {n0} rows kept with buyable = 0")
        R = run_audit(d)
        check(not R.fail, "label_audit PASSES on the clean panel" + (f": {R.fail}" if R.fail else ""))
        la = json.loads((d / "panel_oc" / "label_audit.json").read_text())
        check(la["open_to_close"]["identity_failures"] == 0 and la["open_to_close"]["manifest"]["family_id"] == "oc_v1",
              "identity holds on every row; experiment family oc_v1 recorded")
        led = [json.loads(x) for x in (d / "panel_oc" / "research_ledger.jsonl").read_text().splitlines()]
        check(any(e.get("kind") == "panel_audit" and e.get("passed") for e in led), "audit logged in the research ledger")
        clean = P.copy()
        shutil.copy(pp, d / "panel_A0.parquet")
        R = run_audit(d, compare_to=d / "panel_A0.parquet")
        check(not R.fail, "--compare-to an identical panel: pre-existing labels identical")

        def sabotage(fn, expect, name):
            Q = fn(clean.copy())
            Q.to_parquet(pp, index=False)
            R = run_audit(d)
            hit = [f for f in R.fail if expect in f]
            check(bool(hit), f"sabotage '{name}' -> FAIL: {hit[0][:90] if hit else R.fail}")

        def close_to_close(Q):
            Q["label_oc_5"] = Q["label_fwd_ret_5d"].astype(float)
            return Q
        sabotage(close_to_close, "identity", "oc_5 built close T -> close T+5")

        def shift_a_day(Q):
            Q["label_oc_2"] = Q.groupby("symbol")["label_oc_2"].shift(-1)
            return Q
        sabotage(shift_a_day, "label_oc_2 mismatches", "oc_2 shifted one day")

        def unlock(Q):
            Q.loc[Q["label_buyable_o1"] == 0, "label_buyable_o1"] = 1.0
            return Q
        sabotage(unlock, "label_buyable_o1 mismatches", "locked opens marked buyable")

        def inject_etf(Q):
            e = Q[Q["symbol"] == "STK00"].copy()
            e["symbol"] = "GOLDBEES"
            return pd.concat([Q, e], ignore_index=True)
        sabotage(inject_etf, "ETF / fund symbols", "an ETF row in the panel")

        Q = clean.copy()
        m = Q.index % 7 == 0
        Q.loc[m, "label_exit_ret"] = Q.loc[m, "label_exit_ret"] + 0.01
        Q.to_parquet(pp, index=False)
        R = run_audit(d, compare_to=d / "panel_A0.parquet")
        check(any("label code changed" in f for f in R.fail), "old labels changed on 14% of rows vs --compare-to -> FAIL")
    finally:
        shutil.rmtree(d)


def test_module_origin():
    print("0. the new modules are the ones loaded (no shadowing by the parent folder)")
    here = HERE.parent.resolve()
    for m in (PB, LA, RC):
        check(Path(m.__file__).resolve().parent == here, f"{m.__name__} from {Path(m.__file__).resolve().parent}")
    check(hasattr(RC, "is_etf_symbol") and PB.PANEL_BUILD_VERSION.endswith("v32"), "research_common has the v32 rules")


def main():
    test_module_origin()
    test_hand_example()
    test_locks()
    test_perturbation()
    test_pending()
    test_vs_reference()
    test_universe()
    test_firewall()
    test_golden()
    test_end_to_end()
    print(f"\nVERIFIED  {PB.PANEL_BUILD_VERSION} + {LA.LABEL_AUDIT_VERSION}")


if __name__ == "__main__":
    main()
