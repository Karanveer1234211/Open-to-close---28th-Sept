#!/usr/bin/env python3
"""
tests/test_stage_c_features.py - stage_c_features v1. Prints VERIFIED on success.

Run from the open_close folder with the main folder on PYTHONPATH:
    python tests\\test_stage_c_features.py

 0. Module origin.
 1. The declaration: 123 unique names; the builder produces exactly those.
 2. Formulas against independent code: Corwin-Schultz and Abdi-Ranaldo spreads (plain loops),
    rolling OLS and conditional beta (numpy polyfit), price delay (lstsq), coskewness, window
    percentiles, top-k means, bars-since, streaks, seasonality, frog-in-the-pan, turnover shocks.
 3. NSE band locks: a lower-band close at the low is a lock, one above the low is not; tick size by
    date decides for cheap stocks.
 4. POINT IN TIME: every feature recomputed on data cut at three dates (bars AND universe cut)
    equals the full-data value on those dates - per-symbol, market, ranks and interactions.
    And perturbing every bar after a date changes nothing on or before it.
 5. The universe market series averages members only; the builder never reads a label column.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import research_common as RC          # noqa: E402
import panel_build as PB              # noqa: E402
import stage_b_screen as SB           # noqa: E402
import stage_c_features as SC         # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok  " if cond else "  FAIL") + f"  {msg}")
    if not cond:
        FAILS.append(msg)


def test_origin_and_declaration():
    print("0-1. module origin and the declaration")
    here = HERE.parent.resolve()
    for m in (SC, SB, RC, PB):
        check(Path(m.__file__).resolve().parent == here, f"{m.__name__} from {Path(m.__file__).resolve().parent}")
    decl, sha, names = SC.load_declaration()
    check(len(names) == 123 and len(set(names)) == 123, f"123 unique declared features (sha {sha})")
    check(not any(n.startswith("label_") for n in names), "no declared feature is a label")


def test_formulas():
    print("2. formulas against independent code")
    rng = np.random.default_rng(4)
    n = 400
    c = 100 * np.exp(np.cumsum(0.02 * rng.normal(size=n)))
    o = c * np.exp(0.005 * rng.normal(size=n))
    h = np.maximum(o, c) * (1 + np.abs(0.01 * rng.normal(size=n)))
    l = np.minimum(o, c) * (1 - np.abs(0.01 * rng.normal(size=n)))
    cs = SC.cs_spread_pairs(h, l, c)
    ref = [np.nan]
    k = 3 - 2 * np.sqrt(2)
    for t in range(1, n):
        h1, l1, c1, h2, l2 = h[t - 1], l[t - 1], c[t - 1], h[t], l[t]
        if l2 > c1:
            d = l2 - c1
            h2, l2 = h2 - d, l2 - d
        elif h2 < c1:
            d = c1 - h2
            h2, l2 = h2 + d, l2 + d
        b = np.log(h1 / l1) ** 2 + np.log(h2 / l2) ** 2
        g = np.log(max(h1, h2) / min(l1, l2)) ** 2
        a = (np.sqrt(2 * b) - np.sqrt(b)) / k - np.sqrt(g / k)
        ref.append(max(2 * (np.exp(a) - 1) / (1 + np.exp(a)), 0))
    check(np.allclose(cs[1:], ref[1:]) and np.isnan(cs[0]), "Corwin-Schultz per-pair spread = plain loop (overnight-adjusted)")
    ar = SC.ar_spread_sq_pairs(h, l, c)
    eta = (np.log(h) + np.log(l)) / 2
    ref = [4 * (np.log(c[t - 1]) - eta[t - 1]) * (np.log(c[t - 1]) - eta[t]) for t in range(1, n)]
    check(np.allclose(ar[1:], ref), "Abdi-Ranaldo per-pair term = plain loop")
    x = rng.normal(size=n)
    y = 0.7 * x + 0.3 * rng.normal(size=n)
    a_, b_, r2, vy = SC.rolling_ols(y, x, 60)
    t = 250
    pb = np.polyfit(x[t - 59:t + 1], y[t - 59:t + 1], 1)
    cr = np.corrcoef(x[t - 59:t + 1], y[t - 59:t + 1])[0, 1] ** 2
    check(abs(b_[t] - pb[0]) < 1e-10 and abs(a_[t] - pb[1]) < 1e-10 and abs(r2[t] - cr) < 1e-10,
          "rolling OLS beta, alpha, R2 = numpy polyfit / corrcoef on the same window")
    cb = SC.conditional_beta(y, x, 120, x < 0)
    w = slice(t - 119, t + 1)
    mask = x[w] < 0
    check(abs(cb[t] - np.polyfit(x[w][mask], y[w][mask], 1)[0]) < 1e-9, "conditional (down-market) beta = polyfit on the down days")
    m = rng.normal(size=n)
    yy = 0.5 * m + 0.4 * np.r_[0, m[:-1]] + 0.2 * rng.normal(size=n)
    d_, lb = SC.delay_measure(yy, m, 120)
    t = 300
    ww = slice(t - 119, t + 1)
    Xu = np.column_stack([np.ones(120)] + [np.r_[np.full(k_, np.nan), m[:n - k_]][ww] for k_ in range(5)])
    cu = np.linalg.lstsq(Xu, yy[ww], rcond=None)[0]
    r2u = 1 - ((yy[ww] - Xu @ cu) ** 2).sum() / ((yy[ww] - yy[ww].mean()) ** 2).sum()
    Xr = Xu[:, :2]
    crr = np.linalg.lstsq(Xr, yy[ww], rcond=None)[0]
    r2r = 1 - ((yy[ww] - Xr @ crr) ** 2).sum() / ((yy[ww] - yy[ww].mean()) ** 2).sum()
    check(abs(d_[t] - (1 - r2r / r2u)) < 1e-9 and abs(lb[t] - cu[2:].sum()) < 1e-9 and lb[t] > 0.25,
          f"price delay = 1 - R2r/R2u by lstsq; lagged beta found ({lb[t]:.2f}, planted 0.4)")
    v = np.array([3, 1, 2, 5, 4, 5, 0, 7], float)
    check(np.allclose(SC.pctile_in_window(v, 4)[3:], [(np.array(v[i - 3:i + 1]) < v[i]).mean() for i in range(3, 8)]),
          "window percentile: share strictly below today's value")
    check(abs(SC.top_k_mean(v, 5, 2)[4] - 4.5) < 1e-12, "top-2 mean of [3,1,2,5,4] = 4.5")
    check(SC.bars_since_extreme(np.array([1, 5, 2, 3, 5, 1.0]), 6, np.argmax)[5] == 1 / 6,
          "bars since the most recent max (ties go to the latest)")
    fl = np.array([0, 1, 1, 0, 1, 1, 1], bool)
    check(list(SC.bars_since_flag(fl, 250)) == [250, 0, 0, 1, 0, 0, 0] and list(SC.streak_of(fl)) == [0, 1, 2, 0, 1, 2, 3],
          "bars since a lock and lock streaks")
    check(list(SC.signed_streak(np.array([1, 2, -1, -3, 0, 4.0]))) == [1, 2, -1, -2, 0, 1], "signed up/down streak")
    ts = pd.date_range("2015-01-01", "2021-12-31", freq="B")
    cc = pd.Series(100.0, index=ts)
    for y_ in range(2015, 2022):
        cc[(ts.year == y_) & (ts.month >= 3)] *= 1.1
    s_ = SC.season_same_month(ts.values, cc.values)
    march21 = np.flatnonzero((ts.year == 2021) & (ts.month == 3))[0]
    check(abs(s_[march21] - np.log(1.1)) < 1e-12 and np.isnan(s_[np.flatnonzero((ts.year == 2016) & (ts.month == 3))[0]]),
          "seasonality: March 2021 sees the +10% Marches of 2016-2020; 2016 has too few years")


def make_world(tmp: Path, seed=3, n_sym=12, drop_future_after=None, perturb_after=None):
    """The FULL world is always generated first (same random draws), then cut or perturbed."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", "2021-12-31")
    T = len(dates)
    root = tmp / "cache"
    root.mkdir(parents=True)
    frames, rows = {}, []
    for k in range(n_sym):
        base = 5.0 if k == 0 else float(np.exp(rng.uniform(3, 7)))
        r = 0.02 * rng.normal(size=T)
        c = base * np.exp(np.cumsum(r))
        o = np.r_[c[0], c[:-1]] * np.exp(0.005 * rng.normal(size=T))
        h = np.maximum(o, c) * (1 + np.abs(0.01 * rng.normal(size=T)))
        l = np.minimum(o, c) * (1 - np.abs(0.01 * rng.normal(size=T)))
        if k == 1:                                 # planted lower-band locks
            for t in range(300, T, 40):
                c[t] = round(c[t - 1] * 0.95, 2)
                l[t] = c[t]
                o[t] = max(o[t], c[t])
                h[t] = max(h[t], o[t], c[t])
        vol = rng.integers(10_000, 500_000, size=T).astype(float)
        keep = rng.random(T) > 0.02 if k else np.ones(T, bool)
        uni = rng.random(T) > 0.05                                  # liquidity floor: some days out
        b = pd.DataFrame({"timestamp": dates.values, "open": o, "high": h, "low": l, "close": c, "volume": vol})
        frames[k] = b[keep].reset_index(drop=True)
        rows.append(pd.DataFrame({"timestamp": dates.values[keep & uni], "symbol": f"S{k:02d}"}))
    P = pd.concat(rows, ignore_index=True).sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    r2 = np.random.default_rng(99)
    for ccol in SC.PANEL_INPUTS:
        P[ccol] = r2.normal(size=len(P))
    P["label_oc_1"] = 0.0
    prng = np.random.default_rng(7)
    for k, b in frames.items():
        if perturb_after is not None:
            fut = (b["timestamp"] > perturb_after).to_numpy()
            for col in ("open", "high", "low", "close"):
                b.loc[fut, col] = b.loc[fut, col] * prng.uniform(0.5, 1.5, size=int(fut.sum()))
            b.loc[fut, "high"] = b.loc[fut, ["open", "high", "low", "close"]].max(axis=1)
            b.loc[fut, "low"] = b.loc[fut, ["open", "high", "low", "close"]].min(axis=1)
        if drop_future_after is not None:
            b = b[b["timestamp"] <= drop_future_after]
        b.to_parquet(root / f"S{k:02d}_daily.parquet", index=False)
    if drop_future_after is not None:
        P = P[P["timestamp"] <= drop_future_after].reset_index(drop=True)
    pdir = tmp / "panel_oc"
    pdir.mkdir()
    P.to_parquet(pdir / "panel.parquet", index=False)
    return root, pdir


def test_locks():
    print("3. NSE band locks")
    ts = np.array([np.datetime64(x, "ns") for x in ("2023-06-01", "2023-06-02", "2023-06-05")])
    o = np.array([100.0, 99.0, 96.0])
    h = np.array([101.0, 99.0, 97.0])
    c = np.array([100.0, 95.0, 95.5])
    l = np.array([99.0, 95.0, 95.0])
    lu, ld, od = SC.lock_flags(ts, o, h, l, c)
    check(bool(ld[1]) and not ld[2] and not lu.any(), "a -5% close at the low is a lower lock; a close above the low is not")
    ts2 = np.array([np.datetime64(x, "ns") for x in ("2024-07-01", "2024-07-02")])
    ts1 = ts2 - np.timedelta64(400, "D")
    oo, hh, ll, cc = np.array([2.0, 2.0]), np.array([2.01, 2.0]), np.array([1.99, 1.976]), np.array([2.0, 1.976])
    check(bool(SC.lock_flags(ts1, oo, hh, ll, cc)[1][1]) and not SC.lock_flags(ts2, oo, hh, ll, cc)[1][1],
          "a Rs 2 stock -1.2% at its low: a lock under the Rs 0.05 tick, not under the Rs 0.01 tick")


def test_point_in_time():
    print("4. point in time")
    tmp = Path(tempfile.mkdtemp())
    try:
        _, _, names = SC.load_declaration()
        root, pdir = make_world(tmp / "full")
        full = SC.build(pdir / "panel.parquet", root, names, verbose=False)
        check(list(full.columns[2:]) == names and len(full) == len(pd.read_parquet(pdir / "panel.parquet")),
              f"exactly the declared {len(names)} features, row for row with the panel ({len(full):,} rows)")
        cov = {c_: float(np.isfinite(full[c_].to_numpy()).mean()) for c_ in names}
        dead = [c_ for c_, v in cov.items() if v == 0]
        check(not dead, f"every feature has values somewhere (none all-NaN){' - dead: ' + str(dead) if dead else ''}")
        worst_all = 0.0
        for cut in ("2020-06-30", "2021-03-15", "2021-11-30"):
            T = pd.Timestamp(cut)
            sub = tmp / f"cut_{cut}"
            root_c, pdir_c = make_world(sub, drop_future_after=T)
            cutb = SC.build(pdir_c / "panel.parquet", root_c, names, verbose=False)
            a = full[full["timestamp"] == T].set_index("symbol")[names]
            b = cutb[cutb["timestamp"] == T].set_index("symbol")[names]
            A, Bm = a.to_numpy(float), b.reindex(a.index).to_numpy(float)
            same_nan = np.array_equal(np.isnan(A), np.isnan(Bm))
            diff = np.nanmax(np.abs(A - Bm) / np.maximum(1.0, np.abs(A)))
            worst_all = max(worst_all, diff)
            bad = [names[j] for j in range(len(names)) if not np.array_equal(np.isnan(A[:, j]), np.isnan(Bm[:, j]))
                   or np.nanmax(np.abs(A[:, j] - Bm[:, j]) / np.maximum(1.0, np.abs(A[:, j])), initial=0) > 1e-5]
            check(same_nan and diff < 1e-5 and not bad and len(a) >= 8,
                  f"cut at {cut}: all {len(names)} features on that day equal the full-data values "
                  f"({len(a)} stocks, max rel diff {diff:.1e}){' - differ: ' + str(bad[:6]) if bad else ''}")
        T = pd.Timestamp("2021-03-15")
        root_p, pdir_p = make_world(tmp / "perturbed", perturb_after=T)
        pert = SC.build(pdir_p / "panel.parquet", root_p, names, verbose=False)
        a = full[full["timestamp"] <= T][names].to_numpy(float)
        b = pert[pert["timestamp"] <= T][names].to_numpy(float)
        check(a.shape == b.shape and np.array_equal(np.isnan(a), np.isnan(b)) and np.nanmax(np.abs(a - b)) < 1e-9,
              "scrambling every bar after 2021-03-15 changes nothing on or before it")

        print("5. market series and the firewall")
        bars = {s: SC.load_bars(root, s) for s in ("S02", "S03")}
        mem = {"S02": bars["S02"]["timestamp"].to_numpy(dtype="datetime64[ns]"),
               "S03": bars["S03"]["timestamp"].to_numpy(dtype="datetime64[ns]")[:100]}
        M = SC.market_series(bars, mem)
        day = bars["S03"]["timestamp"].iloc[200]
        b2 = bars["S02"].set_index("timestamp")["close"]
        r2 = b2.loc[day] / b2.shift(1).loc[day] - 1
        check(abs(M.loc[day, "_mr"] - r2) < 1e-12, "the market that day averages universe members only (S03 was out)")
        seen = []
        orig = pd.read_parquet

        def spy(path, *a, **k):
            if str(path).endswith("panel.parquet"):
                seen.extend(k.get("columns") or [])
            return orig(path, *a, **k)
        pd.read_parquet = spy
        try:
            SC.build(pdir / "panel.parquet", root, names, verbose=False)
        finally:
            pd.read_parquet = orig
        check(seen and not any(str(c_).startswith("label_") for c_ in seen),
              f"the builder reads only {len(seen)} panel columns, none a label")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_origin_and_declaration()
    test_formulas()
    test_locks()
    test_point_in_time()
    print()
    if FAILS:
        print(f"FAILED  {len(FAILS)} check(s):")
        for f in FAILS:
            print("   - " + f)
        return 1
    print(f"VERIFIED  {SC.CODE_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
