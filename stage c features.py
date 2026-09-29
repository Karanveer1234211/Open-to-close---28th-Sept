#!/usr/bin/env python3
"""
stage_c_features.py - build the 123 stage-C features declared in
STAGE_C_DECLARATION.json, point in time, aligned row for row with the panel.

    python stage_c_features.py --panel %CACHE_DAILY_ROOT%\\panel_oc --root %CACHE_DAILY_ROOT%

WHAT GOES IN
------------
* the raw daily cache (open, high, low, close, volume) of every panel symbol;
* universe membership = the panel's rows (the liquidity floor is trailing, so
  membership on day T is known on day T);
* six panel columns, for interactions only: INDIAVIX_ret_1d, MKT_D_drawdown_252,
  D_dist_from_52wh, X_turnover_med, D_mdi14_diff1, D_WQ_44.
Nothing starting label_ is ever read.

POINT IN TIME
-------------
Every per-symbol feature on day T uses that symbol's bars up to and including
T. The market series (for betas, residuals, market state) is the equal-weight
mean over the stocks in the universe on each day. Per-date ranks rank only the
stocks in the universe that day. The test suite recomputes everything on data
cut at several dates and requires identical values on those dates.

THE GROUPS (formulas and sources: STAGE_C_DECLARATION.json)
  A overnight/intraday decomposition   F relative position to indicators
  B market model, residuals, delay     G momentum variants, seasonality
  C lottery / MAX                      H interactions
  D liquidity, spread estimators       I market state from the universe
  E NSE price-band (circuit) state     J per-date ranks

OUTPUT: <panel>\\stage_c\\features_c.parquet (timestamp, symbol, 123 features;
same rows, same order as panel.parquet), features_c_meta.json; ledger entries
stage_c_declared (before) and stage_c_features (after).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import research_common as RC       # first: from a subfolder, this folder's copy must win
import panel_build as PB           # noqa: E402,F401
import stage_b_screen as SB        # noqa: E402
import stage_d_forensics as SF     # noqa: E402  (tick sizes, the band test)

CODE_VERSION = "stage_c_features v1"
DECL_FILE = "STAGE_C_DECLARATION.json"
PANEL_INPUTS = ["INDIAVIX_ret_1d", "MKT_D_drawdown_252", "D_dist_from_52wh", "X_turnover_med",
                "D_mdi14_diff1", "D_WQ_44"]
EPS = 1e-12


class PreconditionError(SystemExit):
    pass


def _log(msg: str) -> None:
    print(f"{dt.datetime.now():%H:%M:%S}  {msg}", flush=True)


def load_declaration() -> Tuple[dict, str, List[str]]:
    p = HERE / DECL_FILE
    if not p.exists():
        raise PreconditionError(f"{DECL_FILE} not found next to {Path(__file__).name}")
    raw = p.read_bytes()
    decl = json.loads(raw.decode("utf-8"))
    if decl.get("stage") != "C":
        raise PreconditionError(f"{DECL_FILE} is not the stage-C declaration")
    names = [f["name"] for g in decl["groups"].values() for f in g]
    return decl, hashlib.sha256(raw).hexdigest()[:16], names


# ----------------------------------------------------------------------
# small, testable pieces
# ----------------------------------------------------------------------
def _s(x) -> pd.Series:
    return pd.Series(np.asarray(x, dtype=float))


def roll(x, n, how="mean", minp=None):
    r = _s(x).rolling(n, min_periods=minp or n)
    return getattr(r, how)().to_numpy()


def rolling_ols(y, x, n):
    """alpha, beta, r2, var_y over a trailing window of n (pairs with both values)."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(x)
    yy, xx = np.where(ok, y, np.nan), np.where(ok, x, np.nan)
    my, mx = roll(yy, n), roll(xx, n)
    mxy, mxx, myy = roll(xx * yy, n), roll(xx * xx, n), roll(yy * yy, n)
    cov = mxy - mx * my
    vx = mxx - mx * mx
    vy = myy - my * my
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(vx > EPS, cov / vx, np.nan)
        r2 = np.where((vx > EPS) & (vy > EPS), cov * cov / (vx * vy), np.nan)
    alpha = my - beta * mx
    return alpha, beta, r2, vy


def conditional_beta(y, x, n, cond, min_obs=20):
    """OLS slope over the days in each trailing n-window where cond holds."""
    y, x = np.asarray(y, float), np.asarray(x, float)
    ok = np.isfinite(y) & np.isfinite(x)
    full = roll(ok.astype(float), n, "sum")
    c = (cond & ok).astype(float)
    y0, x0 = np.where(ok, y, 0.0), np.where(ok, x, 0.0)
    S = roll(c, n, "sum")
    Sx, Sy = roll(c * x0, n, "sum"), roll(c * y0, n, "sum")
    Sxx, Sxy = roll(c * x0 * x0, n, "sum"), roll(c * x0 * y0, n, "sum")
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = Sxy / S - (Sx / S) * (Sy / S)
        var = Sxx / S - (Sx / S) ** 2
        b = np.where((S >= min_obs) & (var > EPS) & (full == n), cov / var, np.nan)
    return b


def delay_measure(y, m, n, lags=4):
    """Hou-Moskowitz: 1 - R2(y on m_t) / R2(y on m_t..m_t-lags); and the sum of the lag coefficients."""
    y, m = np.asarray(y, float), np.asarray(m, float)
    T = len(y)
    out_d, out_l = np.full(T, np.nan), np.full(T, np.nan)
    if T < n + lags:
        return out_d, out_l
    cols = [np.r_[np.full(k, np.nan), m[:T - k]] for k in range(lags + 1)]
    X = np.column_stack([np.ones(T)] + cols)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    from numpy.lib.stride_tricks import sliding_window_view as swv
    Xw = swv(X, (n, X.shape[1]))[:, 0]                   # (W, n, k)
    yw = swv(y, n)                                       # (W, n)
    okw = swv(ok, n).all(axis=1)
    W = Xw.shape[0]
    idx = np.arange(n - 1, n - 1 + W)
    Xw, yw = np.where(okw[:, None, None], Xw, 0.0), np.where(okw[:, None], yw, 0.0)
    XtX = np.einsum("wni,wnj->wij", Xw, Xw)
    Xty = np.einsum("wni,wn->wi", Xw, yw)
    good = okw & (np.abs(np.linalg.det(XtX)) > 1e-18)
    coef = np.full((W, X.shape[1]), np.nan)
    if good.any():
        coef[good] = np.linalg.solve(XtX[good], Xty[good][..., None])[..., 0]
    ybar = yw.mean(axis=1)
    sst = (yw * yw).sum(axis=1) - n * ybar ** 2
    fit = np.einsum("wni,wi->wn", Xw, np.nan_to_num(coef))
    sse_u = ((yw - fit) ** 2).sum(axis=1)
    r2_u = 1 - sse_u / np.where(sst > EPS, sst, np.nan)
    Xr = Xw[:, :, :2]
    XtXr = np.einsum("wni,wnj->wij", Xr, Xr)
    Xtyr = np.einsum("wni,wn->wi", Xr, yw)
    goodr = okw & (np.abs(np.linalg.det(XtXr)) > 1e-18)
    cr = np.zeros((W, 2))
    if goodr.any():
        cr[goodr] = np.linalg.solve(XtXr[goodr], Xtyr[goodr][..., None])[..., 0]
    sse_r = ((yw - np.einsum("wni,wi->wn", Xr, cr)) ** 2).sum(axis=1)
    r2_r = 1 - sse_r / np.where(sst > EPS, sst, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(good & goodr & (r2_u > EPS), 1 - r2_r / r2_u, np.nan)
    out_d[idx] = d
    out_l[idx] = np.where(good, coef[:, 2:].sum(axis=1), np.nan)
    return out_d, out_l


def cs_spread_pairs(h, l, c):
    """Corwin-Schultz spread for each pair (t-1, t), overnight-adjusted, floored at 0; NaN at t=0."""
    h, l, c = (np.asarray(v, float) for v in (h, l, c))
    h1, l1, c1 = np.r_[np.nan, h[:-1]], np.r_[np.nan, l[:-1]], np.r_[np.nan, c[:-1]]
    h2, l2 = h.copy(), l.copy()
    up = l2 > c1
    dn = h2 < c1
    sh_up = np.where(up, l2 - c1, 0.0)
    sh_dn = np.where(dn, c1 - h2, 0.0)
    h2, l2 = h2 - sh_up + sh_dn, l2 - sh_up + sh_dn
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.log(h1 / l1) ** 2 + np.log(h2 / l2) ** 2
        gamma = np.log(np.maximum(h1, h2) / np.minimum(l1, l2)) ** 2
        k = 3 - 2 * np.sqrt(2)
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return np.where(np.isfinite(s), np.maximum(s, 0.0), np.nan)


def ar_spread_sq_pairs(h, l, c):
    """Abdi-Ranaldo per-pair 4(c_{t-1} - eta_{t-1})(c_{t-1} - eta_t); NaN at t=0."""
    h, l, c = (np.asarray(v, float) for v in (h, l, c))
    with np.errstate(invalid="ignore", divide="ignore"):
        eta = (np.log(h) + np.log(l)) / 2
        cl = np.log(c)
    return np.r_[np.nan, 4 * (cl[:-1] - eta[:-1]) * (cl[:-1] - eta[1:])]


def pctile_in_window(x, n):
    """Share of the n values ending at t (t included) that are strictly below x_t."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    from numpy.lib.stride_tricks import sliding_window_view as swv
    w = swv(x, n)
    good = np.isfinite(w).all(axis=1)
    out[n - 1:] = np.where(good, (w < w[:, -1:]).mean(axis=1), np.nan)
    return out


def top_k_mean(x, n, k):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    from numpy.lib.stride_tricks import sliding_window_view as swv
    w = swv(x, n)
    good = np.isfinite(w).all(axis=1)
    out[n - 1:] = np.where(good, np.sort(np.where(good[:, None], w, 0.0), axis=1)[:, -k:].mean(axis=1), np.nan)
    return out


def bars_since_extreme(x, n, fn=np.argmax):
    """Bars since the most recent max (or min) of the n values ending at t, divided by n."""
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    from numpy.lib.stride_tricks import sliding_window_view as swv
    w = swv(x, n)[:, ::-1]                       # most recent first
    good = np.isfinite(w).all(axis=1)
    pos = fn(np.where(good[:, None], w, 0.0), axis=1)
    out[n - 1:] = np.where(good, pos / n, np.nan)
    return out


def bars_since_flag(flag, cap=250):
    f = np.asarray(flag, bool)
    idx = np.arange(len(f))
    last = np.maximum.accumulate(np.where(f, idx, -1))
    return np.where(last >= 0, np.minimum(idx - last, cap), cap).astype(float)


def streak_of(flag):
    f = np.asarray(flag, bool).astype(int)
    grp = np.cumsum(f == 0)
    return pd.Series(f).groupby(grp).cumsum().to_numpy().astype(float)


def signed_streak(ret):
    s = np.sign(np.nan_to_num(np.asarray(ret, float)))
    out = np.zeros(len(s))
    for i in range(len(s)):
        if s[i] == 0:
            out[i] = 0
        elif i > 0 and np.sign(out[i - 1]) == s[i]:
            out[i] = out[i - 1] + s[i]
        else:
            out[i] = s[i]
    return out


def season_same_month(ts, c, years=5, min_years=2):
    """Mean log month return of the same calendar month in the previous `years` years (>= min_years)."""
    d = pd.DatetimeIndex(ts)
    s = pd.Series(np.asarray(c, float), index=d)
    me = s.groupby([d.year, d.month]).last()
    mret = np.log(me / me.shift(1))
    lookup = {k: v for k, v in mret.items() if np.isfinite(v)}
    out = np.full(len(d), np.nan)
    for i, (y, m) in enumerate(zip(d.year, d.month)):
        vals = [lookup[(y - k, m)] for k in range(1, years + 1) if (y - k, m) in lookup]
        if len(vals) >= min_years:
            out[i] = float(np.mean(vals))
    return out


def lock_flags(ts, o, h, l, c):
    """Upper/lower-lock closes and lower-locked opens, with NSE's tick size for the date."""
    o, h, l, c = (np.asarray(v, float) for v in (o, h, l, c))
    cp = np.r_[np.nan, c[:-1]]
    tk = SF.tick_size(np.asarray(ts, dtype="datetime64[ns]"), np.nan_to_num(cp, nan=1.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        up = c / cp - 1
        dn = 1 - c / cp
        odn = 1 - o / cp
    lock_up = SF._at_band(up, cp, tk) & (up > 0) & (c >= h * (1 - 1e-9))
    lock_dn = SF._at_band(dn, cp, tk) & (dn > 0) & (c <= l * (1 + 1e-9))
    open_dn = SF._at_band(odn, cp, tk) & (odn > 0) & (o <= l * (1 + 1e-9))
    return lock_up, lock_dn, open_dn


# ----------------------------------------------------------------------
# per-symbol features
# ----------------------------------------------------------------------
def symbol_features(ts, o, h, l, c, v, mk: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """All per-symbol C features on the symbol's full bar history; mk holds the market series
    aligned to these bars (m, m_on, m_id: log returns; NaN where no market that day)."""
    o, h, l, c, v = (np.asarray(x, float) for x in (o, h, l, c, v))
    n = len(c)
    cp = np.r_[np.nan, c[:-1]]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = c / cp - 1
        lr = np.log(c / cp)
        lon = np.log(o / cp)
        lid = np.log(c / o)
        tv = c * v
        tv = np.where(tv > 0, tv, np.nan)
    F: Dict[str, np.ndarray] = {}
    # A
    for k in (5, 20, 60):
        F[f"C_on_sum_{k}"] = roll(lon, k, "sum")
        F[f"C_id_sum_{k}"] = roll(lid, k, "sum")
    for k in (20, 60):
        F[f"C_tug_{k}"] = F[f"C_on_sum_{k}"] - F[f"C_id_sum_{k}"]
    a_on, a_id = roll(np.abs(lon), 20, "sum"), roll(np.abs(lid), 20, "sum")
    with np.errstate(invalid="ignore", divide="ignore"):
        F["C_on_absshare_20"] = a_on / (a_on + a_id)
        F["C_on_id_volratio_60"] = roll(lon, 60, "std") / roll(lid, 60, "std")
    F["C_on_posfrac_20"] = roll(np.where(np.isfinite(lon), (lon > 0).astype(float), np.nan), 20)
    F["C_id_posfrac_20"] = roll(np.where(np.isfinite(lid), (lid > 0).astype(float), np.nan), 20)
    F["C_on_excess_20"] = F["C_on_sum_20"] - roll(mk["m_on"], 20, "sum")
    F["C_id_excess_20"] = F["C_id_sum_20"] - roll(mk["m_id"], 20, "sum")
    # B
    m = mk["m"]
    a60, b60, r2_60, vy60 = rolling_ols(lr, m, 60)
    F["C_beta_60"], F["C_r2_60"] = b60, r2_60
    F["C_idio_vol_60"] = np.sqrt(np.maximum(vy60 * (1 - r2_60), 0.0))
    F["C_beta_down_120"] = conditional_beta(lr, m, 120, m < 0)
    F["C_beta_updown_120"] = conditional_beta(lr, m, 120, m > 0) - F["C_beta_down_120"]
    F["C_beta_on_60"] = rolling_ols(lon, mk["m_on"], 60)[1]
    F["C_beta_id_60"] = rolling_ols(lid, mk["m_id"], 60)[1]
    F["C_delay_120"], F["C_lagbeta_120"] = delay_measure(lr, m, 120)
    ok = np.isfinite(lr) & np.isfinite(m)
    rr, mm = np.where(ok, lr, np.nan), np.where(ok, m, np.nan)
    mr, mmn = roll(rr, 120), roll(mm, 120)
    Erm, Emm, Ermm, Err = roll(rr * mm, 120), roll(mm * mm, 120), roll(rr * mm * mm, 120), roll(rr * rr, 120)
    num = Ermm - 2 * mmn * Erm - mr * Emm + 2 * mr * mmn ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        sd_r = np.sqrt(np.maximum(Err - mr ** 2, 0))
        var_m = np.maximum(Emm - mmn ** 2, 0)
        F["C_coskew_120"] = np.where((sd_r > EPS) & (var_m > EPS), num / (sd_r * var_m), np.nan)
    e = lr - a60 - b60 * m
    F["C_resid_1d"] = e
    F["C_resid_5d"] = roll(e, 5, "sum")
    F["C_resid_20d"] = roll(e, 20, "sum")
    with np.errstate(invalid="ignore", divide="ignore"):
        iv = np.where(F["C_idio_vol_60"] > EPS, F["C_idio_vol_60"], np.nan)
        F["C_resid_5d_z"] = F["C_resid_5d"] / (iv * np.sqrt(5))
        F["C_resid_20d_z"] = F["C_resid_20d"] / (iv * np.sqrt(20))
        e21 = np.r_[np.full(21, np.nan), e[:-21]] if n > 21 else np.full(n, np.nan)
        sm, sd = roll(e21, 230, "sum"), roll(e21, 230, "std")
        F["C_resid_mom_z"] = np.where(sd > EPS, sm / (sd * np.sqrt(230)), np.nan)
    F["C_idio_skew_60"] = _s(e).rolling(60, min_periods=60).skew().to_numpy()
    F["C_max1_resid_21"] = roll(e, 21, "max")
    # C
    F["C_max1_21"] = roll(r, 21, "max")
    F["C_max5_21"] = top_k_mean(r, 21, 5)
    F["C_min1_21"] = roll(r, 21, "min")
    F["C_maxmin_21"] = F["C_max1_21"] + F["C_min1_21"]
    with np.errstate(invalid="ignore", divide="ignore"):
        sd60 = roll(r, 60, "std")
        F["C_max_to_vol"] = np.where(sd60 > EPS, F["C_max1_21"] / sd60, np.nan)
    # D
    cs = cs_spread_pairs(h, l, c)
    ar2 = ar_spread_sq_pairs(h, l, c)
    for k in (20, 60):
        F[f"C_cs_spread_{k}"] = roll(cs, k)
        F[f"C_ar_spread_{k}"] = np.sqrt(np.maximum(roll(ar2, k), 0.0))
    F["C_zero_ret_frac_60"] = roll(np.where(np.isfinite(cp), (c == cp).astype(float), np.nan), 60)
    frozen = ((o == h) & (h == l) & (l == c)).astype(float)
    F["C_frozen_frac_20"] = roll(frozen, 20)
    with np.errstate(invalid="ignore", divide="ignore"):
        med20 = np.r_[np.nan, roll(tv, 20, "median")[:-1]]
        F["C_turn_shock_1"] = np.log(tv / med20)
        med60 = np.r_[np.full(5, np.nan), roll(tv, 60, "median")[:-5]] if n > 5 else np.full(n, np.nan)
        F["C_turn_shock_5"] = np.log(roll(tv, 5) / med60)
        F["C_turn_cv_20"] = roll(tv, 20, "std") / roll(tv, 20)
        ami = roll(np.abs(r) / tv, 20)
        F["C_amihud_shock"] = np.log(ami / roll(ami, 250, "median"))
    ltv = np.log(tv)
    F["C_pv_corr_20"] = _s(r).rolling(20, min_periods=20).corr(_s(ltv)).to_numpy()
    F["C_absret_turn_corr_20"] = _s(np.abs(r)).rolling(20, min_periods=20).corr(_s(ltv)).to_numpy()
    # E
    lu, ld, od = lock_flags(ts, o, h, l, c)
    F["C_lock_up"], F["C_lock_dn"] = lu.astype(float), ld.astype(float)
    F["C_n_lock_up_20"] = roll(lu.astype(float), 20, "sum")
    F["C_n_lock_dn_20"] = roll(ld.astype(float), 20, "sum")
    F["C_n_lock_dn_60"] = roll(ld.astype(float), 60, "sum")
    F["C_days_since_lock_dn"] = bars_since_flag(ld)
    F["C_days_since_lock_up"] = bars_since_flag(lu)
    F["C_streak_lock_dn"] = streak_of(ld)
    F["C_streak_lock_up"] = streak_of(lu)
    F["C_open_lock_dn_20"] = roll(od.astype(float), 20, "sum")
    F["C_band_proxy_250"] = roll(np.abs(r), 250, "max")
    with np.errstate(invalid="ignore", divide="ignore"):
        F["C_ret_to_band"] = np.where(F["C_band_proxy_250"] > EPS, r / F["C_band_proxy_250"], np.nan)
    # F
    tr = np.fmax(h - l, np.fmax(np.abs(h - cp), np.abs(l - cp)))
    atr = roll(tr, 14)
    atr = np.where(atr > EPS, atr, np.nan)
    ema20 = _s(c).ewm(span=20, adjust=False, min_periods=20).mean().to_numpy()
    ema50 = _s(c).ewm(span=50, adjust=False, min_periods=50).mean().to_numpy()
    typ = (h + l + c) / 3
    with np.errstate(invalid="ignore", divide="ignore"):
        F["C_dist_ema20_atr"] = (c - ema20) / atr
        F["C_dist_sma50_atr"] = (c - roll(c, 50)) / atr
        F["C_dist_sma200_atr"] = (c - roll(c, 200)) / atr
        for k in (20, 60):
            vw = roll(typ * v, k, "sum") / roll(v, k, "sum")
            F[f"C_dist_vwap{k}_atr"] = (c - vw) / atr
        F["C_ema20_ema50_atr"] = (ema20 - ema50) / atr
        F["C_close_typical_atr"] = (c - typ) / atr
        rng_ = h - l
        clv = np.where(rng_ > EPS, ((c - l) - (h - c)) / rng_, 0.0)
        clv = np.where(np.isfinite(c) & np.isfinite(h) & np.isfinite(l), clv, np.nan)
        F["C_clv"] = clv
        F["C_clv_mean_5"] = roll(clv, 5)
        F["C_clv_mean_20"] = roll(clv, 20)
        F["C_upper_shadow_atr"] = (h - np.fmax(o, c)) / atr
        F["C_lower_shadow_atr"] = (np.fmin(o, c) - l) / atr
        F["C_range_pctile_250"] = pctile_in_window(rng_ / c, 250)
        F["C_turn_pctile_250"] = pctile_in_window(tv, 250)
        F["C_absret_pctile_250"] = pctile_in_window(np.abs(r), 250)
        F["C_ret_z_60"] = np.where(sd60 > EPS, r / sd60, np.nan)
        sdon, sdid = roll(lon, 60, "std"), roll(lid, 60, "std")
        F["C_on_z_60"] = np.where(sdon > EPS, lon / sdon, np.nan)
        F["C_id_z_60"] = np.where(sdid > EPS, lid / sdid, np.nan)
        absd = np.abs(c - cp)
        for k in (10, 20):
            ck = np.r_[np.full(k, np.nan), c[:-k]] if n > k else np.full(n, np.nan)
            den = roll(absd, k, "sum")
            F[f"C_eff_{k}"] = np.where(den > EPS, np.abs(c - ck) / den, np.nan)
    F["C_streak"] = signed_streak(np.where(np.isfinite(cp), c - cp, 0.0))
    F["C_days_since_hi250"] = bars_since_extreme(c, 250, np.argmax)
    F["C_days_since_lo250"] = bars_since_extreme(c, 250, np.argmin)
    # G
    def lagc(k):
        return np.r_[np.full(k, np.nan), c[:-k]] if n > k else np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        F["C_mom_12_1"] = np.log(lagc(21) / lagc(252))
        F["C_mom_6_1"] = np.log(lagc(21) / lagc(126))
        pret = np.log(lagc(21) / lagc(251))
        pos = np.where(np.isfinite(r), (r > 0).astype(float), np.nan)
        neg = np.where(np.isfinite(r), (r < 0).astype(float), np.nan)
        pos21 = np.r_[np.full(21, np.nan), pos[:-21]] if n > 21 else np.full(n, np.nan)
        neg21 = np.r_[np.full(21, np.nan), neg[:-21]] if n > 21 else np.full(n, np.nan)
        F["C_fip_id"] = np.sign(pret) * (roll(neg21, 230) - roll(pos21, 230))
        F["C_ret_2_5"] = np.log(lagc(1) / lagc(5))
    F["C_season_month"] = season_same_month(ts, c)
    F["_r"], F["_lon"], F["_lid"], F["_m"] = r, lon, lid, m
    return F


# ----------------------------------------------------------------------
# the build
# ----------------------------------------------------------------------
def load_bars(root: Path, sym: str) -> Optional[pd.DataFrame]:
    from data_quality import _paths
    fp, _ = _paths(root, sym)
    if not Path(fp).exists():
        return None
    b = pd.read_parquet(fp, columns=["timestamp", "open", "high", "low", "close", "volume"])
    b["timestamp"] = SB._naive(b["timestamp"])
    return b.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def market_series(bars: Dict[str, pd.DataFrame], members: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Equal-weight universe series by date, from each member's own bars on that date."""
    parts = []
    for s, b in bars.items():
        mem = members.get(s)
        if mem is None or b is None or len(b) < 2:
            continue
        ts = b["timestamp"].to_numpy(dtype="datetime64[ns]")
        o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        cp = np.r_[np.nan, c[:-1]]
        lu, ld, _ = lock_flags(ts, o, h, l, c)
        keep = np.isin(ts, mem)
        with np.errstate(invalid="ignore", divide="ignore"):
            parts.append(pd.DataFrame({"timestamp": ts[keep], "r": (c / cp - 1)[keep], "on": (o / cp - 1)[keep],
                                       "id": (c / o - 1)[keep], "lu": lu[keep].astype(float),
                                       "ld": ld[keep].astype(float)}))
    P = pd.concat(parts, ignore_index=True)
    P = P[np.isfinite(P["r"])]
    g = P.groupby("timestamp")
    M = pd.DataFrame({"M_disp_1d": g["r"].std(ddof=1), "M_breadth_up": g["r"].apply(lambda x: float((x > 0).mean())),
                      "M_share_lock_dn": g["ld"].mean(), "M_share_lock_up": g["lu"].mean(),
                      "M_on_1d": g["on"].mean(), "M_id_1d": g["id"].mean(), "_mr": g["r"].mean()}).sort_index()
    M["M_on_sum_20"] = M["M_on_1d"].rolling(20, min_periods=20).sum()
    M["M_id_sum_20"] = M["M_id_1d"].rolling(20, min_periods=20).sum()
    mu, sd = M["M_disp_1d"].rolling(60, min_periods=60).mean(), M["M_disp_1d"].rolling(60, min_periods=60).std()
    M["M_disp_z_60"] = (M["M_disp_1d"] - mu) / sd
    idx = pd.DatetimeIndex(M.index)
    M["M_cal_days_to_month_end"] = ((idx + pd.offsets.MonthEnd(0)) - idx).days.astype(float)
    M["m"] = np.log1p(M["_mr"])
    M["m_on"] = np.log1p(M["M_on_1d"])
    M["m_id"] = np.log1p(M["M_id_1d"])
    return M


def build(pp: Path, root: Path, names: List[str], verbose: bool = True) -> pd.DataFrame:
    t0 = time.perf_counter()
    import pyarrow.parquet as pq
    schema = pq.ParquetFile(pp).schema_arrow.names
    miss = [c for c in PANEL_INPUTS if c not in schema]
    if miss:
        raise PreconditionError(f"panel lacks the interaction inputs {miss}")
    cols = ["timestamp", "symbol"] + PANEL_INPUTS
    assert not any(c.startswith("label_") for c in cols)          # the label firewall, by construction
    P = pd.read_parquet(pp, columns=cols)
    P["timestamp"] = SB._naive(P["timestamp"])
    P["symbol"] = P["symbol"].astype(str)
    N = len(P)
    rows_of = {s: np.asarray(ix) for s, ix in P.groupby("symbol").indices.items()}
    pts = P["timestamp"].to_numpy(dtype="datetime64[ns]")
    members = {s: pts[ix] for s, ix in rows_of.items()}
    bars = {}
    for q, s in enumerate(sorted(members), 1):
        bars[s] = load_bars(root, s)
        if verbose and q % 300 == 0:
            _log(f"  loaded {q}/{len(members)} symbols")
    M = market_series(bars, members)
    if verbose:
        _log(f"market series: {len(M):,} sessions from the universe ({time.perf_counter() - t0:.0f}s)")
    percol: Optional[List[str]] = None
    R = None
    for q, s in enumerate(sorted(rows_of), 1):
        b = bars.get(s)
        if b is None or b.empty:
            continue
        ts = b["timestamp"].to_numpy(dtype="datetime64[ns]")
        mk = {k: M[k].reindex(pd.DatetimeIndex(ts)).to_numpy(float) for k in ("m", "m_on", "m_id")}
        F = symbol_features(ts, *(b[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume")), mk)
        if percol is None:
            percol = list(F.keys())
            R = np.full((N, len(percol)), np.nan, dtype=np.float32)
        ix = rows_of[s]
        want = pts[ix]
        pos = np.minimum(np.searchsorted(ts, want), len(ts) - 1)
        hit = ts[pos] == want
        for j, k in enumerate(percol):
            R[ix[hit], j] = F[k][pos[hit]]
        if verbose and q % 300 == 0:
            _log(f"  features {q}/{len(rows_of)} symbols ({time.perf_counter() - t0:.0f}s)")
    if R is None:
        raise PreconditionError("no symbol had cached bars")
    col = {k: R[:, j].astype(float) for j, k in enumerate(percol)}
    di = pd.DatetimeIndex(M.index)
    pos = np.searchsorted(di.values, pts)
    okd = (pos < len(di)) & (di.values[np.minimum(pos, len(di) - 1)] == pts)
    for mcol in [c for c in M.columns if c.startswith("M_")]:
        v = M[mcol].to_numpy(float)
        col[mcol] = np.where(okd, v[np.minimum(pos, len(di) - 1)], np.nan)
    for c_ in PANEL_INPUTS:
        col[c_] = pd.to_numeric(P[c_], errors="coerce").to_numpy(float)

    def pr(name):
        return pd.Series(col[name]).groupby(pts).rank(pct=True).to_numpy()
    col["I_cgw_1"] = col["_r"] * col["C_turn_shock_1"]
    col["I_cgw_5"] = col["C_resid_5d"] * col["C_turn_shock_5"]
    col["I_gap_turn"] = col["_lon"] * col["C_turn_shock_1"]
    col["I_id_turn"] = col["_lid"] * col["C_turn_shock_1"]
    pr_cs = pr("C_cs_spread_20")
    col["I_rev_spread"] = col["C_resid_5d_z"] * pr_cs
    col["I_max_spread"] = col["C_max1_21"] * pr_cs
    col["I_beta_mkt"] = col["C_beta_60"] * col["_m"]
    col["I_beta_vix"] = col["C_beta_60"] * col["INDIAVIX_ret_1d"]
    pr_iv = pr("C_idio_vol_60")
    col["I_lowvol_dd"] = (1 - pr_iv) * col["MKT_D_drawdown_252"]
    col["I_hi52_turn"] = col["D_dist_from_52wh"] * col["C_turn_shock_5"]
    col["I_tug_turn"] = col["C_tug_20"] * pr("X_turnover_med")
    col["I_lockdn_turn"] = col["C_lock_dn"] * col["C_turn_shock_1"]
    col["I_mdi_vol"] = col["D_mdi14_diff1"] * pr_iv
    col["I_wq44_turn"] = col["D_WQ_44"] * col["C_turn_shock_5"]
    col["I_rev_disp"] = col["_r"] * col["M_disp_1d"]
    col["I_on_turn20"] = col["C_on_sum_20"] * col["C_turn_shock_5"]
    for b_ in ("C_idio_vol_60", "C_beta_60", "C_cs_spread_20", "C_max1_21", "C_resid_5d_z", "C_tug_20",
               "C_turn_shock_1", "C_on_sum_20"):
        col["PR_" + b_] = pr_iv if b_ == "C_idio_vol_60" else (pr_cs if b_ == "C_cs_spread_20" else pr(b_))
    del R
    produced = [c for c in col if c.startswith(("C_", "I_", "M_", "PR_"))]
    if set(produced) != set(names):
        raise PreconditionError(f"produced features differ from the declaration: missing "
                                f"{sorted(set(names) - set(produced))}, extra {sorted(set(produced) - set(names))}")
    res = {"timestamp": P["timestamp"].to_numpy(), "symbol": P["symbol"].to_numpy()}
    for c_ in names:
        v_ = col[c_]
        res[c_] = np.where(np.isfinite(v_), v_, np.nan).astype("float32")
    out = pd.DataFrame(res)
    if verbose:
        _log(f"{len(out):,} rows x {len(names)} C features built ({time.perf_counter() - t0:.0f}s)")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the declared stage-C features")
    ap.add_argument("--panel", required=True, help="panel folder, e.g. %%CACHE_DAILY_ROOT%%\\panel_oc")
    ap.add_argument("--root", required=True, help="raw cache root, e.g. %%CACHE_DAILY_ROOT%%")
    a = ap.parse_args(argv)
    pp = Path(a.panel) / "panel.parquet"
    man, man_sha = SB.load_manifest()
    SB.check_panel(pp, man_sha)
    decl, sha, names = load_declaration()
    RC.ledger_append(pp, {"kind": "stage_c_declared", "tool": CODE_VERSION, "declaration_sha": sha,
                          "experiment_family": decl["family_id"], "features": len(names)})
    print("=" * 76)
    print(f"{CODE_VERSION} | C declaration {sha} | {len(names)} features | panel {pp}")
    print("=" * 76)
    X = build(pp, Path(a.root), names)
    out = pp.parent / "stage_c"
    out.mkdir(exist_ok=True)
    X.to_parquet(out / "features_c.parquet", index=False)
    cover = {c_: float(np.isfinite(X[c_].to_numpy()).mean()) for c_ in names}
    meta = {"tool": CODE_VERSION, "declaration_sha": sha, "rows": int(len(X)), "features": names,
            "coverage": cover, "built_at": dt.datetime.now().isoformat(), "panel": str(pp)}
    (out / "features_c_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    n = RC.ledger_append(pp, {"kind": "stage_c_features", "tool": CODE_VERSION, "declaration_sha": sha,
                              "experiment_family": decl["family_id"], "rows": int(len(X)), "features": len(names)})
    low = sorted(cover.items(), key=lambda kv: kv[1])[:8]
    print(f"\n  lowest coverage: " + ", ".join(f"{k} {v:.0%}" for k, v in low))
    print(f"  output: {out / 'features_c.parquet'}\n  research ledger: entry {n} (stage_c_features)")
    print("=" * 76)
    print("C FEATURES BUILT. Next: stage_c_screen.py (B's rule + leak screens on the C features).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
