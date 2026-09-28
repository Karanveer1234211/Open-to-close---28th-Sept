#!/usr/bin/env python3
"""
research_common.py - discipline shared by every research tool.

Four things live here so that no two tools can define them differently:

    TIMING CONTRACT   what is known when, checked against the data itself
    REGIME POSTERIOR  Gaussian-mixture assignment that never imputes
    LOCKBOX           the final period no research step may look at
    RESEARCH LEDGER   an append-only count of every experiment and every
                      lockbox look

THE PROBLEM THIS MODULE EXISTS FOR
==================================
The pipeline is now good at preventing classic lookahead - using tomorrow's
price. The larger risk has moved: RESEARCH-SELECTION LEAKAGE. Run a thousand
reasonable experiments, keep what survives, call it an edge. Nothing in any
single experiment is wrong; the selection across them is. The lockbox and
the ledger are the defence, and they only work if every tool uses the same
ones.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------
# TIMING CONTRACT
# ----------------------------------------------------------------------
TIMING_CONTRACT = (
    "Prediction is made at the CLOSE of session T. Every feature uses data "
    "through the close of T and nothing later. The target covers sessions "
    "T+1..T+H, entered at the close of T. Same-day market state (breadth, "
    "median move, dispersion) is therefore legitimate: it is known at T close.")


def timing_contract(entry: str = "close") -> str:
    """The timing statement for a report, matching its entry world."""
    if entry == "open":
        return ("Prediction is made at the CLOSE of session T. Every feature uses data through the "
                "close of T and nothing later. The trade is entered at the OPEN of session T+1 "
                "(the first price a watchlist produced after the close can get) and covers sessions "
                "T+1..T+H with gap-aware stop/target fills. The overnight move from the close of T to "
                "the open of T+1 is NOT part of the target.")
    return TIMING_CONTRACT


def entry_of_target(target: str) -> str:
    return "open" if str(target).endswith("_o1") else "close"


class TimingContractError(RuntimeError):
    pass


def assert_timing_contract(close: np.ndarray, sym_code: np.ndarray,
                           fwd_ret: np.ndarray, *, max_same: float = 0.10,
                           min_next: float = 0.15, sample: int = 300_000,
                           seed: int = 0) -> Dict[str, float]:
    """
    Check the label timing against the DATA, not against anyone's reading
    of the code.

    If the forward return spans T+1..T+H, it contains day T+1's return and
    not day T's. So:
        corr(fwd_ret[T], ret1[T])    should be ~0     (not in the window)
        corr(fwd_ret[T], ret1[T+1])  should be ~0.45  (1 of 5 days)
    A label that leaked day T would push the first figure to ~0.45 too. The
    thresholds leave room for genuine short-term reversal, which can make the
    first figure mildly negative.

    Rows must be in (timestamp, symbol) order, which is also time order
    within each symbol.
    """
    s = pd.Series(np.asarray(close, dtype="float64"))
    g = s.groupby(np.asarray(sym_code))
    r1 = (s / g.shift(1) - 1).to_numpy()
    r1_next = pd.Series(r1).groupby(np.asarray(sym_code)).shift(-1).to_numpy()
    fr = np.asarray(fwd_ret, dtype="float64")
    ok = np.isfinite(fr) & np.isfinite(r1) & np.isfinite(r1_next)
    idx = np.where(ok)[0]
    if len(idx) < 1000:
        raise TimingContractError(f"only {len(idx)} rows to check timing on")
    rng = np.random.default_rng(seed)
    if len(idx) > sample:
        idx = rng.choice(idx, sample, replace=False)

    def sp(a, b):     # Spearman: returns are fat-tailed
        return float(pd.Series(a).rank().corr(pd.Series(b).rank()))

    c_same = sp(fr[idx], r1[idx])
    c_next = sp(fr[idx], r1_next[idx])
    out = {"corr_fwd_vs_same_day": round(c_same, 4),
           "corr_fwd_vs_next_day": round(c_next, 4), "rows_checked": int(len(idx))}
    if abs(c_same) > max_same or c_next < min_next:
        raise TimingContractError(
            f"label timing does not match the contract.\n"
            f"  corr(forward return, SAME-day return) = {c_same:+.3f} "
            f"(must be within +/-{max_same})\n"
            f"  corr(forward return, NEXT-day return) = {c_next:+.3f} "
            f"(must exceed {min_next})\n"
            f"  Contract: {TIMING_CONTRACT}")
    return out


def forward_return_for_check(close: np.ndarray, sym_code: np.ndarray,
                             horizon: int) -> np.ndarray:
    """Fallback when the panel carries no forward-return label."""
    s = pd.Series(np.asarray(close, dtype="float64"))
    return (s.groupby(np.asarray(sym_code)).shift(-horizon) / s - 1).to_numpy()


# ----------------------------------------------------------------------
# REGIME POSTERIOR THAT NEVER IMPUTES
# ----------------------------------------------------------------------
def gmm_proba(gm, X: np.ndarray) -> np.ndarray:
    """
    Component probabilities using ONLY the observed dimensions of each row.

    Replacing a missing input with 0 - the centre of a centred rank - claims
    the stock is "average" on that axis. It is not; it is unknown. The
    previous code did exactly that and then assigned such rows to regimes
    with unearned confidence.

    The marginal of a Gaussian mixture over a subset of dimensions is itself
    a Gaussian mixture with the same weights and the corresponding sub-means
    and sub-covariances. So a row missing ADX is scored on its other axes,
    exactly, with no imputation. A row missing everything gets the prior
    weights: an honest "don't know", with correspondingly low confidence.
    """
    from scipy.stats import multivariate_normal
    X = np.asarray(X, dtype="float64")
    miss = np.isnan(X)
    K = gm.n_components
    out = np.empty((len(X), K))
    full = ~miss.any(axis=1)
    if full.any():
        out[full] = gm.predict_proba(X[full])
    part = np.where(~full)[0]
    if len(part):
        pats, inv = np.unique(miss[part], axis=0, return_inverse=True)
        inv = np.asarray(inv).reshape(-1)
        logw = np.log(np.clip(gm.weights_, 1e-300, None))
        for pi, pat in enumerate(pats):
            rows = part[inv == pi]
            obs = ~pat
            if not obs.any():
                out[rows] = gm.weights_
                continue
            Xo = X[np.ix_(rows, np.where(obs)[0])]
            ll = np.empty((len(rows), K))
            for k in range(K):
                mu = gm.means_[k][obs]
                cov = gm.covariances_[k][np.ix_(obs, obs)]
                ll[:, k] = logw[k] + multivariate_normal.logpdf(
                    Xo, mean=mu, cov=cov, allow_singular=True)
            ll -= ll.max(axis=1, keepdims=True)
            p = np.exp(ll)
            out[rows] = p / p.sum(axis=1, keepdims=True)
    return out


def missing_report(X: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    miss = np.isnan(X)
    anym = miss.any(axis=1)
    allm = miss.all(axis=1)
    conf = proba.max(axis=1)
    return {
        "rows": int(len(X)),
        "pct_rows_any_missing": float(anym.mean()),
        "pct_rows_all_missing": float(allm.mean()),
        "mean_conf_complete": float(conf[~anym].mean()) if (~anym).any() else float("nan"),
        "mean_conf_partial": float(conf[anym & ~allm].mean()) if (anym & ~allm).any() else float("nan"),
        "mean_conf_all_missing": float(conf[allm].mean()) if allm.any() else float("nan"),
    }


def align_components(ref_means: np.ndarray, means: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Map each new component to the reference component it most resembles.

    Returns perm with perm[j] = reference label of new component j, and the
    mean matched centroid distance - the DRIFT. Large drift means a regime ID
    no longer describes the same state, and per-regime conclusions do not
    transfer across time.
    """
    from scipy.optimize import linear_sum_assignment
    cost = ((ref_means[:, None, :] - means[None, :, :]) ** 2).sum(-1)
    r, c = linear_sum_assignment(cost)
    perm = np.empty(len(c), dtype=int)
    perm[c] = r
    return perm, float(np.sqrt(cost[r, c]).mean())


def permute_proba(proba: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Reorder probability columns so column k is reference regime k."""
    out = np.empty_like(proba)
    out[:, perm] = proba
    return out


# ----------------------------------------------------------------------
# LOCKBOX
# ----------------------------------------------------------------------
DEFAULT_LOCKBOX_FRACTION = 0.15


def lockbox_bounds(sessions: np.ndarray, frac: float, horizon: int
                   ) -> Tuple[Optional[np.datetime64], Optional[np.datetime64]]:
    """
    (research_end, lockbox_start). Research may use sessions <= research_end.

    The gap between them is the label horizon: a research row within H
    sessions of the lockbox has a target window reaching INTO the lockbox, so
    it is excluded too. Every research tool must use this one function, or
    one tool's lockbox is another's research data.
    """
    sessions = np.sort(np.asarray(sessions))
    if not frac or frac <= 0:
        return sessions[-1], None
    lb_i = int(len(sessions) * (1 - frac))
    lb_i = max(horizon + 2, min(lb_i, len(sessions) - 1))
    return sessions[lb_i - horizon - 1], sessions[lb_i]


# ----------------------------------------------------------------------
# RESEARCH LEDGER
# ----------------------------------------------------------------------
def ledger_path(panel_path: Path) -> Path:
    return Path(panel_path).parent / "research_ledger.jsonl"


def ledger_append(panel_path: Path, record: dict) -> int:
    """Append one experiment; return how many are now on record."""
    p = ledger_path(panel_path)
    rec = {"at": dt.datetime.now().isoformat(), **record}
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return ledger_count(panel_path)


def ledger_entries(panel_path: Path) -> list:
    p = ledger_path(panel_path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def ledger_count(panel_path: Path, kind: Optional[str] = None) -> int:
    e = ledger_entries(panel_path)
    return len([x for x in e if kind is None or x.get("kind") == kind])


# ----------------------------------------------------------------------
# EVIDENCE: date-block bootstrap on per-DAY outcomes
# ----------------------------------------------------------------------
DEFAULT_COST_BPS = 35.0
BOOT_BLOCK = 10          # sessions; >= 2x the 5-session label horizon


def daily_topn_values(pred: pd.DataFrame, n: int, value_col: str,
                      score_col: str = "p_cal") -> pd.Series:
    """One number per trading day: the mean outcome of that day's top-n."""
    d = pred.dropna(subset=[score_col, value_col]).copy()
    d["_rk"] = d.groupby("timestamp")[score_col].rank(ascending=False, method="first")
    return d[d["_rk"] <= n].groupby("timestamp")[value_col].mean().sort_index()


def block_bootstrap_mean(x, *, B: int = 2000, block: int = BOOT_BLOCK,
                         seed: int = 0) -> Dict[str, float]:
    """
    95% interval for the mean of a DATE-ORDERED series, by circular
    moving-block bootstrap.

    The binomial standard error sqrt(p(1-p)/n) treats every trade as an
    independent coin. They are not: 5-session labels overlap, so today's and
    tomorrow's outcomes share four days of price path, and the top names on
    one day tend to move together. Resampling BLOCKS of consecutive days
    keeps that dependence inside each resample, so the interval widens to
    reflect what was actually learned. The day is the unit, because the
    decision is made once per day.
    """
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    n = len(x)
    out = {"mean": float(x.mean()) if n else float("nan"), "lo": float("nan"),
           "hi": float("nan"), "n_dates": int(n), "block": int(block)}
    if n < 2 * block:
        return out
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(B, nb))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(B, -1)[:, :n]
    means = x[idx].mean(axis=1)
    out["lo"], out["hi"] = float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
    return out


def topn_evidence(pred: pd.DataFrame, n: int, *, cost_bps: float = DEFAULT_COST_BPS,
                  B: int = 2000, seed: int = 0,
                  score_col: str = "p_cal") -> Dict[str, object]:
    """
    Hit rate AND net return of a daily top-n, each with a block-bootstrap CI.

    There is no breakeven constant. With ATR-scaled barriers every trade's
    payoff depends on its own stock's volatility, so a single breakeven
    probability cannot be right: cost is a bigger share of a quiet stock's
    bracket. Net return per trade - realised bracket return minus cost -
    is the economic question asked directly. 'clears' means the LOWER end of
    the net-return interval is above zero.
    """
    hit = block_bootstrap_mean(daily_topn_values(pred, n, "y", score_col).to_numpy(),
                               B=B, seed=seed)
    res = {"top_n": n, "hit": hit}
    if "exit_ret" in pred.columns and pred["exit_ret"].notna().any():
        d = pred.assign(net=pred["exit_ret"] - cost_bps / 1e4)
        net = block_bootstrap_mean(daily_topn_values(d, n, "net", score_col).to_numpy(),
                                   B=B, seed=seed)
        res["net"] = net
        res["clears"] = bool(np.isfinite(net["lo"]) and net["lo"] > 0)
    else:
        res["net"] = None
        res["clears"] = None
    return res


def ranking_tie_report(pred: pd.DataFrame, n: int = 3, cal_col: str = "p_cal",
                       raw_col: str = "p_raw") -> Dict[str, float]:
    """
    How often did ranking by the CALIBRATED score leave the daily top-n to an
    arbitrary tie-break?

    Isotonic calibration maps ranges of raw scores onto flat steps; with a
    weak signal a single step can hold 10% of all rows. Ranking by it, ties
    at the top were broken by row order - alphabetical by symbol. Measures:
    share of days where more than n names shared the n-th best calibrated
    value, and the average overlap between the calibrated top-n and the
    raw-score top-n.
    """
    d = pred.dropna(subset=[cal_col, raw_col])
    tied, overlap = [], []
    for _, g in d.groupby("timestamp"):
        if len(g) <= n:
            continue
        c = g[cal_col].to_numpy()
        kth = np.sort(c)[-n]
        tied.append(float((c >= kth).sum() > n))
        a = set(g.nlargest(n, raw_col).index)
        g2 = g.assign(_rk=g[cal_col].rank(ascending=False, method="first"))
        b = set(g2.index[g2["_rk"] <= n])
        overlap.append(len(a & b) / n)
    return {"days": len(tied), "share_days_tie_decided": float(np.mean(tied)) if tied else float("nan"),
            "mean_overlap_with_raw_topn": float(np.mean(overlap)) if overlap else float("nan"),
            "distinct_calibrated_values": int(d[cal_col].nunique())}


# ----------------------------------------------------------------------
# PANEL-LEVEL LEAKAGE SCREEN
# ----------------------------------------------------------------------
def screen_feature_leakage(values: np.ndarray, close: np.ndarray, sym_code: np.ndarray,
                           *, horizons=(1, 2, 3, 4, 5), threshold: float = 0.15,
                           sample: int = 200_000, seed: int = 0,
                           _cache: Optional[dict] = None) -> Dict[str, float]:
    """
    Does a feature at T correlate with a FUTURE single-day return?

    features_daily's canary proves the per-symbol feature CODE never reads a
    future bar - on synthetic data. It cannot see an exogenous series joined
    one day off, a cross-sectional step that mixed dates, or features a
    research tool derives itself. This checks the finished panel instead.

    A legitimate daily feature predicts tomorrow's return at |rho| of a few
    hundredths. One that contains tomorrow's close sits near 0.3 or above,
    because it IS partly tomorrow's return. Rows must be in (timestamp,
    symbol) order.
    """
    c = _cache if _cache is not None else {}
    key = id(close)
    if key not in c:
        s = pd.Series(np.asarray(close, dtype="float64"))
        r1 = (s / s.groupby(sym_code).shift(1) - 1).to_numpy()
        fut = {h: pd.Series(r1).groupby(sym_code).shift(-h).to_numpy() for h in horizons}
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(r1), min(sample, len(r1)), replace=False)
        c[key] = (idx, {h: pd.Series(fut[h][idx]).rank() for h in horizons})
    idx, fr = c[key]
    fv = pd.Series(np.asarray(values, dtype="float64")[idx]).rank()
    out = {f"rho_t+{h}": float(fv.corr(fr[h])) for h in horizons}
    worst = max((abs(v) for v in out.values() if np.isfinite(v)), default=0.0)
    out["max_abs_future_rho"] = worst
    out["suspect"] = bool(worst > threshold)
    return out


# ----------------------------------------------------------------------
# LABEL FIREWALL - enforced where features are CONSUMED
# ----------------------------------------------------------------------
# Columns that look forward but are not named label_*. features_daily emits
# ret_5d_close_pct = close[t+5]/close[t]-1 and its leak canary exempts it on
# purpose. It was STORED in panel.parquet; panel_feature_columns excluded it,
# but the feature atlas built its list from the parquet schema and would have
# used the forward return as a feature. One upstream filter is not a firewall.
FORWARD_COLUMNS = frozenset({"ret_5d_close_pct"})


class LabelLeakError(RuntimeError):
    pass


# ======================================================================
# UNIVERSE RULE - ETFs and other funds are not stocks
# ======================================================================
# Decided 21 Sep 2026 ("exclude ETFs"), never implemented: the tradability
# audit (28 Sep) found 206 ETF picks in the engine's walk-forward top-3
# (MAFANG alone 45). The panel now drops them at the universe stage. Patterns
# catch the systematic names; the explicit set holds fund symbols with no
# marker; a user file (<cache root>/universe_exclude.txt, one symbol per line
# or a CSV with a SYMBOL column) adds anything the rule misses.
ETF_RULE_VERSION = "etf-v1"
ETF_SUFFIXES = ("BEES", "ETF")
ETF_CONTAINS = ("ETF",)
ETF_PREFIXES = ("SETF",)
ETF_EXPLICIT = frozenset({
    "MON100", "N100", "MAFANG", "MASPTOP50", "MAHKTECH", "MONQ50", "MOM100", "MOM50",
    "ICICIB22", "NV20", "GOLDSHARE",
})
UNIVERSE_EXCLUDE_FILE = "universe_exclude.txt"


def is_etf_symbol(sym: str, extra=frozenset()) -> bool:
    s = str(sym).upper()
    return (s in ETF_EXPLICIT or s in extra or s.endswith(ETF_SUFFIXES) or
            s.startswith(ETF_PREFIXES) or any(k in s for k in ETF_CONTAINS))


def load_symbol_list(path) -> set:
    """Symbols from a text file (one per line, '#' comments) or a CSV with a SYMBOL column."""
    p = Path(path)
    if not p.exists():
        return set()
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        col = next((c for c in df.columns if str(c).strip().upper() == "SYMBOL"), None)
        if col is None:
            raise ValueError(f"{p} has no SYMBOL column")
        return {str(x).strip().upper() for x in df[col].dropna() if str(x).strip()}
    txt = p.read_text(encoding="utf-8-sig", errors="ignore")
    return {ln.strip().upper() for ln in txt.splitlines()
            if ln.strip() and not ln.strip().startswith("#")}


# ======================================================================
# PRICE-BAND LOCKS - could the stock actually be bought?
# ======================================================================
# NSE caps most stocks' daily move at a 2/5/10/20% band. A stock that opens AT
# its upper band with open = high had no sellers in the opening auction: a
# next-open buyer is not filled. Bands are not in the cache, so this is a
# proxy from daily bars; tolerance = max(0.15 pp, 1.02 ticks of 0.05 at the
# reference price) allows for the exchange rounding band prices to a tick.
# Validated on real bars (ELIN, TTML, TANLA, LLOYDSENGG, TVSELECT lock;
# RPOWER, EXICOM, PCJEWELLER, PNBGILTS do not) in tradability_audit v1.
PRICE_BANDS = (0.02, 0.05, 0.10, 0.20)
BAND_TOL_PP = 0.0015
BAND_TICK = 0.05
BAND_TICK_SLACK = 1.02


def at_price_band(move, ref) -> np.ndarray:
    """True where `move` (a fraction, signed toward the band) sits on a 2/5/10/20% band of `ref`."""
    move, ref = np.asarray(move, dtype="float64"), np.asarray(ref, dtype="float64")
    with np.errstate(invalid="ignore", divide="ignore"):
        tol = np.maximum(BAND_TOL_PP, BAND_TICK_SLACK * BAND_TICK / ref)
        hit = np.zeros(move.shape, dtype=bool)
        for b in PRICE_BANDS:
            hit |= np.abs(move - b) <= tol
    return hit & np.isfinite(move) & np.isfinite(ref) & (ref > 0)


def opens_at_upper_band(open_next, high_next, close_t) -> np.ndarray:
    """Next session opens AT its high AND at an upper band vs today's close: unbuyable at that open."""
    o, h, c = (np.asarray(x, dtype="float64") for x in (open_next, high_next, close_t))
    with np.errstate(invalid="ignore", divide="ignore"):
        same = np.abs(o - h) <= 1e-6 * np.maximum(np.abs(o), np.abs(h))
        return same & at_price_band(o / c - 1, c)


def is_forbidden_feature(name: str) -> bool:
    return name.startswith("label_") or name in FORWARD_COLUMNS


def assert_no_label_leak(columns, where: str) -> None:
    """Raise if any label or known forward-looking column is about to be used as a feature."""
    bad = sorted({c for c in columns if is_forbidden_feature(str(c))})
    if bad:
        raise LabelLeakError(
            f"{where}: label / forward-looking columns in the FEATURE set: {bad}. "
            f"These encode the outcome being predicted. Refusing to continue.")


# ======================================================================
# CANONICAL STATE ENGINE - the only definition of stock and market state
# ======================================================================
# feature_atlas and regime_research each had their own: 7 vs 8 stock
# dimensions (the atlas had no shock), market return close-to-close vs
# intraday, trend as a 20-day sum vs a mean. Regimes built on different
# definitions cannot be compared. Both now call these functions and record
# STATE_ENGINE_VERSION; nothing else may define state.
STATE_ENGINE_VERSION = ("state-v1: 8 stock dims as per-date centred ranks "
                        "(shock = |intraday return|); market from close-to-close "
                        "returns within unbroken session runs")

STOCK_STATE_SPEC = (
    ("trend",          "D_ema20_angle_deg",     "rank"),
    ("trend_strength", "D_adx14",               "rank"),
    ("vol_level",      "D_realvol_20",          "rank"),
    ("vol_change",     "D_realvol_ratio_20_60", "rank"),
    ("momentum",       "D_rsi14",               "rank"),
    ("participation",  "D_dvol_z20",            "rank"),
    ("location",       "D_pos_in_52w_range",    "rank"),
    ("shock",          "D_intraday_ret_pct",    "absrank"),
)
STOCK_STATE_NAMES = tuple(f"st_{n}" for n, _, _ in STOCK_STATE_SPEC)
MIN_STATE_DIMS = 6

MARKET_FEATURES = ("mk_ret1", "mk_breadth", "mk_disp",
                   "mk_trend20", "mk_vol20", "mk_breadth20", "mk_disp20")
MARKET_REGIME_INPUTS = ("mk_trend20", "mk_vol20", "mk_breadth20", "mk_disp20")

GMM_PARAMS = {"covariance_type": "full", "reg_covar": 1e-4}


class StateInputError(RuntimeError):
    pass


def session_segments(date_code: np.ndarray, sym_code: np.ndarray) -> np.ndarray:
    """
    Id of each row's UNBROKEN RUN of consecutive sessions within its symbol.

    The liquidity floor drops a stock when turnover falls below it and
    re-admits it later, so one symbol's panel rows can jump from March to
    July. A shift or rolling window over panel ROWS then treats four months as
    one day. Group per-stock time operations by this id instead of by symbol
    and no window ever crosses a gap. Rows must be in (timestamp, symbol)
    order; date_code must count sessions (0, 1, 2, ...).
    """
    d = pd.Series(np.asarray(date_code, dtype=np.int64))
    prev = d.groupby(sym_code).shift(1)
    brk = (prev.isna() | ((d - prev) != 1)).astype(np.int64)
    run = brk.groupby(sym_code).cumsum().to_numpy()
    key = np.asarray(sym_code, dtype=np.int64) * 10_000_000 + run
    return pd.factorize(key)[0]


def daily_returns(close: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Close-to-close return; NaN on the first session of every unbroken run."""
    s = pd.Series(np.asarray(close, dtype="float64"))
    return (s / s.groupby(seg).shift(1) - 1).to_numpy()


def stock_state(get_col, date_code: np.ndarray):
    """
    The canonical stock-state matrix.

    get_col(name) returns that column as an array in row order, or None if
    absent. Each dimension is a per-date cross-sectional rank centred on 0
    (-0.5..+0.5): RELATIVE state - 'high RSI' means high versus peers today.
    A missing input is dropped AND reported; fewer than MIN_STATE_DIMS
    dimensions raises, instead of silently building regimes from a few axes.
    """
    cols, names, missing = [], [], []
    for nm, col, tf in STOCK_STATE_SPEC:
        v = get_col(col)
        if v is None:
            missing.append(col)
            continue
        v = np.asarray(v, dtype="float64")
        v = np.where(np.isfinite(v), v, np.nan)
        if tf == "absrank":
            v = np.abs(v)
        r = pd.Series(v).groupby(date_code).rank(pct=True).to_numpy() - 0.5
        cols.append(r.astype("float32")); names.append(f"st_{nm}")
    if len(names) < MIN_STATE_DIMS:
        raise StateInputError(
            f"only {len(names)} of {len(STOCK_STATE_SPEC)} stock-state inputs "
            f"exist (missing {missing}); at least {MIN_STATE_DIMS} are required")
    return np.column_stack(cols), names, missing


def market_state(close: np.ndarray, date_code: np.ndarray, seg: np.ndarray,
                 n_dates: int, window: int = 20) -> Dict[str, np.ndarray]:
    """
    Canonical market state, one value per session (index = date_code).

    From close-to-close returns of the panel's names each session - the
    market's actual daily move, overnight gap included - computed within
    unbroken runs, so a re-admitted stock's multi-month jump never enters a
    day's median or dispersion. Trailing windows only.
    """
    r = daily_returns(close, seg)
    ok = np.isfinite(r)
    d = pd.DataFrame({"d": np.asarray(date_code)[ok], "r": r[ok]})
    g = d.groupby("d")["r"]
    idx = pd.RangeIndex(n_dates)
    ret1 = g.median().reindex(idx)
    breadth = (d.assign(p=(d["r"] > 0).astype(float)).groupby("d")["p"].mean()
               .reindex(idx))
    disp = g.std().reindex(idx)
    mp = max(3, window // 2)
    out = {"mk_ret1": ret1, "mk_breadth": breadth, "mk_disp": disp,
           "mk_trend20": ret1.rolling(window, min_periods=mp).sum(),
           "mk_vol20": ret1.rolling(window, min_periods=mp).std(),
           "mk_breadth20": breadth.rolling(window, min_periods=mp).mean(),
           "mk_disp20": disp.rolling(window, min_periods=mp).mean()}
    return {k: v.to_numpy("float64") for k, v in out.items()}


def _complete(X: np.ndarray, rows: np.ndarray) -> np.ndarray:
    return rows[np.isfinite(X[rows]).all(axis=1)]


def choose_k_bic(X: np.ndarray, k_range, rows: np.ndarray, seed: int):
    """K by BIC on complete rows. BIC is a clustering criterion only."""
    from sklearn.mixture import GaussianMixture
    rows = _complete(X, rows)
    bic = {}
    for k in k_range:
        if int(k) >= len(rows):
            continue
        bic[int(k)] = float(GaussianMixture(int(k), random_state=seed, max_iter=200,
                                            **GMM_PARAMS).fit(X[rows]).bic(X[rows]))
    return min(bic, key=bic.get), bic


def fit_state_gmm(X: np.ndarray, k: int, rows: np.ndarray, seed: int):
    from sklearn.mixture import GaussianMixture
    rows = _complete(X, rows)
    return GaussianMixture(int(k), random_state=seed, n_init=2, max_iter=300,
                           **GMM_PARAMS).fit(X[rows])


# ----------------------------------------------------------------------
# ENTRY MODES - which label columns a tool reads
# ----------------------------------------------------------------------
# "close": entry at the signal day's close (the original research world).
# "open":  entry at the NEXT session's open with gap-aware fills - the entry
#          you can actually get when signals are produced after the close.
# Tools read the mode's columns and RENAME them to the canonical (close-mode)
# names in memory, so everything downstream is identical in both worlds.
ENTRY_LABELS = {
    "close": {"label_tp_before_sl": "label_tp_before_sl", "label_first_touch": "label_first_touch",
              "label_exit_ret": "label_exit_ret", "label_mfe_5d": "label_mfe_5d",
              "label_mae_5d": "label_mae_5d", "label_days_to_tp": "label_days_to_tp",
              "label_days_to_sl": "label_days_to_sl",
              "label_exit_ret_atr2p0_1p0": "label_exit_ret_atr2p0_1p0"},
    "open": {"label_tp_before_sl": "label_tp_before_sl_o1", "label_first_touch": "label_first_touch_o1",
             "label_exit_ret": "label_exit_ret_o1", "label_mfe_5d": "label_mfe_5d_o1",
             "label_mae_5d": "label_mae_5d_o1", "label_days_to_tp": "label_days_to_tp_o1",
             "label_days_to_sl": "label_days_to_sl_o1",
             "label_exit_ret_atr2p0_1p0": "label_exit_ret_atr2p0_1p0_o1"},
}


def read_panel_labels(panel_path, canonical: list, entry: str = "close", extra: list | None = None,
                      filters=None) -> pd.DataFrame:
    """Read columns from the panel, mapping label columns for the entry mode to canonical names."""
    if entry not in ENTRY_LABELS:
        raise ValueError(f"entry must be one of {list(ENTRY_LABELS)}")
    m = ENTRY_LABELS[entry]
    actual = [m.get(c, c) for c in canonical]
    cols = list(dict.fromkeys(actual + list(extra or [])))
    df = pd.read_parquet(panel_path, columns=cols, filters=filters)
    return df.rename(columns={m[c]: c for c in canonical if c in m and m[c] != c})
