#!/usr/bin/env python3
"""
stage_d_forensics.py - do the stage-D passes survive realistic trade scoring?

    python stage_d_forensics.py --panel %CACHE_DAILY_ROOT%\\panel_oc --root %CACHE_DAILY_ROOT% ^
        --isin C:\\QuantData\\nse\\BhavCopy_NSE_CM_0_0_0_20260928_F_0000.csv.zip

THE RULES ARE IN FORENSICS_DECLARATION.json
===========================================
Three corrections, applied in order to each stage-D target's daily top-3.
Every one can only BLOCK a pass:
  1. ETFs out by ISIN ('INF...' = mutual-fund unit), from NSE's own files;
     each day's top-3 is re-drawn from stage D's scores without them.
  2. An exit session that closes LOCKED at the lower band cannot be sold at
     the close; the trade exits at the first later open that is not itself
     locked (up to 20 sessions).
  3. One tick per round trip on top of 35 bp, at NSE's tick size for the date
     (Rs 0.05 before 2024-06-10; the price-banded schedules after).
Then the stage-D gate, unchanged: top-3 net 98.75% lower bound > 0, mean
from 2024 > 0, excess over buying everything 98.75% lower bound > 0.

WHY THESE AND NOT STRICTER
--------------------------
The open auction fills everyone at one price, so the entry crosses no
spread; one tick per round trip covers the exit. Buying at an open locked at
the LOWER band is allowed - sellers are queued, a buy fills. The benchmark
(buy everything) pays its own tick cost but is not re-scored for its locked
exits, which could only make the excess test easier.

WHAT IT CANNOT SEE
------------------
Survivorship (delisted stocks are not in the cache) and trade-for-trade /
surveillance restrictions at the time of each trade. Today's series from the
bhavcopy is reported, descriptively. Both need NSE's historical bhavcopies.

INTEGRITY STOP
--------------
Every pick's outcome is re-derived from the raw cache. If fewer than 99.5%
match stage D's to 1e-6, the tool stops: the cache is no longer the data
stage D was scored on.

OUTPUT: <panel>\\stage_d\\<STAGED run>\\forensics\\FORENSICS_YYYYMMDD_NNN\\
    forensics_report.md, forensics.json, trades.csv (every re-scored trade),
    etf_classification.csv (every panel symbol: ISIN, series, ETF, how decided)
and ledger entries stage_d_forensics_declared (before) and stage_d_forensics.
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

import research_common as RC     # first: from a subfolder, this folder's copy must win
import panel_build as PB         # noqa: E402,F401
import stage_b_screen as SB      # noqa: E402
import stage_d_gate as SD        # noqa: E402

CODE_VERSION = "stage_d_forensics v1"
DECL_FILE = "FORENSICS_DECLARATION.json"
FAMILY_ID = SB.FAMILY_ID
CFG = {
    "etf_rule": "isin_INF_else_name_rule",
    "lock_exit": "first_open_not_locked_at_lower_band",
    "lock_search_sessions": 20,
    "tick_cost": "one_tick_per_round_trip",
    "tick_schedule": ["2024-06-10", "2025-04-15"],
    "cost_bps": 35.0,
    "top_n": 3,
    "interval_level": 0.9875, "boot_B": 10000, "boot_block": 10, "seed": 7,
    "recency_start": "2024-01-01",
    "min_outcome_match": 0.995,
}
T2T_SERIES = {"BE", "BZ", "BT", "ST", "SM", "IL", "GS", "BL"}   # descriptive only
PRICE_BUCKETS = [0, 10, 20, 50, 100, 250, float("inf")]


class PreconditionError(SystemExit):
    pass


class IntegrityError(RuntimeError):
    pass


def _log(msg: str) -> None:
    print(f"{dt.datetime.now():%H:%M:%S}  {msg}", flush=True)


# ----------------------------------------------------------------------
# rules
# ----------------------------------------------------------------------
def tick_size(dates, prices) -> np.ndarray:
    """NSE equity tick size for each (date, price)."""
    d = np.asarray(dates, dtype="datetime64[ns]")
    p = np.asarray(prices, dtype=float)
    t1, t2 = (np.datetime64(x, "ns") for x in CFG["tick_schedule"])
    out = np.full(p.shape, 0.05)
    mid = (d >= t1) & (d < t2)
    out[mid & (p < 250)] = 0.01
    late = d >= t2
    sched = np.select([p < 250, p <= 1000, p <= 5000, p <= 10000, p <= 20000],
                      [0.01, 0.05, 0.10, 0.50, 1.00], 5.00)
    out[late] = sched[late]
    return out


def _at_band(move, ref, tick) -> np.ndarray:
    """research_common.at_price_band, but 'within one tick of the band' uses the tick size in force
    that day (Rs 0.01 below Rs 250 from 2024-06-10) instead of a fixed Rs 0.05 - fewer false locks."""
    move, ref, tick = (np.asarray(x, dtype=float) for x in (move, ref, tick))
    with np.errstate(invalid="ignore", divide="ignore"):
        tol = np.maximum(RC.BAND_TOL_PP, RC.BAND_TICK_SLACK * tick / ref)
        hit = np.zeros(np.broadcast(move, ref).shape, dtype=bool)
        for b in RC.PRICE_BANDS:
            hit |= np.abs(move - b) <= tol
    return hit & np.isfinite(move) & np.isfinite(ref) & (ref > 0)


def band_down(x, prev, tick=RC.BAND_TICK) -> np.ndarray:
    """x sits on a 2/5/10/20% band BELOW prev."""
    x, prev = np.asarray(x, dtype=float), np.asarray(prev, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _at_band(1.0 - x / prev, prev, tick)


def band_up(x, prev, tick=RC.BAND_TICK) -> np.ndarray:
    x, prev = np.asarray(x, dtype=float), np.asarray(prev, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return _at_band(x / prev - 1.0, prev, tick)


def _eq(a, b) -> np.ndarray:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.abs(a - b) <= 1e-9 * np.maximum(np.abs(a), np.abs(b))


def rescore_symbol(o, h, l, c, i_sig: np.ndarray, horizon: int, search: int, ts=None) -> Dict[str, np.ndarray]:
    """For signals at bar indices i_sig: the stage-D outcome re-derived, and the outcome
    when an exit locked at the lower band is replaced by the first sellable later open.
    ts (bar dates) sets the tick size used to recognise a band; without it, Rs 0.05."""
    n = len(c)
    prev = np.r_[np.nan, c[:-1]]
    tk = tick_size(ts, prev) if ts is not None else np.full(n, RC.BAND_TICK)
    m = len(i_sig)
    out = {k: np.full(m, np.nan) for k in ("entry", "exit_close", "outcome", "exit_px", "outcome_lock")}
    for k in ("locked_exit", "still_locked", "no_later_bar", "frozen_entry", "entry_lower_band",
              "exit_upper_band"):
        out[k] = np.zeros(m, bool)
    out["exit_shift"] = np.zeros(m, int)
    for q, i in enumerate(i_sig):
        if i < 0 or i + 1 >= n:
            continue
        e = o[i + 1]
        out["entry"][q] = e
        out["frozen_entry"][q] = bool(_eq(o[i + 1], h[i + 1]) & _eq(o[i + 1], l[i + 1]) & _eq(o[i + 1], c[i + 1]))
        out["entry_lower_band"][q] = bool(band_down(o[i + 1], c[i], tk[i + 1]))
        j = i + horizon
        if j >= n or not (np.isfinite(e) and e > 0):
            continue
        out["exit_close"][q] = c[j]
        out["outcome"][q] = c[j] / e - 1.0
        out["exit_upper_band"][q] = bool(band_up(c[j], c[j - 1], tk[j]) & (c[j] >= h[j] * (1 - 1e-9)))
        px = c[j]
        if band_down(c[j], c[j - 1], tk[j]) and c[j] <= l[j] * (1 + 1e-9):
            out["locked_exit"][q] = True
            if j + 1 >= n:
                out["no_later_bar"][q] = True
            else:
                found = False
                last = min(j + search, n - 1)
                for k in range(j + 1, last + 1):
                    if not (band_down(o[k], c[k - 1], tk[k]) and o[k] <= l[k] * (1 + 1e-9)):
                        px, found = o[k], True
                        out["exit_shift"][q] = k - j
                        break
                if not found:
                    px = c[last]
                    out["still_locked"][q] = True
                    out["exit_shift"][q] = last - j
        out["exit_px"][q] = px
        out["outcome_lock"][q] = px / e - 1.0
    return out


# ----------------------------------------------------------------------
# ETFs by ISIN
# ----------------------------------------------------------------------
SYM_COLS = ("SYMBOL", "TCKRSYMB")
ISIN_COLS = ("ISIN", "ISIN NUMBER", "ISINNUMBER", "ISIN CODE", "ISIN_CODE")
SERIES_COLS = ("SERIES", "SCTYSRS")


def load_isin_files(paths: List[str]) -> pd.DataFrame:
    parts = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise PreconditionError(f"ISIN file not found: {p}")
        df = pd.read_csv(p, dtype=str, compression="infer")
        cols = {str(c).strip().upper(): c for c in df.columns}
        sc = next((cols[k] for k in SYM_COLS if k in cols), None)
        ic = next((cols[k] for k in ISIN_COLS if k in cols), None)
        rc = next((cols[k] for k in SERIES_COLS if k in cols), None)
        if sc is None or ic is None:
            raise PreconditionError(f"{p.name}: no symbol/ISIN columns found (looked for {SYM_COLS} / {ISIN_COLS})")
        f = pd.DataFrame({"symbol": df[sc].astype(str).str.strip().str.upper(),
                          "isin": df[ic].astype(str).str.strip().str.upper(),
                          "series": df[rc].astype(str).str.strip().str.upper() if rc else "",
                          "source": p.name})
        f = f[(f["symbol"] != "") & (f["isin"].str.len() == 12)]
        parts.append(f)
    all_ = pd.concat(parts, ignore_index=True)
    all_["_eq"] = (all_["series"] != "EQ").astype(int)      # prefer the EQ line for the series column
    return all_.sort_values(["symbol", "_eq"]).drop_duplicates("symbol").drop(columns="_eq")


def classify_symbols(symbols, isin_df: pd.DataFrame, extra: set) -> pd.DataFrame:
    m = isin_df.set_index("symbol")
    rows = []
    for s in sorted(set(symbols)):
        if s in m.index:
            isin, ser = m.at[s, "isin"], m.at[s, "series"]
            rows.append((s, isin, ser, isin.startswith("INF"), "isin"))
        else:
            name = RC.is_etf_symbol(s, extra)
            rows.append((s, "", "", bool(name), "user_list" if s in extra else "name_rule"))
    return pd.DataFrame(rows, columns=["symbol", "isin", "series", "etf", "how"])


# ----------------------------------------------------------------------
# preconditions
# ----------------------------------------------------------------------
def load_declaration() -> Tuple[dict, str]:
    p = HERE / DECL_FILE
    if not p.exists():
        raise PreconditionError(f"{DECL_FILE} not found next to {Path(__file__).name}")
    raw = p.read_bytes()
    decl = json.loads(raw.decode("utf-8"))
    if decl.get("config") != json.loads(json.dumps(CFG)):
        diff = sorted(k for k in set(CFG) | set(decl.get("config", {}))
                      if json.loads(json.dumps(CFG)).get(k) != decl.get("config", {}).get(k))
        raise PreconditionError(f"settings differ from {DECL_FILE} on {diff}; the declaration is fixed")
    return decl, hashlib.sha256(raw).hexdigest()[:16]


def find_run(pp: Path, run: Optional[str]) -> Tuple[str, Path]:
    gates = [e for e in RC.ledger_entries(pp) if e.get("kind") == "stage_d_gate"
             and e.get("experiment_family") == FAMILY_ID]
    if not gates:
        raise PreconditionError("no stage-D run on record for this panel")
    rid = run or gates[-1]["run_id"]
    if rid not in {g["run_id"] for g in gates}:
        raise PreconditionError(f"{rid} is not a stage-D run in this panel's ledger")
    folder = pp.parent / "stage_d" / rid
    for f in ("preds.parquet", "daily.csv", "stage_d.json"):
        if not (folder / f).exists():
            raise PreconditionError(f"{folder / f} is missing")
    return rid, folder


def check_rerun(pp: Path, rid: str, reason: Optional[str]) -> List[dict]:
    prior = [e for e in RC.ledger_entries(pp) if e.get("kind") == "stage_d_forensics" and e.get("d_run") == rid]
    if prior and not reason:
        raise PreconditionError(f"forensics already ran on {rid} ({prior[-1].get('run_id')}). Re-running needs "
                                f"--rerun-reason \"...\" (a verified bug); it goes into the ledger.")
    return prior


# ----------------------------------------------------------------------
# the forensics
# ----------------------------------------------------------------------
def draw_top(preds: pd.DataFrame, etf: set, n: int) -> pd.DataFrame:
    """Each day's top-n by raw score (ties by symbol), ETFs removed."""
    d = preds[~preds["symbol"].isin(etf)].sort_values(["timestamp", "score", "symbol"],
                                                      ascending=[True, False, True])
    d = d.groupby("timestamp", sort=False, observed=True).head(n).copy()
    d["symbol"] = d["symbol"].astype(str)
    d["rank"] = d.groupby("timestamp").cumcount() + 1
    return d


def run_forensics(pp: Path, root: Path, folder: Path, targets: List[str], isin_df: pd.DataFrame,
                  verbose: bool = True, cfg: Optional[dict] = None) -> dict:
    cfg = dict(cfg or CFG)
    t0 = time.perf_counter()
    from data_quality import _paths
    extra = RC.load_symbol_list(root / RC.UNIVERSE_EXCLUDE_FILE)
    import pyarrow.parquet as pq
    names = pq.ParquetFile(pp).schema_arrow.names
    P = pd.read_parquet(pp, columns=["timestamp", "symbol", "close"]
                        + (["X_turnover_med"] if "X_turnover_med" in names else []))
    P["timestamp"] = SB._naive(P["timestamp"])
    cls = classify_symbols(P["symbol"].unique(), isin_df, extra)
    etf = set(cls.loc[cls["etf"], "symbol"])
    missed = sorted(set(cls.loc[cls["etf"] & (cls["how"] == "isin"), "symbol"])
                    - {s for s in cls["symbol"] if RC.is_etf_symbol(s, extra)})
    if verbose:
        _log(f"{len(cls):,} panel symbols | ISIN found for {int((cls['how'] == 'isin').sum()):,} | ETFs: {len(etf)} "
             f"({len(missed)} of them missed by the name rule)")

    preds = pd.read_parquet(folder / "preds.parquet").rename(columns={"outcome": "outcome_stage_d"})
    preds["timestamp"] = SB._naive(preds["timestamp"])
    preds["symbol"] = preds["symbol"].astype(str).astype("category")   # ~5M rows: keep it light
    preds["target"] = preds["target"].astype(str).astype("category")
    daily = pd.read_csv(folder / "daily.csv", parse_dates=["timestamp"])
    cost = cfg["cost_bps"] / 1e4

    # benchmark tick cost: every eligible test row, priced at its signal-day close
    keys = preds[["timestamp", "symbol"]].drop_duplicates().astype({"symbol": str})
    mk = keys.merge(P[["timestamp", "symbol", "close"]], on=["timestamp", "symbol"], how="left")
    mk["tick_cost"] = tick_size(mk["timestamp"].to_numpy(), mk["close"].to_numpy()) / mk["close"]
    mkt_tick = mk.groupby("timestamp")["tick_cost"].mean()

    picks = {}
    for tg in targets:
        pr = preds[preds["target"] == tg]
        picks[tg] = draw_top(pr, etf, cfg["top_n"]).assign(target=tg).astype({"target": str})
    allp = pd.concat(picks.values(), ignore_index=True)

    rec_parts = []
    syms = sorted(allp["symbol"].unique())
    for q, s in enumerate(syms, 1):
        fp, _ = _paths(root, s)
        sub = allp[allp["symbol"] == s]
        if not Path(fp).exists():
            rec_parts.append(sub.assign(match=False, missing_cache=True))
            continue
        b = pd.read_parquet(fp, columns=["timestamp", "open", "high", "low", "close"])
        b["timestamp"] = SB._naive(b["timestamp"])
        b = b.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        ts = b["timestamp"].to_numpy(dtype="datetime64[ns]")
        o, h, l, c = (b[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        for tg, g in sub.groupby("target"):
            sig = g["timestamp"].to_numpy(dtype="datetime64[ns]")
            i = np.searchsorted(ts, sig)
            i = np.where((i < len(ts)) & (ts[np.minimum(i, len(ts) - 1)] == sig), i, -1)
            r = rescore_symbol(o, h, l, c, i, SD.horizon_of(tg), cfg["lock_search_sessions"], ts)
            f = g.copy()
            for k, v in r.items():
                f[k] = v
            f["entry_date"] = [ts[x + 1] if 0 <= x < len(ts) - 1 else np.datetime64("NaT") for x in i]
            rec_parts.append(f.assign(missing_cache=False))
        if verbose and q % 250 == 0:
            _log(f"  {q}/{len(syms)} symbols re-scored")
    T = pd.concat(rec_parts, ignore_index=True)
    for k in ("locked_exit", "still_locked", "no_later_bar", "frozen_entry", "entry_lower_band",
              "exit_upper_band", "missing_cache"):
        T[k] = T[k].astype("boolean").fillna(False).astype(bool) if k in T else False
    T = T.rename(columns={"outcome": "outcome_cache"})
    T["match"] = (np.isfinite(T["outcome_cache"]) & np.isfinite(T["outcome_stage_d"])
                  & (np.abs(T["outcome_cache"] - T["outcome_stage_d"]) <= 1e-6)) | \
                 (~np.isfinite(T["outcome_cache"]) & ~np.isfinite(T["outcome_stage_d"]))
    match_rate = float(T["match"].mean()) if len(T) else float("nan")
    if not match_rate >= cfg["min_outcome_match"]:
        raise IntegrityError(f"only {match_rate:.2%} of picks re-derive stage D's outcome from the raw cache "
                             f"(need {cfg['min_outcome_match']:.1%}) - the cache no longer matches stage D's data")
    T["tick"] = tick_size(pd.to_datetime(T["entry_date"]).to_numpy(), T["entry"].to_numpy(float))
    T["tick_cost"] = T["tick"] / T["entry"]
    T["net_etf"] = T["outcome_stage_d"] - cost
    T["net_lock"] = T["outcome_lock"] - cost
    T["net_final"] = T["outcome_lock"] - cost - T["tick_cost"]
    if "X_turnover_med" in P:
        T = T.merge(P[["timestamp", "symbol", "X_turnover_med"]], on=["timestamp", "symbol"], how="left")

    res = {}
    for tg in targets:
        g = T[T["target"] == tg]
        dd = daily[daily["target"] == tg].set_index("timestamp").sort_index()
        day = g.groupby("timestamp")[["net_etf", "net_lock", "net_final"]].mean().reindex(dd.index)
        mkt_net = dd["market"] - cost - mkt_tick.reindex(dd.index).fillna(0.0)
        excess = day["net_final"] - mkt_net
        res[tg] = evaluate_target(tg, g, dd, day, excess, picks[tg], preds[preds["target"] == tg], etf, cfg)
        if verbose:
            r = res[tg]
            _log(f"{tg}: final net {r['final_net']['mean'] * 1e4:+.1f} bp "
                 f"[{r['final_net']['lo'] * 1e4:+.1f}, {r['final_net']['hi'] * 1e4:+.1f}] -> "
                 f"{'PASS' if r['verdict']['pass'] else 'FAIL'}")
    T["current_series"] = T["symbol"].map(dict(zip(cls["symbol"], cls["series"]))).fillna("")
    return {"results": res, "trades": T, "classification": cls, "etf_missed_by_name_rule": missed,
            "match_rate": match_rate, "seconds": round(time.perf_counter() - t0, 1), "config": cfg}


def evaluate_target(tg, g, dd, day, excess, top, pr, etf, cfg) -> dict:
    lvl, B, blk, seed = cfg["interval_level"], cfg["boot_B"], cfg["boot_block"], cfg["seed"]
    bb = lambda x, L=lvl: SD.block_bootstrap(np.asarray(x, float), L, B, blk, seed)
    r = {"as_run_net": bb(dd["top3_net"]), "etf_net": bb(day["net_etf"]), "lock_net": bb(day["net_lock"]),
         "final_net": bb(day["net_final"]), "final_excess": bb(excess)}
    rec = day.index >= pd.Timestamp(cfg["recency_start"])
    r["final_recent"] = bb(day.loc[rec, "net_final"], 0.95)
    pl = day.index >= pd.Timestamp(SB.LOCKBOX_START_PINNED)
    r["final_post_lockbox_mean"] = float(day.loc[pl, "net_final"].mean())
    r["final_by_year"] = {int(y): float(v) for y, v in day["net_final"].groupby(day.index.year).mean().items()}
    orig = pr.sort_values(["timestamp", "score", "symbol"], ascending=[True, False, True]).groupby(
        "timestamp", observed=True).head(3)
    r["etf_in_original_top3"] = int(orig["symbol"].isin(etf).sum())
    r["picks"] = int(len(g))
    lk = g["locked_exit"].astype(bool)
    r["locked_exits"] = {"count": int(lk.sum()), "share": float(lk.mean()),
                         "still_locked_after_search": int(g["still_locked"].astype(bool).sum()),
                         "mean_cost_bp": float(((g.loc[lk, "outcome_lock"] - g.loc[lk, "outcome_stage_d"]) * 1e4).mean())
                         if lk.any() else 0.0}
    r["frozen_entry_share"] = float(g["frozen_entry"].astype(bool).mean())
    r["entry_at_lower_band_share"] = float(g["entry_lower_band"].astype(bool).mean())
    r["exit_at_upper_band_share"] = float(g["exit_upper_band"].astype(bool).mean())
    r["zero_outcome_share"] = float((g["outcome_stage_d"] == 0).mean())
    r["tick_cost_bp"] = {"mean": float(g["tick_cost"].mean() * 1e4),
                         "share_above_35bp": float((g["tick_cost"] > 0.0035).mean())}
    cut = pd.cut(g["entry"], PRICE_BUCKETS, right=False)
    r["price_buckets"] = {str(k): {"share": float(len(v) / max(len(g), 1)),
                                   "final_net_bp": float(v["net_final"].mean() * 1e4) if len(v) else float("nan")}
                          for k, v in g.groupby(cut, observed=False)}
    vc = g["symbol"].value_counts()
    r["repeat_names"] = {"distinct": int(vc.size), "top10_share": float(vc.head(10).sum() / max(len(g), 1)),
                         "most_picked": {k: int(v) for k, v in vc.head(12).items()}}
    nf = g["net_final"].dropna().sort_values(ascending=False).to_numpy()
    k5 = max(int(len(nf) * 0.05), 1)
    r["tail_top5pct_share_of_net"] = float(nf[:k5].sum() / nf.sum()) if nf.sum() != 0 else float("nan")
    r["mean_final_without_top5pct"] = float(nf[k5:].mean() * 1e4) if len(nf) > k5 else float("nan")
    if "X_turnover_med" in g:
        tv = g["X_turnover_med"].median()
        r["capacity"] = {"median_pick_turnover_rs": float(tv), "one_pct_of_it_rs": float(tv * 0.01)}
    v = {"net_lower_bound_above_zero": bool(np.isfinite(r["final_net"]["lo"]) and r["final_net"]["lo"] > 0),
         "recent_mean_above_zero": bool(np.isfinite(r["final_recent"]["mean"]) and r["final_recent"]["mean"] > 0),
         "excess_lower_bound_above_zero": bool(np.isfinite(r["final_excess"]["lo"]) and r["final_excess"]["lo"] > 0)}
    v["pass"] = all(v.values())
    r["verdict"] = v
    return r


# ----------------------------------------------------------------------
# output
# ----------------------------------------------------------------------
def _bp(x) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x * 1e4:+.1f}"


def _iv(b) -> str:
    return f"{_bp(b['mean'])} [{_bp(b['lo'])}, {_bp(b['hi'])}]"


def write_report(out: dict, targets, run_id, d_run, path: Path, decl_sha: str) -> None:
    R, cls = out["results"], out["classification"]
    L = [f"# Stage D forensics - {run_id}", "",
         f"On stage-D run {d_run} | declaration sha {decl_sha} | {CODE_VERSION} | {dt.datetime.now():%Y-%m-%d %H:%M}",
         "", "**Rules (declared before this run; each can only block):** ETFs out by ISIN; exits locked at the "
         "lower band moved to the first sellable open; one tick per round trip on top of 35 bp. Then the stage-D "
         "gate unchanged.", "",
         f"Integrity: {out['match_rate']:.2%} of picks re-derive stage D's outcome from the raw cache.", "",
         "## 1. The gate after the corrections (bp per trade)", "",
         "| target | as run | ETFs out | + locked exits | + one tick = FINAL [98.75%] | final excess [98.75%] | "
         "final 2024+ | VERDICT |", "|---|---|---|---|---|---|---|---|"]
    for tg in targets:
        r = R[tg]
        v = r["verdict"]
        why = [k for k, ok in (("net interval", v["net_lower_bound_above_zero"]),
                               ("2024+ mean", v["recent_mean_above_zero"]),
                               ("excess interval", v["excess_lower_bound_above_zero"])) if not ok]
        L.append(f"| {tg} | {_bp(r['as_run_net']['mean'])} | {_bp(r['etf_net']['mean'])} | {_bp(r['lock_net']['mean'])} | "
                 f"{_iv(r['final_net'])} | {_iv(r['final_excess'])} | {_bp(r['final_recent']['mean'])} | "
                 f"**{'PASS' if v['pass'] else 'FAIL'}**{'' if v['pass'] else ' (' + ', '.join(why) + ')'} |")
    L += ["", "## 2. ETFs", "",
          f"{len(cls):,} panel symbols; ISIN found for {int((cls['how'] == 'isin').sum()):,}; ETFs "
          f"{int(cls['etf'].sum())}, of which {len(out['etf_missed_by_name_rule'])} the name rule missed: "
          + (", ".join(out["etf_missed_by_name_rule"][:80]) or "none") + ".",
          "", "ETFs in stage D's original top-3: " + "; ".join(f"{tg} {R[tg]['etf_in_original_top3']}" for tg in targets),
          ""]
    L += ["## 3. What the picks are (after ETFs out)", ""]
    for tg in targets:
        r = R[tg]
        lk = r["locked_exits"]
        pb = "; ".join(f"{k} {v['share']:.0%} ({v['final_net_bp']:+.0f} bp)" for k, v in r["price_buckets"].items()
                       if v["share"] > 0)
        cap = r.get("capacity")
        L += [f"**{tg}**", "",
              f"- locked exits: {lk['count']} ({lk['share']:.1%} of picks), costing {lk['mean_cost_bp']:+.0f} bp each on "
              f"average; still locked after 20 sessions: {lk['still_locked_after_search']}",
              f"- one-tick cost: mean {r['tick_cost_bp']['mean']:.1f} bp; above 35 bp on "
              f"{r['tick_cost_bp']['share_above_35bp']:.1%} of picks",
              f"- entry price buckets (share, final net): {pb}",
              f"- entry day frozen (open = high = low = close): {r['frozen_entry_share']:.1%} | bought at an open on "
              f"the lower band: {r['entry_at_lower_band_share']:.1%} | exit closed at the upper band: "
              f"{r['exit_at_upper_band_share']:.1%} | outcome exactly 0: {r['zero_outcome_share']:.1%}",
              f"- names: {r['repeat_names']['distinct']} distinct; top 10 carry {r['repeat_names']['top10_share']:.1%}: "
              + ", ".join(f"{k} {v}" for k, v in r["repeat_names"]["most_picked"].items()),
              f"- tail: the top 5% of trades carry {r['tail_top5pct_share_of_net']:.0%} of final net; without them "
              f"{r['mean_final_without_top5pct']:+.1f} bp",
              (f"- capacity: median pick turnover Rs {cap['median_pick_turnover_rs'] / 1e7:.1f} crore/day; 1% of it "
               f"is Rs {cap['one_pct_of_it_rs'] / 1e5:.1f} lakh per position" if cap else "- capacity: n/a"),
              "- final net by year: " + "; ".join(f"{y} {_bp(v)}" for y, v in r["final_by_year"].items())
              + f" | 2025-03-05 onward {_bp(r['final_post_lockbox_mean'])}", ""]
    T = out["trades"]
    ser = T[T["current_series"].isin(T2T_SERIES)]
    L += ["## 4. Not measurable with this data", "",
          "- **Survivorship**: stocks delisted 2016-2026 are not in the cache.",
          f"- **Trade-for-trade / surveillance at the time**: unknown. Descriptive only - picks whose symbol is in a "
          f"restricted series TODAY: {len(ser)} of {len(T)} ({len(ser) / max(len(T), 1):.1%}).", "",
          "## 5. What this does not say", "",
          "- A PASS keeps the target as a forward paper-test arm. No money before the month-6 check.",
          "- Fills at the open auction for these volatile names are still an assumption; the paper test measures them.",
          ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage D forensics for family oc_v1")
    ap.add_argument("--panel", required=True, help="panel folder, e.g. %%CACHE_DAILY_ROOT%%\\panel_oc")
    ap.add_argument("--root", required=True, help="raw cache root, e.g. %%CACHE_DAILY_ROOT%%")
    ap.add_argument("--isin", required=True, nargs="+", help="NSE bhavcopy CSV/ZIP file(s) with ISINs")
    ap.add_argument("--run", default=None, help="stage-D run id (default: the latest)")
    ap.add_argument("--rerun-reason", default=None)
    a = ap.parse_args(argv)
    pp = Path(a.panel) / "panel.parquet"
    root = Path(a.root)

    man, man_sha = SB.load_manifest()
    targets = list(man["targets"].keys())
    SB.check_panel(pp, man_sha)
    decl, decl_sha = load_declaration()
    d_run, folder = find_run(pp, a.run)
    prior = check_rerun(pp, d_run, a.rerun_reason)
    isin_df = load_isin_files(a.isin)
    out_root = folder / "forensics"
    day = dt.date.today().strftime("%Y%m%d")
    k = 1
    while (out_root / f"FORENSICS_{day}_{k:03d}").exists():
        k += 1
    run_id = f"FORENSICS_{day}_{k:03d}"
    out = out_root / run_id
    out.mkdir(parents=True, exist_ok=False)
    RC.ledger_append(pp, {"kind": "stage_d_forensics_declared", "tool": CODE_VERSION, "run_id": run_id,
                          "d_run": d_run, "declaration_sha": decl_sha, "experiment_family": FAMILY_ID,
                          "isin_files": [Path(x).name for x in a.isin], "rerun_reason": a.rerun_reason})
    print("=" * 76)
    print(f"{CODE_VERSION} | on {d_run} | declaration sha {decl_sha}")
    print("RULES (declared before this run; each can only block): ETFs out by ISIN; exits locked at the lower")
    print("band moved to the first sellable open; one tick per round trip on top of 35 bp; stage-D gate unchanged.")
    print("=" * 76)

    res = run_forensics(pp, root, folder, targets, isin_df)
    R = res["results"]
    res["trades"].to_csv(out / "trades.csv", index=False)
    res["classification"].to_csv(out / "etf_classification.csv", index=False)
    js = {"run_id": run_id, "tool": CODE_VERSION, "d_run": d_run, "declaration_sha": decl_sha,
          "manifest_sha": man_sha, "isin_files": [str(x) for x in a.isin], "match_rate": res["match_rate"],
          "etf_missed_by_name_rule": res["etf_missed_by_name_rule"], "results": R, "config": res["config"],
          "seconds": res["seconds"], "rerun_reason": a.rerun_reason, "prior_runs": [p.get("run_id") for p in prior]}
    (out / "forensics.json").write_text(json.dumps(js, indent=2, default=str), encoding="utf-8")
    write_report(res, targets, run_id, d_run, out / "forensics_report.md", decl_sha)
    passed = [tg for tg in targets if R[tg]["verdict"]["pass"]]
    n = RC.ledger_append(pp, {"kind": "stage_d_forensics", "tool": CODE_VERSION, "run_id": run_id, "d_run": d_run,
                              "declaration_sha": decl_sha, "experiment_family": FAMILY_ID, "passed": passed,
                              "final_net_bp": {tg: [R[tg]["final_net"][x] * 1e4 for x in ("mean", "lo", "hi")]
                                               for tg in targets}, "rerun_reason": a.rerun_reason})
    print()
    print(f"{'target':<12}{'as run':>9}{'ETFs out':>10}{'+locks':>9}{'FINAL [98.75%]':>28}{'excess [98.75%]':>26}"
          f"{'2024+':>8}  verdict")
    for tg in targets:
        r = R[tg]
        print(f"{tg:<12}{_bp(r['as_run_net']['mean']):>9}{_bp(r['etf_net']['mean']):>10}{_bp(r['lock_net']['mean']):>9}"
              f"{_iv(r['final_net']):>28}{_iv(r['final_excess']):>26}{_bp(r['final_recent']['mean']):>8}  "
              f"{'PASS' if r['verdict']['pass'] else 'FAIL'}")
    print(f"\n  ETFs found by ISIN that the name rule missed: {len(res['etf_missed_by_name_rule'])}")
    print(f"  integrity: {res['match_rate']:.2%} of picks re-derive stage D's outcome from the raw cache")
    print(f"  outputs: {out}")
    print(f"  research ledger: entry {n} (stage_d_forensics {run_id})")
    print("=" * 76)
    print("FORENSICS: " + (f"{', '.join(passed)} survive -> paper-test arm(s)." if passed
                           else "no target survives. Do not paper-trade or trade these engines."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
