#!/usr/bin/env python3
"""
stage_d_gate.py - stage D of family oc_v1: can a model on the existing clean
features pick a daily top-3 that makes money after 35 bp, buying at the NEXT
OPEN and selling at the close of T+h?

    python stage_d_gate.py --panel %CACHE_DAILY_ROOT%\\panel_oc
    python stage_d_gate.py --panel %CACHE_DAILY_ROOT%\\panel_oc --rerun-reason "fixing <bug>"

THE RULES ARE IN STAGE_D_DECLARATION.json, NOT HERE
===================================================
Every setting below must equal the declaration's "config" block or the tool
refuses to start, and the declaration's hash goes into the research ledger
BEFORE the first model is fitted. Per target, PASS needs all three:
    1. top-3 net per trade: 98.75% block-bootstrap lower bound > 0
    2. top-3 net, mean over test days from 2024-01-01 > 0
    3. top-3 excess over buying every eligible stock: 98.75% lower bound > 0
A pass makes that target's engine a forward paper-test arm. Nothing more.

WHAT HAPPENS, IN ORDER
----------------------
1. Refuse unless: the panel passed its label audit after its last build, the
   family manifest is unchanged, stage B has run, the declaration matches,
   and stage D has not already run (or a re-run reason is given).
2. Load every ELIGIBLE row (label_buyable_o1 == 1): unbuyable rows are never
   trained on, never scored, never picked.
3. Folds: feasibility_test.splits over every session whose 5-session target
   is resolved - including 2025-03-05 onward, which stage B never read.
4. For each target and fold: train on rows up to the fold's train end (the
   embargo is ASSERTED: the last training label must end before the test
   window starts), on the day-demeaned, 1/99-percentile-clipped target;
   predict every eligible row in the test window.
5. Each test day: rank eligible stocks by raw score, take the top 1/3/5/10
   (ties broken by symbol), score them with the RAW target minus 35 bp, and
   compare with buying every eligible stock that day.
6. Evidence over days: block bootstrap (block 10, 10,000 resamples). Then by
   fold, by year, 2024+, and the post-lockbox stretch (2025-03-05+).

HOW TO READ A 5-SESSION RESULT
------------------------------
Per trade is the declared unit. A top-3 bought every day and held 5 sessions
means ~15 positions open at once, so a 5-session trade's net ties capital up
five times longer than a 1-session trade's. Comparing horizons on return per
day of capital is a stage-E question.

OUTPUT: <panel>\\stage_d\\STAGED_YYYYMMDD_NNN\\
    stage_d.json        declaration hash, folds, every number, verdicts
    stage_d_report.md   the readable report
    daily.csv           per target and test day: top-1/3/5/10 gross, net, excess; market
    picks.csv           per target and test day: the top-10 with score and outcome
    preds.parquet       every test prediction (for stage E)
and ledger entries stage_d_declared (before fitting) and stage_d_gate (results).
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
import panel_build as PB         # noqa: E402,F401
import stage_b_screen as SB      # noqa: E402  (manifest, audit check, clean features, IC helpers)

CODE_VERSION = "stage_d_gate v1"
DECL_FILE = "STAGE_D_DECLARATION.json"
FAMILY_ID = SB.FAMILY_ID
ELIG_COL = SB.ELIG_COL

CFG = {
    "model": "HistGradientBoostingRegressor",
    "hgb": {"loss": "squared_error", "max_iter": 200, "learning_rate": 0.05, "max_leaf_nodes": 31,
            "min_samples_leaf": 200, "l2_regularization": 1.0, "early_stopping": False},
    "label": "demeaned_by_day_winsorized",
    "winsor_pct": [1, 99],
    "fit_cap": 400000,
    "n_splits": 5, "min_train_frac": 0.35, "embargo": 5,
    "top_n_primary": 3, "top_n_all": [1, 3, 5, 10],
    "cost_bps": 35.0,
    "interval_level": 0.9875, "boot_B": 10000, "boot_block": 10,
    "recency_start": "2024-01-01",
    "seed": 7,
}
PROFILE_COLS = ("D_atr_pct", "X_turnover_med")


class PreconditionError(SystemExit):
    pass


class EmbargoError(RuntimeError):
    pass


def _log(msg: str) -> None:
    print(f"{dt.datetime.now():%H:%M:%S}  {msg}", flush=True)


def horizon_of(target: str) -> int:
    m = re.fullmatch(r"label_oc_(\d+)", target)
    if not m:
        raise ValueError(f"not an open-to-close target: {target}")
    return int(m.group(1))


def eligible_mask(buyable: np.ndarray) -> np.ndarray:
    return buyable == 1


# ----------------------------------------------------------------------
# preconditions
# ----------------------------------------------------------------------
def load_declaration() -> Tuple[dict, str]:
    p = HERE / DECL_FILE
    if not p.exists():
        raise PreconditionError(f"{DECL_FILE} not found next to {Path(__file__).name}")
    raw = p.read_bytes()
    decl = json.loads(raw.decode("utf-8"))
    sha = hashlib.sha256(raw).hexdigest()[:16]
    want = json.loads(json.dumps(CFG))
    if decl.get("family_id") != FAMILY_ID or decl.get("stage") != "D":
        raise PreconditionError(f"{DECL_FILE} is not the stage-D declaration of {FAMILY_ID}")
    if decl.get("config") != want:
        diff = sorted(k for k in set(want) | set(decl.get("config", {}))
                      if want.get(k) != decl.get("config", {}).get(k))
        raise PreconditionError(f"the tool's settings differ from {DECL_FILE} on {diff}. The declaration "
                                f"is fixed; the code must match it, not the other way round.")
    return decl, sha


def check_order_and_rerun(pp: Path, reason: Optional[str]) -> List[dict]:
    E = RC.ledger_entries(pp)
    if not [e for e in E if e.get("kind") == "stage_b_screen" and e.get("experiment_family") == FAMILY_ID]:
        raise PreconditionError("stage B has not run on this panel - the family's order is A -> B -> D")
    prior = [e for e in E if e.get("kind") == "stage_d_gate" and e.get("experiment_family") == FAMILY_ID]
    if prior and not reason:
        raise PreconditionError(
            f"stage D has already run on this panel ({prior[-1].get('run_id')}). Re-running is allowed only to "
            f"fix a verified bug: pass --rerun-reason \"...\" - it goes into the ledger. Never re-run to get a "
            f"different answer.")
    return prior


# ----------------------------------------------------------------------
# pieces with independent tests
# ----------------------------------------------------------------------
def make_label(y: np.ndarray, day: np.ndarray, pct=(1, 99)) -> Tuple[np.ndarray, float, float]:
    """Target minus its day's mean (over the rows given), clipped at the given percentiles OF THESE ROWS."""
    y = np.asarray(y, dtype=float)
    u, inv = np.unique(day, return_inverse=True)
    s = np.bincount(inv, y, minlength=len(u))
    c = np.bincount(inv, minlength=len(u))
    dm = y - (s / c)[inv]
    lo, hi = np.percentile(dm, pct)
    return np.clip(dm, lo, hi), float(lo), float(hi)


def block_bootstrap(x, level: float, B: int, block: int, seed: int) -> Dict[str, float]:
    """Circular moving-block bootstrap interval for the mean of a date-ordered series.
    At level 0.95 this reproduces research_common.block_bootstrap_mean exactly (same draws)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    out = {"mean": float(x.mean()) if n else float("nan"), "lo": float("nan"), "hi": float("nan"),
           "n_days": int(n), "level": level, "block": block, "B": B}
    if n < 2 * block:
        return out
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    means = np.empty(B)
    step = 500
    for a in range(0, B, step):
        b = min(step, B - a)
        starts = rng.integers(0, n, size=(b, nb))
        idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(b, -1)[:, :n]
        means[a:a + b] = x[idx].mean(axis=1)
    tail = round((1 - level) / 2 * 100, 10)      # 2.5 and 0.625 exactly, not 2.5000000000000022
    out["lo"], out["hi"] = float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))
    return out


def daily_top_n(day: np.ndarray, sym: np.ndarray, score: np.ndarray, y: np.ndarray, ns) -> dict:
    """Per day: rank by raw score (desc), ties by symbol; mean outcome of each top-n over its known outcomes."""
    o = np.lexsort((sym, -score, day))
    d, s, yy = day[o], score[o], y[o]
    first = np.r_[True, d[1:] != d[:-1]]
    gstart = np.flatnonzero(first)
    days = d[gstart]
    pos = np.arange(len(d)) - np.repeat(gstart, np.diff(np.r_[gstart, len(d)]))
    gi = np.cumsum(first) - 1
    out = {"days": days, "order": o, "pos": pos, "n_names": np.bincount(gi)}
    fin = np.isfinite(yy)
    for n in ns:
        m = pos < n
        cnt = np.bincount(gi[m & fin], minlength=len(days))
        sm = np.bincount(gi[m & fin], yy[m & fin], minlength=len(days))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[f"top{n}"] = np.where(cnt > 0, sm / np.maximum(cnt, 1), np.nan)
        out[f"top{n}_missing"] = np.bincount(gi[m & ~fin], minlength=len(days))
    mc = np.bincount(gi[fin], minlength=len(days))
    ms = np.bincount(gi[fin], yy[fin], minlength=len(days))
    with np.errstate(invalid="ignore", divide="ignore"):
        out["market"] = np.where(mc > 0, ms / np.maximum(mc, 1), np.nan)
    tie = np.zeros(len(days), bool)
    k = 3
    has = out["n_names"] > k
    i3 = gstart[has] + k - 1
    tie[has] = s[i3] == s[i3 + 1]
    out["tie_at_3"] = tie
    return out


# ----------------------------------------------------------------------
# the gate
# ----------------------------------------------------------------------
def run_gate(pp: Path, targets: List[str], verbose: bool = True, cfg: Optional[dict] = None) -> dict:
    import feasibility_test as FT
    from sklearn.ensemble import HistGradientBoostingRegressor
    cfg = dict(cfg or CFG)
    t0 = time.perf_counter()

    feats, excluded = SB.clean_feature_list(pp)
    RC.assert_no_label_leak(feats, "stage_d_gate")
    base = pd.read_parquet(pp, columns=["timestamp", "symbol", ELIG_COL])
    ts = SB._naive(base["timestamp"]).to_numpy(dtype="datetime64[ns]")
    sym = base["symbol"].astype(str).to_numpy()
    buy = pd.to_numeric(base[ELIG_COL], errors="coerce").to_numpy(dtype=float)
    del base
    el = eligible_mask(buy)
    order = np.lexsort((sym, ts))
    sel = order[el[order]]
    ts_s, sym_s = ts[sel], sym[sel]
    rows_panel, rows_elig = int(len(ts)), int(len(sel))
    del ts, sym, buy, el, order
    sessions = np.unique(ts_s)
    day = np.searchsorted(sessions, ts_s)
    Y = {tg: SB._read_col(pp, tg)[sel] for tg in targets}
    last_res = min(int(day[np.isfinite(Y[tg])].max()) for tg in targets)
    fold_sessions = sessions[:last_res + 1]
    folds = FT.splits(fold_sessions, cfg["n_splits"], cfg["embargo"], cfg["min_train_frac"])
    if verbose:
        _log(f"{rows_elig:,} eligible rows of {rows_panel:,} | {len(feats)} clean features (no selection) | "
             f"sessions {pd.Timestamp(sessions[0]).date()}..{pd.Timestamp(fold_sessions[-1]).date()} "
             f"(last resolved) | loading features ...")
    X = np.empty((len(sel), len(feats)), dtype=np.float32)
    for j, f in enumerate(feats):
        X[:, j] = SB._read_col(pp, f)[sel]
    prof = {}
    for c in PROFILE_COLS:
        prof[c] = X[:, feats.index(c)] if c in feats else None
    if verbose:
        _log(f"features loaded ({X.nbytes / 1e9:.2f} GB) in {time.perf_counter() - t0:.0f}s")

    fold_meta, results, preds_parts, picks_parts, daily_parts = [], {}, [], [], []
    sidx = {np.datetime64(s, "ns"): i for i, s in enumerate(sessions)}
    for tg in targets:
        h = horizon_of(tg)
        y = Y[tg]
        score = np.full(len(sel), np.nan)
        fold_of_day = np.zeros(len(sessions), int)
        for f in folds:
            fo = int(f["fold"])
            tr_end = sidx[np.datetime64(pd.Timestamp(f["train_end"]), "ns")]
            te0 = sidx[np.datetime64(pd.Timestamp(f["test_start"]), "ns")]
            te1 = sidx[np.datetime64(pd.Timestamp(f["test_end"]), "ns")]
            fold_of_day[te0:te1 + 1] = fo
            tr = np.flatnonzero((day <= tr_end) & np.isfinite(y))
            if len(tr) == 0:
                raise RuntimeError(f"{tg} fold {fo}: no training rows")
            gap = te0 - int(day[tr].max())
            if gap <= h:
                raise EmbargoError(f"{tg} fold {fo}: last training row is {gap} sessions before the test window; "
                                   f"its {h}-session label would reach into it")
            lab, lo, hi = make_label(y[tr], day[tr], cfg["winsor_pct"])
            rng = np.random.default_rng([cfg["seed"], fo, h])
            keep = (np.arange(len(tr)) if len(tr) <= cfg["fit_cap"]
                    else np.sort(rng.choice(len(tr), cfg["fit_cap"], replace=False)))
            fit_i, fit_lab = tr[keep], lab[keep]
            t1 = time.perf_counter()
            model = HistGradientBoostingRegressor(random_state=cfg["seed"], **cfg["hgb"])
            model.fit(X[fit_i], fit_lab)
            te = np.flatnonzero((day >= te0) & (day <= te1))
            for a in range(0, len(te), 200_000):
                b = te[a:a + 200_000]
                score[b] = model.predict(X[b])
            fm = {"target": tg, "fold": fo, "train_end": str(pd.Timestamp(f["train_end"]).date()),
                  "test_start": str(pd.Timestamp(f["test_start"]).date()),
                  "test_end": str(pd.Timestamp(f["test_end"]).date()), "train_rows": int(len(tr)),
                  "fit_rows": int(len(fit_i)), "test_rows": int(len(te)), "embargo_gap_sessions": int(gap),
                  "label_clip_bp": [lo * 1e4, hi * 1e4], "seconds": round(time.perf_counter() - t1, 1)}
            fold_meta.append(fm)
            if verbose:
                _log(f"{tg} fold {fo}: trained on {len(fit_i):,} of {len(tr):,} rows to {fm['train_end']}, "
                     f"scored {len(te):,} rows {fm['test_start']}..{fm['test_end']} ({fm['seconds']:.0f}s)")
            del model

        tm = np.isfinite(score)
        dt_, st_, yt_, symt_ = day[tm], score[tm], y[tm], sym_s[tm]
        D = daily_top_n(dt_, symt_, st_, yt_, cfg["top_n_all"])
        dates = pd.DatetimeIndex(sessions[D["days"]])
        cost = cfg["cost_bps"] / 1e4
        daily = pd.DataFrame({"timestamp": dates, "target": tg, "fold": fold_of_day[D["days"]],
                              "n_names": D["n_names"], "market": D["market"], "tie_at_3": D["tie_at_3"]})
        for n in cfg["top_n_all"]:
            daily[f"top{n}_gross"] = D[f"top{n}"]
            daily[f"top{n}_net"] = D[f"top{n}"] - cost
            daily[f"top{n}_excess"] = D[f"top{n}"] - D["market"]
            daily[f"top{n}_missing"] = D[f"top{n}_missing"]
        daily_parts.append(daily)

        o, pos = D["order"], D["pos"]
        top10 = pos < 10
        picks_parts.append(pd.DataFrame({
            "timestamp": sessions[dt_[o][top10]], "target": tg, "rank": pos[top10] + 1,
            "symbol": symt_[o][top10], "score": st_[o][top10], "outcome": yt_[o][top10],
            "net": yt_[o][top10] - cost}))
        preds_parts.append(pd.DataFrame({"timestamp": sessions[dt_], "symbol": symt_, "target": tg,
                                         "score": st_.astype("float32"), "outcome": yt_.astype("float32")}))

        fin = np.isfinite(yt_)
        ic, _ = SB.daily_ic(dt_[fin], SB.group_rank(dt_[fin], st_[fin]), SB.group_rank(dt_[fin], yt_[fin]),
                            len(sessions), SB.CFG["min_names"])
        ic_day = ic[D["days"]]
        results[tg] = evaluate(daily, ic_day, cfg)
        if all(v is not None for v in prof.values()):
            top3 = np.zeros(len(dt_), bool)
            top3[o[pos < 3]] = True
            pr = {}
            for c, v in prof.items():
                vt = v[tm]
                pr[c] = {"top3_median": float(np.nanmedian(vt[top3])), "all_median": float(np.nanmedian(vt))}
            results[tg]["pick_profile"] = pr
        if verbose:
            r = results[tg]
            _log(f"{tg}: top-3 net {r['top3_net']['mean'] * 1e4:+.1f} bp "
                 f"[{r['top3_net']['lo'] * 1e4:+.1f}, {r['top3_net']['hi'] * 1e4:+.1f}] -> "
                 f"{'PASS' if r['verdict']['pass'] else 'FAIL'}")
        del score

    return {"results": results, "folds": fold_meta, "features": feats, "excluded": excluded,
            "daily": pd.concat(daily_parts, ignore_index=True),
            "picks": pd.concat(picks_parts, ignore_index=True),
            "preds": pd.concat(preds_parts, ignore_index=True),
            "rows": {"panel": rows_panel, "eligible": rows_elig},
            "sessions": {"first": str(pd.Timestamp(sessions[0]).date()),
                         "last_resolved": str(pd.Timestamp(fold_sessions[-1]).date())},
            "seconds": round(time.perf_counter() - t0, 1), "config": cfg}


def evaluate(daily: pd.DataFrame, ic_day: np.ndarray, cfg: dict) -> dict:
    lvl, B, blk, seed = cfg["interval_level"], cfg["boot_B"], cfg["boot_block"], cfg["seed"]
    d = daily.sort_values("timestamp")
    net3, ex3 = d["top3_net"].to_numpy(), d["top3_excess"].to_numpy()
    r = {"top3_net": block_bootstrap(net3, lvl, B, blk, seed),
         "top3_net_95": block_bootstrap(net3, 0.95, B, blk, seed),
         "top3_excess": block_bootstrap(ex3, lvl, B, blk, seed),
         "top3_excess_95": block_bootstrap(ex3, 0.95, B, blk, seed),
         "market_net": block_bootstrap(d["market"].to_numpy() - cfg["cost_bps"] / 1e4, 0.95, B, blk, seed)}
    rec = d["timestamp"] >= pd.Timestamp(cfg["recency_start"])
    r["recent_top3_net"] = block_bootstrap(d.loc[rec, "top3_net"].to_numpy(), 0.95, B, blk, seed)
    r["recent_top3_excess"] = block_bootstrap(d.loc[rec, "top3_excess"].to_numpy(), 0.95, B, blk, seed)
    pl = d["timestamp"] >= pd.Timestamp(SB.LOCKBOX_START_PINNED)
    r["post_lockbox"] = {"days": int(pl.sum()), "top3_net_mean": float(d.loc[pl, "top3_net"].mean()),
                         "top3_excess_mean": float(d.loc[pl, "top3_excess"].mean()),
                         "market_net_mean": float((d.loc[pl, "market"] - cfg["cost_bps"] / 1e4).mean())}
    r["top_n"] = {int(n): {"net": block_bootstrap(d[f"top{n}_net"].to_numpy(), 0.95, B, blk, seed),
                           "excess_mean": float(d[f"top{n}_excess"].mean())} for n in cfg["top_n_all"]}
    d = d.assign(ic=ic_day, year=d["timestamp"].dt.year)
    r["by_fold"] = {int(k): {"days": int(len(g)), "top3_net": float(g["top3_net"].mean()),
                             "top3_excess": float(g["top3_excess"].mean()),
                             "market_net": float((g["market"] - cfg["cost_bps"] / 1e4).mean()),
                             "rank_ic": float(np.nanmean(g["ic"]))} for k, g in d.groupby("fold")}
    r["by_year"] = {int(k): {"days": int(len(g)), "top3_net": float(g["top3_net"].mean()),
                             "top3_excess": float(g["top3_excess"].mean()),
                             "market_net": float((g["market"] - cfg["cost_bps"] / 1e4).mean())}
                    for k, g in d.groupby("year")}
    r["rank_ic_mean"] = float(np.nanmean(ic_day))
    r["tie_days_at_3"] = float(d["tie_at_3"].mean())
    r["top3_missing_outcomes"] = int(d["top3_missing"].sum())
    v = {"net_lower_bound_above_zero": bool(np.isfinite(r["top3_net"]["lo"]) and r["top3_net"]["lo"] > 0),
         "recent_mean_above_zero": bool(np.isfinite(r["recent_top3_net"]["mean"])
                                        and r["recent_top3_net"]["mean"] > 0),
         "excess_lower_bound_above_zero": bool(np.isfinite(r["top3_excess"]["lo"]) and r["top3_excess"]["lo"] > 0)}
    v["pass"] = all(v.values())
    r["verdict"] = v
    return r


# ----------------------------------------------------------------------
# output
# ----------------------------------------------------------------------
def _next_run_id(out_root: Path) -> str:
    day = dt.date.today().strftime("%Y%m%d")
    n = 1
    while (out_root / f"STAGED_{day}_{n:03d}").exists():
        n += 1
    return f"STAGED_{day}_{n:03d}"


def _bp(x) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x * 1e4:+.1f}"


def _iv(b: dict) -> str:
    return f"{_bp(b['mean'])} [{_bp(b['lo'])}, {_bp(b['hi'])}]"


def write_report(res: dict, targets: List[str], run_id: str, path: Path, decl_sha: str, man_sha: str) -> None:
    R, cfg = res["results"], res["config"]
    L = [f"# Stage D gate - {run_id}", "",
         f"Family {FAMILY_ID} (manifest sha {man_sha}) | declaration sha {decl_sha} | {CODE_VERSION} | "
         f"{dt.datetime.now():%Y-%m-%d %H:%M}", "",
         "**Rule (declared before any fit):** per target, PASS needs (1) top-3 net per trade with a 98.75% "
         "block-bootstrap lower bound above zero, (2) the mean top-3 net from 2024-01-01 above zero, and (3) "
         "top-3 excess over buying every eligible stock with a 98.75% lower bound above zero.", "",
         f"Model: {cfg['model']} (fixed settings, no tuning), {len(res['features'])} clean features with no "
         f"selection, trained on the day-demeaned target clipped at the fold's 1st/99th percentiles; picks "
         f"ranked by raw score. Entry: open T+1; exit: close T+h; cost {cfg['cost_bps']:.0f} bp per round trip. "
         f"Eligible rows only ({res['rows']['eligible']:,} of {res['rows']['panel']:,}). Sessions "
         f"{res['sessions']['first']}..{res['sessions']['last_resolved']}.", "",
         "## 1. The gate", "",
         "| target | top-3 net (98.75%) | top-3 excess (98.75%) | net from 2024 (95%) | buy-everything net | "
         "rank IC | VERDICT |", "|---|---|---|---|---|---|---|"]
    for tg in targets:
        r = R[tg]
        v = r["verdict"]
        why = [] if v["pass"] else [k for k, ok in (("net interval", v["net_lower_bound_above_zero"]),
                                                     ("2024+ mean", v["recent_mean_above_zero"]),
                                                     ("excess interval", v["excess_lower_bound_above_zero"]))
                                    if not ok]
        L.append(f"| {tg} | {_iv(r['top3_net'])} | {_iv(r['top3_excess'])} | {_iv(r['recent_top3_net'])} | "
                 f"{_bp(r['market_net']['mean'])} | {r['rank_ic_mean']:+.4f} | "
                 f"**{'PASS' if v['pass'] else 'FAIL'}**{'' if v['pass'] else ' (' + ', '.join(why) + ')'} |")
    L += ["", "bp per trade. Intervals: circular block bootstrap over trading days, block "
          f"{cfg['boot_block']}, {cfg['boot_B']:,} resamples.", "",
          "## 2. Top-N (descriptive; 95% intervals)", "",
          "| target | " + " | ".join(f"top-{n} net" for n in cfg["top_n_all"]) + " | "
          + " | ".join(f"top-{n} excess" for n in cfg["top_n_all"]) + " |",
          "|---|" + "---|" * (2 * len(cfg["top_n_all"]))]
    for tg in targets:
        tn = R[tg]["top_n"]
        L.append(f"| {tg} | " + " | ".join(_iv(tn[n]["net"]) for n in cfg["top_n_all"]) + " | "
                 + " | ".join(_bp(tn[n]["excess_mean"]) for n in cfg["top_n_all"]) + " |")
    L += ["", "## 3. By fold (top-3 net / excess / buy-everything net, bp; rank IC)", ""]
    for tg in targets:
        bf = R[tg]["by_fold"]
        L.append(f"- **{tg}**: " + "; ".join(
            f"F{k} {_bp(v['top3_net'])} / {_bp(v['top3_excess'])} / {_bp(v['market_net'])} (IC {v['rank_ic']:+.4f})"
            for k, v in sorted(bf.items())))
    L += ["", "## 4. By year (top-3 net / excess, bp)", ""]
    for tg in targets:
        by = R[tg]["by_year"]
        L.append(f"- **{tg}**: " + "; ".join(f"{k} {_bp(v['top3_net'])} / {_bp(v['top3_excess'])}"
                                             for k, v in sorted(by.items())))
    L += ["", "## 5. The stretch stage B never read (2025-03-05 onward)", ""]
    for tg in targets:
        p = R[tg]["post_lockbox"]
        L.append(f"- **{tg}**: {p['days']} days | top-3 net {_bp(p['top3_net_mean'])} | excess "
                 f"{_bp(p['top3_excess_mean'])} | buy-everything net {_bp(p['market_net_mean'])}")
    L += ["", "## 6. Picks and mechanics", ""]
    for tg in targets:
        r = R[tg]
        pp_ = r.get("pick_profile")
        prof = ("" if not pp_ else " | " + "; ".join(
            f"{c} top-3 median {v['top3_median']:.4g} vs all {v['all_median']:.4g}" for c, v in pp_.items()))
        L.append(f"- **{tg}**: days where the 3rd and 4th scores tie: {r['tie_days_at_3']:.2%} | top-3 picks with "
                 f"no outcome (no bar at T+h): {r['top3_missing_outcomes']}{prof}")
    L += ["", "## 7. What this does not say", "",
          "- A PASS makes that target a forward paper-test arm. No money before the month-6 check.",
          "- The walk-forward spans history studied in earlier stages (with other targets); only the forward "
          "paper ledger is untouched evidence.",
          "- Fills: open-auction prices assumed at the open; exits locked at the lower band and slippage are "
          "stage E.",
          "- Per trade is the declared unit. A 5-session hold ties capital up 5x longer than a 1-session hold; "
          "return per day of capital is a stage-E comparison.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage D gate for family oc_v1")
    ap.add_argument("--panel", required=True, help="panel folder, e.g. %%CACHE_DAILY_ROOT%%\\panel_oc")
    ap.add_argument("--rerun-reason", default=None,
                    help="required if stage D already ran on this panel; logged in the ledger")
    a = ap.parse_args(argv)
    pp = Path(a.panel) / "panel.parquet"

    man, man_sha = SB.load_manifest()
    targets = list(man["targets"].keys())
    info = SB.check_panel(pp, man_sha)
    prior = check_order_and_rerun(pp, a.rerun_reason)
    decl, decl_sha = load_declaration()
    out_root = pp.parent / "stage_d"
    run_id = _next_run_id(out_root)
    out = out_root / run_id
    out.mkdir(parents=True, exist_ok=False)
    RC.ledger_append(pp, {"kind": "stage_d_declared", "tool": CODE_VERSION, "run_id": run_id,
                          "experiment_family": FAMILY_ID, "manifest_sha": man_sha, "declaration_sha": decl_sha,
                          "rerun_reason": a.rerun_reason})
    print("=" * 76)
    print(f"{CODE_VERSION} | family {FAMILY_ID} | declaration sha {decl_sha} | panel {pp}")
    print(f"panel: {info['panel_build_version']} built {info['built_at']}, audited {info['audit_at']}")
    print("RULE (declared before any fit), per target, PASS needs all three:")
    print("  1. top-3 net per trade: 98.75% lower bound > 0")
    print("  2. top-3 net, mean from 2024-01-01 > 0")
    print("  3. top-3 excess over buying every eligible stock: 98.75% lower bound > 0")
    print("=" * 76)

    res = run_gate(pp, targets)
    R = res["results"]
    res["daily"].to_csv(out / "daily.csv", index=False)
    res["picks"].to_csv(out / "picks.csv", index=False)
    res["preds"].to_parquet(out / "preds.parquet", index=False)
    js = {"run_id": run_id, "tool": CODE_VERSION, "experiment_family": FAMILY_ID, "manifest_sha": man_sha,
          "declaration_sha": decl_sha, "panel": str(pp), "panel_info": info, "rerun_reason": a.rerun_reason,
          "prior_runs": [p.get("run_id") for p in prior], "results": R, "folds": res["folds"],
          "features": res["features"], "excluded": res["excluded"], "rows": res["rows"],
          "sessions": res["sessions"], "seconds": res["seconds"], "config": res["config"]}
    (out / "stage_d.json").write_text(json.dumps(js, indent=2, default=str), encoding="utf-8")
    write_report(res, targets, run_id, out / "stage_d_report.md", decl_sha, man_sha)
    passed = [tg for tg in targets if R[tg]["verdict"]["pass"]]
    n = RC.ledger_append(pp, {"kind": "stage_d_gate", "tool": CODE_VERSION, "run_id": run_id,
                              "experiment_family": FAMILY_ID, "manifest_sha": man_sha, "declaration_sha": decl_sha,
                              "passed": passed,
                              "top3_net_bp": {tg: [R[tg]["top3_net"][k] * 1e4 for k in ("mean", "lo", "hi")]
                                              for tg in targets},
                              "rerun_reason": a.rerun_reason})

    print()
    print(f"{'target':<12}{'top-3 net [98.75%]':>28}{'excess [98.75%]':>26}{'net 2024+':>11}{'mkt net':>9}"
          f"{'IC':>8}  verdict")
    for tg in targets:
        r = R[tg]
        print(f"{tg:<12}{_iv(r['top3_net']):>28}{_iv(r['top3_excess']):>26}{_bp(r['recent_top3_net']['mean']):>11}"
              f"{_bp(r['market_net']['mean']):>9}{r['rank_ic_mean']:>+8.4f}  "
              f"{'PASS' if r['verdict']['pass'] else 'FAIL'}")
    print(f"\n  outputs: {out}")
    print(f"  research ledger: entry {n} (stage_d_gate {run_id})")
    print("=" * 76)
    if passed:
        print(f"STAGE D: {', '.join(passed)} PASS the declared gate -> forward paper-test arm(s) only. "
              f"No real money before the month-6 check.")
    else:
        print("STAGE D: no target passes the declared gate. Do not trade.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
