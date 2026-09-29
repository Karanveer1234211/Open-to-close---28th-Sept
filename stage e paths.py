#!/usr/bin/env python3
"""
stage_e_paths.py - look INSIDE every trade of the frozen engine.

    python stage_e_paths.py --panel %CACHE_DAILY_ROOT%\\panel_oc --root %CACHE_DAILY_ROOT%

Answers, per trade of the primary rule (engine D, top 3 a day, bought at the
next open), from daily bars over 10 sessions:
  * MFE / MAE - the best and worst point during the 5-session hold, in % and
    in ATR units, and the session each happens;
  * the average path - where the return is made, session by session, split
    into the overnight gap and the trading session;
  * time to target - how often +2% ... +20% is reached, and in how many sessions;
  * stops - how often -3% ... -10% is touched, and how many eventual WINNERS
    touch it first (the ones a stop would have killed);
  * target first or stop first - 36 brackets (% and ATR levels), each scored
    against the plain 5-session exit, with the ambiguous same-day cases counted;
  * trailing stops; giveback (how much of the best price is kept); by rank.

Rules and levels: PATHS_DECLARATION.json (fixed before any path existed).
Same-day target+stop: CONSERVATIVE = stop first (headline), OPTIMISTIC =
target first (bound). A stop cannot fill on a frozen lower-circuit session.
The plain exit is the forensics rule (a close locked at the lower band sells
at the first later open that is not locked). 35 bp per round trip throughout.

This is descriptive. The engines stay frozen: changing the exit rule would be
a new engine variant, judged forward - never by these tables.

OUTPUT: <panel>\\stage_e\\PATHS_YYYYMMDD_NNN\\ paths_report.md, trades_paths.csv,
brackets.csv, time_to_target.csv, stops.csv, path_by_session.csv, stats.json;
ledger stage_e_paths_declared (before) and stage_e_paths_run (after).
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
import freeze_engines as FZ        # noqa: E402
import stage_e_execution as EX     # noqa: E402  (bars, the honest exit, the lock test, watchlists)

CODE_VERSION = "stage_e_paths v1"
DECL_FILE = "PATHS_DECLARATION.json"
CFG = {"engine": "D", "target": "label_oc_5", "picks_per_day": 3, "hold": 5, "window": 10, "reserve_depth": 20,
       "cost_bps": 35.0, "pct_targets": [0.02, 0.03, 0.05, 0.08, 0.10, 0.15], "pct_stops": [0.03, 0.05, 0.08, 0.10],
       "atr_targets": [1.0, 1.5, 2.0, 3.0], "atr_stops": [1.0, 1.5, 2.0], "trailing": [0.05, 0.08, 0.12],
       "hit_levels": [0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20], "atr_window": 14}


class PreconditionError(SystemExit):
    pass


def _log(msg: str) -> None:
    print(f"{dt.datetime.now():%H:%M:%S}  {msg}", flush=True)


def load_declaration() -> Tuple[dict, str]:
    p = HERE / DECL_FILE
    if not p.exists():
        raise PreconditionError(f"{DECL_FILE} not found next to {Path(__file__).name}")
    raw = p.read_bytes()
    decl = json.loads(raw.decode("utf-8"))
    if decl.get("stage") != "E-paths" or decl.get("config") != json.loads(json.dumps(CFG)):
        raise PreconditionError(f"settings differ from {DECL_FILE}; the declaration is fixed")
    return decl, hashlib.sha256(raw).hexdigest()[:16]


# ----------------------------------------------------------------------
# single-trade mechanics (all on daily bars; e = the entry bar, bought at its open)
# ----------------------------------------------------------------------
def frozen_lower(b: EX.Bars, j: int) -> bool:
    return bool(b.o[j] == b.h[j] == b.l[j] == b.c[j] and EX.lower_locked(b, j, "close"))


def _late_sell(b: EX.Bars, start: int, P: float, search: int = 20) -> Tuple[float, str]:
    """First session from `start` that is not frozen at the lower band: sell at its open."""
    n = len(b.c)
    for j in range(start, min(start + search, n)):
        if not frozen_lower(b, j):
            return b.o[j] / P - 1, "stop_late"
    j = min(start + search, n) - 1
    return b.c[j] / P - 1, "stop_late"


def time_exit(b: EX.Bars, e: int, hold: int) -> float:
    x, kind, raw, fl = EX.plan_exit(b, e, hold, None)
    return raw / b.o[e] - 1


def bracket(b: EX.Bars, e: int, hold: int, tp: float, sl: float, tie: str) -> Tuple[float, str, bool]:
    """Target at +tp, stop at -sl (fractions of the entry open), time exit after `hold` sessions."""
    n = len(b.c)
    P = b.o[e]
    TP, SL = P * (1 + tp), P * (1 - sl)
    for k in range(hold):
        j = e + k
        if j >= n:
            break
        frz = frozen_lower(b, j)
        if k > 0:
            if b.o[j] >= TP:
                return b.o[j] / P - 1, "target_gap", False
            if b.o[j] <= SL:
                if frz:
                    return (*_late_sell(b, j + 1, P), False)
                return b.o[j] / P - 1, "stop_gap", False
        hit_tp = b.h[j] >= TP
        hit_sl = b.l[j] <= SL
        if hit_sl and frz:
            return (*_late_sell(b, j + 1, P), False)
        if hit_tp and hit_sl:
            return (-sl, "stop", True) if tie == "conservative" else (tp, "target", True)
        if hit_tp:
            return tp, "target", False
        if hit_sl:
            return -sl, "stop", False
    return time_exit(b, e, hold), "time", False


def trailing(b: EX.Bars, e: int, hold: int, x: float) -> Tuple[float, str]:
    """Stop at (highest close so far, entry price included) x (1 - x); time exit after `hold` sessions."""
    n = len(b.c)
    P = b.o[e]
    peak = P
    for k in range(hold):
        j = e + k
        if j >= n:
            break
        stop = peak * (1 - x)
        frz = frozen_lower(b, j)
        if k > 0 and b.o[j] <= stop:
            return _late_sell(b, j + 1, P) if frz else (b.o[j] / P - 1, "trail_gap")
        if b.l[j] <= stop:
            return _late_sell(b, j + 1, P) if frz else (stop / P - 1, "trail")
        peak = max(peak, b.c[j])
    return time_exit(b, e, hold), "time"


def atr_frac(b: EX.Bars, e: int, n: int) -> float:
    """Mean true range over the n bars ending at the signal day (bar e-1), as a fraction of its close."""
    if e - n < 1:
        return float("nan")
    s = slice(e - n, e)
    pc = b.c[e - n - 1:e - 1]
    tr = np.fmax(b.h[s] - b.l[s], np.fmax(np.abs(b.h[s] - pc), np.abs(b.l[s] - pc)))
    return float(tr.mean() / b.c[e - 1])


def path_record(b: EX.Bars, e: int, cfg: dict) -> dict:
    n = len(b.c)
    P = b.o[e]
    H, W = cfg["hold"], cfg["window"]
    hi_h = b.h[e:min(e + H, n)]
    lo_h = b.l[e:min(e + H, n)]
    hi_w = b.h[e:min(e + W, n)]
    lo_w = b.l[e:min(e + W, n)]
    atr = atr_frac(b, e, cfg["atr_window"])
    rec = {"entry": P, "atr_frac": atr, "final": time_exit(b, e, H),
           "mfe": hi_h.max() / P - 1, "mae": lo_h.min() / P - 1,
           "day_mfe": int(np.argmax(hi_h)) + 1, "day_mae": int(np.argmin(lo_h)) + 1,
           "mfe_10": hi_w.max() / P - 1, "mae_10": lo_w.min() / P - 1}
    for k in range(W):
        j = e + k
        rec[f"close_{k + 1}"] = b.c[j] / P - 1 if j < n else np.nan
        rec[f"overnight_{k + 1}"] = (b.o[j] / b.c[j - 1] - 1) if (k > 0 and j < n) else (0.0 if k == 0 else np.nan)
        rec[f"session_{k + 1}"] = b.c[j] / b.o[j] - 1 if j < n else np.nan
    for L in cfg["hit_levels"]:
        hit = np.flatnonzero(hi_w >= P * (1 + L))
        rec[f"hit_{L}"] = int(hit[0]) + 1 if len(hit) else np.nan
    for L in cfg["pct_stops"]:
        hit = np.flatnonzero(lo_w[:H] <= P * (1 - L))
        rec[f"touch_{L}"] = int(hit[0]) + 1 if len(hit) else np.nan
    return rec


# ----------------------------------------------------------------------
# picks and the study
# ----------------------------------------------------------------------
def build_picks(lists: Dict, bars: Dict[str, EX.Bars], cal: np.ndarray, n: int, depth: int) -> List[dict]:
    pos = {d: i for i, d in enumerate(cal)}
    out = []
    for d in sorted(lists):
        i = pos.get(d)
        if i is None or i + 1 >= len(cal):
            continue
        day = cal[i + 1]
        taken = 0
        for r_, s in enumerate(lists[d][:depth], 1):
            if taken >= n:
                break
            b = bars.get(s)
            e = b.index.get(day) if b is not None else None
            if e is None or not (b.o[e] > 0) or e < 1:
                continue
            out.append({"signal_day": d, "entry_day": day, "symbol": s, "rank": r_, "e": e})
            taken += 1
    return out


def bracket_set(cfg: dict) -> List[Tuple[str, str, float, float]]:
    out = [("pct", f"TP +{tp:.0%} / SL -{sl:.0%}", tp, sl) for tp in cfg["pct_targets"] for sl in cfg["pct_stops"]]
    out += [("atr", f"TP {tp:g} ATR / SL {sl:g} ATR", tp, sl) for tp in cfg["atr_targets"] for sl in cfg["atr_stops"]]
    return out


def run_study(data: dict, cfg: dict, engine: str = "D", verbose=True) -> dict:
    t0 = time.perf_counter()
    picks = build_picks(data["lists"][engine], data["bars"], data["cal"], cfg["picks_per_day"], cfg["reserve_depth"])
    cost = cfg["cost_bps"] / 1e4
    B = bracket_set(cfg)
    rows, brk = [], []
    for p in picks:
        b = data["bars"][p["symbol"]]
        rec = {"signal_day": p["signal_day"], "entry_day": p["entry_day"], "symbol": p["symbol"], "rank": p["rank"],
               **path_record(b, p["e"], cfg)}
        atr = rec["atr_frac"]
        for kind, name, tp, sl in B:
            if kind == "atr":
                if not np.isfinite(atr) or atr <= 0:
                    continue
                tp_, sl_ = tp * atr, sl * atr
            else:
                tp_, sl_ = tp, sl
            rc, how, amb = bracket(b, p["e"], cfg["hold"], tp_, sl_, "conservative")
            ro, _, _ = bracket(b, p["e"], cfg["hold"], tp_, sl_, "optimistic")
            brk.append((len(rows), name, rc, ro, how, amb))
        for x in cfg["trailing"]:
            r_, how = trailing(b, p["e"], cfg["hold"], x)
            brk.append((len(rows), f"trailing {x:.0%} from the highest close", r_, r_, how, False))
        rows.append(rec)
    T = pd.DataFrame(rows)
    T["net"] = T["final"] - cost
    BR = pd.DataFrame(brk, columns=["i", "rule", "ret_cons", "ret_opt", "how", "ambiguous"])
    BR["year"] = pd.DatetimeIndex(T["signal_day"].to_numpy())[BR["i"].to_numpy()].year
    BR["base"] = T["final"].to_numpy()[BR["i"].to_numpy()]
    if verbose:
        _log(f"{engine}: {len(T):,} trades, {len(BR):,} rule outcomes ({time.perf_counter() - t0:.0f}s)")
    return {"trades": T, "rules": BR}


def summarize(study: dict, cfg: dict) -> dict:
    T, BR = study["trades"], study["rules"]
    cost = cfg["cost_bps"] / 1e4
    q = lambda s: {f"p{p}": float(np.nanpercentile(s, p)) for p in (5, 25, 50, 75, 95)} | {"mean": float(np.nanmean(s))}
    atr_ok = T["atr_frac"] > 0
    exc = {"mfe": q(T["mfe"]), "mae": q(T["mae"]), "mfe_atr": q((T["mfe"] / T["atr_frac"])[atr_ok]),
           "mae_atr": q((T["mae"] / T["atr_frac"])[atr_ok]), "final": q(T["final"]),
           "day_mfe_share": {int(k): float(v) for k, v in T["day_mfe"].value_counts(normalize=True).sort_index().items()},
           "day_mae_share": {int(k): float(v) for k, v in T["day_mae"].value_counts(normalize=True).sort_index().items()}}
    W = cfg["window"]
    path = pd.DataFrame({"session": range(1, W + 1),
                         "mean_close_ret": [T[f"close_{k}"].mean() for k in range(1, W + 1)],
                         "median_close_ret": [T[f"close_{k}"].median() for k in range(1, W + 1)],
                         "mean_overnight": [T[f"overnight_{k}"].mean() for k in range(1, W + 1)],
                         "mean_session": [T[f"session_{k}"].mean() for k in range(1, W + 1)],
                         "share_up": [(T[f"close_{k}"] > 0).mean() for k in range(1, W + 1)]})
    ttt = []
    for L in cfg["hit_levels"]:
        h = T[f"hit_{L}"]
        hit5 = h <= cfg["hold"]
        ttt.append({"level": L, "hit_within_hold": float(hit5.mean()), "hit_within_window": float(h.notna().mean()),
                    "median_sessions_to_hit": float(h.median()) if h.notna().any() else float("nan"),
                    "hit_on_session_1": float((h == 1).mean()),
                    "final_net_if_hit_bp": float((T.loc[hit5, "final"].mean() - cost) * 1e4) if hit5.any() else float("nan"),
                    "final_net_if_not_bp": float((T.loc[~hit5, "final"].mean() - cost) * 1e4) if (~hit5).any() else float("nan")})
    stops = []
    win = T["final"] > 0
    for L in cfg["pct_stops"]:
        t_ = T[f"touch_{L}"].notna()
        stops.append({"stop": L, "touched": float(t_.mean()), "winners_touching_first": float((t_ & win).sum() / max(win.sum(), 1)),
                      "losers_touching": float((t_ & ~win).sum() / max((~win).sum(), 1)),
                      "final_net_if_touched_bp": float((T.loc[t_, "final"].mean() - cost) * 1e4) if t_.any() else float("nan")})
    base_by_year = pd.Series(T["final"].to_numpy(), index=pd.DatetimeIndex(T["signal_day"].to_numpy()).year).groupby(level=0).mean()
    rules = []
    for rule, g in BR.groupby("rule", sort=False):
        yr = g.groupby("year")["ret_cons"].mean()
        by = (g.groupby("year")["base"].mean())
        beats = int((yr > by.reindex(yr.index)).sum())
        vc = g["how"].value_counts(normalize=True)
        rules.append({"rule": rule, "trades": int(len(g)),
                      "net_cons_bp": float((g["ret_cons"].mean() - cost) * 1e4),
                      "net_opt_bp": float((g["ret_opt"].mean() - cost) * 1e4),
                      "vs_time_exit_bp": float((g["ret_cons"].mean() - g["base"].mean()) * 1e4),
                      "years_beating_time_exit": f"{beats}/{len(yr)}",
                      "target_first": float(vc.filter(like="target").sum()), "stop_first": float(vc.filter(like="stop").sum() + vc.filter(like="trail").sum()),
                      "time_exit": float(vc.get("time", 0.0)), "ambiguous": float(g["ambiguous"].mean())})
    give = {"mean_mfe": float(T["mfe"].mean()), "mean_final": float(T["final"].mean()),
            "capture": float(T["final"].mean() / T["mfe"].mean()) if T["mfe"].mean() > 0 else float("nan"),
            "up3_closed_negative": float(((T["mfe"] >= 0.03) & (T["final"] < 0)).sum() / max((T["mfe"] >= 0.03).sum(), 1)),
            "up5_closed_below_1pct": float(((T["mfe"] >= 0.05) & (T["final"] < 0.01)).sum() / max((T["mfe"] >= 0.05).sum(), 1))}
    by_rank = T.groupby("rank").agg(trades=("final", "size"), mfe=("mfe", "mean"), mae=("mae", "mean"),
                                    final_net=("net", "mean")).reset_index()
    return {"excursions": exc, "path": path, "time_to_target": pd.DataFrame(ttt), "stops": pd.DataFrame(stops),
            "rules": pd.DataFrame(rules), "giveback": give, "by_rank": by_rank,
            "time_exit_net_bp": float(T["net"].mean() * 1e4), "trades": int(len(T)),
            "base_by_year": {int(k): float(v) for k, v in base_by_year.items()}}


def _pc(x, d=1):
    return "nan" if x is None or not np.isfinite(x) else f"{x * 100:+.{d}f}%"


def write_report(path: Path, S: dict, ens: Optional[dict], sha: str, frozen: dict) -> None:
    E = S["excursions"]
    L = ["# Inside the trades - frozen engine D, top 3 a day", "",
         f"{CODE_VERSION} | declaration {sha} | D = {frozen['D']} | {S['trades']:,} trades | 35 bp per round trip", "",
         "**Read this first.** Descriptive only. Every level was fixed before any path was computed and every one is "
         "shown. The engine stays frozen: changing the exit rule would be a new engine variant, judged forward.", "",
         f"The current rule (sell at the 5th close, honest exits) nets **{S['time_exit_net_bp']:+.1f} bp** per trade here.", "",
         "## 1. How far trades run for (MFE) and against (MAE) you, during the 5 sessions", "",
         "| | 5th pct | 25th | median | 75th | 95th | mean |", "|---|---|---|---|---|---|---|"]
    for k, lab in (("mfe", "best point (MFE)"), ("mae", "worst point (MAE)"), ("final", "where it ends (5th close)"),
                   ("mfe_atr", "MFE in ATRs"), ("mae_atr", "MAE in ATRs")):
        v = E[k]
        f = (lambda x: f"{x:+.2f}") if k.endswith("atr") else _pc
        L.append(f"| {lab} | {f(v['p5'])} | {f(v['p25'])} | {f(v['p50'])} | {f(v['p75'])} | {f(v['p95'])} | {f(v['mean'])} |")
    L += ["", "Session of the best point: " + ", ".join(f"S{k} {v:.0%}" for k, v in E["day_mfe_share"].items())
          + "  |  session of the worst point: " + ", ".join(f"S{k} {v:.0%}" for k, v in E["day_mae_share"].items()), "",
          "## 2. Where the return is made (average path, % from the entry price)", "",
          "| session | mean at close | median | share up | mean overnight gap | mean in-session |", "|---|---|---|---|---|---|"]
    for _, r in S["path"].iterrows():
        L.append(f"| {int(r['session'])}{' (exit)' if r['session'] == 5 else ''} | {_pc(r['mean_close_ret'], 2)} | "
                 f"{_pc(r['median_close_ret'], 2)} | {r['share_up']:.0%} | {_pc(r['mean_overnight'], 2)} | {_pc(r['mean_session'], 2)} |")
    L += ["", "## 3. Time to target: how often +X% is touched, and how fast", "",
          "| target | touched within 5 | within 10 | median sessions | on session 1 | net if touched (bp, plain exit) | net if not |",
          "|---|---|---|---|---|---|---|"]
    for _, r in S["time_to_target"].iterrows():
        L.append(f"| +{r['level']:.0%} | {r['hit_within_hold']:.0%} | {r['hit_within_window']:.0%} | "
                 f"{r['median_sessions_to_hit']:.0f} | {r['hit_on_session_1']:.0%} | {r['final_net_if_hit_bp']:+.0f} | "
                 f"{r['final_net_if_not_bp']:+.0f} |")
    L += ["", "## 4. Stops: how often -X% is touched, and how many eventual winners a stop would have killed", "",
          "| stop | touched | winners touching it first | losers touching it | net if touched (bp, plain exit) |", "|---|---|---|---|---|"]
    for _, r in S["stops"].iterrows():
        L.append(f"| -{r['stop']:.0%} | {r['touched']:.0%} | {r['winners_touching_first']:.0%} | {r['losers_touching']:.0%} | "
                 f"{r['final_net_if_touched_bp']:+.0f} |")
    L += ["", "## 5. Target first or stop first - every declared bracket vs the plain 5-session exit", "",
          "Conservative = stop first when one session touches both (headline); optimistic = target first (bound). "
          "With 39 rules, the best one looks good partly by chance - read the years column.", "",
          "| rule | net bp (conservative) | net bp (optimistic) | vs plain exit | years beating plain exit | target first | stop first | time exit | same-session ambiguous |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in S["rules"].iterrows():
        L.append(f"| {r['rule']} | {r['net_cons_bp']:+.1f} | {r['net_opt_bp']:+.1f} | {r['vs_time_exit_bp']:+.1f} | "
                 f"{r['years_beating_time_exit']} | {r['target_first']:.0%} | {r['stop_first']:.0%} | {r['time_exit']:.0%} | "
                 f"{r['ambiguous']:.1%} |")
    g = S["giveback"]
    L += ["", "## 6. Giveback", "",
          f"Average best point {_pc(g['mean_mfe'])}, average finish {_pc(g['mean_final'])}: the plain exit keeps "
          f"{g['capture']:.0%} of the average best point. Of trades that were up 3% at some point, {g['up3_closed_negative']:.0%} "
          f"finished below entry; of those up 5%, {g['up5_closed_below_1pct']:.0%} finished below +1%.", "",
          "## 7. By rank", "", "| rank | trades | mean MFE | mean MAE | net bp |", "|---|---|---|---|---|"]
    for _, r in S["by_rank"].iterrows():
        L.append(f"| {int(r['rank'])} | {int(r['trades']):,} | {_pc(r['mfe'])} | {_pc(r['mae'])} | {r['final_net'] * 1e4:+.1f} |")
    if ens:
        L += ["", f"The ensemble's top 3, same rules: {ens['trades']:,} trades, MFE {_pc(ens['excursions']['mfe']['mean'])}, "
              f"MAE {_pc(ens['excursions']['mae']['mean'])}, plain-exit net {ens['time_exit_net_bp']:+.1f} bp."]
    L += ["", "## 8. What this does not say", "",
          "- Daily bars cannot order events inside a session: the ambiguous column says how often that matters.",
          "- Stop and target fills assume the level trades; a gap fills at the open, a frozen lower circuit not at all.",
          "- Same history as the research. A rule change is a new engine variant for the forward test.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Inside the trades of the frozen engine")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--isin", nargs="+", default=None)
    ap.add_argument("--rerun-reason", default=None)
    a = ap.parse_args(argv)
    pp = Path(a.panel) / "panel.parquet"
    man_, man_sha = SB.load_manifest()
    SB.check_panel(pp, man_sha)
    man = FZ.verify_frozen(pp)
    decl, sha = load_declaration()
    prior = [e for e in RC.ledger_entries(pp) if e.get("kind") == "stage_e_paths_run"]
    if prior and not a.rerun_reason:
        raise PreconditionError(f"the path study already ran ({prior[-1].get('run_id')}); a re-run needs --rerun-reason")
    out_root = pp.parent / "stage_e"
    day = dt.date.today().strftime("%Y%m%d")
    k = 1
    while (out_root / f"PATHS_{day}_{k:03d}").exists():
        k += 1
    run_id = f"PATHS_{day}_{k:03d}"
    out = out_root / run_id
    out.mkdir(parents=True)
    RC.ledger_append(pp, {"kind": "stage_e_paths_declared", "tool": CODE_VERSION, "run_id": run_id, "declaration_sha": sha,
                          "rerun_reason": a.rerun_reason})
    print("=" * 76)
    print(f"{CODE_VERSION} | declaration {sha} | frozen D {man['engines']['D']['run']}")
    print("Descriptive: MFE/MAE, time to target, stops, target-first vs stop-first. The engines stay frozen.")
    print("=" * 76)
    data = EX.prepare(pp, Path(a.root), man, a.isin)
    st = run_study(data, CFG, "D")
    S = summarize(st, CFG)
    se = summarize(run_study(data, CFG, "ENS", verbose=False), CFG)
    st["trades"].to_csv(out / "trades_paths.csv", index=False)
    S["rules"].to_csv(out / "brackets.csv", index=False)
    S["time_to_target"].to_csv(out / "time_to_target.csv", index=False)
    S["stops"].to_csv(out / "stops.csv", index=False)
    S["path"].to_csv(out / "path_by_session.csv", index=False)
    js = {k_: v for k_, v in S.items() if not isinstance(v, pd.DataFrame)}
    js.update({"run_id": run_id, "tool": CODE_VERSION, "declaration_sha": sha,
               "ensemble": {"trades": se["trades"], "time_exit_net_bp": se["time_exit_net_bp"], "mfe_mean": se["excursions"]["mfe"]["mean"],
                            "mae_mean": se["excursions"]["mae"]["mean"]}})
    (out / "stats.json").write_text(json.dumps(js, indent=2, default=str), encoding="utf-8")
    frozen = {k_: v["run"] for k_, v in man["engines"].items()}
    write_report(out / "paths_report.md", S, se, sha, frozen)
    n = RC.ledger_append(pp, {"kind": "stage_e_paths_run", "tool": CODE_VERSION, "run_id": run_id, "declaration_sha": sha,
                              "trades": S["trades"], "time_exit_net_bp": S["time_exit_net_bp"],
                              "rerun_reason": a.rerun_reason})
    E = S["excursions"]
    print(f"\n{S['trades']:,} trades | plain exit {S['time_exit_net_bp']:+.1f} bp | median MFE {_pc(E['mfe']['p50'])}, "
          f"median MAE {_pc(E['mae']['p50'])} | keeps {S['giveback']['capture']:.0%} of the average best point")
    print(f"  outputs: {out}\n  research ledger: entry {n} (stage_e_paths_run {run_id})")
    print("=" * 76)
    return 0


if __name__ == "__main__":
    sys.exit(main())
