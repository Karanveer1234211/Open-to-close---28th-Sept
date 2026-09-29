#!/usr/bin/env python3
"""
stage_dx_ensemble.py - the two missing arms of the model-combination test:
  C    stage D's exact model on the 123 stage-C features alone;
  ENS  0.5 x per-date rank of D's score + 0.5 x per-date rank of C's score.

    python stage_dx_ensemble.py --panel %CACHE_DAILY_ROOT%\\panel_oc

Rules: STAGE_DX_DECLARATION.json (arms, decision, closure) + STAGE_D_DECLARATION.json
(model, re-verified). D is the existing stage-D run (not refitted). Whether C or ENS
replaces D is decided only by compare_d_variants (corrected exits, paired, 98.75%).
If neither does on any target, the combination question is closed for oc_v1.

Output: <panel>\\stage_d\\STAGEDXC_YYYYMMDD_NNN\\ (C-only) and STAGEDXE_YYYYMMDD_NNN\\ (ensemble),
both in stage D's format (preds.parquet float64, daily.csv, picks.csv, stage_d.json);
ledger stage_dx_declared (before any fit), stage_dx_c_run, stage_dx_ens_run.
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
import stage_d_gate as SD          # noqa: E402
import stage_c_features as SC      # noqa: E402
import stage_c_screen as SCS       # noqa: E402
import stage_dc_model as SDC       # noqa: E402

CODE_VERSION = "stage_dx_ensemble v1"   # never run before release; + D-vs-C score correlation (descriptive)
DECL_FILE = "STAGE_DX_DECLARATION.json"
FAMILY_ID = SB.FAMILY_ID
CFG = {"c_only_model": "stage_d_model_on_C_features_only", "ensemble": "mean_of_per_date_percentile_ranks",
       "ensemble_weights": [0.5, 0.5], "ensemble_members": ["D", "C"], "switch_interval_level": 0.9875}


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
    if decl.get("stage") != "DX" or decl.get("config") != json.loads(json.dumps(CFG)):
        raise PreconditionError(f"settings differ from {DECL_FILE}; the declaration is fixed")
    return decl, hashlib.sha256(raw).hexdigest()[:16]


def check_preconditions(pp: Path, reason: Optional[str]) -> dict:
    E = RC.ledger_entries(pp)
    base = [e["run_id"] for e in E if e.get("kind") == "stage_d_gate"]
    if not base:
        raise PreconditionError("stage D has not run on this panel - it is the baseline and the ENS member")
    d_run = base[-1]
    d_folder = pp.parent / "stage_d" / d_run
    if not (d_folder / "preds.parquet").exists():
        raise PreconditionError(f"{d_folder / 'preds.parquet'} is missing")
    decl, c_sha, names = SC.load_declaration()
    SCS.check_features(pp, c_sha, names)
    scr = [e for e in E if e.get("kind") == "stage_c_screen" and e.get("declaration_sha") == c_sha]
    if not scr or scr[-1].get("leak_suspects", 1) != 0:
        raise PreconditionError("the C screen must have run on these features with no leak suspects")
    prior = [e for e in E if e.get("kind") == "stage_dx_ens_run"]
    if prior and not reason:
        raise PreconditionError(f"DX already ran ({prior[-1].get('run_id')}); the closure rule allows a re-run only "
                                f"with --rerun-reason for a verified bug")
    return {"d_run": d_run, "d_folder": d_folder, "c_sha": c_sha, "names": names, "prior": prior}


def ensemble_frame(d_preds: pd.DataFrame, c_preds: pd.DataFrame, w=(0.5, 0.5)) -> pd.DataFrame:
    """Per target and day: w0 x pct-rank(D score) + w1 x pct-rank(C score), over rows both scored.
    Outcomes come from the C run (float64); the rows are the same eligible rows."""
    d = d_preds[["timestamp", "symbol", "target", "score"]].rename(columns={"score": "score_d"})
    c = c_preds[["timestamp", "symbol", "target", "score", "outcome"]].rename(columns={"score": "score_c"})
    for f in (d, c):
        f["timestamp"] = SB._naive(f["timestamp"])
        f["symbol"] = f["symbol"].astype(str)
        f["target"] = f["target"].astype(str)
    m = d.merge(c, on=["timestamp", "symbol", "target"], how="inner")
    m = m[np.isfinite(m["score_d"].astype(float)) & np.isfinite(m["score_c"].astype(float))].copy()
    g = m.groupby(["target", "timestamp"])
    m["rank_d"] = g["score_d"].rank(pct=True, method="average")
    m["rank_c"] = g["score_c"].rank(pct=True, method="average")
    m["score"] = w[0] * m["rank_d"] + w[1] * m["rank_c"]
    return m


def score_rank_corr(frame: pd.DataFrame) -> Dict[str, float]:
    """Descriptive: per target, the mean over days of the rank correlation between D's and C's scores."""
    out = {}
    for tg, g in frame.groupby("target"):
        c = g.groupby("timestamp").apply(lambda x: x["rank_d"].corr(x["rank_c"]) if len(x) > 2 else np.nan)
        out[str(tg)] = float(np.nanmean(c.to_numpy(float)))
    return out


def outputs_from_scores(frame: pd.DataFrame, targets: List[str], fold_by_day: Dict, cfg: dict) -> dict:
    """Stage D's daily/picks/preds/evaluation from (timestamp, symbol, target, score, outcome) rows."""
    cost = cfg["cost_bps"] / 1e4
    results, daily_parts, picks_parts, preds_parts = {}, [], [], []
    for tg in targets:
        f = frame[frame["target"] == tg]
        sessions = np.unique(f["timestamp"].to_numpy(dtype="datetime64[ns]"))
        day = np.searchsorted(sessions, f["timestamp"].to_numpy(dtype="datetime64[ns]"))
        sym = f["symbol"].astype(str).to_numpy()
        sc = f["score"].to_numpy(float)
        y = f["outcome"].to_numpy(float)
        D = SD.daily_top_n(day, sym, sc, y, cfg["top_n_all"])
        ts_days = pd.DatetimeIndex(sessions[D["days"]])
        daily = pd.DataFrame({"timestamp": ts_days, "target": tg,
                              "fold": [int(fold_by_day.get(t, 0)) for t in ts_days],
                              "n_names": D["n_names"], "market": D["market"], "tie_at_3": D["tie_at_3"]})
        for n in cfg["top_n_all"]:
            daily[f"top{n}_gross"] = D[f"top{n}"]
            daily[f"top{n}_net"] = D[f"top{n}"] - cost
            daily[f"top{n}_excess"] = D[f"top{n}"] - D["market"]
            daily[f"top{n}_missing"] = D[f"top{n}_missing"]
        daily_parts.append(daily)
        o, pos = D["order"], D["pos"]
        t10 = pos < 10
        picks_parts.append(pd.DataFrame({"timestamp": sessions[day[o][t10]], "target": tg, "rank": pos[t10] + 1,
                                         "symbol": sym[o][t10], "score": sc[o][t10], "outcome": y[o][t10],
                                         "net": y[o][t10] - cost}))
        preds_parts.append(pd.DataFrame({"timestamp": sessions[day], "symbol": sym, "target": tg,
                                         "score": sc, "outcome": y}))
        fin = np.isfinite(y)
        ic, _ = SB.daily_ic(day[fin], SB.group_rank(day[fin], sc[fin]), SB.group_rank(day[fin], y[fin]),
                            len(sessions), SB.CFG["min_names"])
        results[tg] = SD.evaluate(daily, ic[D["days"]], cfg)
    return {"results": results, "daily": pd.concat(daily_parts, ignore_index=True),
            "picks": pd.concat(picks_parts, ignore_index=True), "preds": pd.concat(preds_parts, ignore_index=True)}


def _next_id(root: Path, prefix: str) -> str:
    day = dt.date.today().strftime("%Y%m%d")
    k = 1
    while (root / f"{prefix}_{day}_{k:03d}").exists():
        k += 1
    return f"{prefix}_{day}_{k:03d}"


def _write(out: Path, res: dict, meta: dict) -> None:
    out.mkdir(parents=True)
    res["daily"].to_csv(out / "daily.csv", index=False)
    res["picks"].to_csv(out / "picks.csv", index=False)
    res["preds"].to_parquet(out / "preds.parquet", index=False)
    (out / "stage_d.json").write_text(json.dumps({**meta, "results": res["results"]}, indent=2, default=str),
                                      encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage DX: C-only arm and the declared D+C rank ensemble")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--rerun-reason", default=None)
    a = ap.parse_args(argv)
    pp = Path(a.panel) / "panel.parquet"
    man, man_sha = SB.load_manifest()
    targets = list(man["targets"].keys())
    SB.check_panel(pp, man_sha)
    pre = check_preconditions(pp, a.rerun_reason)
    decl, sha = load_declaration()
    _, d_sha = SD.load_declaration()
    root = pp.parent / "stage_d"
    c_id, e_id = _next_id(root, "STAGEDXC"), _next_id(root, "STAGEDXE")
    RC.ledger_append(pp, {"kind": "stage_dx_declared", "tool": CODE_VERSION, "declaration_sha": sha,
                          "stage_d_declaration_sha": d_sha, "c_declaration_sha": pre["c_sha"], "d_run": pre["d_run"],
                          "c_run": c_id, "ens_run": e_id, "rerun_reason": a.rerun_reason})
    print("=" * 76)
    print(f"{CODE_VERSION} | DX declaration {sha} | stage-D model {d_sha} | baseline {pre['d_run']}")
    print("C = stage D's model on the 123 C features alone; ENS = 0.5 rank(D) + 0.5 rank(C). Decision and")
    print("closure: compare_d_variants, same rule as D2 / D_C. If neither replaces D, the question is closed.")
    print("=" * 76)
    t0 = time.perf_counter()
    rc = SDC.run_dc(pp, targets, pre["names"], include_base=False)
    if rc["n_base"] != 0 or rc["n_c"] != len(pre["names"]):
        raise RuntimeError("the C-only arm must use the C features alone")
    meta_c = {"run_id": c_id, "tool": CODE_VERSION, "arm": "C", "declaration_sha": sha, "features": rc["features"],
              "folds": rc["folds"], "config": rc["config"], "rows": rc["rows"]}
    _write(root / c_id, rc, meta_c)
    RC.ledger_append(pp, {"kind": "stage_dx_c_run", "tool": CODE_VERSION, "run_id": c_id, "declaration_sha": sha,
                          "experiment_family": FAMILY_ID})
    d_preds = pd.read_parquet(pre["d_folder"] / "preds.parquet")
    d_daily = pd.read_csv(pre["d_folder"] / "daily.csv", parse_dates=["timestamp"])
    fold_by_day = dict(zip(pd.DatetimeIndex(d_daily["timestamp"]), d_daily["fold"]))
    ens = ensemble_frame(d_preds, rc["preds"], tuple(CFG["ensemble_weights"]))
    er = outputs_from_scores(ens, targets, fold_by_day, SD.CFG)
    score_corr = score_rank_corr(ens)
    meta_e = {"run_id": e_id, "tool": CODE_VERSION, "arm": "ENS", "declaration_sha": sha,
              "members": {"D": pre["d_run"], "C": c_id}, "weights": CFG["ensemble_weights"], "config": SD.CFG,
              "rows": int(len(ens)), "d_vs_c_daily_score_rank_corr": score_corr}
    _write(root / e_id, er, meta_e)
    n = RC.ledger_append(pp, {"kind": "stage_dx_ens_run", "tool": CODE_VERSION, "run_id": e_id, "c_run": c_id,
                              "d_run": pre["d_run"], "declaration_sha": sha, "experiment_family": FAMILY_ID,
                              "rerun_reason": a.rerun_reason})
    print()
    for tg in targets:
        print(f"{tg:<12} C {SD._iv(rc['results'][tg]['top3_net']):<28} ENS {SD._iv(er['results'][tg]['top3_net'])} "
              f"(stage-D scoring) | D vs C scores: mean daily rank corr {score_corr[tg]:+.2f}")
    print(f"\n  outputs: {root / c_id} and {root / e_id} ({time.perf_counter() - t0:.0f}s)")
    print(f"  research ledger: entry {n} (stage_dx_ens_run {e_id})")
    print("=" * 76)
    print("DX DONE. Next: compare_d_variants.py - D vs D2 vs D_C vs C vs ENS, corrected exits, the declared rule.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
