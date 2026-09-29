#!/usr/bin/env python3
"""
stage_dc_model.py - stage D's exact model on all 205 clean features PLUS all
123 stage-C features. No selection (D2 showed selection cost information).

    python stage_dc_model.py --panel %CACHE_DAILY_ROOT%\\panel_oc

Refuses unless: the panel is audited, stage D ran, stage D's model declaration
still verifies, the C features were built from the current C declaration and
match the panel row for row, and the C screen ran with NO leak suspects.
Whether D_C replaces D is decided only by compare_d_variants (corrected exits,
paired, 98.75%, the declared rule).

Output: <panel>\\stage_d\\STAGEDC_YYYYMMDD_NNN\\ in stage D's format (preds.parquet
in float64, daily.csv, picks.csv, stage_d.json); ledger stage_dc_declared, stage_dc_run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

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

CODE_VERSION = "stage_dc_model v1.1"   # v1.1: run_dc(include_base=False) serves the C-only arm; D_C unchanged
FAMILY_ID = SB.FAMILY_ID


class PreconditionError(SystemExit):
    pass


def _log(msg: str) -> None:
    print(f"{dt.datetime.now():%H:%M:%S}  {msg}", flush=True)


def check_preconditions(pp: Path, reason: Optional[str]) -> dict:
    E = RC.ledger_entries(pp)
    if not [e for e in E if e.get("kind") == "stage_d_gate"]:
        raise PreconditionError("stage D has not run on this panel - D_C is compared against it")
    decl, sha, names = SC.load_declaration()
    SCS.check_features(pp, sha, names)
    scr = [e for e in E if e.get("kind") == "stage_c_screen" and e.get("declaration_sha") == sha]
    if not scr:
        raise PreconditionError("the C screen has not run on these features - run stage_c_screen.py first")
    if scr[-1].get("leak_suspects", 1) != 0:
        raise PreconditionError(f"the C screen flagged possible leaks: {scr[-1].get('suspect_features')}. D_C does "
                                f"not run until they are explained.")
    prior = [e for e in E if e.get("kind") == "stage_dc_run"]
    if prior and not reason:
        raise PreconditionError(f"D_C already ran ({prior[-1].get('run_id')}); a re-run needs --rerun-reason")
    return {"c_sha": sha, "names": names, "screen_run": scr[-1].get("run_id"), "prior": prior}


def run_dc(pp: Path, targets: List[str], c_names: List[str], verbose: bool = True, cfg: Optional[dict] = None,
           include_base: bool = True) -> dict:
    import feasibility_test as FT
    from sklearn.ensemble import HistGradientBoostingRegressor
    cfg = dict(cfg or SD.CFG)
    t0 = time.perf_counter()
    base_feats, excluded = SB.clean_feature_list(pp) if include_base else ([], {})
    feats = base_feats + list(c_names)
    RC.assert_no_label_leak(feats, "stage_dc_model")
    if len(set(feats)) != len(feats):
        raise PreconditionError("a C feature name collides with a panel feature")
    b = pd.read_parquet(pp, columns=["timestamp", "symbol", SB.ELIG_COL])
    ts = SB._naive(b["timestamp"]).to_numpy(dtype="datetime64[ns]")
    sym = b["symbol"].astype(str).to_numpy()
    buy = pd.to_numeric(b[SB.ELIG_COL], errors="coerce").to_numpy(dtype=float)
    del b
    el = SD.eligible_mask(buy)
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
    X = np.empty((len(sel), len(feats)), dtype=np.float32)
    read_c = SCS.reader_for(pp)
    for j, f in enumerate(feats):
        X[:, j] = (SB._read_col(pp, f) if j < len(base_feats) else read_c(f))[sel]
    if verbose:
        _log(f"{rows_elig:,} eligible rows | {len(base_feats)} clean + {len(c_names)} C = {len(feats)} features "
             f"({X.nbytes / 1e9:.2f} GB, {time.perf_counter() - t0:.0f}s)")
    sidx = {np.datetime64(s, "ns"): i for i, s in enumerate(sessions)}
    fold_of_day = np.zeros(len(sessions), int)
    fold_meta, results, daily_parts, picks_parts, preds_parts = [], {}, [], [], []
    cost = cfg["cost_bps"] / 1e4
    for tg in targets:
        h = SD.horizon_of(tg)
        y = Y[tg]
        score = np.full(len(sel), np.nan)
        for f in folds:
            fo = int(f["fold"])
            tr_end = sidx[np.datetime64(pd.Timestamp(f["train_end"]), "ns")]
            te0 = sidx[np.datetime64(pd.Timestamp(f["test_start"]), "ns")]
            te1 = sidx[np.datetime64(pd.Timestamp(f["test_end"]), "ns")]
            fold_of_day[te0:te1 + 1] = fo
            tr = np.flatnonzero((day <= tr_end) & np.isfinite(y))
            gap = te0 - int(day[tr].max())
            if gap <= h:
                raise SD.EmbargoError(f"{tg} fold {fo}: embargo gap {gap} <= horizon {h}")
            lab, lo, hi = SD.make_label(y[tr], day[tr], cfg["winsor_pct"])
            rng = np.random.default_rng([cfg["seed"], fo, h])
            keep = (np.arange(len(tr)) if len(tr) <= cfg["fit_cap"]
                    else np.sort(rng.choice(len(tr), cfg["fit_cap"], replace=False)))
            t1 = time.perf_counter()
            model = HistGradientBoostingRegressor(random_state=cfg["seed"], **cfg["hgb"])
            model.fit(X[tr[keep]], lab[keep])
            te = np.flatnonzero((day >= te0) & (day <= te1))
            for a_ in range(0, len(te), 200_000):
                bb = te[a_:a_ + 200_000]
                score[bb] = model.predict(X[bb])
            fold_meta.append({"target": tg, "fold": fo, "train_end": str(pd.Timestamp(f["train_end"]).date()),
                              "test_start": str(pd.Timestamp(f["test_start"]).date()),
                              "test_end": str(pd.Timestamp(f["test_end"]).date()), "train_rows": int(len(tr)),
                              "fit_rows": int(len(keep)), "test_rows": int(len(te)), "embargo_gap_sessions": int(gap),
                              "label_clip_bp": [lo * 1e4, hi * 1e4], "seconds": round(time.perf_counter() - t1, 1)})
            if verbose:
                _log(f"{tg} fold {fo}: fit on {len(keep):,} rows ({fold_meta[-1]['seconds']:.0f}s)")
            del model
        tm = np.isfinite(score)
        dt_, st_, yt_, symt_ = day[tm], score[tm], y[tm], sym_s[tm]
        D = SD.daily_top_n(dt_, symt_, st_, yt_, cfg["top_n_all"])
        daily = pd.DataFrame({"timestamp": pd.DatetimeIndex(sessions[D["days"]]), "target": tg,
                              "fold": fold_of_day[D["days"]], "n_names": D["n_names"], "market": D["market"],
                              "tie_at_3": D["tie_at_3"]})
        for n in cfg["top_n_all"]:
            daily[f"top{n}_gross"] = D[f"top{n}"]
            daily[f"top{n}_net"] = D[f"top{n}"] - cost
            daily[f"top{n}_excess"] = D[f"top{n}"] - D["market"]
            daily[f"top{n}_missing"] = D[f"top{n}_missing"]
        daily_parts.append(daily)
        o, pos = D["order"], D["pos"]
        t10 = pos < 10
        picks_parts.append(pd.DataFrame({"timestamp": sessions[dt_[o][t10]], "target": tg, "rank": pos[t10] + 1,
                                         "symbol": symt_[o][t10], "score": st_[o][t10], "outcome": yt_[o][t10],
                                         "net": yt_[o][t10] - cost}))
        preds_parts.append(pd.DataFrame({"timestamp": sessions[dt_], "symbol": symt_, "target": tg,
                                         "score": st_, "outcome": yt_}))
        fin = np.isfinite(yt_)
        ic, _ = SB.daily_ic(dt_[fin], SB.group_rank(dt_[fin], st_[fin]), SB.group_rank(dt_[fin], yt_[fin]),
                            len(sessions), SB.CFG["min_names"])
        results[tg] = SD.evaluate(daily, ic[D["days"]], cfg)
        if verbose:
            r = results[tg]
            _log(f"{tg}: D_C top-3 net {SD._iv(r['top3_net'])} (stage-D scoring; the decision is compare_d_variants')")
    return {"results": results, "folds": fold_meta, "features": feats, "n_base": len(base_feats),
            "n_c": len(c_names), "daily": pd.concat(daily_parts, ignore_index=True),
            "picks": pd.concat(picks_parts, ignore_index=True), "preds": pd.concat(preds_parts, ignore_index=True),
            "rows": {"panel": rows_panel, "eligible": rows_elig}, "seconds": round(time.perf_counter() - t0, 1),
            "config": cfg}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage D_C: stage D's model on 205 + 123 C features")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--rerun-reason", default=None)
    a = ap.parse_args(argv)
    pp = Path(a.panel) / "panel.parquet"
    man, man_sha = SB.load_manifest()
    targets = list(man["targets"].keys())
    info = SB.check_panel(pp, man_sha)
    pre = check_preconditions(pp, a.rerun_reason)
    _, d_sha = SD.load_declaration()
    out_root = pp.parent / "stage_d"
    day = dt.date.today().strftime("%Y%m%d")
    k = 1
    while (out_root / f"STAGEDC_{day}_{k:03d}").exists():
        k += 1
    run_id = f"STAGEDC_{day}_{k:03d}"
    out = out_root / run_id
    out.mkdir(parents=True)
    RC.ledger_append(pp, {"kind": "stage_dc_declared", "tool": CODE_VERSION, "run_id": run_id, "c_declaration_sha":
                          pre["c_sha"], "stage_d_declaration_sha": d_sha, "c_screen_run": pre["screen_run"],
                          "rerun_reason": a.rerun_reason})
    print("=" * 76)
    print(f"{CODE_VERSION} | C declaration {pre['c_sha']} | stage-D model declaration {d_sha} | panel {pp}")
    print("Stage D's exact model on all clean features + all C features. The decision is compare_d_variants'.")
    print("=" * 76)
    res = run_dc(pp, targets, pre["names"])
    res["daily"].to_csv(out / "daily.csv", index=False)
    res["picks"].to_csv(out / "picks.csv", index=False)
    res["preds"].to_parquet(out / "preds.parquet", index=False)
    js = {"run_id": run_id, "tool": CODE_VERSION, "c_declaration_sha": pre["c_sha"], "stage_d_declaration_sha": d_sha,
          "panel_info": info, "results": res["results"], "folds": res["folds"], "features": res["features"],
          "n_base": res["n_base"], "n_c": res["n_c"], "rows": res["rows"], "seconds": res["seconds"],
          "config": res["config"], "rerun_reason": a.rerun_reason}
    (out / "stage_d.json").write_text(json.dumps(js, indent=2, default=str), encoding="utf-8")
    n = RC.ledger_append(pp, {"kind": "stage_dc_run", "tool": CODE_VERSION, "run_id": run_id,
                              "experiment_family": FAMILY_ID, "c_declaration_sha": pre["c_sha"],
                              "top3_net_bp_stage_d_scoring": {tg: res["results"][tg]["top3_net"]["mean"] * 1e4
                                                              for tg in targets}, "rerun_reason": a.rerun_reason})
    print()
    for tg in targets:
        print(f"{tg:<12} D_C top-3 net {SD._iv(res['results'][tg]['top3_net'])} (stage-D scoring)")
    print(f"\n  outputs: {out}\n  research ledger: entry {n} (stage_dc_run {run_id})")
    print("=" * 76)
    print("D_C DONE. Next: compare_d_variants.py (D vs D2 vs D_C, corrected exits, the declared rule).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
