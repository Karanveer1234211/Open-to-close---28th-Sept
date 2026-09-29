#!/usr/bin/env python3
"""
stage_d2_nested.py - stage D's model, but on stage B's strongest features,
re-selected INSIDE each fold so no test day ever chooses a feature.

    python stage_d2_nested.py --panel %CACHE_DAILY_ROOT%\\panel_oc

Rules: STAGE_D2_DECLARATION.json (selection) + STAGE_D_DECLARATION.json (model,
re-verified at start and used unchanged). Per fold, on that fold's training rows:
  per-day rank IC of every stock-level feature x target, Newey-West t (lag 20),
  BH q pooled over the fold's cells, sign in >= 4 of 5 training blocks ->
  pass; one feature per family (|rho| >= 0.80, strongest |t| leads); plus all
  market-level columns. Then stage D's exact model on those features.

Output: <panel>\\stage_d\\STAGED2_YYYYMMDD_NNN\\ in stage D's format (preds.parquet,
daily.csv, picks.csv, stage_d.json) plus selection.json (every fold's picks),
and ledger entries stage_d2_declared (before fitting) and stage_d2_run.
Whether D2 replaces D is decided ONLY by compare_d_variants (exit-corrected,
paired, declared rule).
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

CODE_VERSION = "stage_d2_nested v1.1"   # v1.1: preds.parquet in float64 so outside audits reproduce to 1e-10
DECL_FILE = "STAGE_D2_DECLARATION.json"
FAMILY_ID = SB.FAMILY_ID
ELIG_COL = SB.ELIG_COL
SEL = {
    "selection": "nested_B_rule_family_leaders_plus_market_level",
    "min_names": 15, "min_dates": 60, "hac_lag": 20,
    "q_max": 0.10, "blocks": 5, "min_sign_blocks": 4,
    "market_level_share": 0.95,
    "family_corr": 0.80, "family_sample_dates": 60,
    "switch_interval_level": 0.9875,
    "seed": 7,
}


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
    if decl.get("stage") != "D2" or decl.get("config") != json.loads(json.dumps(SEL)):
        diff = sorted(k for k in set(SEL) | set(decl.get("config", {}))
                      if json.loads(json.dumps(SEL)).get(k) != decl.get("config", {}).get(k))
        raise PreconditionError(f"selection settings differ from {DECL_FILE} on {diff}; the declaration is fixed")
    return decl, hashlib.sha256(raw).hexdigest()[:16]


def check_order_and_rerun(pp: Path, reason: Optional[str]) -> List[dict]:
    E = RC.ledger_entries(pp)
    if not [e for e in E if e.get("kind") == "stage_d_gate" and e.get("experiment_family") == FAMILY_ID]:
        raise PreconditionError("stage D has not run on this panel - D2 is compared against it")
    prior = [e for e in E if e.get("kind") == "stage_d2_run" and e.get("experiment_family") == FAMILY_ID]
    if prior and not reason:
        raise PreconditionError(f"D2 already ran on this panel ({prior[-1].get('run_id')}). Re-running needs "
                                f"--rerun-reason \"...\" (a verified bug); it goes into the ledger.")
    return prior


# ----------------------------------------------------------------------
# the nested screen
# ----------------------------------------------------------------------
def market_level_mask(X: np.ndarray, rows: np.ndarray, day: np.ndarray, share: float) -> np.ndarray:
    """Columns constant across stocks on >= share of days (days with >= 2 values), over the given rows."""
    out = np.zeros(X.shape[1], bool)
    for j in range(X.shape[1]):
        x = X[rows, j]
        f = np.isfinite(x)
        if f.sum() < 2:
            continue
        g = pd.DataFrame({"d": day[f], "x": x[f]}).groupby("d")["x"].agg(["min", "max", "size"])
        g = g[g["size"] >= 2]
        out[j] = len(g) > 0 and float((g["max"] == g["min"]).mean()) >= share
    return out


def nested_screen(X: np.ndarray, feats: List[str], day: np.ndarray, Y: Dict[str, np.ndarray],
                  rows: np.ndarray, n_days: int, sel: dict) -> dict:
    """Stage B's rule on the given (training) rows only. Returns, per target, the passing features,
    the family leaders, and the stats; plus the market-level columns."""
    d = day[rows]
    tdays = np.unique(d)
    blocks = np.array_split(tdays, sel["blocks"])
    block_of = np.full(n_days, -1)
    for b, ds in enumerate(blocks):
        block_of[ds] = b
    mkt = market_level_mask(X, rows, d, sel["market_level_share"])      # column by column: no full copy of X
    yt = {tg: Y[tg][rows] for tg in Y}
    yfin = {tg: np.isfinite(v) for tg, v in yt.items()}
    ry_full = {tg: SB.group_rank(d[yfin[tg]], yt[tg][yfin[tg]]) for tg in yt}
    cells = []
    for j, f in enumerate(feats):
        if mkt[j]:
            continue
        x = X[rows, j].astype(float)
        fx = np.isfinite(x)
        for tg in yt:
            m = fx & yfin[tg]
            if m.sum() == 0:
                continue
            dm = d[m]
            rx = SB.group_rank(dm, x[m])
            ry = ry_full[tg] if m.sum() == yfin[tg].sum() else SB.group_rank(dm, yt[tg][m])
            ic, _ = SB.daily_ic(dm, rx, ry, n_days, sel["min_names"])
            ok = np.isfinite(ic)
            n_ok = int(ok.sum())
            if n_ok < sel["min_dates"]:
                continue
            ics = ic[ok]
            mean = float(ics.mean())
            t = SB.hac_t(ics, sel["hac_lag"])
            bm = [float(np.nanmean(ic[ok & (block_of == b)])) if (ok & (block_of == b)).any() else float("nan")
                  for b in range(sel["blocks"])]
            agree = int(sum(1 for v in bm if np.isfinite(v) and np.sign(v) == np.sign(mean) and mean != 0))
            cells.append({"feature": f, "j": j, "target": tg, "n_days": n_ok, "mean_ic": mean, "t_hac": t,
                          "p_value": SB.p_two_sided(t), "sign_agree_blocks": agree})
    C = pd.DataFrame(cells)
    out = {"market_level": [feats[j] for j in np.flatnonzero(mkt)], "targets": {}}
    if C.empty:
        for tg in yt:
            out["targets"][tg] = {"passers": [], "leaders": [], "stats": C}
        return out
    C["q_value"] = SB.bh_qvalues(C["p_value"].to_numpy(float))
    C["passes"] = (C["q_value"] <= sel["q_max"]) & (C["sign_agree_blocks"] >= sel["min_sign_blocks"])
    rng = np.random.default_rng(sel["seed"])
    samp_days = np.sort(rng.choice(tdays, min(sel["family_sample_dates"], len(tdays)), replace=False))
    samp = np.isin(d, samp_days)
    for tg in yt:
        P = C[(C["target"] == tg) & C["passes"]]
        P = P.reindex(P["t_hac"].abs().sort_values(ascending=False).index)
        order = P["feature"].tolist()
        vals = {f: X[rows[samp], feats.index(f)].astype("float32") for f in order}
        fam = SB.assign_families(order, vals, d[samp], sel["family_corr"]) if order else {}
        leaders = [f for f in order if fam.get(f) == f]
        out["targets"][tg] = {"passers": order, "leaders": leaders, "families": fam,
                              "stats": C[C["target"] == tg]}
    return out


# ----------------------------------------------------------------------
# the run
# ----------------------------------------------------------------------
def run_variant(pp: Path, targets: List[str], verbose: bool = True, cfg: Optional[dict] = None,
                sel: Optional[dict] = None) -> dict:
    import feasibility_test as FT
    from sklearn.ensemble import HistGradientBoostingRegressor
    cfg = dict(cfg or SD.CFG)
    sel = dict(sel or SEL)
    t0 = time.perf_counter()
    feats, excluded = SB.clean_feature_list(pp)
    RC.assert_no_label_leak(feats, "stage_d2_nested")
    base = pd.read_parquet(pp, columns=["timestamp", "symbol", ELIG_COL])
    ts = SB._naive(base["timestamp"]).to_numpy(dtype="datetime64[ns]")
    sym = base["symbol"].astype(str).to_numpy()
    buy = pd.to_numeric(base[ELIG_COL], errors="coerce").to_numpy(dtype=float)
    del base
    el = SD.eligible_mask(buy)
    order = np.lexsort((sym, ts))
    selr = order[el[order]]
    ts_s, sym_s = ts[selr], sym[selr]
    rows_panel, rows_elig = int(len(ts)), int(len(selr))
    del ts, sym, buy, el, order
    sessions = np.unique(ts_s)
    day = np.searchsorted(sessions, ts_s)
    Y = {tg: SB._read_col(pp, tg)[selr] for tg in targets}
    last_res = min(int(day[np.isfinite(Y[tg])].max()) for tg in targets)
    fold_sessions = sessions[:last_res + 1]
    folds = FT.splits(fold_sessions, cfg["n_splits"], cfg["embargo"], cfg["min_train_frac"])
    X = np.empty((len(selr), len(feats)), dtype=np.float32)
    for j, f in enumerate(feats):
        X[:, j] = SB._read_col(pp, f)[selr]
    if verbose:
        _log(f"{rows_elig:,} eligible rows | {len(feats)} clean features | features loaded "
             f"({time.perf_counter() - t0:.0f}s)")
    sidx = {np.datetime64(s, "ns"): i for i, s in enumerate(sessions)}
    scores = {tg: np.full(len(selr), np.nan) for tg in targets}
    fold_of_day = np.zeros(len(sessions), int)
    fold_meta, selection = [], []
    anyfin = np.zeros(len(selr), bool)
    for tg in targets:
        anyfin |= np.isfinite(Y[tg])
    for f in folds:
        fo = int(f["fold"])
        tr_end = sidx[np.datetime64(pd.Timestamp(f["train_end"]), "ns")]
        te0 = sidx[np.datetime64(pd.Timestamp(f["test_start"]), "ns")]
        te1 = sidx[np.datetime64(pd.Timestamp(f["test_end"]), "ns")]
        fold_of_day[te0:te1 + 1] = fo
        t1 = time.perf_counter()
        rows = np.flatnonzero((day <= tr_end) & anyfin)
        S = nested_screen(X, feats, day, Y, rows, len(sessions), sel)
        if verbose:
            _log(f"fold {fo}: screened on {len(rows):,} training rows to {pd.Timestamp(f['train_end']).date()} "
                 f"({time.perf_counter() - t1:.0f}s): " + ", ".join(
                     f"{tg[-4:]} {len(S['targets'][tg]['passers'])} pass -> {len(S['targets'][tg]['leaders'])} "
                     f"leaders" for tg in targets) + f"; +{len(S['market_level'])} market-level")
        for tg in targets:
            h = SD.horizon_of(tg)
            y = Y[tg]
            use = S["targets"][tg]["leaders"] + S["market_level"]
            idx = [feats.index(u) for u in use]
            selection.append({"fold": fo, "target": tg, "n_passers": len(S["targets"][tg]["passers"]),
                              "leaders": S["targets"][tg]["leaders"], "market_level": S["market_level"],
                              "n_features_used": len(idx)})
            tr = np.flatnonzero((day <= tr_end) & np.isfinite(y))
            gap = te0 - int(day[tr].max())
            if gap <= h:
                raise SD.EmbargoError(f"{tg} fold {fo}: embargo gap {gap} <= horizon {h}")
            te = np.flatnonzero((day >= te0) & (day <= te1))
            if not idx:
                fold_meta.append({"target": tg, "fold": fo, "features": 0, "note": "nothing selected: no picks"})
                continue
            lab, lo, hi = SD.make_label(y[tr], day[tr], cfg["winsor_pct"])
            rng = np.random.default_rng([cfg["seed"], fo, h])
            keep = (np.arange(len(tr)) if len(tr) <= cfg["fit_cap"]
                    else np.sort(rng.choice(len(tr), cfg["fit_cap"], replace=False)))
            t2 = time.perf_counter()
            model = HistGradientBoostingRegressor(random_state=cfg["seed"], **cfg["hgb"])
            model.fit(X[np.ix_(tr[keep], idx)], lab[keep])
            for a in range(0, len(te), 200_000):
                b = te[a:a + 200_000]
                scores[tg][b] = model.predict(X[np.ix_(b, idx)])
            fold_meta.append({"target": tg, "fold": fo, "features": len(idx), "train_rows": int(len(tr)),
                              "fit_rows": int(len(keep)), "test_rows": int(len(te)), "embargo_gap_sessions": int(gap),
                              "train_end": str(pd.Timestamp(f["train_end"]).date()),
                              "test_start": str(pd.Timestamp(f["test_start"]).date()),
                              "test_end": str(pd.Timestamp(f["test_end"]).date()),
                              "seconds": round(time.perf_counter() - t2, 1)})
            if verbose:
                _log(f"  {tg} fold {fo}: {len(idx)} features, fit on {len(keep):,} rows ({fold_meta[-1]['seconds']:.0f}s)")

    results, daily_parts, picks_parts, preds_parts = {}, [], [], []
    cost = cfg["cost_bps"] / 1e4
    for tg in targets:
        score, y = scores[tg], Y[tg]
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
                                         "score": st_, "outcome": yt_}))        # float64: exact, auditable
        fin = np.isfinite(yt_)
        ic, _ = SB.daily_ic(dt_[fin], SB.group_rank(dt_[fin], st_[fin]), SB.group_rank(dt_[fin], yt_[fin]),
                            len(sessions), SB.CFG["min_names"])
        results[tg] = SD.evaluate(daily, ic[D["days"]], cfg)
        if verbose:
            r = results[tg]
            _log(f"{tg}: D2 top-3 net {r['top3_net']['mean'] * 1e4:+.1f} bp "
                 f"[{r['top3_net']['lo'] * 1e4:+.1f}, {r['top3_net']['hi'] * 1e4:+.1f}] (stage-D scoring; the "
                 f"decision is compare_d_variants')")
    return {"results": results, "folds": fold_meta, "selection": selection, "features": feats,
            "daily": pd.concat(daily_parts, ignore_index=True), "picks": pd.concat(picks_parts, ignore_index=True),
            "preds": pd.concat(preds_parts, ignore_index=True), "rows": {"panel": rows_panel, "eligible": rows_elig},
            "sessions": {"first": str(pd.Timestamp(sessions[0]).date()),
                         "last_resolved": str(pd.Timestamp(fold_sessions[-1]).date())},
            "seconds": round(time.perf_counter() - t0, 1), "config": cfg, "selection_config": sel}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage D2: stage D's model on nested stage-B selections")
    ap.add_argument("--panel", required=True, help="panel folder, e.g. %%CACHE_DAILY_ROOT%%\\panel_oc")
    ap.add_argument("--rerun-reason", default=None)
    a = ap.parse_args(argv)
    pp = Path(a.panel) / "panel.parquet"
    man, man_sha = SB.load_manifest()
    targets = list(man["targets"].keys())
    info = SB.check_panel(pp, man_sha)
    prior = check_order_and_rerun(pp, a.rerun_reason)
    decl, sha = load_declaration()
    _, d_sha = SD.load_declaration()
    out_root = pp.parent / "stage_d"
    day = dt.date.today().strftime("%Y%m%d")
    k = 1
    while (out_root / f"STAGED2_{day}_{k:03d}").exists():
        k += 1
    run_id = f"STAGED2_{day}_{k:03d}"
    out = out_root / run_id
    out.mkdir(parents=True, exist_ok=False)
    RC.ledger_append(pp, {"kind": "stage_d2_declared", "tool": CODE_VERSION, "run_id": run_id,
                          "experiment_family": FAMILY_ID, "declaration_sha": sha, "stage_d_declaration_sha": d_sha,
                          "rerun_reason": a.rerun_reason})
    print("=" * 76)
    print(f"{CODE_VERSION} | D2 declaration {sha} | stage-D model declaration {d_sha} | panel {pp}")
    print("Per fold, on training rows only: stage B's rule -> one feature per family + market-level columns;")
    print("then stage D's exact model. Whether D2 replaces D is decided by compare_d_variants.")
    print("=" * 76)
    res = run_variant(pp, targets)
    res["daily"].to_csv(out / "daily.csv", index=False)
    res["picks"].to_csv(out / "picks.csv", index=False)
    res["preds"].to_parquet(out / "preds.parquet", index=False)
    (out / "selection.json").write_text(json.dumps(res["selection"], indent=1), encoding="utf-8")
    js = {"run_id": run_id, "tool": CODE_VERSION, "experiment_family": FAMILY_ID, "declaration_sha": sha,
          "stage_d_declaration_sha": d_sha, "panel": str(pp), "panel_info": info, "results": res["results"],
          "folds": res["folds"], "rows": res["rows"], "sessions": res["sessions"], "seconds": res["seconds"],
          "config": res["config"], "selection_config": res["selection_config"],
          "rerun_reason": a.rerun_reason, "prior_runs": [p.get("run_id") for p in prior]}
    (out / "stage_d.json").write_text(json.dumps(js, indent=2, default=str), encoding="utf-8")
    n = RC.ledger_append(pp, {"kind": "stage_d2_run", "tool": CODE_VERSION, "run_id": run_id,
                              "experiment_family": FAMILY_ID, "declaration_sha": sha,
                              "top3_net_bp_stage_d_scoring": {tg: res["results"][tg]["top3_net"]["mean"] * 1e4
                                                              for tg in targets},
                              "rerun_reason": a.rerun_reason})
    print()
    for tg in targets:
        s = [x for x in res["selection"] if x["target"] == tg]
        r = res["results"][tg]
        print(f"{tg:<12} features used per fold: {', '.join(str(x['n_features_used']) for x in s):<24} "
              f"top-3 net {SD._iv(r['top3_net'])} (stage-D scoring)")
    print(f"\n  outputs: {out}\n  research ledger: entry {n} (stage_d2_run {run_id})")
    print("=" * 76)
    print("D2 DONE. Next: compare_d_variants.py decides, with corrected exits, whether D2 replaces D.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
