#!/usr/bin/env python3
"""
stage_c_screen.py - stage B's rule on the 123 stage-C features, as their own
family, plus two leak screens that must be clean before stage D_C may run.

    python stage_c_screen.py --panel %CACHE_DAILY_ROOT%\\panel_oc

SCREEN (declared in STAGE_C_DECLARATION.json, identical to stage B)
  per-day Spearman IC vs label_oc_1/2/3/5, Newey-West t (lag 20), BH q <= 0.10
  across all C cells, sign in >= 4 of 5 folds; research rows only (the pinned
  lockbox from 2025-03-05 is never read); eligible rows only. Market-level
  columns (group I) are listed, not screened. BH is over the C family alone -
  stage B's cells are not in it, so C gets its own error budget.

LEAK SCREENS (either one stops stage D_C)
  1. |IC| >= 0.15 against a target (stage B's alarm);
  2. |IC| >= 0.15 against label_gap1 - the next overnight gap, which a feature
     formed at the close of T cannot know;
  3. |rho| > 0.15 with any of the next 5 single-day returns (smoke test's screen).

OUTPUT: <panel>\\stage_c\\screen\\STAGEC_YYYYMMDD_NNN\\ screen_c.csv, leak_c.csv,
stage_c_screen.json, stage_c_report.md; ledger entry stage_c_screen.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import research_common as RC       # first: from a subfolder, this folder's copy must win
import panel_build as PB           # noqa: E402,F401
import stage_b_screen as SB        # noqa: E402
import stage_c_features as SC      # noqa: E402

CODE_VERSION = "stage_c_screen v1"
LEAK_IC = 0.15


class PreconditionError(SystemExit):
    pass


def _log(msg: str) -> None:
    print(f"{dt.datetime.now():%H:%M:%S}  {msg}", flush=True)


def features_file(pp: Path) -> Path:
    return pp.parent / "stage_c" / "features_c.parquet"


def check_features(pp: Path, sha: str, names: List[str]) -> None:
    """The C file must be built from THIS declaration and match the panel row for row."""
    import pyarrow.parquet as pq
    fp = features_file(pp)
    mp = fp.with_name("features_c_meta.json")
    if not fp.exists() or not mp.exists():
        raise PreconditionError(f"no C features at {fp} - run stage_c_features.py first")
    meta = json.loads(mp.read_text(encoding="utf-8"))
    if meta.get("declaration_sha") != sha:
        raise PreconditionError(f"features_c.parquet was built from declaration {meta.get('declaration_sha')}, "
                                f"not the current {sha}")
    have = [c for c in pq.ParquetFile(fp).schema_arrow.names if c not in ("timestamp", "symbol")]
    if have != names:
        raise PreconditionError("features_c.parquet columns differ from the declared C list")
    a = pd.read_parquet(fp, columns=["timestamp", "symbol"])
    b = pd.read_parquet(pp, columns=["timestamp", "symbol"])
    if len(a) != len(b) or not (SB._naive(a["timestamp"]).to_numpy() == SB._naive(b["timestamp"]).to_numpy()).all() \
            or not (a["symbol"].astype(str).to_numpy() == b["symbol"].astype(str).to_numpy()).all():
        raise PreconditionError("features_c.parquet rows do not match panel.parquet row for row - rebuild the C features")


def reader_for(pp: Path):
    import pyarrow.parquet as pq
    fp = features_file(pp)

    def read(name):
        return pd.to_numeric(pq.read_table(fp, columns=[name]).column(0).to_pandas(),
                             errors="coerce").to_numpy(dtype="float64")
    return read


def future_return_screen(pp: Path, names: List[str], read, research_end) -> pd.DataFrame:
    base = pd.read_parquet(pp, columns=["timestamp", "symbol", "close"])
    ts = SB._naive(base["timestamp"]).to_numpy(dtype="datetime64[ns]")
    sym = base["symbol"].astype(str).to_numpy()
    keep = ts <= np.datetime64(pd.Timestamp(research_end), "ns")
    order = np.lexsort((sym, ts))
    order = order[keep[order]]
    code = pd.factorize(sym[order])[0]
    close = base["close"].to_numpy(float)[order]
    cache: dict = {}
    rows = []
    for f in names:
        r = RC.screen_feature_leakage(read(f)[order], close, code, _cache=cache)
        rows.append({"feature": f, **r})
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage C screen + leak screens")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--rerun-reason", default=None)
    a = ap.parse_args(argv)
    pp = Path(a.panel) / "panel.parquet"
    man, man_sha = SB.load_manifest()
    targets = list(man["targets"].keys())
    SB.check_panel(pp, man_sha)
    decl, sha, names = SC.load_declaration()
    check_features(pp, sha, names)
    prior = [e for e in RC.ledger_entries(pp) if e.get("kind") == "stage_c_screen"]
    if prior and not a.rerun_reason:
        raise PreconditionError(f"the C screen already ran ({prior[-1].get('run_id')}); a re-run needs --rerun-reason")
    read = reader_for(pp)
    t0 = time.perf_counter()
    print("=" * 76)
    print(f"{CODE_VERSION} | C declaration {sha} | {len(names)} features, their own BH family")
    print("RULE: BH q <= 0.10 across all C cells AND sign in >= 4 of 5 folds (stage B's rule).")
    print("LEAK STOPS: |IC| >= 0.15 vs a target or vs label_gap1; |rho| > 0.15 vs the next 5 daily returns.")
    print("=" * 76)
    res = SB.run_screen(pp, targets, verbose=True, feats=names, reader=read)
    _log("leak screen 1: against the next overnight gap (label_gap1)")
    gap = SB.run_screen(pp, ["label_gap1"], verbose=False, feats=names, reader=read)
    _log("leak screen 2: against the next 5 single-day returns")
    fut = future_return_screen(pp, names, read, res["research_end"])
    C = res["cells"]
    G = gap["cells"][["feature", "mean_ic"]].rename(columns={"mean_ic": "ic_vs_gap1"})
    L = fut.merge(G, on="feature", how="left")
    tmax = C.groupby("feature")["mean_ic"].apply(lambda s: float(np.nanmax(np.abs(s))) if s.notna().any() else 0.0)
    L["max_abs_ic_vs_targets"] = L["feature"].map(tmax).fillna(0.0)
    L["suspect_target"] = L["max_abs_ic_vs_targets"] >= LEAK_IC
    L["suspect_gap1"] = L["ic_vs_gap1"].abs() >= LEAK_IC
    L["suspect_future"] = L["suspect"].astype(bool)
    L["leak_suspect"] = L[["suspect_target", "suspect_gap1", "suspect_future"]].any(axis=1)
    suspects = L.loc[L["leak_suspect"], "feature"].tolist()
    summ = SB.summarize(res, targets)
    group_of = {f["name"]: g for g, fs in decl["groups"].items() for f in fs}
    C["group"] = C["feature"].map(group_of)
    by_group = {g: {tg: int(C[(C["group"] == g) & (C["target"] == tg)]["passes"].sum()) for tg in targets}
                for g in decl["groups"]}

    out_root = pp.parent / "stage_c" / "screen"
    day = dt.date.today().strftime("%Y%m%d")
    k = 1
    while (out_root / f"STAGEC_{day}_{k:03d}").exists():
        k += 1
    run_id = f"STAGEC_{day}_{k:03d}"
    out = out_root / run_id
    out.mkdir(parents=True)
    C.to_csv(out / "screen_c.csv", index=False)
    L.to_csv(out / "leak_c.csv", index=False)
    js = {"run_id": run_id, "tool": CODE_VERSION, "declaration_sha": sha, "summary": summ, "by_group": by_group,
          "market_level": res["market_level"], "leak_suspects": suspects, "research_end": res["research_end"],
          "folds": res["folds"], "seconds": round(time.perf_counter() - t0, 1), "rerun_reason": a.rerun_reason}
    (out / "stage_c_screen.json").write_text(json.dumps(js, indent=2, default=str), encoding="utf-8")
    Lr = ["# Stage C screen - " + run_id, "", f"{CODE_VERSION} | C declaration {sha} | {len(names)} features", "",
          "**Rule:** BH q <= 0.10 across all C cells (their own family) AND sign in >= 4 of 5 folds. Research rows "
          "only; eligible rows only. Marginal evidence only - whether C helps the MODEL is stage D_C's question.", "",
          "## Summary", "", "| target | tested | q <= 0.10 | stable 4/5 | PASS | families |", "|---|---|---|---|---|---|"]
    for tg in targets:
        s_ = summ[tg]
        Lr.append(f"| {tg} | {s_['tested']} | {s_['q_pass']} | {s_['stable']} | **{s_['passes']}** | {s_['families']} |")
    Lr += ["", "## Passes by group", "", "| group | " + " | ".join(targets) + " |", "|---|" + "---|" * len(targets)]
    for g, d in by_group.items():
        Lr.append(f"| {g} | " + " | ".join(str(d[tg]) for tg in targets) + " |")
    for tg in targets:
        T = C[(C["target"] == tg) & C["passes"]].copy()
        if T.empty:
            continue
        T = T.reindex(T["t_hac"].abs().sort_values(ascending=False).index).head(12)
        Lr += ["", f"### {tg} - strongest passing C features", "",
               "| feature | group | IC | t | folds | quintile | quintile excess | net after cost |", "|---|---|---|---|---|---|---|---|"]
        for _, r in T.iterrows():
            Lr.append(f"| {r['feature']} | {r['group']} | {r['mean_ic']:+.4f} | {r['t_hac']:+.1f} | {int(r['sign_agree_folds'])}/5 | "
                      f"{r['long_side']} | {r['quintile_excess_bp']:+.1f} | {r['quintile_net_after_cost_bp']:+.1f} |")
    Lr += ["", "## Leak screens", "",
           f"Suspects: {', '.join(suspects) if suspects else 'none'}.", "",
           f"Largest |IC| vs the targets: {L['max_abs_ic_vs_targets'].max():.4f}; vs label_gap1: "
           f"{L['ic_vs_gap1'].abs().max():.4f}; largest |rho| with a future day: {L['max_abs_future_rho'].max():.4f}.", ""]
    (out / "stage_c_report.md").write_text("\n".join(Lr), encoding="utf-8")
    n = RC.ledger_append(pp, {"kind": "stage_c_screen", "tool": CODE_VERSION, "run_id": run_id,
                              "declaration_sha": sha, "passes": {tg: summ[tg]["passes"] for tg in targets},
                              "leak_suspects": len(suspects), "suspect_features": suspects,
                              "rerun_reason": a.rerun_reason})
    print()
    print(f"{'target':<12}{'tested':>8}{'q<=0.10':>9}{'stable':>8}{'PASS':>6}{'families':>10}")
    for tg in targets:
        s_ = summ[tg]
        print(f"{tg:<12}{s_['tested']:>8}{s_['q_pass']:>9}{s_['stable']:>8}{s_['passes']:>6}{s_['families']:>10}")
    print("\npasses by group: " + "; ".join(f"{g.split('_')[0]} " + "/".join(str(d[tg]) for tg in targets)
                                           for g, d in by_group.items()))
    print(f"\nLEAK SCREENS: largest |IC| vs targets {L['max_abs_ic_vs_targets'].max():.4f}, vs next gap "
          f"{L['ic_vs_gap1'].abs().max():.4f}, largest |rho| with a future day {L['max_abs_future_rho'].max():.4f}")
    print(f"  suspects: {', '.join(suspects) if suspects else 'none'}")
    print(f"  outputs: {out}\n  research ledger: entry {n} (stage_c_screen {run_id})")
    print("=" * 76)
    print("C SCREEN DONE. " + ("LEAK SUSPECTS FOUND - stage D_C will refuse to run. Send the report."
                               if suspects else "No leak suspects. Next: stage_dc_model.py."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
