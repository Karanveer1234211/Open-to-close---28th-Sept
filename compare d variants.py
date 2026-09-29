#!/usr/bin/env python3
"""
compare_d_variants.py - score stage-D runs IDENTICALLY, with honest exits, and
decide by the declared rule whether a variant replaces the baseline.

    python compare_d_variants.py --panel %CACHE_DAILY_ROOT%\\panel_oc --root %CACHE_DAILY_ROOT%
    python compare_d_variants.py --panel ... --root ... --runs STAGED_20260928_001 STAGED2_20260929_001 ^
        --isin C:\\QuantData\\nse\\BhavCopy_NSE_CM_0_0_0_20260928_F_0000.csv.zip

The first run is the BASELINE (default: the latest stage-D run); the others
(default: the latest D2 run) are variants. For every run and target:
  * each day's top-3 is re-drawn from that run's own scores without ETFs
    (by ISIN when a bhavcopy is given, else the name rule + universe_exclude.txt);
  * an exit that closes locked at the lower band moves to the first later open
    that is not itself locked (forensics rule 2, tick-aware band test);
  * net = outcome - 35 bp; excess = net minus buying every eligible stock.
Every pick's outcome is first re-derived from the raw cache (>= 99.5% or stop).

Decision (STAGE_D2_DECLARATION.json): a variant replaces the baseline for a
target only if it passes the stage-D gate under this corrected scoring AND the
paired daily difference (variant - baseline) has a 98.75% block-bootstrap lower
bound above zero. Otherwise the baseline stays.

Output: <panel>\\stage_d\\compare\\COMPARE_YYYYMMDD_NNN\\ compare_report.md,
compare.json, daily_corrected.csv; ledger entry stage_d_compare.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import research_common as RC       # first: from a subfolder, this folder's copy must win
import panel_build as PB           # noqa: E402,F401
import stage_b_screen as SB        # noqa: E402
import stage_d_gate as SD          # noqa: E402
import stage_d_forensics as SF     # noqa: E402
import stage_d2_nested as SD2      # noqa: E402

CODE_VERSION = "compare_d_variants v1.2"   # v1.2: C-only and ENS arms (stage DX); pick-agreement diagnostic
RUN_KINDS = ("stage_d_gate", "stage_d2_run", "stage_dc_run", "stage_dx_c_run", "stage_dx_ens_run")


class PreconditionError(SystemExit):
    pass


def _log(msg: str) -> None:
    print(f"{dt.datetime.now():%H:%M:%S}  {msg}", flush=True)


def resolve_runs(pp: Path, runs: Optional[List[str]]) -> List[str]:
    E = RC.ledger_entries(pp)
    known = {e["run_id"]: e["kind"] for e in E if e.get("kind") in RUN_KINDS}
    if runs:
        bad = [r for r in runs if r not in known]
        if bad:
            raise PreconditionError(f"not stage-D runs in this panel's ledger: {bad}")
        return runs
    base = [e["run_id"] for e in E if e.get("kind") == "stage_d_gate"]
    var = [[e["run_id"] for e in E if e.get("kind") == k]
           for k in ("stage_d2_run", "stage_dc_run", "stage_dx_c_run", "stage_dx_ens_run")]
    var = [v[-1] for v in var if v]
    if not base or not var:
        raise PreconditionError("need a stage-D run and a D2 or D_C run on record (or pass --runs)")
    return [base[-1]] + var


def label_of(run_id: str) -> str:
    for pre, lab in (("STAGEDXE", "ENS"), ("STAGEDXC", "C"), ("STAGEDC", "DC"), ("STAGED2", "D2")):
        if run_id.startswith(pre):
            return lab
    return "D"


def pick_agreement(pa: pd.DataFrame, pb: pd.DataFrame, tg: str) -> dict:
    """Descriptive: corrected net of picks both runs chose, baseline-only picks and variant-only picks."""
    a = pa[pa["target"] == tg][["timestamp", "symbol", "net_corr"]]
    b = pb[pb["target"] == tg][["timestamp", "symbol", "net_corr"]]
    m = a.merge(b, on=["timestamp", "symbol"], how="outer", suffixes=("_a", "_b"), indicator=True)
    out = {}
    for key, sel, col in (("both", "both", "net_corr_a"), ("baseline_only", "left_only", "net_corr_a"),
                          ("variant_only", "right_only", "net_corr_b")):
        v = m.loc[m["_merge"] == sel, col].astype(float)
        out[key] = {"picks": int(len(v)), "mean_net_bp": float(v.mean() * 1e4) if v.notna().any() else float("nan")}
    return out


def score_run(pp: Path, root: Path, folder: Path, targets: List[str], etf: set, cfg: dict,
              verbose: bool = True) -> dict:
    """Corrected daily top-3 of one run: ETFs re-drawn out, locked exits moved to the first sellable open."""
    from data_quality import _paths
    preds = pd.read_parquet(folder / "preds.parquet").rename(columns={"outcome": "outcome_stage_d"})
    preds["timestamp"] = SB._naive(preds["timestamp"])
    preds["symbol"] = preds["symbol"].astype(str).astype("category")
    preds["target"] = preds["target"].astype(str).astype("category")
    daily = pd.read_csv(folder / "daily.csv", parse_dates=["timestamp"])
    picks = pd.concat([SF.draw_top(preds[preds["target"] == tg], etf, 3).assign(target=tg).astype({"target": str})
                       for tg in targets], ignore_index=True)
    parts = []
    for s in sorted(picks["symbol"].unique()):
        fp, _ = _paths(root, s)
        sub = picks[picks["symbol"] == s]
        if not Path(fp).exists():
            parts.append(sub.assign(outcome_cache=np.nan, outcome_lock=np.nan, locked_exit=False))
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
            r = SF.rescore_symbol(o, h, l, c, i, SD.horizon_of(tg), SF.CFG["lock_search_sessions"], ts)
            parts.append(g.assign(outcome_cache=r["outcome"], outcome_lock=r["outcome_lock"],
                                  locked_exit=r["locked_exit"]))
    T = pd.concat(parts, ignore_index=True)
    T["locked_exit"] = T["locked_exit"].astype("boolean").fillna(False).astype(bool)
    ok = (np.isfinite(T["outcome_cache"]) & np.isfinite(T["outcome_stage_d"])
          & (np.abs(T["outcome_cache"] - T["outcome_stage_d"]) <= 1e-6)) | \
         (~np.isfinite(T["outcome_cache"]) & ~np.isfinite(T["outcome_stage_d"]))
    match = float(ok.mean()) if len(T) else float("nan")
    if not match >= SF.CFG["min_outcome_match"]:
        raise SF.IntegrityError(f"{folder.name}: only {match:.2%} of picks re-derive from the raw cache")
    cost = cfg["cost_bps"] / 1e4
    T["net_corr"] = T["outcome_lock"] - cost
    out = {"match_rate": match, "targets": {}, "picks": T}
    for tg in targets:
        g = T[T["target"] == tg]
        dd = daily[daily["target"] == tg].set_index("timestamp").sort_index()
        day = pd.DataFrame({"net_stage_d": dd["top3_net"],
                            "net_corr": g.groupby("timestamp")["net_corr"].mean().reindex(dd.index)})
        day["excess_corr"] = day["net_corr"] - (dd["market"] - cost)
        out["targets"][tg] = {"daily": day, "locked_exits": int(g["locked_exit"].sum()), "picks": int(len(g))}
    return out


def gate(day: pd.DataFrame, cfg: dict) -> dict:
    lvl, B, blk, seed = cfg["interval_level"], cfg["boot_B"], cfg["boot_block"], cfg["seed"]
    r = {"net": SD.block_bootstrap(day["net_corr"].to_numpy(), lvl, B, blk, seed),
         "excess": SD.block_bootstrap(day["excess_corr"].to_numpy(), lvl, B, blk, seed),
         "stage_d_scoring_mean": float(day["net_stage_d"].mean())}
    rec = day.index >= pd.Timestamp(cfg["recency_start"])
    r["recent_mean"] = float(day.loc[rec, "net_corr"].mean())
    r["post_lockbox_mean"] = float(day.loc[day.index >= pd.Timestamp(SB.LOCKBOX_START_PINNED), "net_corr"].mean())
    r["by_year"] = {int(y): float(v) for y, v in day["net_corr"].groupby(day.index.year).mean().items()}
    r["pass"] = bool(np.isfinite(r["net"]["lo"]) and r["net"]["lo"] > 0 and r["recent_mean"] > 0
                     and np.isfinite(r["excess"]["lo"]) and r["excess"]["lo"] > 0)
    return r


def compare(scored: Dict[str, dict], runs: List[str], targets: List[str], cfg: dict, level: float) -> dict:
    base = runs[0]
    out = {"baseline": base, "runs": {}, "decisions": {}}
    for r in runs:
        out["runs"][r] = {tg: gate(scored[r]["targets"][tg]["daily"], cfg) for tg in targets}
        for tg in targets:
            out["runs"][r][tg]["locked_exits"] = scored[r]["targets"][tg]["locked_exits"]
    for r in runs[1:]:
        out["decisions"][r] = {}
        for tg in targets:
            a = scored[base]["targets"][tg]["daily"]["net_corr"]
            b = scored[r]["targets"][tg]["daily"]["net_corr"]
            j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna().sort_index()
            xa = scored[base]["targets"][tg]["daily"]["excess_corr"]
            xb = scored[r]["targets"][tg]["daily"]["excess_corr"]
            jx = pd.concat([xa.rename("a"), xb.rename("b")], axis=1).dropna()
            diff = SD.block_bootstrap((j["b"] - j["a"]).to_numpy(), level, cfg["boot_B"], cfg["boot_block"], cfg["seed"])
            pa = scored[base]["picks"]
            pb = scored[r]["picks"]
            ka = set(zip(pa.loc[pa["target"] == tg, "timestamp"], pa.loc[pa["target"] == tg, "symbol"]))
            kb = set(zip(pb.loc[pb["target"] == tg, "timestamp"], pb.loc[pb["target"] == tg, "symbol"]))
            overlap = len(ka & kb) / max(len(ka), 1)
            vpass = out["runs"][r][tg]["pass"]
            replace = bool(vpass and np.isfinite(diff["lo"]) and diff["lo"] > 0)
            out["decisions"][r][tg] = {"paired_diff": diff, "variant_passes_gate": vpass,
                                       "pick_overlap_with_baseline": overlap, "replaces_baseline": replace,
                                       "pick_agreement": pick_agreement(pa, pb, tg),
                                       "daily_pnl_corr": float(j["a"].corr(j["b"])),
                                       "daily_excess_corr": float(jx["a"].corr(jx["b"]))}
    return out


def _bp(x) -> str:
    return "nan" if x is None or not np.isfinite(x) else f"{x * 1e4:+.1f}"


def _iv(b) -> str:
    return f"{_bp(b['mean'])} [{_bp(b['lo'])}, {_bp(b['hi'])}]"


def write_report(C: dict, scored: dict, runs: List[str], targets: List[str], path: Path, run_id: str,
                 etf_note: str, labels: Dict[str, str]) -> None:
    L = [f"# Stage-D variants compared - {run_id}", "", f"{CODE_VERSION} | {dt.datetime.now():%Y-%m-%d %H:%M}", "",
         "**Scoring (identical for every run):** each day's top-3 re-drawn from the run's own scores without ETFs "
         f"({etf_note}); exits that close locked at the lower band moved to the first sellable open; 35 bp.", "",
         "**Decision (declared before D2 was fitted):** a variant replaces the baseline for a target only if it "
         "passes the stage-D gate under this scoring AND its paired daily difference over the baseline has a "
         "98.75% lower bound above zero.", "",
         "Integrity: " + "; ".join(f"{labels[r]} {scored[r]['match_rate']:.2%}" for r in runs)
         + " of picks re-derive from the raw cache.", "",
         "## 1. Each run under corrected scoring (bp per trade)", "",
         "| target | run | stage-D scoring | corrected net [98.75%] | corrected excess [98.75%] | 2024+ | "
         "since 2025-03-05 | locked exits | gate |", "|---|---|---|---|---|---|---|---|---|"]
    for tg in targets:
        for r in runs:
            g = C["runs"][r][tg]
            L.append(f"| {tg} | {labels[r]} | {_bp(g['stage_d_scoring_mean'])} | {_iv(g['net'])} | {_iv(g['excess'])} | "
                     f"{_bp(g['recent_mean'])} | {_bp(g['post_lockbox_mean'])} | {g['locked_exits']} | "
                     f"{'PASS' if g['pass'] else 'FAIL'} |")
    L += ["", "## 2. Decisions (paired: variant minus baseline, same days)", ""]
    for r in runs[1:]:
        L += [f"**{labels[r]} vs {labels[runs[0]]}**", "",
              "| target | paired difference [98.75%] | variant passes gate | same picks | DECISION |",
              "|---|---|---|---|---|"]
        for tg in targets:
            d = C["decisions"][r][tg]
            L.append(f"| {tg} | {_iv(d['paired_diff'])} | {'yes' if d['variant_passes_gate'] else 'no'} | "
                     f"{d['pick_overlap_with_baseline']:.0%} | **{'REPLACE' if d['replaces_baseline'] else 'KEEP BASELINE'}** |")
        L.append("")
    L += ["## 3. Pick agreement (descriptive - decides nothing)", "",
          "Each day's corrected top-3 of the baseline and of each variant: stocks both chose, baseline-only, "
          "variant-only; mean corrected net per pick (bp). Daily P&L correlation with the baseline, and the same "
          "with the market's move removed (excess) - the plain one is inflated by shared market exposure.", ""]
    for r in runs[1:]:
        L += [f"**{labels[r]} vs {labels[runs[0]]}**", "",
              "| target | both (n, net) | baseline-only (n, net) | variant-only (n, net) | daily P&L corr | excess corr |",
              "|---|---|---|---|---|---|"]
        for tg in targets:
            d = C["decisions"][r][tg]
            ag = d["pick_agreement"]
            cell = lambda k: f"{ag[k]['picks']:,}, {ag[k]['mean_net_bp']:+.1f}"
            L.append(f"| {tg} | {cell('both')} | {cell('baseline_only')} | {cell('variant_only')} | {d['daily_pnl_corr']:.2f} | "
                     f"{d['daily_excess_corr']:.2f} |")
        L.append("")
    L += ["## 4. Corrected net by year (bp)", ""]
    for tg in targets:
        for r in runs:
            by = C["runs"][r][tg]["by_year"]
            L.append(f"- **{tg} {labels[r]}**: " + "; ".join(f"{y} {_bp(v)}" for y, v in by.items()))
    L += ["", "## 5. What this does not say", "",
          "- Tick costs and trade-for-trade restrictions are not in this scoring (stage_d_forensics covers ticks).",
          "- Survivorship is unmeasured. Only the forward paper ledger is untouched evidence.", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Compare stage-D runs with corrected exits")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--root", required=True, help="raw cache root")
    ap.add_argument("--runs", nargs="+", default=None, help="baseline first (default: latest D, latest D2)")
    ap.add_argument("--isin", nargs="+", default=None, help="NSE bhavcopy file(s); optional")
    a = ap.parse_args(argv)
    pp = Path(a.panel) / "panel.parquet"
    root = Path(a.root)
    man, man_sha = SB.load_manifest()
    targets = list(man["targets"].keys())
    SB.check_panel(pp, man_sha)
    decl, d2sha = SD2.load_declaration()
    level = decl["config"]["switch_interval_level"]
    runs = resolve_runs(pp, a.runs)
    folders = {r: pp.parent / "stage_d" / r for r in runs}
    for r, f in folders.items():
        for x in ("preds.parquet", "daily.csv"):
            if not (f / x).exists():
                raise PreconditionError(f"{f / x} is missing")
    extra = RC.load_symbol_list(root / RC.UNIVERSE_EXCLUDE_FILE)
    syms = pd.read_parquet(pp, columns=["symbol"])["symbol"].astype(str).unique()
    if a.isin:
        cls = SF.classify_symbols(syms, SF.load_isin_files(a.isin), extra)
        etf_note = "by ISIN from " + ", ".join(Path(x).name for x in a.isin)
    else:
        cls = pd.DataFrame({"symbol": syms, "etf": [RC.is_etf_symbol(s, extra) for s in syms]})
        etf_note = "by the name rule - no bhavcopy given, so ETFs with plain names stay in"
    etf = set(cls.loc[cls["etf"], "symbol"])
    labels = {r: label_of(r) + f" ({r})" for r in runs}
    t0 = time.perf_counter()
    scored = {}
    for r in runs:
        _log(f"scoring {labels[r]} ...")
        scored[r] = score_run(pp, root, folders[r], targets, etf, SD.CFG)
    C = compare(scored, runs, targets, SD.CFG, level)
    out_root = pp.parent / "stage_d" / "compare"
    day = dt.date.today().strftime("%Y%m%d")
    k = 1
    while (out_root / f"COMPARE_{day}_{k:03d}").exists():
        k += 1
    run_id = f"COMPARE_{day}_{k:03d}"
    out = out_root / run_id
    out.mkdir(parents=True)
    rows = []
    for r in runs:
        for tg in targets:
            d = scored[r]["targets"][tg]["daily"].reset_index()
            rows.append(d.assign(run=r, target=tg))
    pd.concat(rows, ignore_index=True).to_csv(out / "daily_corrected.csv", index=False)
    (out / "compare.json").write_text(json.dumps({"run_id": run_id, "tool": CODE_VERSION, "runs": runs,
                                                  "etf_rule": etf_note, "d2_declaration_sha": d2sha,
                                                  "comparison": C, "seconds": round(time.perf_counter() - t0, 1)},
                                                 indent=2, default=str), encoding="utf-8")
    write_report(C, scored, runs, targets, out / "compare_report.md", run_id, etf_note, labels)
    dec = {r: {tg: C["decisions"][r][tg]["replaces_baseline"] for tg in targets} for r in runs[1:]}
    n = RC.ledger_append(pp, {"kind": "stage_d_compare", "tool": CODE_VERSION, "run_id": run_id, "runs": runs,
                              "d2_declaration_sha": d2sha, "decisions": dec, "etf_rule": etf_note})
    print()
    print(f"{'target':<12}{'run':<26}{'corrected net [98.75%]':>30}{'2024+':>9}  gate")
    for tg in targets:
        for r in runs:
            g = C["runs"][r][tg]
            print(f"{tg:<12}{labels[r]:<26}{_iv(g['net']):>30}{_bp(g['recent_mean']):>9}  {'PASS' if g['pass'] else 'FAIL'}")
    for r in runs[1:]:
        print(f"\n{labels[r]} vs {labels[runs[0]]} (paired, 98.75%):")
        for tg in targets:
            d = C["decisions"][r][tg]
            print(f"  {tg:<12}{_iv(d['paired_diff']):>28}  same picks {d['pick_overlap_with_baseline']:.0%}  -> "
                  f"{'REPLACE' if d['replaces_baseline'] else 'KEEP BASELINE'}")
    print(f"\n  ETFs: {etf_note}\n  outputs: {out}\n  research ledger: entry {n} (stage_d_compare {run_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
