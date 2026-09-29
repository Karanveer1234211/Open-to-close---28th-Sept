#!/usr/bin/env python3
"""
tests/test_stage_dx.py - stage_dx_ensemble v1 + stage_dc_model v1.1 + compare_d_variants v1.2.
Prints VERIFIED on success.

Run from the open_close folder with the main folder on PYTHONPATH:
    python tests\\test_stage_dx.py

 0. Module origin; the declaration must match.
 1. DX end to end on a planted world: the C arm uses the 123 C features alone with stage D's settings;
    the ensemble's every daily top-3 is re-derived by independent pandas (per-date percentile ranks of
    D's and C's scores, 50/50, ties by symbol); stage D's output format; ledger order; re-run guard;
    refusal without a clean C screen.
 2. The comparison v1.2: resolves and labels C and ENS runs; the pick-agreement diagnostic matches a
    hand example and independent pandas on real scored picks; an identical variant agrees 100%.
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import research_common as RC                 # noqa: E402
import stage_b_screen as SB                  # noqa: E402
import stage_d_gate as SD                    # noqa: E402
import stage_c_features as SC                # noqa: E402
import stage_c_screen as SCS                 # noqa: E402
import stage_dc_model as SDC                 # noqa: E402
import stage_dx_ensemble as SX               # noqa: E402
import compare_d_variants as CV              # noqa: E402
import stage_d_forensics as SF               # noqa: E402
import test_stage_c_screen_dc as TW          # noqa: E402  (clean / leaky worlds with C features)
import test_stage_d_forensics as TF          # noqa: E402  (world with a raw cache)

FAILS = []
TARGETS = ["label_oc_1", "label_oc_2", "label_oc_3", "label_oc_5"]


def check(cond, msg):
    print(("  ok  " if cond else "  FAIL") + f"  {msg}")
    if not cond:
        FAILS.append(msg)


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **k)
    return out, buf.getvalue()


def add_d_run(pp: Path, run_id="STAGED_X"):
    res, _ = quiet(SD.run_gate, pp, TARGETS, verbose=False)
    f = pp.parent / "stage_d" / run_id
    f.mkdir(parents=True, exist_ok=True)
    res["preds"].to_parquet(f / "preds.parquet", index=False)
    res["daily"].to_csv(f / "daily.csv", index=False)


def test_dx():
    print("0-1. DX end to end")
    here = HERE.parent.resolve()
    for mo in (SX, SDC, CV, SC, SCS, SB, SD, RC):
        check(Path(mo.__file__).resolve().parent == here, f"{mo.__name__} from {Path(mo.__file__).resolve().parent}")
    decl, sha = SX.load_declaration()
    check(decl["stage"] == "DX" and len(sha) == 16, f"settings match the DX declaration (sha {sha})")
    tmp = Path(tempfile.mkdtemp())
    try:
        pdir, pp, names = TW.world(tmp / "clean")
        add_d_run(pp)
        try:
            quiet(SX.main, ["--panel", str(pdir)])
            check(False, "refused before the C screen has run")
        except SystemExit as e:
            check("C screen" in str(e), "refused before the C screen has run clean")
        quiet(SCS.main, ["--panel", str(pdir)])
        rc, log = quiet(SX.main, ["--panel", str(pdir)])
        croot = pdir / "stage_d"
        crun = sorted(croot.glob("STAGEDXC_*"))[-1]
        erun = sorted(croot.glob("STAGEDXE_*"))[-1]
        for run in (crun, erun):
            for fn in ("preds.parquet", "daily.csv", "picks.csv", "stage_d.json"):
                check((run / fn).exists(), f"{run.name[:8]} output {fn}")
        cj = json.loads((crun / "stage_d.json").read_text())
        check(cj["features"] == names and cj["config"] == json.loads(json.dumps(SD.CFG)),
              "the C arm used exactly the 123 C features, with stage D's settings unchanged")
        dp = pd.read_parquet(croot / "STAGED_X" / "preds.parquet")
        cp = pd.read_parquet(crun / "preds.parquet")
        m = dp[["timestamp", "symbol", "target", "score"]].merge(
            cp[["timestamp", "symbol", "target", "score", "outcome"]], on=["timestamp", "symbol", "target"],
            suffixes=("_d", "_c"))
        m["e"] = 0.5 * m.groupby(["target", "timestamp"])["score_d"].rank(pct=True) + \
            0.5 * m.groupby(["target", "timestamp"])["score_c"].rank(pct=True)
        ed = pd.read_csv(erun / "daily.csv", parse_dates=["timestamp"])
        worst = 0.0
        for tg in TARGETS:
            g = m[m["target"] == tg].sort_values(["timestamp", "e", "symbol"], ascending=[True, False, True])
            t3 = g.groupby("timestamp").head(3).groupby("timestamp")["outcome"].mean() - 0.0035
            dd = ed[ed["target"] == tg].set_index("timestamp")["top3_net"].reindex(t3.index)
            worst = max(worst, float(np.nanmax(np.abs(t3.to_numpy() - dd.to_numpy()))))
        check(worst < 1e-12, f"every ensemble daily top-3 re-derived by pandas from the two runs' scores "
                             f"(max diff {worst:.1e})")
        ej = json.loads((erun / "stage_d.json").read_text())
        sc_ = ej["d_vs_c_daily_score_rank_corr"]
        ref = {}
        for tg in TARGETS:
            g = m[m["target"] == tg].copy()
            g["rd"] = g.groupby("timestamp")["score_d"].rank(pct=True)
            g["rc"] = g.groupby("timestamp")["score_c"].rank(pct=True)
            ref[tg] = float(np.nanmean(g.groupby("timestamp").apply(lambda x: x["rd"].corr(x["rc"])).to_numpy(float)))
        check(all(abs(sc_[tg] - ref[tg]) < 1e-12 for tg in TARGETS),
              "D-vs-C daily score rank correlation re-derived by pandas (" +
              ", ".join(f"{tg[-4:]} {sc_[tg]:+.2f}" for tg in TARGETS) + ")")
        ep = pd.read_parquet(erun / "preds.parquet")
        check(ep["score"].between(0, 1).all() and str(ep["score"].dtype) == "float64",
              "ensemble scores are rank averages in [0, 1], stored in float64")
        kinds = [e.get("kind") for e in RC.ledger_entries(pp)]
        check(kinds.index("stage_dx_declared") < kinds.index("stage_dx_c_run") < kinds.index("stage_dx_ens_run"),
              "ledger: declared -> C run -> ENS run")
        try:
            quiet(SX.main, ["--panel", str(pdir)])
            check(False, "a second DX run without --rerun-reason is refused (closure rule)")
        except SystemExit as e:
            check("closure rule" in str(e), "a second DX run without --rerun-reason is refused (closure rule)")
        runs = CV.resolve_runs(pp, None)
        check(runs[0] == "STAGED_X" and crun.name in runs and erun.name in runs and
              CV.label_of(crun.name) == "C" and CV.label_of(erun.name) == "ENS",
              f"the comparison resolves D as baseline plus the C and ENS runs, labelled C and ENS")
        orig = SX.CFG
        try:
            SX.CFG = dict(orig, ensemble_weights=[0.6, 0.4])
            try:
                SX.load_declaration()
                check(False, "different ensemble weights are refused")
            except SystemExit:
                check(True, "different ensemble weights are refused (the declaration is fixed)")
        finally:
            SX.CFG = orig
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_compare_diagnostic():
    print("2. comparison v1.2: pick agreement")
    a = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-03"] * 3), "target": "t",
                      "symbol": ["A", "B", "C", "A", "B", "C"], "net_corr": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]})
    b = pd.DataFrame({"timestamp": pd.to_datetime(["2024-01-02"] * 3 + ["2024-01-03"] * 3), "target": "t",
                      "symbol": ["B", "C", "D", "A", "E", "F"], "net_corr": [0.02, 0.03, -0.01, 0.04, 0.10, 0.20]})
    ag = CV.pick_agreement(a, b, "t")
    ok = (ag["both"]["picks"] == 3 and abs(ag["both"]["mean_net_bp"] - 300) < 1e-9
          and ag["baseline_only"]["picks"] == 3 and abs(ag["baseline_only"]["mean_net_bp"] - 400) < 1e-9
          and ag["variant_only"]["picks"] == 3 and abs(ag["variant_only"]["mean_net_bp"] - (-100 + 1000 + 2000) / 3) < 1e-9)
    check(ok, "hand example: both / baseline-only / variant-only counts and mean nets")
    tmp = Path(tempfile.mkdtemp())
    try:
        root, pdir, ip = TF.make_world(tmp)
        pp = pdir / "panel.parquet"
        base = pd.read_parquet(pdir / "stage_d" / "STAGED_T_001" / "preds.parquet")
        var = base.copy()
        m5 = var["target"] == "label_oc_5"
        var.loc[m5, "score"] = np.where(var.loc[m5, "symbol"].str.startswith("N"), var.loc[m5, "outcome"], -1.0)
        import test_compare_d as TCD
        TCD.add_run(pdir, "STAGEDXC_T_001", var)
        TCD.add_run(pdir, "STAGEDXE_T_001", base.copy())
        led = RC.ledger_entries(pp)
        kinds_fix = {"STAGEDXC_T_001": "stage_dx_c_run", "STAGEDXE_T_001": "stage_dx_ens_run"}
        lp = RC.ledger_path(pp)
        lines = [json.loads(x) for x in lp.read_text().splitlines()]
        for e in lines:
            if e.get("run_id") in kinds_fix:
                e["kind"] = kinds_fix[e["run_id"]]
        lp.write_text("\n".join(json.dumps(e) for e in lines) + "\n")
        rc, log = quiet(CV.main, ["--panel", str(pdir), "--root", str(root), "--isin", str(ip),
                                  "--runs", "STAGED_T_001", "STAGEDXC_T_001", "STAGEDXE_T_001"])
        out = sorted((pdir / "stage_d" / "compare").iterdir())[-1]
        J = json.loads((out / "compare.json").read_text())["comparison"]
        same = J["decisions"]["STAGEDXE_T_001"]
        check(all(same[tg]["pick_agreement"]["baseline_only"]["picks"] == 0 and
                  same[tg]["pick_agreement"]["variant_only"]["picks"] == 0 and abs(same[tg]["daily_pnl_corr"] - 1) < 1e-12
                  for tg in TARGETS), "an identical variant: every pick shared, daily P&L correlation 1.00")
        check(all(abs(same[tg]["daily_excess_corr"] - 1) < 1e-12 for tg in TARGETS),
              "an identical variant: market-removed correlation 1.00 too")
        dc_ = pd.read_csv(out / "daily_corrected.csv", parse_dates=["timestamp"])
        xa = dc_[(dc_["run"] == "STAGED_T_001") & (dc_["target"] == "label_oc_5")].set_index("timestamp")["excess_corr"]
        xb = dc_[(dc_["run"] == "STAGEDXC_T_001") & (dc_["target"] == "label_oc_5")].set_index("timestamp")["excess_corr"]
        jx = pd.concat([xa, xb], axis=1).dropna()
        check(abs(jx.iloc[:, 0].corr(jx.iloc[:, 1]) - J["decisions"]["STAGEDXC_T_001"]["label_oc_5"]["daily_excess_corr"]) < 1e-12,
              "market-removed correlation re-derived by pandas from daily_corrected.csv")
        cv5 = J["decisions"]["STAGEDXC_T_001"]["label_oc_5"]["pick_agreement"]
        etf = set(SF.classify_symbols(pd.read_parquet(pp, columns=["symbol"])["symbol"].unique(),
                                      SF.load_isin_files([str(ip)]), set()).query("etf")["symbol"])
        sa, _ = quiet(CV.score_run, pp, root, pdir / "stage_d" / "STAGED_T_001", TARGETS, etf, SD.CFG, False)
        sb, _ = quiet(CV.score_run, pp, root, pdir / "stage_d" / "STAGEDXC_T_001", TARGETS, etf, SD.CFG, False)
        A = sa["picks"][sa["picks"]["target"] == "label_oc_5"].set_index(["timestamp", "symbol"])["net_corr"]
        B = sb["picks"][sb["picks"]["target"] == "label_oc_5"].set_index(["timestamp", "symbol"])["net_corr"]
        both, bo, vo = A.index.intersection(B.index), A.index.difference(B.index), B.index.difference(A.index)
        ok = (cv5["both"]["picks"] == len(both) and cv5["baseline_only"]["picks"] == len(bo) and cv5["variant_only"]["picks"] == len(vo)
              and abs(cv5["variant_only"]["mean_net_bp"] - B.loc[vo].mean() * 1e4) < 1e-9
              and abs(cv5["baseline_only"]["mean_net_bp"] - A.loc[bo].mean() * 1e4) < 1e-9)
        check(ok, f"oc_5 agreement re-derived by pandas from scored picks (variant-only {len(vo)} picks at "
                  f"{cv5['variant_only']['mean_net_bp']:+.0f} bp vs baseline-only {cv5['baseline_only']['mean_net_bp']:+.0f} bp)")
        rep = (out / "compare_report.md").read_text(encoding="utf-8")
        check("Pick agreement" in rep and "C (STAGEDXC_T_001)" in rep and "ENS (STAGEDXE_T_001)" in rep and "excess corr" in rep,
              "report carries the agreement section and the C / ENS labels")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_dx()
    test_compare_diagnostic()
    print()
    if FAILS:
        print(f"FAILED  {len(FAILS)} check(s):")
        for f in FAILS:
            print("   - " + f)
        return 1
    print(f"VERIFIED  {SX.CODE_VERSION} + {SDC.CODE_VERSION} + {CV.CODE_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
