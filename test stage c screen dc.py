#!/usr/bin/env python3
"""
tests/test_stage_c_screen_dc.py - stage_c_screen v1 + stage_dc_model v1 + compare v1.1.
Prints VERIFIED on success.

Run from the open_close folder with the main folder on PYTHONPATH:
    python tests\\test_stage_c_screen_dc.py

 0. Module origin; stage B v1.2 screens an external source exactly as it screens the panel.
 1. Clean world: a C feature carrying a real oc_3 signal passes the C screen; market-level C
    columns are recognised; no leak suspects; D_C then runs on clean + all 123 C features with
    stage D's settings unchanged, in stage D's output format; ledger order.
 2. Leaky world: a C feature that knows the next overnight gap, and one that knows tomorrow's
    return, are both caught; D_C refuses to run.
 3. Guards: a C file built from another declaration, or with rows out of panel order, is refused;
    the comparison tool now resolves D_C runs and labels them.
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

import research_common as RC          # noqa: E402
import stage_b_screen as SB           # noqa: E402
import stage_d_gate as SD             # noqa: E402
import stage_c_features as SC         # noqa: E402
import stage_c_screen as SCS          # noqa: E402
import stage_dc_model as SDC          # noqa: E402
import compare_d_variants as CV       # noqa: E402
import test_stage_d as TD             # noqa: E402

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


def world(tmp: Path, leaky=False, misalign=False, wrong_sha=False):
    pdir = TD.make_world(tmp)
    pp = pdir / "panel.parquet"
    P = pd.read_parquet(pp)
    rng = np.random.default_rng(21)
    P["label_gap1"] = 0.01 * rng.normal(size=len(P))
    P.to_parquet(pp, index=False)
    m = json.loads((pdir / "panel_meta.json").read_text())
    m["built_at"] = (pd.Timestamp.now() - pd.Timedelta(hours=2)).isoformat()
    (pdir / "panel_meta.json").write_text(json.dumps(m))
    _, man_sha = SB.load_manifest()
    RC.ledger_append(pp, {"kind": "panel_audit", "passed": True, "manifest_sha": man_sha})
    RC.ledger_append(pp, {"kind": "stage_b_screen", "experiment_family": "oc_v1", "run_id": "B"})
    RC.ledger_append(pp, {"kind": "stage_d_gate", "experiment_family": "oc_v1", "run_id": "STAGED_X"})
    decl, sha, names = SC.load_declaration()
    day_code = pd.factorize(P["timestamp"])[0]
    cols = {"timestamp": P["timestamp"].to_numpy(), "symbol": P["symbol"].to_numpy()}
    for nm in names:
        if nm.startswith("M_"):
            cols[nm] = rng.normal(size=day_code.max() + 1)[day_code].astype("float32")
        else:
            cols[nm] = rng.normal(size=len(P)).astype("float32")
    F = pd.DataFrame(cols)
    # a real but realistic oc_3 signal (IC about 0.1, below the 0.15 leak alarm)
    F["C_on_sum_5"] = (P["D_e3"] + 8.0 * rng.normal(size=len(P))).astype("float32")
    if leaky:
        F["C_id_sum_5"] = (P["label_gap1"] + 0.001 * rng.normal(size=len(P))).astype("float32")
        s = P.sort_values(["symbol", "timestamp"])
        nxt = (s.groupby("symbol")["close"].shift(-1) / s["close"] - 1).reindex(P.index)
        F["C_tug_20"] = nxt.astype("float32")
    if misalign:
        F = F.sample(frac=1.0, random_state=1).reset_index(drop=True)
    cdir = pdir / "stage_c"
    cdir.mkdir()
    F.to_parquet(cdir / "features_c.parquet", index=False)
    (cdir / "features_c_meta.json").write_text(json.dumps({"declaration_sha": "0" * 16 if wrong_sha else sha}))
    RC.ledger_append(pp, {"kind": "stage_c_features", "declaration_sha": sha})
    return pdir, pp, names


def main():
    print("0. module origin; stage B screens an external source the same way")
    here = HERE.parent.resolve()
    for mo in (SCS, SDC, SC, SB, SD, CV, RC):
        check(Path(mo.__file__).resolve().parent == here, f"{mo.__name__} from {Path(mo.__file__).resolve().parent}")
    tmp = Path(tempfile.mkdtemp())
    try:
        pdir, pp, names = world(tmp / "clean")
        a, _ = quiet(SB.run_screen, pp, TARGETS, False, ["D_e5"], None)
        b, _ = quiet(SB.run_screen, pp, TARGETS, False, ["D_e5"], lambda f: SB._read_col(pp, f))
        ca, cb = a["cells"].set_index("target"), b["cells"].set_index("target")
        check(np.allclose(ca["mean_ic"], cb["mean_ic"], atol=0, rtol=0) and np.allclose(ca["t_hac"], cb["t_hac"]),
              "an external reader gives exactly stage B's numbers for the same column")

        print("1. clean world")
        rc, log = quiet(SCS.main, ["--panel", str(pdir)])
        out = sorted((pdir / "stage_c" / "screen").iterdir())[-1]
        J = json.loads((out / "stage_c_screen.json").read_text())
        C = pd.read_csv(out / "screen_c.csv")
        r3 = C[(C["feature"] == "C_on_sum_5") & (C["target"] == "label_oc_3")].iloc[0]
        check(bool(r3["passes"]) and 0.03 < r3["mean_ic"] < 0.15,
              f"the planted C signal passes on oc_3 (IC {r3['mean_ic']:+.3f}, below the leak alarm)")
        noise = C[~C["feature"].isin(["C_on_sum_5"])]
        check(int(noise["passes"].sum()) <= max(3, int(0.02 * len(noise))),
              f"pure-noise C features: {int(noise['passes'].sum())} of {len(noise)} cells pass (BH at q <= 0.10)")
        check(set(J["market_level"]) == {n for n in names if n.startswith("M_")}, "all 10 market-state columns recognised as market-level")
        check(J["leak_suspects"] == [] and "No leak suspects" in log, "no leak suspects in a clean world")
        check((out / "stage_c_report.md").exists() and (out / "leak_c.csv").exists(), "report and leak table written")
        rc, log = quiet(SDC.main, ["--panel", str(pdir)])
        run = sorted((pdir / "stage_d").glob("STAGEDC_*"))[-1]
        for fn in ("preds.parquet", "daily.csv", "picks.csv", "stage_d.json"):
            check((run / fn).exists(), f"D_C output {fn}")
        js = json.loads((run / "stage_d.json").read_text())
        n_base = len(SB.clean_feature_list(pp)[0])
        check(js["n_c"] == 123 and js["n_base"] == n_base and len(js["features"]) == n_base + 123,
              f"D_C used {n_base} clean + 123 C features, no selection")
        check(js["config"] == json.loads(json.dumps(SD.CFG)), "D_C model settings are stage D's, unchanged")
        pr = pd.read_parquet(run / "preds.parquet")
        check(str(pr["score"].dtype) == "float64", "predictions stored in float64 (exactly auditable)")
        kinds = [e.get("kind") for e in RC.ledger_entries(pp)]
        check(kinds.index("stage_c_screen") < kinds.index("stage_dc_declared") < kinds.index("stage_dc_run"),
              "ledger: C screen -> D_C declared -> D_C run")
        try:
            quiet(SDC.main, ["--panel", str(pdir)])
            check(False, "a second D_C run without --rerun-reason is refused")
        except SystemExit as e:
            check("already ran" in str(e), "a second D_C run without --rerun-reason is refused")
        runs = CV.resolve_runs(pp, None)
        check(runs[0] == "STAGED_X" and runs[-1] == run.name, f"compare resolves D as baseline and the D_C run ({runs})")

        print("2. leaky world")
        pdir2, pp2, _ = world(tmp / "leaky", leaky=True)
        rc, log = quiet(SCS.main, ["--panel", str(pdir2)])
        J2 = json.loads((sorted((pdir2 / "stage_c" / "screen").iterdir())[-1] / "stage_c_screen.json").read_text())
        check("C_id_sum_5" in J2["leak_suspects"], "a feature that knows the next overnight gap is caught")
        check("C_tug_20" in J2["leak_suspects"], "a feature that knows tomorrow's return is caught")
        check("LEAK SUSPECTS FOUND" in log, "the console says D_C will refuse")
        try:
            quiet(SDC.main, ["--panel", str(pdir2)])
            check(False, "D_C refuses after a leaky screen")
        except SystemExit as e:
            check("flagged possible leaks" in str(e), "D_C refuses after a leaky screen")

        print("3. guards")
        pdir3, pp3, n3 = world(tmp / "misaligned", misalign=True)
        try:
            SCS.check_features(pp3, SC.load_declaration()[1], n3)
            check(False, "a C file out of panel row order is refused")
        except SystemExit as e:
            check("row for row" in str(e), "a C file out of panel row order is refused")
        pdir4, pp4, n4 = world(tmp / "wrongsha", wrong_sha=True)
        try:
            SCS.check_features(pp4, SC.load_declaration()[1], n4)
            check(False, "a C file from another declaration is refused")
        except SystemExit as e:
            check("built from declaration" in str(e), "a C file from another declaration is refused")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if FAILS:
        print(f"FAILED  {len(FAILS)} check(s):")
        for f in FAILS:
            print("   - " + f)
        return 1
    print(f"VERIFIED  {SCS.CODE_VERSION} + {SDC.CODE_VERSION} + {CV.CODE_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
