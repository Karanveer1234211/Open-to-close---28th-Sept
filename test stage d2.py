#!/usr/bin/env python3
"""
tests/test_stage_d2.py - stage_d2_nested v1. Prints VERIFIED on success.

Run from the open_close folder with the main folder on PYTHONPATH:
    python tests\\test_stage_d2.py

 0. Module origin.
 1. The declaration: settings must match; stage D's model declaration is re-verified.
 2. The nested screen on planted arrays: a real signal passes with the IC scipy computes;
    noise does not; a signal whose sign flips across training blocks fails stability; a
    near-copy of the real signal joins its family (only the leader is used); a market-level
    column is kept but never screened.
 3. End to end on a planted panel: selection happens on training rows ONLY (instrumented: the
    latest row the screen sees is before each fold's test window); a feature that predicts
    only inside the last test window is never selected; the planted per-target signals are;
    the model settings are stage D's exactly; stage D's output format; ledger order;
    refusal before stage D has run; re-run guard.
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

import research_common as RC      # noqa: E402
import panel_build as PB          # noqa: E402
import stage_b_screen as SB       # noqa: E402
import stage_d_gate as SD         # noqa: E402
import stage_d2_nested as SD2     # noqa: E402
import test_stage_d as TD         # noqa: E402  (its planted panel)

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


def test_origin():
    print("0. module origin")
    here = HERE.parent.resolve()
    for m in (SD2, SD, SB, RC, PB):
        check(Path(m.__file__).resolve().parent == here, f"{m.__name__} from {Path(m.__file__).resolve().parent}")


def test_declaration():
    print("1. the declaration")
    decl, sha = SD2.load_declaration()
    check(decl["stage"] == "D2" and len(sha) == 16, f"selection settings match the D2 declaration (sha {sha})")
    _, dsha = SD.load_declaration()
    check(len(dsha) == 16, f"stage D's model declaration still verifies (sha {dsha})")
    orig = SD2.SEL
    try:
        SD2.SEL = dict(orig, q_max=0.2)
        try:
            SD2.load_declaration()
            check(False, "a changed selection setting is refused")
        except SystemExit as e:
            check("q_max" in str(e), "a changed selection setting is refused (names q_max)")
    finally:
        SD2.SEL = orig


def test_screen():
    print("2. the nested screen on planted arrays")
    from scipy.stats import spearmanr
    rng = np.random.default_rng(2)
    nD, per = 400, 60
    day = np.repeat(np.arange(nD), per)
    R = len(day)
    real = rng.normal(size=R)
    copy = real + 0.05 * rng.normal(size=R)
    noise = rng.normal(size=(R, 6))
    flip = rng.normal(size=R)
    sgn = np.where(day < nD * 0.4, 1.0, -1.0)
    mkt = np.repeat(rng.normal(size=nD), per)
    X = np.column_stack([real, copy, flip, mkt, noise]).astype(np.float32)
    feats = ["D_real", "D_copy", "D_flip", "MKT_x"] + [f"D_noise{i}" for i in range(6)]
    Y = {tg: 0.3 * real + 0.3 * sgn * flip + rng.normal(size=R) for tg in TARGETS}
    y = Y["label_oc_1"]
    S = SD2.nested_screen(X, feats, day, Y, np.arange(R), nD, SD2.SEL)
    st = S["targets"]["label_oc_1"]["stats"].set_index("feature")
    ref = np.mean([spearmanr(real[day == k], y[day == k]).statistic for k in range(nD)])
    check(abs(st.at["D_real", "mean_ic"] - ref) < 1e-12, f"IC of the real signal = scipy's ({ref:+.4f})")
    t = S["targets"]["label_oc_1"]
    check("D_real" in t["passers"] or "D_copy" in t["passers"], "the real signal passes")
    npass = sum(f.startswith("D_noise") for tg in TARGETS for f in S["targets"][tg]["passers"])
    check(npass <= 1, f"noise: {npass} of 24 noise cells pass (BH at q <= 0.10 allows the odd fluke)")
    check("D_flip" not in t["passers"] and int(st.at["D_flip", "sign_agree_blocks"]) < 4,
          f"a signal that flips sign across training blocks fails stability ({int(st.at['D_flip', 'sign_agree_blocks'])}/5)")
    lead = t["leaders"]
    check(len({"D_real", "D_copy"} & set(lead)) == 1 and {"D_real", "D_copy"} <= set(t["passers"]),
          f"the near-copy joins the real signal's family; only one leads ({lead})")
    check(S["market_level"] == ["MKT_x"] and "MKT_x" not in st.index, "the market-level column is kept, never screened")


def test_end_to_end():
    print("3. end to end on a planted panel")
    tmp = Path(tempfile.mkdtemp())
    try:
        pdir = TD.make_world(tmp)
        pp = pdir / "panel.parquet"
        P = pd.read_parquet(pp)
        # a feature that predicts oc_1 ONLY inside the last test window: nested selection must never see it
        last_start = pd.Timestamp("2025-06-01")
        z = np.random.default_rng(9).normal(size=len(P))
        late = (P["timestamp"] >= last_start).to_numpy()
        P["D_future_only"] = z
        P.loc[late, "label_oc_1"] = P.loc[late, "label_oc_1"] + 0.03 * z[late]
        P.to_parquet(pp, index=False)
        m = json.loads((pdir / "panel_meta.json").read_text())
        m["built_at"] = (pd.Timestamp.now() - pd.Timedelta(hours=2)).isoformat()
        (pdir / "panel_meta.json").write_text(json.dumps(m))
        _, sha = SB.load_manifest()
        RC.ledger_append(pp, {"kind": "panel_audit", "passed": True, "manifest_sha": sha})
        try:
            quiet(SD2.main, ["--panel", str(pdir)])
            check(False, "refused before stage D has run")
        except SystemExit as e:
            check("stage D has not run" in str(e), "refused before stage D has run")
        RC.ledger_append(pp, {"kind": "stage_b_screen", "experiment_family": "oc_v1", "run_id": "B"})
        RC.ledger_append(pp, {"kind": "stage_d_gate", "experiment_family": "oc_v1", "run_id": "STAGED_X"})

        seen = []
        orig = SD2.nested_screen

        def spy(X, feats, day, Y, rows, n_days, sel):
            seen.append(int(day[rows].max()))
            return orig(X, feats, day, Y, rows, n_days, sel)
        SD2.nested_screen = spy
        try:
            rc, log = quiet(SD2.main, ["--panel", str(pdir)])
        finally:
            SD2.nested_screen = orig
        out = sorted((pdir / "stage_d").glob("STAGED2_*"))[-1]
        for fn in ("preds.parquet", "daily.csv", "picks.csv", "stage_d.json", "selection.json"):
            check((out / fn).exists(), f"output {fn}")
        J = json.loads((out / "stage_d.json").read_text())
        Sel = pd.DataFrame(json.loads((out / "selection.json").read_text()))
        F = pd.DataFrame([f for f in J["folds"] if "test_start" in f])
        sessions = np.unique(pd.read_parquet(pp, columns=["timestamp"])["timestamp"].to_numpy())
        ok = True
        for fo, mx in zip(sorted(F["fold"].unique()), seen):
            ts0 = pd.Timestamp(F.loc[F["fold"] == fo, "test_start"].iloc[0])
            ok &= pd.Timestamp(sessions[mx]) < ts0
        check(ok and len(seen) == 5, "every fold's screen saw only rows before its test window (5 folds, instrumented)")
        check(not any("D_future_only" in L for L in Sel["leaders"]),
              "a feature that predicts only inside the last test window is never selected")
        e5 = Sel[Sel["target"] == "label_oc_5"]
        check(all("D_e5" in L for L in e5["leaders"]), "the planted oc_5 signal is selected in every fold")
        check(all("MKT_level" in M for M in Sel["market_level"]), "market-level columns kept in every fold")
        check(not any("D_lock" in L for L in Sel["leaders"]),
              "a feature that works only on unbuyable rows is never selected (the screen sees eligible rows only)")
        check(J["config"] == json.loads(json.dumps(SD.CFG)), "model settings are stage D's, unchanged")
        check(J["results"]["label_oc_5"]["verdict"]["pass"] and not J["results"]["label_oc_1"]["verdict"]["pass"],
              "planted truths come through: oc_5 passes, noise oc_1 does not (stage-D scoring)")
        led = [e.get("kind") for e in RC.ledger_entries(pp)]
        check(led.index("stage_d2_declared") < led.index("stage_d2_run"), "ledger: declared before the run")
        try:
            quiet(SD2.main, ["--panel", str(pdir)])
            check(False, "a second run without --rerun-reason is refused")
        except SystemExit as e:
            check("already ran" in str(e), "a second run without --rerun-reason is refused")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    test_origin()
    test_declaration()
    test_screen()
    test_end_to_end()
    print()
    if FAILS:
        print(f"FAILED  {len(FAILS)} check(s):")
        for f in FAILS:
            print("   - " + f)
        return 1
    print(f"VERIFIED  {SD2.CODE_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
