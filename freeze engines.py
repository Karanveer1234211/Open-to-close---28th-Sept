#!/usr/bin/env python3
"""
freeze_engines.py - freeze the two engines chosen at the end of family oc_v1:
    D   (primary arm)     stage D's model on the 205 clean features
    ENS (challenger)      0.5 x rank(D) + 0.5 x rank(C-only)
both on label_oc_5.

    python freeze_engines.py --panel %CACHE_DAILY_ROOT%\\panel_oc

WHAT "FROZEN" MEANS
-------------------
The SHA-256 of every file the engines depend on is written to
<panel>\\frozen\\FROZEN_ENGINES.json and logged in the research ledger:
  * the walk-forward runs themselves (preds.parquet, daily.csv, stage_d.json
    of D, of the ensemble and of its C-only member);
  * the code and declarations that define them (stage D, C features, the
    ensemble, the panel build, the shared research code);
  * the panel's and the C features' metadata.
verify_frozen() recomputes every hash; the execution backtest and the paper
pipeline refuse to run if a single byte changed. A change is a new engine.

ROLES (fixed here, before any paper trade)
  D is the arm judged for real money at month 6. The ensemble runs alongside
  for information and can replace D only through a comparison declared in
  advance. Neither is picked by which looks better.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import research_common as RC     # first: from a subfolder, this folder's copy must win
import stage_b_screen as SB      # noqa: E402

CODE_VERSION = "freeze_engines v1"
LOCAL_FILES = ["stage_d_gate.py", "STAGE_D_DECLARATION.json", "stage_dx_ensemble.py", "STAGE_DX_DECLARATION.json",
               "stage_dc_model.py", "stage_c_features.py", "STAGE_C_DECLARATION.json", "stage_c_screen.py",
               "stage_b_screen.py", "stage_d_forensics.py", "research_common.py", "panel_build.py",
               "EXPERIMENT_OC_FAMILY.json"]
MAIN_MODULES = ["features_daily", "data_quality", "signal_engine", "feasibility_test"]
RUN_FILES = ["preds.parquet", "daily.csv", "stage_d.json"]


class PreconditionError(SystemExit):
    pass


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_path(pp: Path) -> Path:
    return pp.parent / "frozen" / "FROZEN_ENGINES.json"


def collect(pp: Path, d_run: str, ens_run: str, c_run: str) -> Dict[str, str]:
    files: Dict[str, str] = {}
    for f in LOCAL_FILES:
        p = HERE / f
        if not p.exists():
            raise PreconditionError(f"{f} not found next to {Path(__file__).name}")
        files[f"code/{f}"] = sha(p)
    for mname in MAIN_MODULES:
        mod = __import__(mname)
        files[f"code/main/{Path(mod.__file__).name}"] = sha(Path(mod.__file__))
    root = pp.parent / "stage_d"
    for tag, rid in (("D", d_run), ("ENS", ens_run), ("C", c_run)):
        for f in RUN_FILES:
            p = root / rid / f
            if not p.exists():
                raise PreconditionError(f"{p} is missing")
            files[f"runs/{tag}/{rid}/{f}"] = sha(p)
    for rel in ("panel_meta.json", "stage_c/features_c_meta.json"):
        p = pp.parent / rel
        if not p.exists():
            raise PreconditionError(f"{p} is missing")
        files[f"panel/{rel}"] = sha(p)
    return files


def verify_frozen(pp: Path) -> dict:
    """Recompute every hash in the freeze manifest; raise on any difference. Returns the manifest."""
    fp = frozen_path(pp)
    if not fp.exists():
        raise PreconditionError("the engines are not frozen - run freeze_engines.py first")
    man = json.loads(fp.read_text(encoding="utf-8"))
    now = collect(pp, man["engines"]["D"]["run"], man["engines"]["ENS"]["run"], man["engines"]["ENS"]["c_run"])
    bad = sorted(k for k in man["files"] if now.get(k) != man["files"][k])
    if bad:
        raise PreconditionError(f"frozen files changed since the freeze: {bad[:8]}{' ...' if len(bad) > 8 else ''}")
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Freeze engines D and ENS")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--d-run", default=None)
    ap.add_argument("--ens-run", default=None)
    ap.add_argument("--rerun-reason", default=None)
    a = ap.parse_args(argv)
    pp = Path(a.panel) / "panel.parquet"
    man_, man_sha = SB.load_manifest()
    SB.check_panel(pp, man_sha)
    E = RC.ledger_entries(pp)
    if frozen_path(pp).exists() and not a.rerun_reason:
        raise PreconditionError("engines already frozen; re-freezing needs --rerun-reason (it creates new engines)")
    d_runs = [e["run_id"] for e in E if e.get("kind") == "stage_d_gate"]
    ens = [e for e in E if e.get("kind") == "stage_dx_ens_run"]
    if not d_runs or not ens:
        raise PreconditionError("need a stage-D run and an ensemble run on record")
    d_run = a.d_run or d_runs[-1]
    ens_e = next((e for e in reversed(ens) if e["run_id"] == (a.ens_run or ens[-1]["run_id"])), None)
    if ens_e is None:
        raise PreconditionError(f"{a.ens_run} is not an ensemble run in the ledger")
    if ens_e.get("d_run") and ens_e["d_run"] != d_run:
        raise PreconditionError(f"the ensemble was built on D run {ens_e['d_run']}, not {d_run}")
    files = collect(pp, d_run, ens_e["run_id"], ens_e["c_run"])
    manifest = {
        "tool": CODE_VERSION, "frozen_at": dt.datetime.now().isoformat(), "family": "oc_v1", "target": "label_oc_5",
        "engines": {"D": {"run": d_run, "role": "primary - the arm judged for real money at month 6"},
                    "ENS": {"run": ens_e["run_id"], "c_run": ens_e["c_run"], "d_run": d_run,
                            "role": "challenger - runs alongside; replaces D only through a comparison declared in advance"}},
        "rules": ["no retraining, no feature, setting or code change: any change is a new engine",
                  "the paper-test confirmation rule is declared before the first paper trade"],
        "files": files, "rerun_reason": a.rerun_reason}
    fp = frozen_path(pp)
    fp.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(manifest, indent=2).encode("utf-8")
    fp.write_bytes(raw)
    n = RC.ledger_append(pp, {"kind": "engines_frozen", "tool": CODE_VERSION, "manifest_sha": hashlib.sha256(raw).hexdigest()[:16],
                              "d_run": d_run, "ens_run": ens_e["run_id"], "c_run": ens_e["c_run"],
                              "files": len(files), "rerun_reason": a.rerun_reason})
    print("=" * 76)
    print(f"{CODE_VERSION}: D = {d_run} (primary), ENS = {ens_e['run_id']} (challenger), target label_oc_5")
    print(f"  {len(files)} files hashed -> {fp}")
    print(f"  research ledger: entry {n} (engines_frozen)")
    print("=" * 76)
    print("FROZEN. Any change to these files is a new engine; later tools verify every hash.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
