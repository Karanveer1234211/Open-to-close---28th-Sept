#!/usr/bin/env python3
"""
tests/test_stage_e_paths.py - stage_e_paths v1. Prints VERIFIED on success.

Run from the open_close folder with the main folder on PYTHONPATH:
    python tests\\test_stage_e_paths.py

 0. Module origin; the declaration must match.
 1. Mechanics on hand-built paths: target first, stop first, both in one session (conservative =
    stop, optimistic = target, flagged ambiguous), gaps through the target and the stop, a stop
    that cannot fill on a frozen lower circuit, the plain time exit, a trailing stop, MFE / MAE and
    their sessions, sessions to target, ATR.
 2. Invariants on the planted world: a bracket that can never trigger equals the plain exit on
    every trade; conservative <= optimistic always; hit and touch rates fall as levels widen; the
    picks and plain-exit returns equal the execution simulator's research-convention trades.
 3. End to end: refused before a freeze; outputs, report, ledger, re-run guard.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import research_common as RC          # noqa: E402
import stage_b_screen as SB           # noqa: E402
import freeze_engines as FZ           # noqa: E402
import stage_e_execution as EX        # noqa: E402
import stage_e_paths as SP            # noqa: E402
import test_stage_e as TE             # noqa: E402

FAILS = []


def check(cond, msg):
    print(("  ok  " if cond else "  FAIL") + f"  {msg}")
    if not cond:
        FAILS.append(msg)


def quiet(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **k)
    return out, buf.getvalue()


def bars(o, h, l, c, start="2023-03-01"):
    ts = pd.bdate_range(start, periods=len(o)).values.astype("datetime64[ns]")
    o, h, l, c = (np.asarray(x, float) for x in (o, h, l, c))
    return EX.Bars(ts, o, h, l, c, np.full(len(o), 0.02), {t: i for i, t in enumerate(ts)})


def test_mechanics():
    print("1. mechanics on hand-built paths")
    # bar 0 = signal day; entry at bar 1's open = 100
    b = bars([99, 100, 101, 102, 103, 104, 105, 106], [100, 101, 104, 103, 104, 105, 106, 107],
             [98, 99, 100, 101, 102, 103, 104, 105], [99, 100.5, 102, 102.5, 103, 104, 105, 106])
    r, how, amb = SP.bracket(b, 1, 5, 0.03, 0.05, "conservative")
    check(r == 0.03 and how == "target" and not amb, "target first: +3% touched in session 2 -> +3%")
    b2 = bars([99, 100, 99, 97, 96, 95, 95, 95], [100, 101, 100, 98, 97, 96, 96, 96],
              [98, 99, 96, 94, 95, 94, 94, 94], [99, 99.5, 97, 96, 95.5, 95, 95, 95])
    r, how, amb = SP.bracket(b2, 1, 5, 0.05, 0.05, "conservative")
    check(r == -0.05 and how == "stop", "stop first: -5% touched in session 3 -> -5%")
    b3 = bars([99, 100, 100, 100, 100, 100], [100, 101, 106, 101, 101, 101], [98, 99, 94, 99, 99, 99], [99, 100, 100, 100, 100, 100])
    rc, _, ac = SP.bracket(b3, 1, 5, 0.05, 0.05, "conservative")
    ro, _, ao = SP.bracket(b3, 1, 5, 0.05, 0.05, "optimistic")
    check(rc == -0.05 and ro == 0.05 and ac and ao, "one session touches both: conservative -5%, optimistic +5%, flagged ambiguous")
    b4 = bars([99, 100, 101, 108, 108, 108], [100, 101, 102, 109, 109, 109], [98, 99, 100, 107, 107, 107], [99, 100, 101, 108, 108, 108])
    r, how, _ = SP.bracket(b4, 1, 5, 0.05, 0.05, "conservative")
    check(abs(r - 0.08) < 1e-12 and how == "target_gap", "a gap through the target fills at the open (+8%, better)")
    b5 = bars([99, 100, 99, 92, 92, 92], [100, 101, 100, 93, 93, 93], [98, 99, 98, 91, 91, 91], [99, 100, 99, 92, 92, 92])
    r, how, _ = SP.bracket(b5, 1, 5, 0.10, 0.05, "conservative")
    check(abs(r + 0.08) < 1e-12 and how == "stop_gap", "a gap through the stop fills at the open (-8%, worse)")
    c6 = [99, 100, 100, 95.0, 95.0 * 0.95, 88.0, 90.0]
    o6 = [99, 100, 100, 99.0, 95.0 * 0.95, 88.0, 89.0]
    h6 = [100, 101, 101, 99.5, 95.0 * 0.95, 89.0, 91.0]
    l6 = [98, 99, 99, 95.0, 95.0 * 0.95, 87.0, 88.0]
    b6 = bars(o6, h6, l6, c6)
    r, how, _ = SP.bracket(b6, 1, 5, 0.20, 0.06, "conservative")
    check(abs(r - (88.0 / 100 - 1)) < 1e-12 and how == "stop_late",
          "a stop reached only on a frozen lower-circuit session sells at the next open that trades (-12%)")
    r, how, _ = SP.bracket(b, 1, 5, 0.50, 0.50, "conservative")
    check(how == "time" and r == SP.time_exit(b, 1, 5) == 104 / 100 - 1, "no level reached: the plain exit at the 5th close (+4%)")
    b7 = bars([99, 100, 104, 108, 110, 104, 100], [100, 101, 105, 109, 111, 105, 101], [98, 99, 103, 107, 104, 103, 99],
              [99, 104, 108, 110, 106, 104, 100])
    r, how = SP.trailing(b7, 1, 5, 0.05)
    check(abs(r - (110 * 0.95 / 100 - 1)) < 1e-12 and how == "trail",
          "trailing 5%: highest close 110, stopped at 104.5 when the low reaches it")
    rec = SP.path_record(b, 1, dict(SP.CFG, window=6))
    check(abs(rec["mfe"] - 0.05) < 1e-12 and rec["day_mfe"] == 5 and abs(rec["mae"] + 0.01) < 1e-12 and rec["day_mae"] == 1,
          "MFE +5% in session 5, MAE -1% in session 1")
    check(rec["hit_0.03"] == 2 and np.isnan(rec["touch_0.03"]), "sessions to +3%: 2; a -3% stop never touched")
    bb = bars([10] * 20, [11] * 20, [9] * 20, [10] * 20)
    check(abs(SP.atr_frac(bb, 16, 14) - 0.2) < 1e-12, "ATR: a steady 2-point range on a 10 price = 20%")


def main():
    print("0. module origin and the declaration")
    here = HERE.parent.resolve()
    for mo in (SP, EX, FZ, SB, RC):
        check(Path(mo.__file__).resolve().parent == here, f"{mo.__name__} from {Path(mo.__file__).resolve().parent}")
    decl, sha = SP.load_declaration()
    check(decl["stage"] == "E-paths", f"settings match the path declaration (sha {sha})")
    test_mechanics()
    tmp = Path(tempfile.mkdtemp())
    try:
        root, pdir, pp, ip = TE.world(tmp)
        try:
            quiet(SP.main, ["--panel", str(pdir), "--root", str(root)])
            check(False, "refused before the engines are frozen")
        except SystemExit as e:
            check("not frozen" in str(e), "refused before the engines are frozen")
        quiet(FZ.main, ["--panel", str(pdir)])
        man = FZ.verify_frozen(pp)
        data, _ = quiet(EX.prepare, pp, root, man, None, False)
        print("2. invariants on the planted world")
        st, _ = quiet(SP.run_study, data, SP.CFG, "D", False)
        T, BR = st["trades"], st["rules"]
        worst = 0.0
        for i, p in T.iterrows():
            b = data["bars"][p["symbol"]]
            e = b.index[p["entry_day"]]
            r, how, _ = SP.bracket(b, e, 5, 10.0, 0.99, "conservative")
            worst = max(worst, abs(r - p["final"]))
        check(worst == 0.0, f"a bracket that can never trigger equals the plain exit on all {len(T):,} trades")
        check(bool((BR["ret_cons"] <= BR["ret_opt"] + 1e-15).all()), "conservative <= optimistic on every trade and rule")
        S = SP.summarize(st, SP.CFG)
        h = S["time_to_target"]["hit_within_window"].to_numpy()
        tch = S["stops"]["touched"].to_numpy()
        check(bool(np.all(np.diff(h) <= 1e-12)) and bool(np.all(np.diff(tch) <= 1e-12)),
              "touch rates fall as targets and stops widen")
        rf = EX.simulate(replace(EX.primary_scenario(), name="flat", costs="flat35", integer_shares=False, liquidity_cap=None),
                         data["lists"]["D"], data["bars"], data["adv"], data["cal"])
        a = rf["trades"].set_index(["signal_day", "symbol"])["raw_ret"].sort_index()
        bpath = T.set_index(["signal_day", "symbol"])["final"].sort_index()
        same = a.index.equals(bpath.index) and float(np.max(np.abs(a.to_numpy() - bpath.to_numpy()))) < 1e-15
        check(same, f"picks and plain-exit returns = the execution simulator's research-convention trades ({len(a):,})")
        plain = T[T["final"] == T["close_5"]]
        check(bool(((plain["mae"] <= plain["final"] + 1e-15) & (plain["final"] <= plain["mfe"] + 1e-15)).all()),
              "MAE <= finish <= MFE on every trade without a locked exit")
        print("3. end to end")
        rc, log = quiet(SP.main, ["--panel", str(pdir), "--root", str(root), "--isin", str(ip)])
        out = sorted((pdir / "stage_e").glob("PATHS_*"))[-1]
        for fn in ("paths_report.md", "trades_paths.csv", "brackets.csv", "time_to_target.csv", "stops.csv",
                   "path_by_session.csv", "stats.json"):
            check((out / fn).exists(), f"output {fn}")
        rep = (out / "paths_report.md").read_text(encoding="utf-8")
        for phrase in ("MFE", "Where the return is made", "Time to target", "winners a stop would have killed",
                       "Target first or stop first", "Giveback", "By rank"):
            check(phrase in rep, f"report section: {phrase}")
        br = pd.read_csv(out / "brackets.csv")
        check(len(br) == 6 * 4 + 4 * 3 + 3, f"{len(br)} rules: 24 % brackets, 12 ATR brackets, 3 trailing stops")
        kinds = [e.get("kind") for e in RC.ledger_entries(pp)]
        check(kinds.index("stage_e_paths_declared") < kinds.index("stage_e_paths_run"), "ledger: declared -> run")
        try:
            quiet(SP.main, ["--panel", str(pdir), "--root", str(root)])
            check(False, "a second run without --rerun-reason is refused")
        except SystemExit as e:
            check("already ran" in str(e), "a second run without --rerun-reason is refused")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if FAILS:
        print(f"FAILED  {len(FAILS)} check(s):")
        for f in FAILS:
            print("   - " + f)
        return 1
    print(f"VERIFIED  {SP.CODE_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
