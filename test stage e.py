#!/usr/bin/env python3
"""
tests/test_stage_e.py - freeze_engines v1 + stage_e_execution v1. Prints VERIFIED on success.

Run from the open_close folder with the main folder on PYTHONPATH:
    python tests\\test_stage_e.py

 0. Module origin; the declaration must match.
 1. Freeze: every hash recorded; a changed byte in a frozen run is refused; a second freeze is refused.
 2. Zerodha costs on a hand example (Rs 1,00,000 buy, Rs 1,10,000 sell).
 3. Exits by hand: time exit at the close; a close locked at the lower band sells at the first
    unlocked open; stop-loss fills at the stop intraday, at the open on a gap, never on a frozen
    lower-circuit day.
 4. The simulator on a planted world with a raw cache:
      - with the research convention (flat 35 bp, fractional shares, no cap) every trade's return
        equals the honest-exit outcome minus 35 bp for the same stock and day - the research numbers;
      - the books balance: equity = cash + positions every day; final equity = capital + every
        trade's P&L;
      - sizing: each new position is 1/15 of equity; never more than 15 open plus locked extensions;
      - no look-ahead: changing every watchlist after a date changes nothing up to the next open;
      - the 2% liquidity cap binds; impact makes big capital earn less per trade;
      - no-duplicates never holds the same stock twice.
 5. End to end: refused before a freeze; outputs and report; ledger order; re-run guard.
"""

from __future__ import annotations

import contextlib
import io
import json
import math
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

import research_common as RC            # noqa: E402
import stage_b_screen as SB             # noqa: E402
import stage_d_forensics as SF          # noqa: E402
import freeze_engines as FZ             # noqa: E402
import stage_e_execution as EX          # noqa: E402
import test_stage_d_forensics as TF     # noqa: E402

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


def world(tmp: Path):
    root, pdir, ip = TF.make_world(tmp)
    pp = pdir / "panel.parquet"
    base = pd.read_parquet(pdir / "stage_d" / "STAGED_T_001" / "preds.parquet")
    for rid in ("STAGEDXC_T_001", "STAGEDXE_T_001"):
        f = pdir / "stage_d" / rid
        f.mkdir(parents=True)
        v = base.copy()
        if rid.endswith("E_T_001"):
            v["score"] = v["score"] + 0.001 * np.random.default_rng(3).normal(size=len(v))
        v.to_parquet(f / "preds.parquet", index=False)
        shutil.copy(pdir / "stage_d" / "STAGED_T_001" / "daily.csv", f / "daily.csv")
        (f / "stage_d.json").write_text("{}", encoding="utf-8")
    RC.ledger_append(pp, {"kind": "stage_dx_ens_run", "experiment_family": "oc_v1", "run_id": "STAGEDXE_T_001",
                          "c_run": "STAGEDXC_T_001", "d_run": "STAGED_T_001"})
    (pdir / "stage_c").mkdir(exist_ok=True)
    (pdir / "stage_c" / "features_c_meta.json").write_text("{}", encoding="utf-8")
    return root, pdir, pp, ip


def test_costs_and_exits():
    print("2. Zerodha costs by hand")
    sc = EX.primary_scenario()
    b = EX.buy_costs(100000.0, sc)
    s = EX.sell_costs(110000.0, sc, True, 100000.0)
    hb = 100 + 15 + 3.22 + 0.1 + 0.18 * (3.22 + 0.1)
    hs = 110 + 3.542 + 0.11 + 0.18 * (3.542 + 0.11) + 15.93
    check(abs(sum(b.values()) - hb) < 1e-9 and abs(sum(s.values()) - hs) < 1e-9,
          f"buy Rs {sum(b.values()):.4f} (hand {hb:.4f}), sell Rs {sum(s.values()):.4f} (hand {hs:.4f}) incl. DP Rs 15.93")
    f = EX.sell_costs(110000.0, replace(sc, costs="flat35"), True, 100000.0)
    check(abs(sum(f.values()) - 350.0) < 1e-9 and sum(EX.buy_costs(1e5, replace(sc, costs="flat35")).values()) == 0,
          "flat35: exactly 35 bp of the buy value per round trip")

    print("3. exits by hand")
    ts = pd.bdate_range("2023-03-01", periods=12).values.astype("datetime64[ns]")
    o = np.array([100, 100, 102, 104, 105, 106, 101.5, 95.665, 93, 94, 95, 96.0])
    c = np.array([100, 101, 103, 104, 106, 106, 100.7, 95.665, 94, 95, 96, 97.0])
    h = np.maximum(o, c) * 1.01
    l = np.minimum(o, c) * 0.99
    c[6] = 106 * 0.95                                  # bar 6 closes locked at -5%
    l[6] = c[6]
    o[7] = h[7] = l[7] = c[7] = c[6] * 0.95            # bar 7 frozen at the lower band
    bb = EX.Bars(ts, o, h, l, c, np.full(12, 0.02), {t: i for i, t in enumerate(ts)})
    x, kind, raw, fl = EX.plan_exit(bb, 1, 5, None)
    check(x == 5 and kind == "close" and raw == c[5] and not fl["locked_exit"], "time exit at the close of the 5th session")
    x, kind, raw, fl = EX.plan_exit(bb, 2, 5, None)
    check(fl["locked_exit"] and x == 8 and kind == "open" and raw == o[8],
          "a close locked at the lower band sells at the first later open that is not locked (skipping a frozen day)")
    x, kind, raw, fl = EX.plan_exit(bb, 5, 3, 0.05)
    check(fl["stopped"] and x == 6 and kind == "stop" and abs(raw - o[5] * 0.95) < 1e-12,
          "stop 5%: the low reaches it -> filled at the stop price")
    o2 = o.copy()
    o2[3] = 90.0
    l2 = l.copy()
    l2[3] = 89.0
    bb2 = EX.Bars(ts, o2, np.maximum(h, o2), l2, c, np.full(12, 0.02), bb.index)
    x, kind, raw, fl = EX.plan_exit(bb2, 1, 5, 0.05)
    check(fl["stopped"] and x == 3 and kind == "open" and raw == 90.0, "stop 5%: a gap below it fills at the open")
    x, kind, raw, fl = EX.plan_exit(bb, 7, 3, 0.02)
    check(x != 7, "a frozen lower-circuit day cannot fill a stop")


def main():
    print("0. module origin and the declaration")
    here = HERE.parent.resolve()
    for mo in (EX, FZ, SF, SB, RC):
        check(Path(mo.__file__).resolve().parent == here, f"{mo.__name__} from {Path(mo.__file__).resolve().parent}")
    decl, sha = EX.load_declaration()
    check(decl["stage"] == "E" and len(sha) == 16, f"settings match the execution declaration (sha {sha})")
    orig = EX.CFG
    try:
        EX.CFG = dict(orig, n_per_day=5)
        try:
            EX.load_declaration()
            check(False, "a changed setting is refused")
        except SystemExit:
            check(True, "a changed setting is refused (the declaration is fixed)")
    finally:
        EX.CFG = orig
    test_costs_and_exits()
    tmp = Path(tempfile.mkdtemp())
    try:
        root, pdir, pp, ip = world(tmp)
        print("1. freeze")
        try:
            quiet(EX.main, ["--panel", str(pdir), "--root", str(root)])
            check(False, "the execution backtest refuses unfrozen engines")
        except SystemExit as e:
            check("not frozen" in str(e), "the execution backtest refuses unfrozen engines")
        quiet(FZ.main, ["--panel", str(pdir)])
        man = FZ.verify_frozen(pp)
        check(man["engines"]["D"]["run"] == "STAGED_T_001" and man["engines"]["ENS"]["c_run"] == "STAGEDXC_T_001"
              and len(man["files"]) >= 25, f"frozen: D, ENS and its C member; {len(man['files'])} files hashed")
        fp = pdir / "stage_d" / "STAGEDXE_T_001" / "preds.parquet"
        raw = fp.read_bytes()
        fp.write_bytes(raw + b"\0")
        try:
            FZ.verify_frozen(pp)
            check(False, "one changed byte in a frozen run is refused")
        except SystemExit as e:
            check("changed since the freeze" in str(e), "one changed byte in a frozen run is refused")
        fp.write_bytes(raw)
        try:
            quiet(FZ.main, ["--panel", str(pdir)])
            check(False, "a second freeze is refused")
        except SystemExit as e:
            check("already frozen" in str(e), "a second freeze is refused")

        print("4. the simulator")
        data, _ = quiet(EX.prepare, pp, root, man, None, False)
        P0 = EX.primary_scenario()
        flat = replace(P0, name="flat", costs="flat35", integer_shares=False, liquidity_cap=None)
        rf = EX.simulate(flat, data["lists"]["D"], data["bars"], data["adv"], data["cal"])
        t = rf["trades"]
        worst = 0.0
        for _, r in t.iterrows():
            b = data["bars"][r["symbol"]]
            i = b.index[r["signal_day"]]
            o = SF.rescore_symbol(b.o, b.h, b.l, b.c, np.array([i]), 5, 20, b.ts)
            worst = max(worst, abs(r["ret"] - (o["outcome_lock"][0] - 0.0035)))
        check(len(t) > 1000 and worst < 1e-12, f"research convention: all {len(t):,} trades = honest-exit outcome - 35 bp "
                                               f"(max diff {worst:.1e})")
        rp = EX.simulate(P0, data["lists"]["D"], data["bars"], data["adv"], data["cal"])
        eq, tp = rp["equity"], rp["trades"]
        ident = float((eq["equity"] - eq["cash"] - eq["invested"]).abs().max())
        check(ident < 1e-6, "every day: equity = cash + positions at the close")
        check(rp["open_at_end"] == 0 and abs(eq["equity"].iloc[-1] - (P0.capital + tp["pnl"].sum())) < 1e-4,
              f"final equity = capital + every trade's P&L (Rs {eq['equity'].iloc[-1]:,.2f})")
        st_ = rp["stats"]
        check(st_["n_trades"] == len(tp) and abs(st_["sum_ret"] / st_["n_trades"] - tp["ret"].mean()) < 1e-12
              and st_["wins"] == int((tp["ret"] > 0).sum()), "lightweight trade counters = the full trade table")
        first = tp[tp["entry_day"] == tp["entry_day"].min()]
        check(len(first) == 3 and all(abs(v - P0.capital / 15) <= px + 1e-6 for v, px in zip(first["buy_value"], first["entry_fill"])),
              "first day: 3 buys of 1/15 of equity each (within one share)")
        check(eq["positions"].max() <= 15 + int(tp["locked_exit"].sum()), f"never more than 15 open (+ locked extensions); "
                                                                           f"max {int(eq['positions'].max())}")
        cut = data["cal"][len(data["cal"]) // 2]
        L2 = {d: (v if d <= cut else list(reversed(v))) for d, v in data["lists"]["D"].items()}
        r2 = EX.simulate(P0, L2, data["bars"], data["adv"], data["cal"])
        nxt = data["cal"][np.searchsorted(data["cal"], cut) + 1]
        a_ = rp["equity"]["equity"][rp["equity"].index < pd.Timestamp(nxt)]
        b_ = r2["equity"]["equity"][r2["equity"].index < pd.Timestamp(nxt)]
        check(len(a_) > 100 and np.array_equal(a_.to_numpy(), b_.to_numpy()),
              "no look-ahead: reversing every watchlist after a date changes nothing before the next open")
        adv2 = dict(data["adv"])
        victim = tp.loc[tp["entry_fill"] < 1000, "symbol"].iloc[0]
        for k in list(adv2):
            if k[1] == victim:
                adv2[k] = 100000.0
        r3 = EX.simulate(P0, data["lists"]["D"], data["bars"], adv2, data["cal"])
        tv = r3["trades"][r3["trades"]["symbol"] == victim]
        check(len(tv) and (tv["buy_value"] <= 0.02 * 100000.0 + 1e-6).all() and tv["capped"].all(),
              f"liquidity cap binds: {victim} orders held to 2% of its turnover (Rs 2,000)")
        small = EX.simulate(replace(P0, capital=2e5), data["lists"]["D"], data["bars"], data["adv"], data["cal"])
        mid = EX.simulate(replace(P0, capital=5e7, liquidity_cap=None), data["lists"]["D"], data["bars"], data["adv"], data["cal"])
        big = EX.simulate(replace(P0, capital=5e8, liquidity_cap=None), data["lists"]["D"], data["bars"], data["adv"], data["cal"])
        ms, mm, mb = (x["trades"]["ret"].mean() * 1e4 for x in (small, mid, big))
        dp_bp = small["costs"]["dp"] / small["trades"]["buy_value"].sum() * 1e4
        check(mb < mm, f"impact: Rs 50 crore earns less per trade than Rs 5 crore ({mb:+.1f} vs {mm:+.1f} bp)")
        check(ms < mm and dp_bp > 5, f"fixed costs: Rs 2 lakh earns less per trade than Rs 5 crore ({ms:+.1f} vs {mm:+.1f} bp) - "
                                     f"the Rs 15.93 DP charge alone costs {dp_bp:.1f} bp on its small positions")
        nd = EX.simulate(replace(P0, duplicates=False), data["lists"]["D"], data["bars"], data["adv"], data["cal"])
        T = nd["trades"]
        clash = 0
        for s_, g in T.groupby("symbol"):
            g = g.sort_values("entry_day")
            clash += int((g["entry_day"].to_numpy()[1:] <= g["exit_day"].to_numpy()[:-1]).sum())
        check(clash == 0, "no-duplicates: the same stock is never held twice at once")

        print("5. end to end")
        rc, log = quiet(EX.main, ["--panel", str(pdir), "--root", str(root), "--isin", str(ip)])
        out = sorted((pdir / "stage_e").iterdir())[-1]
        for fn in ("execution_report.md", "summary.csv", "equity_primary.csv", "trades_primary.csv", "monthly_primary.csv",
                   "grid.csv", "capacity.csv", "rank.csv", "regimes.csv", "costs_primary.csv", "stats.json"):
            check((out / fn).exists(), f"output {fn}")
        S = pd.read_csv(out / "summary.csv")
        check(S["trades"].gt(0).all() and S["avg_trade_bp"].notna().all(), "every simulation reports its trades and average")
        check(len(S) == 1 + 1 + 1 + 1 + 1 + 1 + 3 + 8 + 30 + 10, f"{len(S)} simulations: primary, ensemble, split, flat35, stress, "
                                                               f"no-duplicates, 3 stops, 8 capitals, 30 grid cells, 10 random")
        rep = (out / "execution_report.md").read_text(encoding="utf-8")
        for phrase in ("Read this first", "Construction grid - DESCRIPTIVE", "Capacity", "Costs", "Regimes",
                       "Per-trade net by watchlist rank", "Bootstrap range"):
            check(phrase in rep, f"report section: {phrase}")
        kinds = [e.get("kind") for e in RC.ledger_entries(pp)]
        check(kinds.index("engines_frozen") < kinds.index("stage_e_declared") < kinds.index("stage_e_run"),
              "ledger: frozen -> declared -> run")
        try:
            quiet(EX.main, ["--panel", str(pdir), "--root", str(root)])
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
    print(f"VERIFIED  {EX.CODE_VERSION} + {FZ.CODE_VERSION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
