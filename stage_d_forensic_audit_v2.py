#!/usr/bin/env python3
"""
stage_d_forensic_audit.py

Independent forensic audit of an EXISTING Stage-D run.

This script does NOT fit a model and does NOT change the Stage-D result.
It tries to answer:

1) Are label_oc_1/2/3/5 independently reproducible from OHLC?
2) Do daily Top-N returns independently reproduce daily.csv?
3) Do picks agree with the independent ranking?
4) Are any selected stocks unbuyable?
5) Are missing outcomes materially affecting the result?
6) Do fold dates / embargoes agree with the declared protocol?
7) Do the reported winsorization cut-points agree with independently
   recomputed training labels?
8) Does the Stage-D source tree contain obvious future-looking feature
   construction patterns that deserve manual inspection?

IMPORTANT:
- A PASS here is evidence about the mechanics, not proof that every feature
  is leak-free.
- Static source scans are warnings, not automatic proof of leakage.
- The strongest feature-provenance check requires inspecting the actual feature
  construction functions.

Typical Windows usage:

python stage_d_forensic_audit.py ^
  --panel C:\\QuantData\\cache_daily\\panel_oc ^
  --open-close C:\\QuantData\\cache_daily\\open_close ^
  --stage-d C:\\QuantData\\cache_daily\\panel_oc\\stage_d\\STAGED_20260928_001

If --stage-d is omitted, the newest STAGED_* directory is selected.

Optional:
  --source-root C:\\Users\\karanvis\\PyCharmProjects\\bigmove_deploy
  --strict

Outputs:
  forensic_audit_report.md
  forensic_audit.json
  target_reconstruction_mismatches.csv
  ranking_mismatches.csv

Exit code:
  0 = no hard failures
  1 = hard failure(s)
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


TARGETS = ["label_oc_1", "label_oc_2", "label_oc_3", "label_oc_5"]
COST_BPS = 35.0
COST = COST_BPS / 1e4
HORIZONS = {t: int(t.rsplit("_", 1)[1]) for t in TARGETS}

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"

FAILS: List[str] = []
WARNS: List[str] = []
RESULTS: Dict[str, object] = {}


def ok(name: str, detail: str = ""):
    print(f"[PASS] {name}" + (f" — {detail}" if detail else ""))
    RESULTS[name] = {"status": PASS, "detail": detail}


def warn(name: str, detail: str):
    print(f"[WARN] {name} — {detail}")
    WARNS.append(name)
    RESULTS[name] = {"status": WARN, "detail": detail}


def fail(name: str, detail: str):
    print(f"[FAIL] {name} — {detail}")
    FAILS.append(name)
    RESULTS[name] = {"status": FAIL, "detail": detail}


def find_col(columns, candidates):
    lower = {str(c).lower(): c for c in columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def load_stage_dir(panel: Path, supplied: Path | None) -> Path:
    if supplied:
        p = supplied
        if not p.exists():
            raise FileNotFoundError(f"Stage-D directory not found: {p}")
        return p

    root = (panel.parent if panel.is_file() else panel) / "stage_d"
    runs = sorted([p for p in root.glob("STAGED_*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No STAGED_* run found under {root}")
    return runs[-1]


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def naive_ts(s):
    x = pd.to_datetime(s, errors="coerce")
    try:
        return x.dt.tz_localize(None)
    except TypeError:
        return x


def resolve_panel_parquet(panel: Path) -> Path:
    """Accept either the panel_oc directory or panel.parquet itself."""
    if panel.is_file():
        return panel
    p = panel / "panel.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Panel parquet not found: {p}")
    return p


def _prepare_ohlc(panel_file: Path) -> pd.DataFrame:
    """Load only the columns needed for an independent label reconstruction."""
    cols = pd.read_parquet(panel_file, columns=[]).columns
    ts_col = find_col(cols, ["timestamp", "date", "datetime"])
    sym_col = find_col(cols, ["symbol", "ticker", "tradingsymbol"])
    open_col = find_col(cols, ["open", "Open", "OPEN"])
    close_col = find_col(cols, ["close", "Close", "CLOSE"])

    if not all([ts_col, sym_col, open_col, close_col]):
        raise RuntimeError(
            "Could not identify timestamp/symbol/open/close columns. "
            f"Found candidates: timestamp={ts_col}, symbol={sym_col}, "
            f"open={open_col}, close={close_col}"
        )

    df = pd.read_parquet(
        panel_file,
        columns=[ts_col, sym_col, open_col, close_col]
    ).rename(columns={
        ts_col: "timestamp",
        sym_col: "symbol",
        open_col: "_open",
        close_col: "_close",
    })

    df["timestamp"] = naive_ts(df["timestamp"])
    df["symbol"] = df["symbol"].astype(str)
    df["_open"] = pd.to_numeric(df["_open"], errors="coerce")
    df["_close"] = pd.to_numeric(df["_close"], errors="coerce")
    df["_row"] = np.arange(len(df))

    if df["timestamp"].isna().any():
        raise RuntimeError(
            f"{int(df['timestamp'].isna().sum()):,} OHLC rows have invalid timestamps"
        )

    return df


def _global_session_map(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[pd.Timestamp, int]]:
    """
    Build the global trading-session calendar from the panel.

    This is deliberately NOT a per-symbol shift.

    For a prediction row at session T, T+1 and T+h are determined from the
    global set of trading sessions. The stock's OHLC is then looked up at
    those exact timestamps. Therefore a missing stock row cannot silently
    turn into "the next observed stock row".
    """
    sessions = pd.DatetimeIndex(sorted(df["timestamp"].drop_duplicates()))
    pos = {ts: i for i, ts in enumerate(sessions)}

    # Detect duplicate symbol/session rows because they make exact lookup
    # ambiguous and could hide a reconstruction problem.
    dup = df.duplicated(["symbol", "timestamp"], keep=False)
    if dup.any():
        sample = df.loc[dup, ["symbol", "timestamp"]].head(10)
        raise RuntimeError(
            f"{int(dup.sum()):,} duplicate symbol/session rows found; "
            "exact calendar reconstruction is ambiguous. Sample:\n"
            f"{sample.to_string(index=False)}"
        )

    return df, pos


def independent_target_from_ohlc(
    panel_file: Path,
    target: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Reconstruct close[T+h] / open[T+1] - 1 using the GLOBAL session calendar.

    For every prediction row at T:
      entry timestamp = global session T+1
      exit timestamp  = global session T+h
      target          = Close(symbol, T+h) / Open(symbol, T+1) - 1

    This is intentionally different from groupby(symbol).shift(...):
    a suspended/missing stock session must produce NaN, not silently advance
    to the next row available for that stock.
    """
    df = _prepare_ohlc(panel_file)
    df, session_pos = _global_session_map(df)

    h = HORIZONS[target]
    sessions = pd.DatetimeIndex(sorted(df["timestamp"].unique()))

    # Exact OHLC lookup by (symbol, session). Reindexing against explicit
    # future timestamps makes missing stock sessions visible as NaN.
    ohlc = df.set_index(["symbol", "timestamp"])[["_open", "_close"]]

    base = df[["_row", "timestamp", "symbol"]].copy()
    base["_session_pos"] = base["timestamp"].map(session_pos)
    base["_t1_pos"] = base["_session_pos"] + 1
    base["_th_pos"] = base["_session_pos"] + h

    valid_t1 = base["_t1_pos"] < len(sessions)
    valid_th = base["_th_pos"] < len(sessions)

    base["_t1_ts"] = pd.NaT
    base["_th_ts"] = pd.NaT
    base.loc[valid_t1, "_t1_ts"] = sessions[base.loc[valid_t1, "_t1_pos"].astype(int)]
    base.loc[valid_th, "_th_ts"] = sessions[base.loc[valid_th, "_th_pos"].astype(int)]

    entry_key = pd.MultiIndex.from_arrays(
        [base["symbol"], base["_t1_ts"]], names=["symbol", "timestamp"]
    )
    exit_key = pd.MultiIndex.from_arrays(
        [base["symbol"], base["_th_ts"]], names=["symbol", "timestamp"]
    )

    base["_entry_open"] = ohlc["_open"].reindex(entry_key).to_numpy()
    base["_exit_close"] = ohlc["_close"].reindex(exit_key).to_numpy()
    base["_recon"] = base["_exit_close"] / base["_entry_open"] - 1.0

    out = base.set_index("_row")["_recon"].sort_index()
    out.name = target + "_reconstructed"
    return base, out


def write_reconstruction_sample(
    base: pd.DataFrame,
    target: str,
    outdir: Path,
    n: int = 25,
):
    """Write an auditable table showing the exact T/T+1/T+h arithmetic."""
    sample = base[
        ["_row", "timestamp", "symbol", "_t1_ts", "_th_ts",
         "_entry_open", "_exit_close", "_recon"]
    ].copy()

    sample = sample.rename(columns={
        "_row": "panel_row",
        "timestamp": "T",
        "_t1_ts": "T_plus_1",
        "_th_ts": f"T_plus_{HORIZONS[target]}",
        "_entry_open": "entry_open_T_plus_1",
        "_exit_close": f"exit_close_T_plus_{HORIZONS[target]}",
        "_recon": "reconstructed_return",
    })

    sample["formula"] = (
        f"Close[T+{HORIZONS[target]}] / Open[T+1] - 1"
    )

    resolved = sample["reconstructed_return"].notna()
    sample = sample.loc[resolved].head(n)

    # Human-readable percentage for easy spot checking.
    sample["reconstructed_return_pct"] = sample["reconstructed_return"] * 100.0

    path = outdir / f"{target}_reconstruction_sample.csv"
    sample.to_csv(path, index=False)
    return path


def compare_target(panel_file: Path, target: str, outdir: Path):
    print(f"\n--- Target reconstruction: {target} ---")

    base_panel = pd.read_parquet(
        panel_file,
        columns=["timestamp", "symbol", "label_buyable_o1", target]
    )
    recon_detail, recon = independent_target_from_ohlc(panel_file, target)

    base_panel[target] = pd.to_numeric(base_panel[target], errors="coerce")
    recon = recon.reindex(base_panel.index)

    a = base_panel[target].to_numpy(dtype=float)
    b = recon.to_numpy(dtype=float)

    both = np.isfinite(a) & np.isfinite(b)
    mism = both & (np.abs(a - b) > 1e-10)
    one_missing = np.isfinite(a) ^ np.isfinite(b)

    sample_path = write_reconstruction_sample(
        recon_detail, target, outdir, n=25
    )
    print(f"  Arithmetic sample: {sample_path.name}")

    if mism.any():
        idx = np.flatnonzero(mism)
        sample = base_panel.iloc[idx[:100]].copy()
        sample["reconstructed"] = b[idx[:100]]
        sample["abs_diff"] = np.abs(a[idx[:100]] - b[idx[:100]])
        sample.to_csv(
            outdir / f"{target}_reconstruction_mismatches.csv",
            index=False,
        )
        fail(
            f"{target}_reconstruction",
            f"{int(mism.sum()):,} numeric mismatches; sample written to "
            f"{target}_reconstruction_mismatches.csv"
        )
    elif one_missing.any():
        warn(
            f"{target}_missingness",
            f"{int(one_missing.sum()):,} rows have a value on one side only"
        )
    else:
        ok(
            f"{target}_reconstruction",
            f"{int(both.sum()):,} resolved rows match the GLOBAL-session formula"
        )

    # Explicitly report whether any stock/session rows are absent from the
    # panel for the requested future dates. These become NaN rather than being
    # silently advanced to a later stock row.
    missing_entry = recon_detail["_entry_open"].isna()
    missing_exit = recon_detail["_exit_close"].isna()
    if missing_entry.any() or missing_exit.any():
        warn(
            f"{target}_future_ohlc_missing",
            f"entry-open missing={int(missing_entry.sum()):,}, "
            f"exit-close missing={int(missing_exit.sum()):,}"
        )
def compare_target(panel: Path, target: str, outdir: Path):
    print(f"\n--- Target reconstruction: {target} ---")

    base = pd.read_parquet(panel, columns=["timestamp", "symbol", "label_buyable_o1", target])
    recon = independent_target_from_ohlc(panel, target)

    base[target] = pd.to_numeric(base[target], errors="coerce")
    recon = recon.reindex(base.index)

    a = base[target].to_numpy(dtype=float)
    b = recon.to_numpy(dtype=float)

    both = np.isfinite(a) & np.isfinite(b)
    mism = both & (np.abs(a - b) > 1e-10)
    one_missing = np.isfinite(a) ^ np.isfinite(b)

    # A small tolerance is allowed only for floating point representation.
    if mism.any():
        idx = np.flatnonzero(mism)
        sample = base.iloc[idx[:100]].copy()
        sample["reconstructed"] = b[idx[:100]]
        sample["abs_diff"] = np.abs(a[idx[:100]] - b[idx[:100]])
        sample.to_csv(outdir / f"{target}_reconstruction_mismatches.csv", index=False)
        fail(
            f"{target}_reconstruction",
            f"{int(mism.sum()):,} numeric mismatches; sample written to "
            f"{target}_reconstruction_mismatches.csv"
        )
    elif one_missing.any():
        # Missingness can legitimately differ at the tail if the panel target
        # construction has explicit handling. Report rather than silently pass.
        warn(
            f"{target}_missingness",
            f"{int(one_missing.sum()):,} rows have a value on one side only"
        )
    else:
        ok(f"{target}_reconstruction", f"{int(both.sum()):,} resolved rows match")

    RESULTS[f"{target}_counts"] = {
        "panel_resolved": int(np.isfinite(a).sum()),
        "reconstructed_resolved": int(np.isfinite(b).sum()),
        "both_resolved": int(both.sum()),
        "numeric_mismatches": int(mism.sum()),
        "one_side_missing": int(one_missing.sum()),
    }

    return base, recon


def independent_daily_ranking(stage_dir: Path, panel: Path):
    """
    Rebuild Top-1/3/5/10 from preds.parquet independently of stage_d_gate.py.

    This intentionally uses only the stored raw score + outcome in preds and
    independently re-implements the daily sorting/aggregation.
    """
    pred_path = stage_dir / "preds.parquet"
    daily_path = stage_dir / "daily.csv"
    picks_path = stage_dir / "picks.csv"

    if not pred_path.exists() or not daily_path.exists() or not picks_path.exists():
        fail("output_files", "preds.parquet, daily.csv and/or picks.csv missing")
        return

    preds = pd.read_parquet(pred_path)
    daily = pd.read_csv(daily_path)
    picks = pd.read_csv(picks_path)

    preds["timestamp"] = naive_ts(preds["timestamp"])
    daily["timestamp"] = naive_ts(daily["timestamp"])
    picks["timestamp"] = naive_ts(picks["timestamp"])

    required = {"timestamp", "symbol", "target", "score", "outcome"}
    if not required.issubset(preds.columns):
        fail("preds_schema", f"missing {sorted(required - set(preds.columns))}")
        return

    ranking_mismatches = []

    for target in TARGETS:
        p = preds[preds["target"] == target].copy()
        d0 = daily[daily["target"] == target].copy()
        pk0 = picks[picks["target"] == target].copy()

        if p.empty:
            fail(f"{target}_preds", "no predictions found")
            continue

        p["symbol"] = p["symbol"].astype(str)
        p["score"] = pd.to_numeric(p["score"], errors="coerce")
        p["outcome"] = pd.to_numeric(p["outcome"], errors="coerce")

        # Stable deterministic ordering: score descending, symbol ascending.
        p = p.sort_values(
            ["timestamp", "score", "symbol"],
            ascending=[True, False, True],
            kind="mergesort"
        )

        rows = []
        for ts, g in p.groupby("timestamp", sort=True):
            g = g.reset_index(drop=True)
            row = {"timestamp": ts}

            for n in [1, 3, 5, 10]:
                top = g.iloc[:n]
                vals = top["outcome"].to_numpy(dtype=float)
                finite = np.isfinite(vals)
                row[f"top{n}_gross"] = (
                    float(np.mean(vals[finite])) if finite.any() else np.nan
                )
                row[f"top{n}_missing"] = int((~finite).sum())
                row[f"top{n}_net"] = (
                    row[f"top{n}_gross"] - COST
                    if np.isfinite(row[f"top{n}_gross"]) else np.nan
                )

            allv = g["outcome"].to_numpy(dtype=float)
            finite = np.isfinite(allv)
            market = float(np.mean(allv[finite])) if finite.any() else np.nan
            row["market"] = market

            for n in [1, 3, 5, 10]:
                row[f"top{n}_excess"] = (
                    row[f"top{n}_gross"] - market
                    if np.isfinite(row[f"top{n}_gross"]) and np.isfinite(market)
                    else np.nan
                )

            rows.append(row)

        calc = pd.DataFrame(rows)

        merged = d0.merge(calc, on="timestamp", suffixes=("_reported", "_calc"), how="outer")

        for n in [1, 3, 5, 10]:
            for metric in ["gross", "net", "excess"]:
                c = f"top{n}_{metric}"
                r = c + "_reported"
                q = c + "_calc"
                if r not in merged or q not in merged:
                    continue
                a = pd.to_numeric(merged[r], errors="coerce").to_numpy()
                b = pd.to_numeric(merged[q], errors="coerce").to_numpy()
                both = np.isfinite(a) & np.isfinite(b)
                bad = both & (np.abs(a - b) > 1e-10)
                if bad.any():
                    ranking_mismatches.append(
                        pd.DataFrame({
                            "target": target,
                            "timestamp": merged.loc[bad, "timestamp"],
                            "field": c,
                            "reported": a[bad],
                            "independent": b[bad],
                            "abs_diff": np.abs(a[bad] - b[bad]),
                        })
                    )

        # Check every reported Top-3 pick is exactly the independent top-3.
        expected = (
            p.groupby("timestamp", sort=True)
            .head(3)[["timestamp", "symbol", "score"]]
            .copy()
        )
        expected["rank"] = expected.groupby("timestamp").cumcount() + 1

        actual = pk0[pk0["rank"].astype(int) <= 3][
            ["timestamp", "symbol", "score", "rank"]
        ].copy()
        actual["symbol"] = actual["symbol"].astype(str)
        expected["symbol"] = expected["symbol"].astype(str)

        em = expected.merge(
            actual,
            on=["timestamp", "rank"],
            how="outer",
            suffixes=("_expected", "_reported"),
            indicator=True,
        )
        bad_pick = (
            (em["_merge"] != "both")
            | (em["symbol_expected"] != em["symbol_reported"])
        )
        if bad_pick.any():
            ranking_mismatches.append(
                em.loc[bad_pick].assign(target=target)
            )

        print(
            f"{target}: independent ranking checked "
            f"{len(calc):,} days / {len(p):,} predictions"
        )

    if ranking_mismatches:
        mm = pd.concat(ranking_mismatches, ignore_index=True)
        mm.to_csv(stage_dir / "ranking_mismatches.csv", index=False)
        fail(
            "independent_daily_ranking",
            f"{len(mm):,} mismatch records; see ranking_mismatches.csv"
        )
    else:
        ok(
            "independent_daily_ranking",
            "daily.csv and Top-3 picks reproduce independently from preds.parquet"
        )


def check_eligibility(stage_dir: Path, panel: Path):
    picks = pd.read_csv(stage_dir / "picks.csv")
    picks["timestamp"] = naive_ts(picks["timestamp"])
    picks["symbol"] = picks["symbol"].astype(str)

    p = pd.read_parquet(
        panel,
        columns=["timestamp", "symbol", "label_buyable_o1"]
    )
    p["timestamp"] = naive_ts(p["timestamp"])
    p["symbol"] = p["symbol"].astype(str)

    m = picks.merge(
        p,
        on=["timestamp", "symbol"],
        how="left",
        indicator=True,
    )

    bad = (m["_merge"] != "both") | (pd.to_numeric(m["label_buyable_o1"], errors="coerce") != 1)

    if bad.any():
        bad_rows = m.loc[bad]
        bad_rows.to_csv(stage_dir / "unbuyable_picks.csv", index=False)
        fail(
            "pick_eligibility",
            f"{int(bad.sum()):,} reported picks are missing or unbuyable"
        )
    else:
        ok(
            "pick_eligibility",
            f"all {len(picks):,} stored Top-10 picks are buyable"
        )


def check_missing_outcomes(stage_dir: Path):
    daily = pd.read_csv(stage_dir / "daily.csv")
    for t in TARGETS:
        d = daily[daily["target"] == t].copy()
        if d.empty:
            continue

        miss = pd.to_numeric(d["top3_missing"], errors="coerce").fillna(0)
        days = len(d)
        slots = days * 3
        missing_slots = int(miss.sum())

        # Reconstruct a conservative sensitivity:
        # treating each missing pick as a zero-return trade instead of dropping it.
        gross = pd.to_numeric(d["top3_gross"], errors="coerce")
        adjusted = gross * 3.0 / (3.0 + miss)
        # This is not a production estimate; it only answers whether missing
        # outcomes could plausibly explain the headline result.
        adj_mean = float(np.nanmean(adjusted))
        headline = float(np.nanmean(gross))

        impact_bp = (headline - adj_mean) * 1e4

        if missing_slots:
            warn(
                f"{t}_missing_outcomes",
                f"{missing_slots:,}/{slots:,} Top-3 slots missing "
                f"({missing_slots / max(slots,1):.3%}); "
                f"zero-return sensitivity reduces mean gross by about "
                f"{impact_bp:.1f} bp"
            )
        else:
            ok(f"{t}_missing_outcomes", "zero missing Top-3 outcomes")


def check_folds(stage_dir: Path):
    js = load_json(stage_dir / "stage_d.json")
    cfg = js.get("config", {})
    folds = js.get("folds", [])

    if not folds:
        fail("fold_metadata", "no fold metadata in stage_d.json")
        return

    bad = []
    for f in folds:
        target = f["target"]
        h = HORIZONS[target]
        train_end = pd.Timestamp(f["train_end"])
        test_start = pd.Timestamp(f["test_start"])
        gap = int(f["embargo_gap_sessions"])

        if not (train_end < test_start):
            bad.append((target, f["fold"], "train_end >= test_start"))
        if gap <= h:
            bad.append((target, f["fold"], f"gap={gap} <= horizon={h}"))

    if bad:
        fail("fold_embargo", str(bad[:20]))
    else:
        ok(
            "fold_embargo",
            f"{len(folds)} target-fold records have chronological train/test "
            f"ordering and gap > target horizon"
        )

    # Confirm key declared settings are present and expected.
    expected = {
        "cost_bps": 35.0,
        "fit_cap": 400000,
        "n_splits": 5,
        "embargo": 5,
        "winsor_pct": [1, 99],
        "top_n_primary": 3,
        "top_n_all": [1, 3, 5, 10],
    }
    mism = {
        k: (cfg.get(k), v)
        for k, v in expected.items()
        if cfg.get(k) != v
    }
    if mism:
        fail("declared_config", str(mism))
    else:
        ok("declared_config", "key Stage-D settings match the declaration")


def check_winsorization(stage_dir: Path, panel: Path):
    js = load_json(stage_dir / "stage_d.json")
    folds = js.get("folds", [])

    # Load only fields needed for all target checks.
    cols = ["timestamp", "symbol", "label_buyable_o1"] + TARGETS
    p = pd.read_parquet(panel, columns=cols)
    p["timestamp"] = naive_ts(p["timestamp"])
    p["symbol"] = p["symbol"].astype(str)
    p["label_buyable_o1"] = pd.to_numeric(p["label_buyable_o1"], errors="coerce")

    mism = []
    for f in folds:
        t = f["target"]
        train_end = pd.Timestamp(f["train_end"])
        sub = p[
            (p["label_buyable_o1"] == 1)
            & (p["timestamp"] <= train_end)
            & np.isfinite(pd.to_numeric(p[t], errors="coerce"))
        ][["timestamp", t]].copy()

        if sub.empty:
            mism.append((t, f["fold"], "no training rows"))
            continue

        y = pd.to_numeric(sub[t], errors="coerce").to_numpy(float)
        day = pd.factorize(sub["timestamp"])[0]

        # Independent day demeaning over exactly the eligible training rows.
        sums = np.bincount(day, y)
        counts = np.bincount(day)
        dm = y - sums[day] / counts[day]

        lo, hi = np.percentile(dm, [1, 99])
        reported = np.asarray(f["label_clip_bp"], dtype=float) / 1e4

        if abs(lo - reported[0]) > 1e-10 or abs(hi - reported[1]) > 1e-10:
            mism.append(
                (t, f["fold"], lo, hi, reported[0], reported[1])
            )

    if mism:
        fail(
            "training_label_winsorization",
            f"{len(mism)} fold mismatch(es): {mism[:10]}"
        )
    else:
        ok(
            "training_label_winsorization",
            f"independently reproduced 1/99% day-demeaned clip points "
            f"for {len(folds)} target-fold records"
        )


def static_source_scan(source_root: Path):
    """
    Look for patterns that deserve manual inspection.

    This is intentionally conservative. A negative shift in a target-building
    file is often legitimate. A negative shift in a feature-construction file
    is suspicious. We therefore classify, not automatically fail.
    """
    if not source_root.exists():
        warn("source_scan", f"source root not found: {source_root}")
        return

    py_files = list(source_root.rglob("*.py"))
    if not py_files:
        warn("source_scan", "no Python files found")
        return

    suspicious = []

    patterns = [
        ("negative_shift", re.compile(r"\.shift\s*\(\s*-\s*\d+")),
        ("negative_diff_like", re.compile(r"\bshift\s*\(\s*-\s*")),
        ("centered_rolling", re.compile(r"rolling\s*\([^)]*center\s*=\s*True", re.I)),
        ("bfill", re.compile(r"\.bfill\s*\(")),
        ("backward_fill", re.compile(r"\bbackfill\s*\(")),
        ("future_named_feature", re.compile(r"\b(future|fwd|forward|lead)[A-Za-z0-9_]*\s*=", re.I)),
    ]

    for p in py_files:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        # Ignore virtual environments and obvious caches.
        if any(part.lower() in {"venv", ".venv", "__pycache__", ".git"} for part in p.parts):
            continue

        for name, rx in patterns:
            for m in rx.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                snippet = text.splitlines()[line - 1].strip()[:180]
                suspicious.append({
                    "file": str(p),
                    "line": line,
                    "pattern": name,
                    "snippet": snippet,
                })

    if suspicious:
        df = pd.DataFrame(suspicious)
        out = source_root / "stage_d_forensic_source_warnings.csv"
        df.to_csv(out, index=False)
        warn(
            "source_scan",
            f"{len(suspicious)} suspicious construction pattern(s); "
            f"manual review required. Details: {out}"
        )
        RESULTS["source_scan_records"] = suspicious[:200]
    else:
        ok("source_scan", "no obvious future-looking construction patterns found")


def compare_run_metadata(stage_dir: Path):
    js = load_json(stage_dir / "stage_d.json")
    decl_path_candidates = [
        stage_dir.parent.parent / "STAGE_D_DECLARATION.json",
        stage_dir.parent.parent / "STAGE D DECLARATION.json",
    ]

    # Also look next to this audit script.
    decl_path_candidates += [
        Path(__file__).with_name("STAGE_D_DECLARATION.json"),
        Path(__file__).with_name("STAGE D DECLARATION.json"),
        Path(__file__).with_name("STAGE D DECLARATION(1).json"),
    ]

    decl = None
    for p in decl_path_candidates:
        if p.exists():
            try:
                decl = load_json(p)
                break
            except Exception:
                pass

    if decl is None:
        warn(
            "declaration_file",
            "could not locate the original declaration; stage_d.json config was checked instead"
        )
        return

    if decl.get("config") != js.get("config"):
        fail(
            "declaration_vs_run",
            "stage_d.json config differs from located Stage-D declaration"
        )
    else:
        ok("declaration_vs_run", "run config matches located declaration")


def main():
    ap = argparse.ArgumentParser(description="Independent forensic audit of an existing Stage-D run")
    ap.add_argument("--panel", default=r"C:\QuantData\cache_daily\panel_oc", help="panel_oc directory or panel.parquet")
    ap.add_argument("--stage-d", default=None, help="specific STAGED_* directory")
    ap.add_argument("--open-close", default=r"C:\QuantData\cache_daily\open_close", help="raw open_close directory; checked for presence and recorded, not used as a shortcut for labels")
    ap.add_argument("--source-root", default=None, help="project root containing feature construction code")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="return exit code 1 for warnings as well as hard failures",
    )
    args = ap.parse_args()

    panel = Path(args.panel)
    panel_file = resolve_panel_parquet(panel)
    open_close = Path(args.open_close) if args.open_close else None
    stage_dir = load_stage_dir(panel_file.parent if panel_file.name == "panel.parquet" else panel, Path(args.stage_d) if args.stage_d else None)

    print("=" * 78)
    print("STAGE D FORENSIC AUDIT")
    print("=" * 78)
    print(f"Panel dir: {panel_file.parent}")
    print(f"Panel    : {panel_file}")
    print(f"OpenClose: {open_close}")
    print(f"Stage D  : {stage_dir}")
    print()

    RESULTS["panel"] = str(panel_file)
    RESULTS["open_close"] = str(open_close) if open_close else None
    RESULTS["stage_d"] = str(stage_dir)
    if open_close is not None:
        if open_close.exists():
            ok("open_close_path", f"found: {open_close}")
        else:
            warn("open_close_path", f"not found: {open_close}")

    # Basic files.
    needed = ["stage_d.json", "daily.csv", "picks.csv", "preds.parquet"]
    missing = [x for x in needed if not (stage_dir / x).exists()]
    if missing:
        fail("output_files", f"missing: {missing}")
        print("\nHARD FAILURES — cannot continue reliably.")
        raise SystemExit(1)

    compare_run_metadata(stage_dir)
    check_folds(stage_dir)

    # Independent target reconstruction.
    for t in TARGETS:
        try:
            compare_target(panel_file, t, stage_dir)
        except Exception as e:
            fail(f"{t}_reconstruction", f"exception: {type(e).__name__}: {e}")

    independent_daily_ranking(stage_dir, panel_file)
    check_eligibility(stage_dir, panel_file)
    check_missing_outcomes(stage_dir)
    check_winsorization(stage_dir, panel_file)

    source_root = (
        Path(args.source_root)
        if args.source_root
        else Path(__file__).resolve().parent
    )
    static_source_scan(source_root)

    print("\n" + "=" * 78)
    print("AUDIT SUMMARY")
    print("=" * 78)
    print(f"Hard failures : {len(FAILS)}")
    print(f"Warnings      : {len(WARNS)}")

    report_lines = [
        "# Stage D Forensic Audit",
        "",
        f"Panel: `{panel}`",
        f"Stage-D run: `{stage_dir}`",
        "",
        "## Summary",
        f"- Hard failures: **{len(FAILS)}**",
        f"- Warnings: **{len(WARNS)}**",
        "",
        "## Interpretation",
        "- PASS means the audited mechanical property was independently reproduced.",
        "- WARN means the result needs manual review but is not itself proof of leakage.",
        "- FAIL means the corresponding Stage-D result should not be treated as mechanically verified.",
        "",
    ]

    if FAILS:
        report_lines += ["## Failures", ""]
        report_lines += [f"- {x}" for x in FAILS]
        report_lines.append("")

    if WARNS:
        report_lines += ["## Warnings", ""]
        report_lines += [f"- {x}" for x in WARNS]
        report_lines.append("")

    (stage_dir / "forensic_audit_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    (stage_dir / "forensic_audit.json").write_text(
        json.dumps(
            {
                "hard_failures": FAILS,
                "warnings": WARNS,
                "checks": RESULTS,
                "stage_dir": str(stage_dir),
                "panel": str(panel),
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    if FAILS:
        print("\nRESULT: FORENSIC AUDIT = FAIL")
        raise SystemExit(1)

    if WARNS:
        print("\nRESULT: MECHANICS PASS, BUT MANUAL REVIEW IS REQUIRED")
        raise SystemExit(1 if args.strict else 0)

    print("\nRESULT: FORENSIC AUDIT = PASS")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
