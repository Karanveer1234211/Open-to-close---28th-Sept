# bigmove_deploy — frozen bundle (28 Sept 2026; + tradability_audit v1; + stages A and B in `open_close\`)

All files go FLAT into `C:\Users\karanvsi\PyCharmMiscProject\bigmove_deploy` (they import each other by name). `tests\` stays a subfolder.

## Install
1. Back up your current bigmove_deploy folder.
2. Copy everything from this bundle into it, overwriting. **Keep your own `env_daily.bat` and `watchlist.txt`** - they are not in the bundle.
3. Run once: `setup_once.bat`
4. The existing 23:45 weekday task already points at `daily_run.bat`, so it picks up v3 automatically.
5. Create the monthly retrain task (one-time, in cmd):
   `schtasks /Create /TN "bigmove_monthly_retrain" /TR "C:\Users\karanvsi\PyCharmMiscProject\bigmove_deploy\monthly_retrain.bat" /SC MONTHLY /MO FIRST /D SAT /ST 10:00`

## Automation
| file | what it does |
|---|---|
| daily_run.bat (v4) | 23:45 IST weekdays: data + panel -> Soul v3 -> watchlists for BOTH entry worlds (open world once trained); resolve both on Fridays. Logs to `<root>\logs\daily_YYYY-MM-DD.log` |
| monthly_retrain.bat (v2) | first Saturday, for each entry world: Soul v3 -> engine train -> history report |
| setup_once.bat | installs requirements, checks every library and module imports |
| requirements.txt | Python libraries |

## Daily pipeline (runs automatically)
| file | version | role |
|---|---|---|
| Daily_cache_v27.py | v27 | Kite historical data cache (history anchor 2015-01-01 - never change) |
| data_quality.py | - | quality gate on the cache |
| features_daily.py | - | per-stock daily features (leak canary) |
| panel_build.py | v31 | model-ready panel for the daily pipeline: features + labels (5 close-entry brackets + next-open labels `*_o1`), liquidity floor, label firewall. Writes `panel\`. (v32 lives in `open_close\`.) |
| run_daily.py | - | orchestrator: cache -> gate -> leak gate -> panel -> report |
| daily_report.py | - | one-page daily pipeline verdict |
| soul_v3.py | v3.2 | Soul: personality + memory (build) and reference cards (cards); `--entry open` for the next-open world |
| signal_engine.py | v5 | production model: train / watchlist / resolve / history; `--entry open` = next-open world in engine_open\ |
| meta_label_test.py | v4 | meta-model core (imported by the engine); also a standalone test |
| research_common.py | - | shared discipline: canonical state engine, label firewall, bootstrap, lockbox, entry-world label map |
| feasibility_test.py | v3 | walk-forward folds used by the engine; also the feasibility test (`--entry open`) |

## Checks
| file | when |
|---|---|
| label_audit.py | after any panel rebuild (independent label recomputation) |
| smoke_test.py | after any panel rebuild (columns, timing contract, leakage screen) |
| verify_features.py | after any change to features_daily.py |
| reconcile.py | monthly: `python reconcile.py --sample 100` (independent price source) |
| liquidity_scan.py | when revisiting the Rs 1 crore liquidity floor |

## Research record (rerunnable; not daily)
| file | version | question it answered |
|---|---|---|
| feature_audit.py | - | which features enter (answer: base 5) |
| feature_atlas.py | v5 | which signals work where (`--target label_exit_ret`) |
| regime_research.py | v9 | M0-M4 + lockbox (spent: primary FAIL, excess +46 bp) |
| base_model.py | - | calibrated base model (earliest stage) |
| barrier_scan.py | - | feasible TP/SL brackets |
| specialist_test.py | v1 | one model per stock state (answer: rejected) |
| bucket_report.py | - | models x buckets x brackets x quarters |
| soul_test.py | v2 | does Soul v3 improve the engine? (`--entry open` supported) |
| challenger_test.py | v2 | macro ablation + HGB vs 4-model ensemble (close: HGB stays, macro stays; `--entry open` supported) |
| entry_timing.py | v1.1 | close vs next-open vs realistic entry (close edge does NOT survive: +3 bp realistic); logic shared with panel_build |
| compare_worlds.py | v1 | both entry worlds side by side, from results on disk |
| edge_anatomy.py | v1 | where in the five days the picks' return happens (ran 28 Sep: ON1 excess +136.3 [+124.2, +149.5], 63% of the week; every session negative vs market; gap-then-fade rejected) |
| tradability_audit.py | v1 | could the picks be bought? circuit locks at the close / next open, frozen exits, ETFs; declared rule: A_sub top-3 net 95% interval above zero, else close execution is dead (run pending) |

## Legacy (superseded; kept for reference)
| file | superseded by |
|---|---|
| soul.py, memory.py, soul_features.py | soul_v3.py |
| ledger.py | the engine's paper_ledger.csv (signal_engine resolve) |

## Tests
Run from the code folder, e.g. `python tests\test_signal_engine.py`. Each prints VERIFIED on success.

## Stage A research line - subfolder `open_close\` (28 Sep 2026)
Separate on purpose: the 23:45 daily pipeline keeps using the main folder's v31 code and `panel\`; stage A builds its own panel in `%CACHE_DAILY_ROOT%\panel_oc` (with its own `research_ledger.jsonl`). Run from `open_close\` with `set PYTHONPATH=<main folder>` so its modules win and the rest are found in the main folder.
| file (in open_close\) | version | role |
|---|---|---|
| panel_build.py | v32 | v31 + open-to-close targets `label_oc_1/2/3/5` (open T+1 -> close T+h) + `label_buyable_o1`; ETFs/funds excluded at the universe stage |
| label_audit.py | v2 | v1 + section 5 (open-to-close targets re-derived by a plain loop, convention identity on every row, firewall, ETFs absent, exclusions by year/symbol/market) + `--panel` + `--compare-to` (old labels unchanged) |
| research_common.py | - | main-folder version + ETF rule `etf-v1` + price-band lock proxy |
| EXPERIMENT_OC_FAMILY.json | oc_v1 | the pre-declared target family (4 horizons), stage-B screen rules, stage-D gate; never edited after stage B starts |
| tests\test_panel_oc.py | - | planted-effect tests incl. module-origin (no shadowing), golden checksums of the old labels, 5 sabotaged panels |
| stage_b_screen.py | v1.1 | stage B: target distributions first (N, mean, median, SD, p1-p99, positive share by fold, buy-everything net); then every clean feature x the 4 open-to-close targets (marginal evidence); per-day rank IC, Newey-West t, BH q, fold stability (declared rule q <= 0.10 AND 4/5 folds), descriptive quintile economics (`quintile_*` columns - not a trading result), descriptive families; research rows only (pinned lockbox 2025-03-05), eligible rows only; refuses an unaudited panel or a changed manifest; one run per panel unless `--rerun-reason`. Output `panel_oc\stage_b\STAGEB_*` |
| tests\test_stage_b.py | - | statistics vs independent code; planted world (signal, noise, unstable, unbuyable-only, post-lockbox-only, pre-fold-only, leak, market-level, price levels); power checks; preconditions |

## Optional user file
- `%CACHE_DAILY_ROOT%\universe_exclude.txt` - one symbol per line: extra funds/non-equities for panel_build to exclude (the ETF rule catches the systematic names).

## Not in this bundle (on purpose)
- `env_daily.bat`, `watchlist.txt` - yours (paths, credentials, universe).
- Daily_cache_v21 to v26 and development scratch scripts - superseded.
