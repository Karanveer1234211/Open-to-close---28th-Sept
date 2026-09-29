# bigmove_deploy — frozen bundle (29 Sept 2026; + tradability_audit v1; + stages A, B, C, D, D-forensics, D2, D_C and the variant comparison in `open_close\`)

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
| stage_b_screen.py | v1.2 | stage B: target distributions first (N, mean, median, SD, p1-p99, positive share by fold, buy-everything net); then every clean feature x the 4 open-to-close targets (marginal evidence); per-day rank IC, Newey-West t, BH q, fold stability (declared rule q <= 0.10 AND 4/5 folds), descriptive quintile economics (`quintile_*` columns - not a trading result), descriptive families; research rows only (pinned lockbox 2025-03-05), eligible rows only; refuses an unaudited panel or a changed manifest; one run per panel unless `--rerun-reason`. Output `panel_oc\stage_b\STAGEB_*` |
| tests\test_stage_b.py | - | statistics vs independent code; planted world (signal, noise, unstable, unbuyable-only, post-lockbox-only, pre-fold-only, leak, market-level, price levels); power checks; preconditions |
| stage_d_gate.py | v1 | stage D: per target, HGB (fixed settings, no tuning) on all clean features, no selection; label = target minus the day's eligible mean, clipped at the fold's 1st/99th pct; walk-forward over every resolved session incl. 2025-03-05+ (embargo asserted); daily top-3 by raw score among eligible rows. PASS = 98.75% lower bound of top-3 net > 0 AND 2024+ mean net > 0 AND 98.75% lower bound of excess > 0. Refuses unless audited panel, unchanged manifest, stage B run, settings = declaration; logs the declaration hash before fitting. Output `panel_oc\stage_d\STAGED_*` |
| STAGE_D_DECLARATION.json | oc_v1 D | the stage-D rules and every setting, declared before any fit; the tool refuses if they differ |
| tests\test_stage_d.py | - | label, bootstrap (= research_common at 95%), top-n with ties/missing outcomes; a world with a different truth per target (noise FAIL; skill in a losing market FAIL on net; skill ending 2024 FAIL on recency; persistent PASS); pandas re-derivation of every daily top-3; eligibility / embargo / firewall sabotage |
| stage_d_forensics.py | v1 | re-scores each stage-D target's daily top-3 with three corrections that can only block: ETFs out by ISIN (INF = fund unit, from an NSE bhavcopy), exits closing locked at the lower band moved to the first sellable open, one tick per round trip at NSE's tick size for the date; then the stage-D gate unchanged. Stops if picks no longer re-derive from the raw cache (<99.5%). Output `panel_oc\stage_d\<run>\forensics\FORENSICS_*` |
| FORENSICS_DECLARATION.json | oc_v1 D-forensics | the forensic rules and settings, declared before the run |
| tests\test_stage_d_forensics.py | - | tick schedule at every boundary; locked-exit re-scoring by hand; ISIN files in 3 layouts + zip; a world where each target dies at a different correction (penny ticks, ETFs, locked exits) and a clean one survives; integrity stop; preconditions |
| stage_d2_nested.py | v1.1 | D2: stage D's exact model, but per fold stage B's rule is re-run on that fold's TRAINING rows only (BH q <= 0.10 pooled over the fold's cells + sign in >= 4 of 5 training blocks), one feature per family, plus all market-level columns. Output `panel_oc\stage_d\STAGED2_*` in stage D's format + selection.json |
| STAGE_D2_DECLARATION.json | oc_v1 D2 | the nested selection and the switch rule, declared before D2 was fitted |
| compare_d_variants.py | v1.1 | scores stage-D runs identically (ETFs re-drawn out, exits locked at the lower band moved to the first sellable open) and applies the declared switch rule: a variant replaces the baseline only if it passes the gate under corrected scoring AND its paired daily difference has a 98.75% lower bound above zero. Output `panel_oc\stage_d\compare\COMPARE_*` |
| tests\test_stage_d2.py | - | nested screen vs scipy; stability, families, market-level; end to end: selection sees only training rows (instrumented), a test-window-only feature is never picked, model settings = stage D |
| tests\test_compare_d.py | - | identical variant = exact tie; a better variant replaces on its target only; a locked-exit picker loses once exits are honest; cross-checked against the forensics tool |
| STAGE_C_DECLARATION.json | oc_v1_c | the 123 stage-C features (formula + source each), the screen rule, the leak stops, D_C and the decision rule, declared before any C feature was computed |
| stage_c_features.py | v1 | builds the 123 C features point in time from the raw cache + universe membership (+6 panel columns for interactions): overnight/intraday decomposition, market model/residuals/delay, MAX, spread estimators, NSE band-lock state, relative position to indicators, momentum/seasonality, interactions, universe market state, per-date ranks. Output `panel_oc\stage_c\features_c.parquet`, row for row with the panel |
| stage_c_screen.py | v1 | stage B's rule on the C features as their own BH family + leak stops (|IC| >= 0.15 vs a target or vs label_gap1; |rho| > 0.15 vs the next 5 daily returns). Output `panel_oc\stage_c\screen\STAGEC_*` |
| stage_dc_model.py | v1 | stage D's exact model on all clean features + all 123 C features, no selection; refuses after a leaky C screen. Output `panel_oc\stage_d\STAGEDC_*` |
| tests\test_stage_c_features.py | - | formulas vs independent code; band locks by tick size; POINT IN TIME: all 123 features recomputed on data cut at 3 dates equal the full-data values exactly, and scrambling the future changes nothing |
| tests\test_stage_c_screen_dc.py | - | external-source screen = stage B exactly; clean world passes and D_C runs; gap-knowing and tomorrow-knowing features are caught and D_C refuses; misaligned or foreign C files refused |

## Optional user file
- `%CACHE_DAILY_ROOT%\universe_exclude.txt` - one symbol per line: extra funds/non-equities for panel_build to exclude (the ETF rule catches the systematic names).

## Not in this bundle (on purpose)
- `env_daily.bat`, `watchlist.txt` - yours (paths, credentials, universe).
- Daily_cache_v21 to v26 and development scratch scripts - superseded.
