# Stage-D variants compared - COMPARE_20260929_001

compare_d_variants v1 | 2026-09-29 10:38

**Scoring (identical for every run):** each day's top-3 re-drawn from the run's own scores without ETFs (by the name rule - no bhavcopy given, so ETFs with plain names stay in); exits that close locked at the lower band moved to the first sellable open; 35 bp.

**Decision (declared before D2 was fitted):** a variant replaces the baseline for a target only if it passes the stage-D gate under this scoring AND its paired daily difference over the baseline has a 98.75% lower bound above zero.

Integrity: D (STAGED_20260928_001) 100.00%; D2 (STAGED2_20260929_002) 100.00% of picks re-derive from the raw cache.

## 1. Each run under corrected scoring (bp per trade)

| target | run | stage-D scoring | corrected net [98.75%] | corrected excess [98.75%] | 2024+ | since 2025-03-05 | locked exits | gate |
|---|---|---|---|---|---|---|---|---|
| label_oc_1 | D (STAGED_20260928_001) | +41.7 | -51.1 [-104.9, -8.4] | +1.2 [-51.7, +42.4] | -75.8 | -50.5 | 715 | FAIL |
| label_oc_1 | D2 (STAGED2_20260929_002) | +41.6 | -54.6 [-107.5, -14.1] | -2.3 [-53.9, +37.1] | -69.0 | -49.6 | 727 | FAIL |
| label_oc_2 | D (STAGED_20260928_001) | +54.2 | +32.2 [-5.0, +65.9] | +75.3 [+43.2, +103.8] | +12.0 | +30.2 | 205 | FAIL |
| label_oc_2 | D2 (STAGED2_20260929_002) | +46.4 | +27.5 [-6.0, +58.0] | +70.6 [+41.1, +96.2] | +6.5 | +15.6 | 183 | FAIL |
| label_oc_3 | D (STAGED_20260928_001) | +80.7 | +62.4 [+6.7, +111.0] | +96.1 [+50.8, +133.4] | +62.7 | +55.4 | 173 | PASS |
| label_oc_3 | D2 (STAGED2_20260929_002) | +78.1 | +64.0 [+19.8, +105.7] | +97.7 [+63.5, +128.4] | +46.3 | +39.6 | 137 | PASS |
| label_oc_5 | D (STAGED_20260928_001) | +117.2 | +107.4 [+43.6, +171.1] | +121.6 [+82.0, +164.5] | +105.1 | +85.8 | 167 | PASS |
| label_oc_5 | D2 (STAGED2_20260929_002) | +89.1 | +77.5 [+15.2, +137.8] | +91.7 [+53.8, +130.7] | +83.0 | +91.8 | 161 | PASS |

## 2. Decisions (paired: variant minus baseline, same days)

**D2 (STAGED2_20260929_002) vs D (STAGED_20260928_001)**

| target | paired difference [98.75%] | variant passes gate | same picks | DECISION |
|---|---|---|---|---|
| label_oc_1 | -3.5 [-18.7, +11.5] | no | 52% | **KEEP BASELINE** |
| label_oc_2 | -4.6 [-24.5, +14.3] | no | 41% | **KEEP BASELINE** |
| label_oc_3 | +1.6 [-23.5, +27.9] | yes | 35% | **KEEP BASELINE** |
| label_oc_5 | -29.9 [-64.6, +4.0] | yes | 29% | **KEEP BASELINE** |

## 3. Corrected net by year (bp)

- **label_oc_1 D (STAGED_20260928_001)**: 2020 -23.6; 2021 +77.4; 2022 -66.8; 2023 -124.3; 2024 -97.6; 2025 -60.3; 2026 -66.7
- **label_oc_1 D2 (STAGED2_20260929_002)**: 2020 -44.1; 2021 +41.4; 2022 -51.7; 2023 -125.0; 2024 -78.4; 2025 -55.5; 2026 -74.9
- **label_oc_2 D (STAGED_20260928_001)**: 2020 +71.5; 2021 +76.6; 2022 +45.0; 2023 -8.9; 2024 +11.9; 2025 -1.5; 2026 +31.3
- **label_oc_2 D2 (STAGED2_20260929_002)**: 2020 +53.8; 2021 +77.4; 2022 +29.5; 2023 +7.1; 2024 +28.5; 2025 -39.4; 2026 +40.9
- **label_oc_3 D (STAGED_20260928_001)**: 2020 +94.4; 2021 +150.3; 2022 +25.0; 2023 -21.1; 2024 +100.1; 2025 +35.8; 2026 +47.9
- **label_oc_3 D2 (STAGED2_20260929_002)**: 2020 +93.8; 2021 +133.2; 2022 +59.1; 2023 +18.1; 2024 +79.4; 2025 +26.8; 2026 +27.1
- **label_oc_5 D (STAGED_20260928_001)**: 2020 +126.4; 2021 +143.8; 2022 +100.6; 2023 +65.5; 2024 +211.1; 2025 +20.2; 2026 +74.7
- **label_oc_5 D2 (STAGED2_20260929_002)**: 2020 +28.3; 2021 +148.5; 2022 +13.9; 2023 +103.3; 2024 +130.2; 2025 +26.3; 2026 +96.7

## 4. What this does not say

- Tick costs and trade-for-trade restrictions are not in this scoring (stage_d_forensics covers ticks).
- Survivorship is unmeasured. Only the forward paper ledger is untouched evidence.
