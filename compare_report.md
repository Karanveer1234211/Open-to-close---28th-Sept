# Stage-D variants compared - COMPARE_20260929_003

compare_d_variants v1.2 | 2026-09-29 23:23

**Scoring (identical for every run):** each day's top-3 re-drawn from the run's own scores without ETFs (by the name rule - no bhavcopy given, so ETFs with plain names stay in); exits that close locked at the lower band moved to the first sellable open; 35 bp.

**Decision (declared before D2 was fitted):** a variant replaces the baseline for a target only if it passes the stage-D gate under this scoring AND its paired daily difference over the baseline has a 98.75% lower bound above zero.

Integrity: D (STAGED_20260928_001) 100.00%; D2 (STAGED2_20260929_002) 100.00%; DC (STAGEDC_20260929_001) 100.00%; C (STAGEDXC_20260929_001) 100.00%; ENS (STAGEDXE_20260929_001) 100.00% of picks re-derive from the raw cache.

## 1. Each run under corrected scoring (bp per trade)

| target | run | stage-D scoring | corrected net [98.75%] | corrected excess [98.75%] | 2024+ | since 2025-03-05 | locked exits | gate |
|---|---|---|---|---|---|---|---|---|
| label_oc_1 | D (STAGED_20260928_001) | +41.7 | -51.1 [-104.9, -8.4] | +1.2 [-51.7, +42.4] | -75.8 | -50.5 | 715 | FAIL |
| label_oc_1 | D2 (STAGED2_20260929_002) | +41.6 | -54.6 [-107.5, -14.1] | -2.3 [-53.9, +37.1] | -69.0 | -49.6 | 727 | FAIL |
| label_oc_1 | DC (STAGEDC_20260929_001) | +56.5 | -37.2 [-90.2, +4.2] | +15.1 [-36.6, +55.7] | -58.8 | -41.9 | 824 | FAIL |
| label_oc_1 | C (STAGEDXC_20260929_001) | +42.7 | -59.5 [-118.3, -13.8] | -7.2 [-64.8, +36.4] | -67.5 | -46.2 | 859 | FAIL |
| label_oc_1 | ENS (STAGEDXE_20260929_001) | +45.9 | -50.3 [-106.2, -7.4] | +2.0 [-53.5, +43.6] | -63.4 | -33.6 | 747 | FAIL |
| label_oc_2 | D (STAGED_20260928_001) | +54.2 | +32.2 [-5.0, +65.9] | +75.3 [+43.2, +103.8] | +12.0 | +30.2 | 205 | FAIL |
| label_oc_2 | D2 (STAGED2_20260929_002) | +46.4 | +27.5 [-6.0, +58.0] | +70.6 [+41.1, +96.2] | +6.5 | +15.6 | 183 | FAIL |
| label_oc_2 | DC (STAGEDC_20260929_001) | +70.8 | +48.2 [+9.8, +82.8] | +91.3 [+57.3, +120.5] | +18.7 | +12.9 | 275 | PASS |
| label_oc_2 | C (STAGEDXC_20260929_001) | +59.2 | +24.9 [-16.3, +62.3] | +68.0 [+31.6, +100.8] | -20.7 | -43.4 | 339 | FAIL |
| label_oc_2 | ENS (STAGEDXE_20260929_001) | +67.9 | +44.3 [+7.4, +77.9] | +87.4 [+54.2, +115.8] | +6.7 | -1.6 | 214 | PASS |
| label_oc_3 | D (STAGED_20260928_001) | +80.7 | +62.4 [+6.7, +111.0] | +96.1 [+50.8, +133.4] | +62.7 | +55.4 | 173 | PASS |
| label_oc_3 | D2 (STAGED2_20260929_002) | +78.1 | +64.0 [+19.8, +105.7] | +97.7 [+63.5, +128.4] | +46.3 | +39.6 | 137 | PASS |
| label_oc_3 | DC (STAGEDC_20260929_001) | +81.8 | +61.4 [+10.0, +107.2] | +95.1 [+53.1, +130.9] | +47.9 | +37.1 | 217 | PASS |
| label_oc_3 | C (STAGEDXC_20260929_001) | +75.0 | +51.6 [+0.0, +97.2] | +85.3 [+44.1, +119.7] | +20.2 | -0.4 | 260 | PASS |
| label_oc_3 | ENS (STAGEDXE_20260929_001) | +76.7 | +60.5 [+10.6, +104.2] | +94.2 [+53.1, +129.8] | +49.3 | +18.6 | 171 | PASS |
| label_oc_5 | D (STAGED_20260928_001) | +117.2 | +107.4 [+43.6, +171.1] | +121.6 [+82.0, +164.5] | +105.1 | +85.8 | 167 | PASS |
| label_oc_5 | D2 (STAGED2_20260929_002) | +89.1 | +77.5 [+15.2, +137.8] | +91.7 [+53.8, +130.7] | +83.0 | +91.8 | 161 | PASS |
| label_oc_5 | DC (STAGEDC_20260929_001) | +121.9 | +102.1 [+27.6, +174.3] | +116.3 [+62.3, +167.4] | +112.3 | +64.9 | 208 | PASS |
| label_oc_5 | C (STAGEDXC_20260929_001) | +119.2 | +95.6 [+21.9, +165.3] | +109.8 [+56.9, +160.9] | +59.2 | +43.9 | 240 | PASS |
| label_oc_5 | ENS (STAGEDXE_20260929_001) | +124.9 | +112.0 [+46.7, +177.9] | +126.1 [+82.6, +171.3] | +72.0 | +65.7 | 165 | PASS |

## 2. Decisions (paired: variant minus baseline, same days)

**D2 (STAGED2_20260929_002) vs D (STAGED_20260928_001)**

| target | paired difference [98.75%] | variant passes gate | same picks | DECISION |
|---|---|---|---|---|
| label_oc_1 | -3.5 [-18.7, +11.5] | no | 52% | **KEEP BASELINE** |
| label_oc_2 | -4.6 [-24.5, +14.3] | no | 41% | **KEEP BASELINE** |
| label_oc_3 | +1.6 [-23.5, +27.9] | yes | 35% | **KEEP BASELINE** |
| label_oc_5 | -29.9 [-64.6, +4.0] | yes | 29% | **KEEP BASELINE** |

**DC (STAGEDC_20260929_001) vs D (STAGED_20260928_001)**

| target | paired difference [98.75%] | variant passes gate | same picks | DECISION |
|---|---|---|---|---|
| label_oc_1 | +13.9 [-4.2, +31.7] | no | 43% | **KEEP BASELINE** |
| label_oc_2 | +16.0 [-7.0, +39.3] | yes | 35% | **KEEP BASELINE** |
| label_oc_3 | -0.9 [-24.0, +23.2] | yes | 35% | **KEEP BASELINE** |
| label_oc_5 | -5.3 [-49.3, +36.4] | yes | 28% | **KEEP BASELINE** |

**C (STAGEDXC_20260929_001) vs D (STAGED_20260928_001)**

| target | paired difference [98.75%] | variant passes gate | same picks | DECISION |
|---|---|---|---|---|
| label_oc_1 | -8.4 [-33.3, +14.5] | no | 32% | **KEEP BASELINE** |
| label_oc_2 | -7.2 [-36.2, +19.6] | no | 23% | **KEEP BASELINE** |
| label_oc_3 | -10.7 [-40.0, +18.3] | yes | 20% | **KEEP BASELINE** |
| label_oc_5 | -11.8 [-60.8, +37.0] | yes | 15% | **KEEP BASELINE** |

**ENS (STAGEDXE_20260929_001) vs D (STAGED_20260928_001)**

| target | paired difference [98.75%] | variant passes gate | same picks | DECISION |
|---|---|---|---|---|
| label_oc_1 | +0.8 [-14.1, +15.8] | no | 49% | **KEEP BASELINE** |
| label_oc_2 | +12.2 [-8.2, +32.3] | yes | 44% | **KEEP BASELINE** |
| label_oc_3 | -1.9 [-27.7, +24.0] | yes | 40% | **KEEP BASELINE** |
| label_oc_5 | +4.5 [-32.9, +42.0] | yes | 34% | **KEEP BASELINE** |

## 3. Pick agreement (descriptive - decides nothing)

Each day's corrected top-3 of the baseline and of each variant: stocks both chose, baseline-only, variant-only; mean corrected net per pick (bp). Daily P&L correlation with the baseline, and the same with the market's move removed (excess) - the plain one is inflated by shared market exposure.

**D2 (STAGED2_20260929_002) vs D (STAGED_20260928_001)**

| target | both (n, net) | baseline-only (n, net) | variant-only (n, net) | daily P&L corr | excess corr |
|---|---|---|---|---|---|
| label_oc_1 | 2,561, -84.1 | 2,404, -16.0 | 2,404, -23.2 | 0.85 | 0.82 |
| label_oc_2 | 2,018, +31.2 | 2,947, +32.8 | 2,947, +25.0 | 0.73 | 0.61 |
| label_oc_3 | 1,738, +90.3 | 3,227, +47.3 | 3,227, +49.8 | 0.73 | 0.59 |
| label_oc_5 | 1,424, +92.8 | 3,541, +113.1 | 3,541, +71.1 | 0.65 | 0.41 |

**DC (STAGEDC_20260929_001) vs D (STAGED_20260928_001)**

| target | both (n, net) | baseline-only (n, net) | variant-only (n, net) | daily P&L corr | excess corr |
|---|---|---|---|---|---|
| label_oc_1 | 2,136, -86.2 | 2,829, -24.6 | 2,829, -0.1 | 0.78 | 0.75 |
| label_oc_2 | 1,728, +57.6 | 3,237, +18.6 | 3,237, +43.1 | 0.71 | 0.59 |
| label_oc_3 | 1,747, +81.2 | 3,218, +52.1 | 3,218, +50.7 | 0.72 | 0.58 |
| label_oc_5 | 1,371, +114.1 | 3,594, +104.7 | 3,594, +97.6 | 0.66 | 0.44 |

**C (STAGEDXC_20260929_001) vs D (STAGED_20260928_001)**

| target | both (n, net) | baseline-only (n, net) | variant-only (n, net) | daily P&L corr | excess corr |
|---|---|---|---|---|---|
| label_oc_1 | 1,581, -145.9 | 3,384, -6.8 | 3,384, -19.2 | 0.71 | 0.67 |
| label_oc_2 | 1,147, +62.7 | 3,818, +23.0 | 3,818, +13.6 | 0.64 | 0.50 |
| label_oc_3 | 992, +107.5 | 3,973, +51.1 | 3,973, +37.7 | 0.64 | 0.46 |
| label_oc_5 | 727, +127.5 | 4,238, +103.8 | 4,238, +89.8 | 0.58 | 0.33 |

**ENS (STAGEDXE_20260929_001) vs D (STAGED_20260928_001)**

| target | both (n, net) | baseline-only (n, net) | variant-only (n, net) | daily P&L corr | excess corr |
|---|---|---|---|---|---|
| label_oc_1 | 2,451, -102.1 | 2,514, -1.4 | 2,514, +0.2 | 0.86 | 0.84 |
| label_oc_2 | 2,190, +52.4 | 2,775, +16.2 | 2,775, +37.9 | 0.77 | 0.67 |
| label_oc_3 | 1,995, +84.9 | 2,970, +47.2 | 2,970, +44.1 | 0.75 | 0.61 |
| label_oc_5 | 1,687, +122.5 | 3,278, +99.4 | 3,278, +106.2 | 0.71 | 0.52 |

## 4. Corrected net by year (bp)

- **label_oc_1 D (STAGED_20260928_001)**: 2020 -23.6; 2021 +77.4; 2022 -66.8; 2023 -124.3; 2024 -97.6; 2025 -60.3; 2026 -66.7
- **label_oc_1 D2 (STAGED2_20260929_002)**: 2020 -44.1; 2021 +41.4; 2022 -51.7; 2023 -125.0; 2024 -78.4; 2025 -55.5; 2026 -74.9
- **label_oc_1 DC (STAGEDC_20260929_001)**: 2020 -7.5; 2021 +74.4; 2022 -32.7; 2023 -124.2; 2024 -57.7; 2025 -58.3; 2026 -60.9
- **label_oc_1 C (STAGEDXC_20260929_001)**: 2020 -62.4; 2021 +67.6; 2022 -68.5; 2023 -154.1; 2024 -68.2; 2025 -63.0; 2026 -73.1
- **label_oc_1 ENS (STAGEDXE_20260929_001)**: 2020 -26.9; 2021 +65.9; 2022 -60.5; 2023 -144.4; 2024 -77.5; 2025 -61.2; 2026 -46.4
- **label_oc_2 D (STAGED_20260928_001)**: 2020 +71.5; 2021 +76.6; 2022 +45.0; 2023 -8.9; 2024 +11.9; 2025 -1.5; 2026 +31.3
- **label_oc_2 D2 (STAGED2_20260929_002)**: 2020 +53.8; 2021 +77.4; 2022 +29.5; 2023 +7.1; 2024 +28.5; 2025 -39.4; 2026 +40.9
- **label_oc_2 DC (STAGEDC_20260929_001)**: 2020 +85.0; 2021 +117.5; 2022 +55.3; 2023 +15.5; 2024 +50.1; 2025 -10.3; 2026 +15.0
- **label_oc_2 C (STAGEDXC_20260929_001)**: 2020 +97.0; 2021 +96.0; 2022 +21.1; 2023 +11.1; 2024 +30.3; 2025 -31.0; 2026 -79.2
- **label_oc_2 ENS (STAGEDXE_20260929_001)**: 2020 +104.2; 2021 +116.0; 2022 +56.4; 2023 +4.1; 2024 +41.1; 2025 -39.2; 2026 +23.2
- **label_oc_3 D (STAGED_20260928_001)**: 2020 +94.4; 2021 +150.3; 2022 +25.0; 2023 -21.1; 2024 +100.1; 2025 +35.8; 2026 +47.9
- **label_oc_3 D2 (STAGED2_20260929_002)**: 2020 +93.8; 2021 +133.2; 2022 +59.1; 2023 +18.1; 2024 +79.4; 2025 +26.8; 2026 +27.1
- **label_oc_3 DC (STAGEDC_20260929_001)**: 2020 +113.4; 2021 +147.9; 2022 +39.9; 2023 -17.9; 2024 +89.8; 2025 +13.2; 2026 +37.4
- **label_oc_3 C (STAGEDXC_20260929_001)**: 2020 +120.4; 2021 +108.4; 2022 +28.2; 2023 +36.5; 2024 +72.0; 2025 +2.2; 2026 -28.0
- **label_oc_3 ENS (STAGEDXE_20260929_001)**: 2020 +98.5; 2021 +135.3; 2022 +27.9; 2023 +11.0; 2024 +112.3; 2025 +6.9; 2026 +20.0
- **label_oc_5 D (STAGED_20260928_001)**: 2020 +126.4; 2021 +143.8; 2022 +100.6; 2023 +65.5; 2024 +211.1; 2025 +20.2; 2026 +74.7
- **label_oc_5 D2 (STAGED2_20260929_002)**: 2020 +28.3; 2021 +148.5; 2022 +13.9; 2023 +103.3; 2024 +130.2; 2025 +26.3; 2026 +96.7
- **label_oc_5 DC (STAGEDC_20260929_001)**: 2020 +169.4; 2021 +118.5; 2022 +52.7; 2023 +41.9; 2024 +256.7; 2025 +32.7; 2026 +19.6
- **label_oc_5 C (STAGEDXC_20260929_001)**: 2020 +204.3; 2021 +115.6; 2022 +9.9; 2023 +155.1; 2024 +154.1; 2025 +23.2; 2026 -25.1
- **label_oc_5 ENS (STAGEDXE_20260929_001)**: 2020 +221.9; 2021 +139.2; 2022 +102.1; 2023 +96.0; 2024 +155.9; 2025 +19.8; 2026 +26.7

## 5. What this does not say

- Tick costs and trade-for-trade restrictions are not in this scoring (stage_d_forensics covers ticks).
- Survivorship is unmeasured. Only the forward paper ledger is untouched evidence.
