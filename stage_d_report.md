# Stage D gate - STAGED_20260928_001

Family oc_v1 (manifest sha 1f5c13384e36e759) | declaration sha dcfab6eccc05fcb9 | stage_d_gate v1 | 2026-09-29 00:06

**Rule (declared before any fit):** per target, PASS needs (1) top-3 net per trade with a 98.75% block-bootstrap lower bound above zero, (2) the mean top-3 net from 2024-01-01 above zero, and (3) top-3 excess over buying every eligible stock with a 98.75% lower bound above zero.

Model: HistGradientBoostingRegressor (fixed settings, no tuning), 205 clean features with no selection, trained on the day-demeaned target clipped at the fold's 1st/99th percentiles; picks ranked by raw score. Entry: open T+1; exit: close T+h; cost 35 bp per round trip. Eligible rows only (1,890,199 of 1,902,513). Sessions 2016-06-06..2026-09-15.

## 1. The gate

| target | top-3 net (98.75%) | top-3 excess (98.75%) | net from 2024 (95%) | buy-everything net | rank IC | VERDICT |
|---|---|---|---|---|---|---|
| label_oc_1 | +41.7 [+25.9, +57.4] | +94.0 [+78.3, +109.8] | +10.7 [-5.6, +27.5] | -52.3 | +0.1109 | **PASS** |
| label_oc_2 | +54.2 [+27.2, +82.0] | +97.2 [+75.7, +120.4] | +27.1 [-3.3, +59.8] | -43.1 | +0.1055 | **PASS** |
| label_oc_3 | +80.7 [+38.9, +121.7] | +114.4 [+85.8, +143.4] | +70.5 [+25.9, +116.4] | -33.7 | +0.0975 | **PASS** |
| label_oc_5 | +117.2 [+53.3, +179.2] | +131.3 [+91.8, +173.4] | +116.1 [+44.4, +191.6] | -14.1 | +0.0793 | **PASS** |

bp per trade. Intervals: circular block bootstrap over trading days, block 10, 10,000 resamples.

## 2. Top-N (descriptive; 95% intervals)

| target | top-1 net | top-3 net | top-5 net | top-10 net | top-1 excess | top-3 excess | top-5 excess | top-10 excess |
|---|---|---|---|---|---|---|---|---|
| label_oc_1 | +64.1 [+44.0, +83.6] | +41.7 [+29.4, +54.3] | +25.1 [+15.2, +35.4] | +8.2 [+0.2, +16.3] | +116.4 | +94.0 | +77.4 | +60.4 |
| label_oc_2 | +92.4 [+61.0, +124.6] | +54.2 [+33.3, +76.1] | +41.1 [+22.8, +59.7] | +26.1 [+10.5, +41.7] | +135.5 | +97.2 | +84.2 | +69.2 |
| label_oc_3 | +99.7 [+55.1, +145.1] | +80.7 [+47.8, +113.2] | +62.4 [+33.6, +91.1] | +46.7 [+20.9, +71.6] | +133.4 | +114.4 | +96.1 | +80.4 |
| label_oc_5 | +122.7 [+58.4, +187.1] | +117.2 [+68.0, +166.9] | +104.2 [+56.7, +151.4] | +82.4 [+38.6, +125.5] | +136.8 | +131.3 | +118.4 | +96.5 |

## 3. By fold (top-3 net / excess / buy-everything net, bp; rank IC)

- **label_oc_1**: F1 +60.6 / +114.3 / -53.7 (IC +0.1112); F2 +69.3 / +123.8 / -54.5 (IC +0.1239); F3 +57.2 / +108.5 / -51.3 (IC +0.1171); F4 +28.8 / +81.3 / -52.5 (IC +0.1088); F5 -7.4 / +42.0 / -49.4 (IC +0.0934)
- **label_oc_2**: F1 +75.4 / +115.8 / -40.4 (IC +0.1095); F2 +78.1 / +122.3 / -44.2 (IC +0.1162); F3 +61.8 / +98.9 / -37.0 (IC +0.1091); F4 +37.3 / +84.5 / -47.2 (IC +0.1052); F5 +18.2 / +64.8 / -46.6 (IC +0.0872)
- **label_oc_3**: F1 +108.8 / +135.2 / -26.3 (IC +0.1082); F2 +95.1 / +129.6 / -34.6 (IC +0.1074); F3 +54.9 / +77.3 / -22.4 (IC +0.1005); F4 +104.0 / +145.4 / -41.4 (IC +0.0855); F5 +40.7 / +84.5 / -43.8 (IC +0.0859)
- **label_oc_5**: F1 +147.3 / +142.1 / +5.3 (IC +0.0923); F2 +113.9 / +129.2 / -15.3 (IC +0.0875); F3 +91.1 / +83.9 / +7.2 (IC +0.0773); F4 +147.4 / +177.0 / -29.7 (IC +0.0519); F5 +86.1 / +124.3 / -38.3 (IC +0.0877)

## 4. By year (top-3 net / excess, bp)

- **label_oc_1**: 2020 +43.2 / +98.2; 2021 +103.3 / +158.9; 2022 +52.1 / +106.4; 2023 +52.4 / +100.9; 2024 +27.7 / +82.2; 2025 +4.1 / +53.4; 2026 -4.3 / +43.3
- **label_oc_2**: 2020 +84.3 / +129.7; 2021 +85.2 / +122.0; 2022 +74.3 / +124.7; 2023 +46.9 / +77.8; 2024 +38.0 / +81.7; 2025 +12.2 / +62.8; 2026 +33.0 / +77.0
- **label_oc_3**: 2020 +95.6 / +130.4; 2021 +152.9 / +170.8; 2022 +51.3 / +98.3; 2023 +50.8 / +63.8; 2024 +111.9 / +144.8; 2025 +44.3 / +96.4; 2026 +48.8 / +88.7
- **label_oc_5**: 2020 +131.1 / +140.5; 2021 +154.8 / +134.4; 2022 +115.0 / +155.8; 2023 +70.6 / +46.2; 2024 +225.3 / +239.3; 2025 +25.1 / +78.2; 2026 +90.0 / +121.1

## 5. The stretch stage B never read (2025-03-05 onward)

- **label_oc_1**: 378 days | top-3 net -1.4 | excess +43.5 | buy-everything net -45.0
- **label_oc_2**: 378 days | top-3 net +37.5 | excess +76.7 | buy-everything net -39.2
- **label_oc_3**: 378 days | top-3 net +60.2 | excess +93.8 | buy-everything net -33.5
- **label_oc_5**: 378 days | top-3 net +94.7 | excess +115.0 | buy-everything net -20.3

## 6. Picks and mechanics

- **label_oc_1**: days where the 3rd and 4th scores tie: 0.00% | top-3 picks with no outcome (no bar at T+h): 0 | D_atr_pct top-3 median 5.684 vs all 3.784; X_turnover_med top-3 median 9.29e+07 vs all 1.013e+08
- **label_oc_2**: days where the 3rd and 4th scores tie: 0.00% | top-3 picks with no outcome (no bar at T+h): 0 | D_atr_pct top-3 median 5.404 vs all 3.784; X_turnover_med top-3 median 1.193e+08 vs all 1.013e+08
- **label_oc_3**: days where the 3rd and 4th scores tie: 0.00% | top-3 picks with no outcome (no bar at T+h): 0 | D_atr_pct top-3 median 5.261 vs all 3.784; X_turnover_med top-3 median 8.885e+07 vs all 1.013e+08
- **label_oc_5**: days where the 3rd and 4th scores tie: 0.00% | top-3 picks with no outcome (no bar at T+h): 1 | D_atr_pct top-3 median 5.128 vs all 3.784; X_turnover_med top-3 median 8.323e+07 vs all 1.013e+08

## 7. What this does not say

- A PASS makes that target a forward paper-test arm. No money before the month-6 check.
- The walk-forward spans history studied in earlier stages (with other targets); only the forward paper ledger is untouched evidence.
- Fills: open-auction prices assumed at the open; exits locked at the lower band and slippage are stage E.
- Per trade is the declared unit. A 5-session hold ties capital up 5x longer than a 1-session hold; return per day of capital is a stage-E comparison.
