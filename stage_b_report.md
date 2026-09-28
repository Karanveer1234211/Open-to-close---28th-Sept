# Stage B feature screen - STAGEB_20260928_001

Family oc_v1 (manifest sha 1f5c13384e36e759) | stage_b_screen v1.1 | built 2026-09-28 22:53

**Rule (declared before this run):** a cell passes only with Benjamini-Hochberg q <= 0.10 across every tested cell AND the IC sign equal to the pooled sign in at least 4 of 5 folds.

**What an IC is:** each day, stocks ranked by the feature vs ranked by the target, correlated. Ranking within a day cancels the market, so it measures selection only. It is MARGINAL evidence (each feature on its own), not incremental (beyond other features) - that is stage D's question.

Rows: research only, eligible only (next open buyable). Research ends 2025-02-24; the lockbox from 2025-03-05 is never read. Scored windows: fold 1 2019-07-01..2020-08-18 (281 sessions); fold 2 2020-08-19..2021-10-04 (281 sessions); fold 3 2021-10-05..2022-11-22 (281 sessions); fold 4 2022-11-23..2024-01-09 (281 sessions); fold 5 2024-01-10..2025-02-24 (282 sessions).

Eligible rows scored: 1,094,307 (unbuyable rows skipped in these windows: 7,948). Clean features: 205 (34 market-level, listed separately).

## 1. Target distribution (descriptive, the rows the screen scores)

Per trade, basis points, pooled over every eligible stock-day in the five windows. Market = the equal-weight average of all eligible stocks each day, averaged over days.

| statistic | label_oc_1 | label_oc_2 | label_oc_3 | label_oc_5 |
|---|---|---|---|---|
| N | 1,094,307 | 1,094,307 | 1,094,307 | 1,094,307 |
| mean | -19.4 | -10.1 | -1.0 | +17.8 |
| median | -34.7 | -33.3 | -31.1 | -24.5 |
| SD | 262.8 | 386.2 | 478.9 | 629.7 |
| p1 | -647.4 | -937.5 | -1127.3 | -1435.1 |
| p5 | -393.8 | -565.7 | -688.8 | -877.2 |
| p25 | -157.4 | -214.7 | -258.1 | -325.4 |
| p75 | +94.7 | +162.3 | +214.9 | +305.5 |
| p95 | +410.3 | +625.9 | +789.0 | +1055.1 |
| p99 | +810.0 | +1177.8 | +1474.4 | +1963.7 |
| positive % | 41.6 | 44.6 | 45.9 | 47.6 |
| above 35 bp cost % | 34.8 | 39.8 | 42.0 | 44.6 |
| market mean | -19.1 | -9.8 | -0.5 | +18.8 |
| market net after 35 bp | -54.1 | -44.8 | -35.5 | -16.2 |

Positive-return share by fold (%):

| target | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 |
|---|---|---|---|---|---|
| label_oc_1 | 42.2 | 40.9 | 41.5 | 41.8 | 41.8 |
| label_oc_2 | 44.5 | 45.2 | 44.2 | 45.9 | 43.5 |
| label_oc_3 | 45.4 | 47.5 | 45.2 | 47.9 | 44.1 |
| label_oc_5 | 46.8 | 50.0 | 46.3 | 50.4 | 44.9 |

Buying every eligible stock at the next open, gross, per trade (bp) - the hole every long trade starts in:

| target | fold 1 | fold 2 | fold 3 | fold 4 | fold 5 | pooled | pooled net of 35 bp |
|---|---|---|---|---|---|---|---|
| label_oc_1 | -18.5 | -16.2 | -23.0 | -13.5 | -24.2 | -19.1 | -54.1 |
| label_oc_2 | -17.7 | +7.8 | -19.4 | +4.0 | -23.4 | -9.8 | -44.8 |
| label_oc_3 | -16.2 | +31.9 | -15.9 | +21.1 | -23.4 | -0.5 | -35.5 |
| label_oc_5 | -9.1 | +80.1 | -8.6 | +55.1 | -23.4 | +18.8 | -16.2 |

## 2. Feature screen (declared rule)

| target | tested | q <= 0.10 | stable 4/5 | PASS | +IC | -IC | families |
|---|---|---|---|---|---|---|---|
| label_oc_1 | 171 | 154 | 156 | **153** | 54 | 99 | 82 |
| label_oc_2 | 171 | 157 | 159 | **157** | 58 | 99 | 87 |
| label_oc_3 | 171 | 151 | 162 | **150** | 56 | 94 | 79 |
| label_oc_5 | 171 | 147 | 160 | **146** | 54 | 92 | 77 |

## 3. Strongest passing cells, per target

**Descriptive quintile economics - not a trading result.** The money columns are for the fifth of stocks the IC favours (~150 names a day), per trade, in bp; a top-3 pick is far more extreme and is judged only in stage D. One row per family leader (families are descriptive; every passing cell stays in screen.csv, near-misses included).

### label_oc_1

| feature | IC | t | q | folds (1..5) | spread Q5-Q1 | quintile | quintile gross | quintile excess | quintile net after cost | family size |
|---|---|---|---|---|---|---|---|---|---|---|
| D_WQ_44 | +0.0400 | +27.4 | 1.6e-162 | +0.037 +0.039 +0.041 +0.044 +0.039 | +14.8 | Q5 | -11.6 | +7.5 | -46.6 | 1 |
| D_range_pct | -0.0720 | -27.2 | 2.8e-160 | -0.070 -0.082 -0.071 -0.069 -0.068 | -19.8 | Q1 | -10.1 | +9.0 | -45.1 | 3 |
| D_vol_yz_20 | -0.0753 | -22.8 | 5.8e-113 | -0.071 -0.085 -0.070 -0.077 -0.074 | -17.4 | Q1 | -11.3 | +7.8 | -46.3 | 9 |
| D_days_since_5pct_up | +0.0583 | +22.2 | 6.5e-107 | +0.052 +0.069 +0.058 +0.059 +0.054 | +14.4 | Q5 | -14.0 | +5.1 | -49.0 | 1 |
| X_rank_D_gap_pct | -0.0424 | -21.9 | 1.2e-104 | -0.035 -0.045 -0.037 -0.053 -0.042 | -19.6 | Q1 | -13.0 | +6.1 | -48.0 | 3 |
| D_amihud_60 | -0.0594 | -21.7 | 3.6e-102 | -0.056 -0.068 -0.060 -0.065 -0.048 | -15.0 | Q1 | -11.7 | +7.4 | -46.7 | 6 |
| D_WQ_26 | +0.0326 | +20.9 | 1.1e-95 | +0.027 +0.038 +0.036 +0.030 +0.032 | +10.8 | Q5 | -13.9 | +5.2 | -48.9 | 1 |
| D_n_5pct_up_60 | -0.0584 | -20.1 | 5.8e-88 | -0.054 -0.066 -0.056 -0.060 -0.056 | -12.5 | Q1 | -14.8 | +4.5 | -49.8 | 2 |
| D_WQ_40 | +0.0432 | +19.0 | 3.4e-79 | +0.033 +0.046 +0.049 +0.048 +0.040 | +16.8 | Q5 | -11.3 | +7.8 | -46.3 | 1 |
| D_mdi14_diff1 | +0.0394 | +18.5 | 3e-75 | +0.032 +0.039 +0.040 +0.041 +0.045 | +18.8 | Q5 | -11.0 | +8.1 | -46.0 | 1 |
| D_ret_5d_roll_std | -0.0612 | -18.4 | 1.5e-74 | -0.068 -0.062 -0.053 -0.063 -0.060 | -13.6 | Q1 | -13.4 | +5.7 | -48.4 | 1 |
| D_n_5pct_up_252 | -0.0631 | -18.3 | 3e-73 | -0.062 -0.059 -0.064 -0.064 -0.067 | -12.3 | Q1 | -12.9 | +6.2 | -47.9 | 1 |

### label_oc_2

| feature | IC | t | q | folds (1..5) | spread Q5-Q1 | quintile | quintile gross | quintile excess | quintile net after cost | family size |
|---|---|---|---|---|---|---|---|---|---|---|
| D_WQ_44 | +0.0425 | +21.5 | 4.3e-101 | +0.037 +0.042 +0.048 +0.045 +0.042 | +22.4 | Q5 | +1.1 | +10.9 | -33.9 | 1 |
| D_mdi14_diff1 | +0.0464 | +20.7 | 3.8e-93 | +0.039 +0.052 +0.046 +0.049 +0.046 | +31.3 | Q5 | +1.9 | +11.7 | -33.1 | 1 |
| X_rank_D_range_pct | -0.0640 | -19.9 | 2.2e-86 | -0.066 -0.074 -0.060 -0.052 -0.067 | -23.7 | Q1 | -2.3 | +7.4 | -37.3 | 3 |
| D_rsi14_obv_x | -0.0382 | -17.5 | 3e-67 | -0.032 -0.040 -0.035 -0.044 -0.040 | -23.4 | Q1 | +0.3 | +10.0 | -34.7 | 3 |
| D_WQ_35 | +0.0381 | +16.4 | 2.9e-59 | +0.023 +0.039 +0.043 +0.046 +0.040 | +25.6 | Q5 | +0.7 | +10.5 | -34.3 | 1 |
| D_WQ_26 | +0.0322 | +16.1 | 3.1e-57 | +0.027 +0.039 +0.038 +0.026 +0.032 | +15.1 | Q5 | -2.0 | +7.8 | -37.0 | 1 |
| Comb_GapUp__CPR_Tmr_Above | -0.0324 | -16.1 | 5.8e-57 | -0.020 -0.039 -0.032 -0.038 -0.034 | +nan | Q1 | +nan | +nan | +nan | 1 |
| D_days_since_5pct_up | +0.0517 | +16.0 | 3e-56 | +0.051 +0.062 +0.052 +0.043 +0.051 | +17.6 | Q5 | -5.9 | +4.1 | -40.9 | 1 |
| X_z_D_gap_pct | -0.0331 | -15.9 | 1.2e-55 | -0.023 -0.039 -0.029 -0.040 -0.035 | -18.3 | Q1 | -5.9 | +3.9 | -40.9 | 3 |
| D_WQ_40 | +0.0439 | +15.6 | 1.7e-53 | +0.035 +0.049 +0.052 +0.043 +0.040 | +25.7 | Q5 | +1.9 | +11.7 | -33.1 | 1 |
| D_structure_trend_code | -0.0356 | -15.5 | 3.6e-53 | -0.022 -0.039 -0.038 -0.043 -0.035 | -22.7 | Q1 | +1.7 | +10.9 | -33.3 | 2 |
| D_WQ_29 | +0.0488 | +15.5 | 5.4e-53 | +0.042 +0.059 +0.051 +0.052 +0.041 | +28.5 | Q5 | +0.5 | +10.3 | -34.5 | 1 |

### label_oc_3

| feature | IC | t | q | folds (1..5) | spread Q5-Q1 | quintile | quintile gross | quintile excess | quintile net after cost | family size |
|---|---|---|---|---|---|---|---|---|---|---|
| D_mdi14_diff1 | +0.0431 | +20.2 | 1.7e-89 | +0.034 +0.052 +0.043 +0.045 +0.041 | +36.2 | Q5 | +11.3 | +11.9 | -23.7 | 1 |
| D_WQ_44 | +0.0424 | +17.9 | 2.7e-70 | +0.036 +0.042 +0.049 +0.043 +0.043 | +26.9 | Q5 | +12.4 | +12.9 | -22.6 | 1 |
| D_rsi14_obv_x | -0.0354 | -17.6 | 2.4e-68 | -0.030 -0.041 -0.032 -0.039 -0.035 | -27.2 | Q1 | +10.4 | +10.9 | -24.6 | 3 |
| D_WQ_33 | +0.0353 | +16.0 | 1.6e-56 | +0.018 +0.040 +0.039 +0.043 +0.037 | +26.3 | Q5 | +2.9 | +3.4 | -32.1 | 7 |
| D_WQ_12 | +0.0244 | +15.5 | 5.9e-53 | +0.025 +0.028 +0.028 +0.023 +0.018 | +16.2 | Q5 | +4.1 | +4.6 | -30.9 | 1 |
| D_WQ_35 | +0.0390 | +15.4 | 3e-52 | +0.024 +0.041 +0.047 +0.045 +0.039 | +33.0 | Q5 | +12.3 | +12.7 | -22.7 | 1 |
| D_range_pct | -0.0587 | -15.0 | 6.4e-50 | -0.065 -0.069 -0.055 -0.043 -0.062 | -23.9 | Q1 | +4.5 | +5.0 | -30.5 | 3 |
| Comb_GapUp__CPR_Tmr_Above | -0.0298 | -14.5 | 7.2e-47 | -0.017 -0.040 -0.031 -0.034 -0.028 | +nan | Q1 | +nan | +nan | +nan | 1 |
| D_daily_trend | -0.0298 | -14.0 | 7.8e-44 | -0.020 -0.035 -0.032 -0.035 -0.026 | +nan | Q1 | +nan | +nan | +nan | 4 |
| D_WQ_13 | +0.0342 | +14.0 | 1.2e-43 | +0.037 +0.038 +0.041 +0.032 +0.024 | +30.4 | Q5 | +12.2 | +12.8 | -22.8 | 1 |
| D_WQ_29 | +0.0507 | +13.9 | 2.4e-43 | +0.045 +0.060 +0.055 +0.053 +0.040 | +35.6 | Q5 | +12.4 | +12.9 | -22.6 | 1 |
| D_WQ_40 | +0.0438 | +13.6 | 2.7e-41 | +0.036 +0.050 +0.052 +0.041 +0.041 | +31.0 | Q5 | +13.8 | +14.3 | -21.2 | 1 |

### label_oc_5

| feature | IC | t | q | folds (1..5) | spread Q5-Q1 | quintile | quintile gross | quintile excess | quintile net after cost | family size |
|---|---|---|---|---|---|---|---|---|---|---|
| D_mdi14_diff1 | +0.0381 | +17.3 | 8.6e-66 | +0.033 +0.045 +0.041 +0.037 +0.034 | +43.0 | Q5 | +31.6 | +12.8 | -3.4 | 1 |
| D_WQ_44 | +0.0405 | +14.6 | 5e-47 | +0.035 +0.041 +0.046 +0.037 +0.044 | +30.2 | Q5 | +33.1 | +14.3 | -1.9 | 1 |
| D_rsi14_obv_x | -0.0292 | -13.5 | 6.4e-41 | -0.028 -0.030 -0.030 -0.031 -0.028 | -28.7 | Q1 | +29.7 | +10.9 | -5.3 | 3 |
| D_WQ_33 | +0.0301 | +13.3 | 1.4e-39 | +0.018 +0.033 +0.032 +0.036 +0.032 | +31.2 | Q5 | +23.0 | +4.2 | -12.0 | 7 |
| D_WQ_12 | +0.0240 | +13.2 | 3.5e-39 | +0.025 +0.029 +0.029 +0.021 +0.015 | +23.1 | Q5 | +23.2 | +4.4 | -11.8 | 1 |
| Comb_GapUp__CPR_Tmr_Above | -0.0266 | -12.9 | 2e-37 | -0.018 -0.033 -0.032 -0.026 -0.024 | +nan | Q1 | +nan | +nan | +nan | 1 |
| D_WQ_35 | +0.0364 | +12.1 | 7e-33 | +0.028 +0.034 +0.044 +0.039 +0.037 | +38.9 | Q5 | +33.4 | +14.5 | -1.6 | 1 |
| D_WQ_13 | +0.0331 | +11.7 | 7.9e-31 | +0.038 +0.038 +0.035 +0.031 +0.023 | +38.4 | Q5 | +33.9 | +15.1 | -1.1 | 1 |
| D_cpr_vs_yday_code | -0.0284 | -11.6 | 1.5e-30 | -0.018 -0.036 -0.033 -0.031 -0.024 | -34.9 | Q1 | +82.5 | +17.6 | +47.5 | 1 |
| D_daily_trend | -0.0256 | -11.5 | 3e-30 | -0.019 -0.029 -0.031 -0.029 -0.021 | +nan | Q1 | +nan | +nan | +nan | 4 |
| D_WQ_29 | +0.0490 | +11.4 | 2.5e-29 | +0.046 +0.055 +0.053 +0.050 +0.041 | +43.0 | Q5 | +35.2 | +16.4 | +0.2 | 1 |
| D_WQ_40 | +0.0431 | +11.1 | 3.7e-28 | +0.037 +0.050 +0.049 +0.038 +0.042 | +35.3 | Q5 | +35.2 | +16.4 | +0.2 | 1 |

## 4. Market-level columns (descriptive only: outside the family, no correction, slow series inflate t)

| feature | label_oc_1 corr (t) | label_oc_2 corr (t) | label_oc_3 corr (t) | label_oc_5 corr (t) |
|---|---|---|---|---|
| D_dow | -0.047 (-1.8) | -0.027 (-1.1) | -0.001 (-0.0) | -0.021 (-1.5) |
| DOW_0 | +0.042 (+1.6) | +0.054 (+2.3) | +0.044 (+2.2) | +0.018 (+1.3) |
| DOW_1 | +0.034 (+1.5) | +0.028 (+1.3) | -0.016 (-0.7) | +0.000 (+0.0) |
| DOW_2 | +0.023 (+0.7) | -0.047 (-1.7) | -0.038 (-1.8) | -0.004 (-0.3) |
| DOW_3 | -0.088 (-3.0) | -0.047 (-2.2) | -0.011 (-0.7) | +0.005 (+0.4) |
| DOW_4 | -0.040 (-1.2) | +0.002 (+0.1) | +0.003 (+0.1) | +0.004 (+0.2) |
| DOW_5 | +0.023 (+1.2) | +0.029 (+1.4) | +0.033 (+1.4) | +0.029 (+1.4) |
| DOW_6 | -0.007 (-0.3) | +0.009 (+0.3) | +0.017 (+0.8) | -0.022 (-1.6) |
| X_universe_size | -0.015 (-0.6) | -0.009 (-0.2) | -0.009 (-0.2) | -0.011 (-0.2) |
| MKT_D_realvol_20 | +0.021 (+0.9) | +0.045 (+0.9) | +0.059 (+0.9) | +0.084 (+1.0) |
| MKT_D_rsi14 | -0.007 (-0.3) | +0.031 (+0.6) | +0.052 (+0.8) | +0.076 (+0.9) |
| MKT_D_adx14 | +0.039 (+1.7) | +0.042 (+1.0) | +0.052 (+1.0) | +0.076 (+1.2) |
| MKT_D_dist_from_52wh | -0.023 (-0.9) | -0.010 (-0.2) | -0.006 (-0.1) | -0.001 (-0.0) |
| MKT_D_drawdown_252 | -0.023 (-0.8) | -0.021 (-0.4) | -0.026 (-0.4) | -0.042 (-0.5) |
| NIFTYBANK_ret_1d | +0.010 (+0.3) | +0.039 (+1.2) | +0.044 (+1.0) | +0.078 (+1.2) |
| NIFTY500_ret_1d | +0.009 (+0.2) | +0.053 (+1.5) | +0.065 (+1.3) | +0.102 (+1.3) |
| NIFTYMIDCAP150_ret_1d | +0.014 (+0.3) | +0.058 (+1.7) | +0.068 (+1.5) | +0.098 (+1.4) |
| NIFTYSMLCAP250_ret_1d | -0.006 (-0.1) | +0.058 (+1.8) | +0.074 (+1.5) | +0.106 (+1.5) |
| INDIAVIX_ret_1d | -0.014 (-0.3) | -0.045 (-1.4) | -0.060 (-1.3) | -0.099 (-1.6) |
| NIFTYAUTO_ret_1d | +0.009 (+0.2) | +0.049 (+1.6) | +0.047 (+1.2) | +0.073 (+1.4) |
| NIFTYFMCG_ret_1d | +0.044 (+0.9) | +0.057 (+2.0) | +0.092 (+2.1) | +0.130 (+1.9) |
| NIFTYIT_ret_1d | +0.008 (+0.2) | +0.050 (+1.3) | +0.047 (+1.0) | +0.080 (+1.1) |
| NIFTYMETAL_ret_1d | +0.037 (+1.0) | +0.070 (+2.0) | +0.057 (+1.5) | +0.074 (+1.2) |
| NIFTYPHARMA_ret_1d | +0.015 (+0.3) | +0.054 (+1.6) | +0.075 (+1.8) | +0.107 (+1.7) |
| NIFTYREALTY_ret_1d | +0.014 (+0.4) | +0.058 (+2.0) | +0.068 (+1.7) | +0.097 (+1.8) |
| NIFTYENERGY_ret_1d | -0.050 (-0.9) | +0.010 (+0.3) | +0.016 (+0.4) | +0.053 (+0.8) |
| NIFTYPSUBANK_ret_1d | -0.043 (-0.9) | -0.001 (-0.0) | -0.002 (-0.1) | +0.039 (+0.9) |
| NIFTYFINSERVICE_ret_1d | +0.020 (+0.6) | +0.044 (+1.3) | +0.050 (+1.2) | +0.083 (+1.3) |
| NIFTYMEDIA_ret_1d | +0.008 (+0.2) | +0.054 (+1.4) | +0.039 (+1.0) | +0.075 (+1.2) |
| NIFTYCONSUMPTION_ret_1d | +0.035 (+0.8) | +0.060 (+2.3) | +0.081 (+1.9) | +0.115 (+1.8) |
| NIFTYINFRA_ret_1d | -0.012 (-0.2) | +0.033 (+1.0) | +0.046 (+1.0) | +0.080 (+1.2) |
| USDINR_ret_1d | +0.011 (+0.3) | -0.039 (-1.0) | -0.016 (-0.5) | -0.005 (-0.2) |
| CRUDEOIL_ret_1d | +0.010 (+1.0) | -0.026 (-1.0) | -0.007 (-1.0) | +0.010 (+1.0) |
| GOLD_ret_1d | -0.025 (-0.8) | -0.007 (-0.1) | -0.018 (-0.5) | -0.000 (-0.0) |

## 5. What this does not say

- It does not say anything is tradable. Stage D's gate decides that.
- It is marginal evidence: each feature on its own. Whether a feature adds anything beyond the others is stage D's question.
- Families are descriptive. No feature is dropped for sharing a family.
- The pass list was chosen on these windows, so stage D must reselect inside each fold's training window, not reuse this list over the same dates.
- About 10% of passing cells are expected to be false discoveries (that is what q <= 0.10 means).
