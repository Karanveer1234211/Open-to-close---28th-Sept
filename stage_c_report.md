# Stage C screen - STAGEC_20260929_001

stage_c_screen v1 | C declaration e945f62dd321625c | 123 features

**Rule:** BH q <= 0.10 across all C cells (their own family) AND sign in >= 4 of 5 folds. Research rows only; eligible rows only. Marginal evidence only - whether C helps the MODEL is stage D_C's question.

## Summary

| target | tested | q <= 0.10 | stable 4/5 | PASS | families |
|---|---|---|---|---|---|
| label_oc_1 | 113 | 96 | 102 | **95** | 66 |
| label_oc_2 | 113 | 91 | 91 | **90** | 62 |
| label_oc_3 | 113 | 90 | 94 | **90** | 61 |
| label_oc_5 | 113 | 89 | 94 | **89** | 60 |

## Passes by group

| group | label_oc_1 | label_oc_2 | label_oc_3 | label_oc_5 |
|---|---|---|---|---|
| A_overnight_intraday | 13 | 11 | 11 | 11 |
| B_market_model | 14 | 14 | 13 | 13 |
| C_lottery | 5 | 5 | 5 | 5 |
| D_liquidity | 11 | 10 | 9 | 8 |
| E_price_bands | 10 | 10 | 12 | 12 |
| F_relative_position | 17 | 18 | 18 | 17 |
| G_momentum_seasonal | 3 | 2 | 2 | 2 |
| H_interactions | 14 | 12 | 12 | 13 |
| I_market_state | 0 | 0 | 0 | 0 |
| J_date_ranks | 8 | 8 | 8 | 8 |

### label_oc_1 - strongest passing C features

| feature | group | IC | t | folds | quintile | quintile excess | net after cost |
|---|---|---|---|---|---|---|---|
| C_on_sum_5 | A_overnight_intraday | -0.0542 | -27.6 | 5/5 | Q1 | +8.1 | -46.0 |
| C_on_excess_20 | A_overnight_intraday | -0.0601 | -26.2 | 5/5 | Q1 | +8.9 | -45.2 |
| C_on_sum_20 | A_overnight_intraday | -0.0601 | -26.2 | 5/5 | Q1 | +8.9 | -45.2 |
| PR_C_on_sum_20 | J_date_ranks | -0.0601 | -26.2 | 5/5 | Q1 | +8.9 | -45.2 |
| C_on_sum_60 | A_overnight_intraday | -0.0664 | -25.0 | 5/5 | Q1 | +10.3 | -43.8 |
| C_max1_resid_21 | B_market_model | -0.0564 | -22.1 | 5/5 | Q1 | +5.8 | -48.3 |
| C_idio_vol_60 | B_market_model | -0.0671 | -21.8 | 5/5 | Q1 | +5.8 | -48.2 |
| PR_C_idio_vol_60 | J_date_ranks | -0.0671 | -21.8 | 5/5 | Q1 | +5.8 | -48.2 |
| PR_C_max1_21 | J_date_ranks | -0.0555 | -20.8 | 5/5 | Q1 | +5.2 | -48.9 |
| C_max1_21 | C_lottery | -0.0555 | -20.8 | 5/5 | Q1 | +5.2 | -48.9 |
| I_mdi_vol | H_interactions | +0.0561 | +20.5 | 5/5 | Q5 | +6.4 | -47.7 |
| C_on_z_60 | F_relative_position | -0.0313 | -20.3 | 5/5 | Q1 | +6.7 | -47.4 |

### label_oc_2 - strongest passing C features

| feature | group | IC | t | folds | quintile | quintile excess | net after cost |
|---|---|---|---|---|---|---|---|
| I_mdi_vol | H_interactions | +0.0605 | +19.9 | 5/5 | Q5 | +10.2 | -34.6 |
| C_resid_1d | B_market_model | -0.0442 | -19.1 | 5/5 | Q1 | +7.5 | -37.2 |
| C_ret_to_band | E_price_bands | -0.0458 | -18.6 | 5/5 | Q1 | +7.6 | -37.2 |
| C_on_sum_5 | A_overnight_intraday | -0.0424 | -18.0 | 5/5 | Q1 | +5.0 | -39.7 |
| I_cgw_1 | H_interactions | -0.0325 | -17.8 | 5/5 | Q1 | +2.6 | -42.2 |
| C_ret_z_60 | F_relative_position | -0.0438 | -17.6 | 5/5 | Q1 | +6.4 | -38.4 |
| C_streak | F_relative_position | -0.0380 | -16.6 | 5/5 | Q1 | +12.2 | -32.6 |
| C_on_excess_20 | A_overnight_intraday | -0.0461 | -16.4 | 5/5 | Q1 | +6.2 | -38.6 |
| PR_C_on_sum_20 | J_date_ranks | -0.0461 | -16.4 | 5/5 | Q1 | +6.2 | -38.6 |
| C_on_sum_20 | A_overnight_intraday | -0.0461 | -16.4 | 5/5 | Q1 | +6.2 | -38.6 |
| C_max1_resid_21 | B_market_model | -0.0499 | -16.3 | 5/5 | Q1 | +5.8 | -39.0 |
| I_id_turn | H_interactions | -0.0258 | -15.6 | 5/5 | Q1 | -0.1 | -44.8 |

### label_oc_3 - strongest passing C features

| feature | group | IC | t | folds | quintile | quintile excess | net after cost |
|---|---|---|---|---|---|---|---|
| I_mdi_vol | H_interactions | +0.0548 | +18.7 | 5/5 | Q5 | +10.4 | -25.1 |
| C_ret_to_band | E_price_bands | -0.0433 | -18.1 | 5/5 | Q1 | +8.3 | -27.5 |
| I_cgw_1 | H_interactions | -0.0316 | -17.8 | 5/5 | Q1 | +2.8 | -32.7 |
| C_resid_1d | B_market_model | -0.0421 | -17.7 | 5/5 | Q1 | +8.5 | -27.0 |
| C_ret_z_60 | F_relative_position | -0.0414 | -17.1 | 5/5 | Q1 | +6.5 | -29.0 |
| C_streak | F_relative_position | -0.0363 | -15.0 | 5/5 | Q1 | +12.5 | -23.0 |
| C_id_z_60 | F_relative_position | -0.0319 | -14.8 | 5/5 | Q1 | +3.2 | -32.3 |
| I_id_turn | H_interactions | -0.0246 | -14.7 | 5/5 | Q1 | -0.0 | -35.5 |
| I_rev_disp | H_interactions | -0.0415 | -14.0 | 5/5 | Q1 | +6.0 | -29.5 |
| C_on_sum_5 | A_overnight_intraday | -0.0376 | -13.6 | 5/5 | Q1 | +3.0 | -32.5 |
| I_rev_spread | H_interactions | -0.0437 | -13.1 | 5/5 | Q1 | +15.6 | -19.9 |
| C_max1_resid_21 | B_market_model | -0.0467 | -12.9 | 5/5 | Q1 | +5.0 | -30.5 |

### label_oc_5 - strongest passing C features

| feature | group | IC | t | folds | quintile | quintile excess | net after cost |
|---|---|---|---|---|---|---|---|
| I_cgw_1 | H_interactions | -0.0302 | -16.0 | 5/5 | Q1 | +5.6 | -10.6 |
| I_mdi_vol | H_interactions | +0.0465 | +15.5 | 5/5 | Q5 | +9.5 | -6.7 |
| C_resid_1d | B_market_model | -0.0364 | -15.2 | 5/5 | Q1 | +10.0 | -6.2 |
| C_ret_to_band | E_price_bands | -0.0368 | -14.8 | 5/5 | Q1 | +8.7 | -7.7 |
| C_ret_z_60 | F_relative_position | -0.0355 | -13.9 | 5/5 | Q1 | +6.4 | -9.8 |
| I_id_turn | H_interactions | -0.0230 | -12.7 | 5/5 | Q1 | +3.1 | -13.1 |
| C_id_z_60 | F_relative_position | -0.0277 | -12.1 | 5/5 | Q1 | +4.3 | -11.9 |
| C_streak | F_relative_position | -0.0304 | -11.4 | 5/5 | Q1 | +10.4 | -5.8 |
| I_rev_spread | H_interactions | -0.0422 | -11.4 | 5/5 | Q1 | +21.9 | +5.7 |
| I_rev_disp | H_interactions | -0.0345 | -11.4 | 5/5 | Q1 | +5.6 | -10.6 |
| C_resid_5d_z | B_market_model | -0.0428 | -10.7 | 5/5 | Q1 | +16.4 | +0.2 |
| PR_C_resid_5d_z | J_date_ranks | -0.0428 | -10.7 | 5/5 | Q1 | +16.4 | +0.2 |

## Leak screens

Suspects: none.

Largest |IC| vs the targets: 0.0671; vs label_gap1: 0.1348; largest |rho| with a future day: 0.0886.
