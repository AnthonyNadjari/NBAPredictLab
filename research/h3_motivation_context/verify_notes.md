# H3 adversarial verification notes (2026-10-04)

Verdict after review: **no_edge confirmed** (claim not refuted; numbers reproduce bit for bit).

## 1. Reproduction
* `run.py` re-run with outputs redirected to a scratch folder (repo results untouched): finished in 131 s.
  `slices.csv`, `playoff_favourites.csv`, `rule_bets.csv`, `walkforward_close.csv`, `walkforward_open.csv`,
  `walkforward_close_train2021.csv`, `betting_models.csv`, `line_move.csv` and `preds_close.csv` are
  **byte-identical** to the committed ones.
* Caches rebuilt from scratch from `team_logs.csv` + ESPN files (`h3feat.standings/series_state/playin_elim`,
  `h3data.build(cache=False)`): `context_features.csv`, `series_state.csv`, `eval_dataset.csv` are
  **identical** to the cached files, so the cache is what the code produces.
* `power.py` re-run (seed 2026) to a scratch folder: same values as `results/power.csv`.
* Every number in the claim JSON was checked against these tables (closing best po_slope -0.00022
  [-0.00070,+0.00022], Bonferroni [-0.00095,+0.00050], per-season -0.00047/+0.00025/-0.00023/-0.00044;
  opening best race_diff -0.00037 [-0.00155,+0.00082], -0.00041/-0.00033; spec A recal +0.00040; spec A
  range +0.00014..+0.00055; train-2021 best game7 -0.00007, median +0.00065; playoff-favourite table;
  folk rules; 0/244 model rules with CI > 0, medians -3.0% / -4.8% / -10.3%; line movement). All match.
* The post-hoc margin-SD diagnostic (14.5 vs 13.3 pts, CI [+0.3,+2.0], 64.9% vs 66.1%) is **not produced by
  any committed script**; recomputed independently and it reproduces (14.50 vs 13.29; series-clustered CI
  [+0.38,+2.00]; 64.9% vs 66.1%).

## 2. Leakage (feature code read line by line + independent recounts)
* Standings: W/L recounted independently from `team_logs.csv` with `date < game_date` for 600 random
  games x 2 sides: 0 mismatches; 0 regular-season rows whose W+L includes the game itself.
* Series state recounted independently for all 669 playoff games: 0 mismatches; game_no equals the
  official game-ID game number in 100% of 672 playoff games.
* `last_week` identical for both sides; `rs_end`/planned games are schedule facts (2019-20 approximation is
  training-only). Play-in elimination uses standings on the play-in date (final regular season, known).
* Clinch/elimination/locked logic checked: `best` is optimistic and `worst` pessimistic, so flags can only
  fire late, never early.
* **Minor leak found (no effect):** 4 regular-season 2019-20 games (POR-DET, LAC-DEN, ATL-POR, PHX-GSW, late
  Feb / early Mar 2020) carry an ESPN placeholder tip time, so ESPN dates them one day after the real game.
  The "morning-of" standings for those rows therefore include the game's own result (W off by one).
  Training-only season, 4 of 10,146 rows; no impact on any result. Fix: join on the nba_api game date.
* Minor data quirks (no effect): 4 2022 Finals games use DK "current" lines with zero or negative vig
  (hold-out only); nba_api PLUS_MINUS differs from the ESPN margin in 53 games (used only for the
  point-differential tiebreak and `prev_margin`).

## 3. Protocol
* Walk-forward by season, `season < S` for training; inner leave-one-season-out CV for the penalty and for
  the LightGBM rounds uses training seasons only; feature scaling uses training statistics only.
* Market logit is an input (spec A: free a,b; spec B: offset). Paired per-game bootstrap of the log-loss
  difference; survival requires the Bonferroni CI < 0 and >= 3/4 seasons (2/2 for opening) negative.
* The log-loss bootstrap resamples games, not playoff series, so it is slightly anti-conservative for the
  playoff variants. Nothing passes even so, so the verdict is unaffected.

## 4. Multiple testing
* 33 variants per baseline x 3 runs (close, open, close with training from 2021-22), 38 slices, 8 folk
  rules, 244 model betting rules. All are reported. The best results are reported as selection-biased.
* The only nominal positives are the pooled playoff-favourite gap (cluster CI [-6.6,-0.4] pooled, but
  the never-examined 2018-22 hold-out gives -2.3 pts, z = -0.91) and one of 8 ATS folk rules
  (+11.9% [+0.3%,+24.2%]). The ATS rule fails Bonferroni and gives only +4.3% on 2022-26. Neither
  survives a correction.
* Hold-out independence checked: `research/upsets.py` (where the hint came from) only uses games that have
  `mkt_close`, i.e. 2022-23 onwards, so the 2018-19 to 2021-22 playoffs were not part of the hint search.

## 5. ROI
* Bets are settled at the vig-inclusive decimal odds of the same book row as the de-vigged probability.
  Folk-rule directions are fixed in advance; model bets are out-of-sample walk-forward predictions. The
  playoff CIs resample whole series. No ROI CI is above 0 after vig.

## Caveat on interpretation
Power is genuinely low. I checked the simulation analytically: an effect of -0.3 logit on 353 active test
games gives an expected z of about 1.3, so about 25% power. "no_edge" therefore means "no exploitable
edge can be demonstrated", which is what the README says.
