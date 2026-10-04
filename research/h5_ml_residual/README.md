# H5: can a flexible ML model learn what the betting market misses?

**Question.** If you give LightGBM, penalised logistic regressions and a margin model every
leak-free pre-game feature the engine computes, **plus the market price as the starting point**, do
they find residual structure that beats the price out of sample?

**Answer: no edge against the closing line. Against the opening line there is a small lead, but it
does not pass the strict bar.**

* **Closing line: 0 of 18 variants pass.** The best is the ensemble, at **-0.00016** log-loss
  [95% CI -0.00076, +0.00043]. The median variant is **+0.00024**, which is worse than the market.
  The main LightGBM boosted from the close comes out at +0.00002 (better in 1 of 4 seasons).
  Learning on 4,945 to 8,827 earlier games, the models gain a little on their inner validation
  season and then give it all back on the next season.
* **Opening line: 0 of 14 variants pass the Bonferroni bar.** The best model, `F_move_ridge`,
  predicts the open-to-close line move from pre-game features and adds that to the open. It gets
  **-0.00241** [-0.00448, -0.00039], negative in 2 of 2 seasons. That is about 22% of the
  0.0111 the close itself gains over the open. Its Bonferroni CI [-0.00545, +0.00067] includes 0.
* **Betting: 0 of 130 rules** have a ROI whose 95% CI is above zero after vig. The best
  opening-odds rule (`F_move_ridge`, edge > 2 pp) made **+15.3%** over 305 bets
  [-0.9%, +32.7%]. Taking opening-line model picks at the closing odds loses about **-9%**
  (29 of 56 rules have a CI entirely below 0). The line moves against those picks: adverse selection.
* **The structure used to exist and has faded.** In 2018-19 to 2022-23, 18% of features correlated
  with the closing-line residual at |z| > 1.96. A within-season permutation null gives 3% (p = 0.04).
  The top feature, **opponents' 3P% over the last 10 games** (z = 4.4, permutation p = 0.002), looks
  like the market treating 3-point luck as skill. From 2023-24 the share drops to 6%, which is noise
  (p = 0.26), and the opponent-3P% signal is gone (z = 0.0). This is why every learner trained on
  older seasons helps a bit in 2022-23 and then hurts.
* **Detection power is limited** (synthetic check). We injected a known mispricing worth
  -0.0032 log-loss, which is larger than anything real found here. The learners recovered about a
  quarter of it, and in neither draw was its CI below 0. So "no edge" means no edge of roughly
  that size or larger. A smaller one could still be there unseen.

Run end-to-end (about 6 minutes, no network: all inputs are local caches):

```
python research/h5_ml_residual/run.py              # main study + power check + diagnostics -> results/
python research/h5_ml_residual/run.py --no-power   # skip the synthetic power check (~2.5 min)
python research/h5_ml_residual/h5data.py           # rebuild the cached dataset only
```

## Data (`h5data.py`, cached to `research/data/h5_ml_residual/dataset.csv`)

| seasons | games | price | role |
|---|---|---|---|
| 2018-19 .. 2020-21 | 3,622 | ESPN's stored last pre-game line: Caesars, else Westgate/Wynn/Unibet/consensus. Read from H3's cache (`research/data/h3_motivation_context/espn_20*.csv`), with H3's sanity rule (within 6 pp of the median book). Optional: if the files are missing, training starts in 2021-22. | training only |
| 2021-22 | 1,323 | DraftKings "current" moneyline (ESPN stores no explicit close) | training only |
| 2022-23 .. 2025-26 | **5,197** | exactly `research/upsets.py` `load()`: ESPN close (de-vigged), and the open from 2023-24 | **evaluation** |

* Outcomes and all features come from `src/engine/features.build(history.load())`. Every feature
  uses only games strictly before the row (shift(1), Elo state before the update). The game's own box
  score (`home_pts`, `home_fgm`, ...) is **dropped** in `h5data.py` so it can never reach a model.
* The vig-inclusive decimal odds (close and open) of the same book are kept for the betting view.
* Check: the closing-line log-loss over 2023-24..2025-26 is **0.58162** (mean of the season means),
  matching the protocol. Over all four test seasons, pooled per game, it is 0.59156. 2022-23 is a
  noisy season for every predictor (market 0.623, Elo 0.661).

## Features (`h5models.make_features`): 153 (close) / 148 (open)

* **Per team, home and away:** every engine column, plus the home-minus-away difference of each.
  * last 3/5/10 games: win%, net rating, point diff, ppg, FG%, pace
  * last 10: offensive/defensive rating, 3P%, opponents' 3P%, assists, rebounds, turnovers, opponent ppg, 3PA rate
  * season point diff and win%; EWM form; weighted recent form
  * games played; streak; form acceleration
  * rest days; back-to-back; games in the last 5 days
  * venue splits (last 20 home / road games); shrunk net and form
* **Game:** Elo diff and Elo win probability, `net_diff`, `form_diff`, `rest_diff`, `b2b_diff`, `g5_diff`.
* **Context:** month, playoffs, play-in, neutral site, days since the season opener, weekday.
* **Market, close mode:**
  * logit(close), vig, closing spread and total
  * spread-implied minus ML-implied logit
  * open-to-close move (NaN before 2023-24)
  * `elo_gap` = logit(Elo prob) - logit(market)
* **Market, open mode:** logit(open), opening vig, `elo_gap` against the open. The stored spread and
  total are closing numbers, so the opening models **do not see them**.

## Models (`h5models.py`, `run.py`)

Every candidate takes the market as an input or offset. The fitted model is the market plus a learned
correction. All tuning happens on an **inner time-based split inside the training seasons**: the last
training season, or the last 30% of games by date when only one season is available. The tuning
covers the LightGBM config and rounds (early stopping, patience 200) and the L2 strength. The test
season is never touched.

| id | model |
|---|---|
| **A_lgb** (a)/(b) | LightGBM binary, `init_score = logit(market)`, all features. The config is chosen among stumps, 4 leaves/depth 2 and 8 leaves/depth 3, all with lr 0.01-0.02, min_data_in_leaf 100-300, L2 10-20, feature_fraction 0.5 and bagging 0.7. The model is refit on all training seasons with the chosen number of rounds. |
| **C1_logit_inter** (c) | logit p = b0 + (1+a)·logit(mkt) + interactions with L2. The interactions are market × {rest diff (clipped ±3), back-to-back diff, Oct-Nov, Mar-Apr regular season, playoffs, favourite at home}, plus the main effects. The slope and intercept are free (recalibration spec). |
| **C2_logit_inter_plus** (c) | C1 + team-level gaps: Elo-vs-market, net/form diff, last-5/10 net diff, streak diff, venue split, 3P% diff, pace, win% diff, games played, spread-vs-ML, line move, vig |
| **C1o / C2o** | the same, but in the **offset spec**: *everything*, intercept and slope included, is shrunk toward the raw market (λ up to 100), so heavy shrinkage returns exactly the price. These were added after the first run showed that a free recalibration (R0) costs +0.0004. |
| **D1_margin_shift** (d) | LightGBM regression of the final margin with `init_score` = market margin (-closing spread; for the open, the opening ML mapped to points). The predicted adjustment is converted with Δlogit = 1.702·adj/σ (σ = training residual SD, about 13.2) and added to logit(market ML). |
| **D2_margin_link** (d) | pure margin model: P(win) = logistic(a + b·(market margin + adj)/σ), with a and b fitted on the inner validation rows |
| **E_ensemble** | average logit of A, C2 and D1 |
| **F_move_lgb / F_move_ridge** (open only) | predict the open-to-close move (logit) from pre-game features, then publish logit(open) + predicted move. The close is used only as a **training target** from earlier seasons. Added after the first run (whose results did not include it) and counted in the multiplicity. |
| **G1 / G5_screen** | honest data mining: in each fold, rank all features by \|corr\| with the market residual **on training rows only**, keep the top 1 or 5, and fit an offset logistic |
| R0_platt, R1_lgb_no_market | references: Platt recalibration of the price alone; LightGBM on features without any market input |
| `...@2021+` | robustness: the same models trained from 2021-22 only, the protocol's literal start |

Protocol (as specified):
* walk-forward by test season S, fitting on seasons < S
* paired bootstrap over games (10,000 resamples) of the per-game log-loss difference
* Bonferroni over all variants evaluated against that baseline (18 for the close, 14 for the open)
* pass = Bonferroni CI below 0 **and** negative in at least 3 of 4 seasons (2 of 2 for the open, which only exists from 2023-24)

## Results

### Against the closing line (5,197 games; market 0.59156 pooled; Bonferroni level 99.72%)

| model | delta | 95% CI | Bonferroni CI | 22-23 | 23-24 | 24-25 | 25-26 | pass |
|---|---|---|---|---|---|---|---|---|
| A_lgb | +0.00002 | [-0.00106, +0.00110] | [-0.00158, +0.00170] | -0.00214 | +0.00184 | +0.00010 | +0.00018 | no |
| C1_logit_inter | +0.00058 | [-0.00020, +0.00134] | [-0.00066, +0.00172] | -0.00018 | +0.00229 | -0.00024 | +0.00042 | no |
| C2_logit_inter_plus | +0.00026 | [-0.00051, +0.00102] | [-0.00096, +0.00140] | +0.00039 | +0.00068 | -0.00026 | +0.00025 | no |
| C1o (offset spec) | +0.00064 | [+0.00000, +0.00127] | [-0.00037, +0.00157] | -0.00000 | +0.00229 | -0.00000 | +0.00024 | no |
| C2o (offset spec) | -0.00002 | [-0.00059, +0.00055] | [-0.00091, +0.00087] | -0.00000 | -0.00000 | -0.00024 | +0.00017 | no |
| D1_margin_shift | -0.00006 | [-0.00065, +0.00052] | [-0.00098, +0.00082] | -0.00040 | -0.00054 | +0.00057 | +0.00012 | no |
| D2_margin_link | +0.00233 | [+0.00047, +0.00425] | [-0.00051, +0.00535] | +0.00077 | +0.00407 | +0.00124 | +0.00316 | no |
| G1_screen_top1 | +0.00020 | [-0.00079, +0.00121] | [-0.00135, +0.00170] | -0.00143 | +0.00083 | -0.00017 | +0.00150 | no |
| G5_screen_top5 | -0.00010 | [-0.00152, +0.00132] | [-0.00220, +0.00202] | -0.00167 | +0.00061 | -0.00004 | +0.00063 | no |
| **E_ensemble** (best) | **-0.00016** | [-0.00076, +0.00043] | [-0.00109, +0.00074] | -0.00099 | +0.00027 | -0.00006 | +0.00010 | no |
| R0_platt (ref) | +0.00040 | [-0.00004, +0.00083] | | +0.00010 | +0.00120 | +0.00006 | +0.00022 | no |
| R1_lgb_no_market (ref) | +0.02238 | [+0.01652, +0.02835] | | +0.01357 | +0.02799 | +0.02363 | +0.02393 | no |
| A / C1 / C2 / D1 / D2 / E @2021+ | +0.00004 / +0.00093 / +0.00078 / +0.00030 / +0.00247 / +0.00021 | | | | | | | no |

Best variant -0.00016, median variant +0.00024. In C1o/C2o a "-0.00000" season is one where the
inner split chose λ = 100, so the model *is* the market. LightGBM stopped early at 156, 269, 48 and
40 rounds (lr 0.01). With the 2021+ training window it stopped at 3-29 rounds, which is almost
exactly the market.

### Against the opening line (2,630 games, 2024-25 + 2025-26; open 0.59204, close on the same games 0.58095)

| model | delta vs open | 95% CI | Bonferroni CI (99.64%) | 24-25 | 25-26 | pass |
|---|---|---|---|---|---|---|
| **F_move_ridge** | **-0.00241** | **[-0.00448, -0.00039]** | [-0.00545, +0.00067] | -0.00337 | -0.00144 | **no (passes 95% only)** |
| F_move_lgb | -0.00093 | [-0.00207, +0.00020] | [-0.00265, +0.00077] | -0.00133 | -0.00053 | no |
| D1_margin_shift | -0.00066 | [-0.00137, +0.00005] | [-0.00172, +0.00045] | -0.00011 | -0.00121 | no |
| A_lgb (b) | -0.00045 | [-0.00160, +0.00071] | [-0.00217, +0.00122] | +0.00015 | -0.00105 | no |
| E_ensemble | -0.00045 | [-0.00149, +0.00055] | [-0.00194, +0.00107] | -0.00006 | -0.00084 | no |
| C2o (offset spec) | +0.00001 | [-0.00106, +0.00109] | | +0.00025 | -0.00023 | no |
| C1o / C1 / C2 / D2 / G1 / G5 | +0.00043 / +0.00104 / +0.00056 / +0.00051 / +0.00067 / +0.00044 | | | | | no |
| R0_platt / R1_lgb_no_market (ref) | +0.00061 / +0.01514 | | | | | no |

The **best achievable improvement over the opening line** is therefore about **-0.0024 log-loss**,
roughly 22% of the open-to-close gain (0.0111). That is not significant under the strict bar.
Every opening-line model is still 0.009-0.012 **worse than the close** (`open_models_vs_close.csv`):
publishing late is worth about five times more than any model.

**Where the opening-line gain comes from** (`open_diagnostics.py`, `results/open_diagnostics.txt`):

* **It is not a constant shift.** Adding the average training-season move to the open changes
  log-loss by only -0.00005. The game-specific part of the prediction carries the gain (-0.00231).
  Predicted and actual moves correlate at +0.18 and +0.22 in the two test seasons.
* **Main drivers:**
  * `elo_gap`: the line drifts toward where Elo disagrees with the open (23% of the LightGBM move model's gain).
  * The size of the favourite: big opening favourites drift toward the dog.
  * Back-to-back fatigue: the line moves against a team on the second night, as rest decisions come out.
* **Timing check.** The time ESPN's "open" was posted is unknown. If it was posted before the previous
  night's games ended, features that include last night's game would be "news" the open could not
  contain. On games where **neither team played the day before** (1,880 of 2,630), the gain is
  -0.00187 [-0.00420, +0.00050], still negative in both seasons but no longer significant. On
  back-to-back games it is -0.00375 [-0.00798, +0.00026].

### Betting view (flat 1u at vig-inclusive odds when the model probability exceeds 1/odds + thr, thr = 0/2/4/6 pp; rules with fewer than 20 bets are skipped)

| set | rules | median ROI | best rule with 200+ bets | rules with CI > 0 |
|---|---|---|---|---|
| closing-line models at closing odds | 43 | -5.0% | G5_screen_top5, thr 2 pp: 518 bets, +7.5% [-5.6%, +21.0%] | 0 |
| opening-line models at opening odds | 31 | +1.5% | F_move_ridge, thr 2 pp: 305 bets, +15.3% [-0.9%, +32.7%] (thr 0: 1,009 bets, +7.1% [-2.6%, +16.9%]) | 0 |
| opening-line models at **closing** odds | 56 | -9.2% | all negative; 29 of 56 have a CI entirely below 0 | 0 |

Whether the opening price can actually be taken when the bot publishes is unknown (see the timing
check). A follower who takes an opening-model pick later in the day, at a price that has moved, is
systematically on the wrong side of informed money.

### Feature importances (gain share, summed over folds; `results/importance_*.csv`)

* **A_lgb vs close.** Nothing dominates, a sign it is fitting small, unstable patterns:
  * `diff_last10_opp_fg3_pct` 7.7%
  * `home_net_shrunk` 3.8%
  * `home_team_home_point_diff` 2.6%
  * `away_last10_three_point_rate` 2.4%
  * `away_last3_fg_pct` 2.4%
* **D (margin) vs close:** `away_last10_win_pct` 6.0%, `diff_last10_opp_fg3_pct` 5.7%, `away_elo` 4.3%.
* **F_move_lgb (open move):** `elo_gap` 23%, `mkt_logit` 11%, then home-court point diff,
  back-to-back and net-rating diffs at about 2-3% each. This is the only model whose importances are
  concentrated and make sense.
* **G screens** picked `diff_last10_opp_fg3_pct` as the top-1 feature in all 4 closing-line folds.
  It helped in 2022-23 (-0.00143) and hurt afterwards (+0.00083, -0.00017, +0.00150).

### Residual structure that faded (`decay.py`, `results/decay.txt`, exploratory)

z = corr(feature, home_win - p_close)·√n for the 150 features. The null is built by shuffling
residuals within each season 500 times, which keeps the correlation between features.

| period | games | share of features with \|z\| > 1.96 | null median / 95% | p | max \|z\| | null 95% of max | p |
|---|---|---|---|---|---|---|---|
| 2018-19 .. 2022-23 | 6,200 | **18.0%** | 3.3% / 16.7% | 0.04 | **4.36** (opp 3P% diff) | 3.45 | **0.002** |
| 2023-24 .. 2025-26 | 3,942 | 6.0% | 3.3% / 17.4% | 0.26 | 2.67 | 3.31 | 0.38 |

Per-season correlation of the opponent-3P% difference with the residual:

| 18-19 | 19-20 | 20-21 | 21-22 | 22-23 | 23-24 | 24-25 | 25-26 |
|---|---|---|---|---|---|---|---|
| +0.042 | +0.082 | +0.054 | +0.035 | +0.066 | +0.013 | +0.025 | -0.035 |

The direction fits a known bias: teams whose opponents recently shot well from three, or whose own
recent shooting was hot, were priced as if that shooting were skill. The z-profiles of the two
periods correlate at -0.31 across features, so the old pattern has partly reversed. Two
explanations are possible and the data cannot separate them:
* the market learned;
* the older price sources were less sharp (2018-21 is Caesars' last stored line, 2021-22 DraftKings
  "current").

Either way, a model trained on the old structure would not have made money in 2023-26.

### Power check (synthetic, `results/power.csv`, 2 draws per size, indicative)

Outcomes were re-drawn from sigmoid(logit(close) + k·s(X)). Here s mixes back-to-back, rest and a
favourite-longshot term, scaled to SD 1, so the market misses a known amount.

| injected k (logit SD) | oracle gain | A_lgb | C2o | detected (95% CI < 0) |
|---|---|---|---|---|
| 0 | 0 | +0.00119 | +0.00012 | - |
| 0.1 | -0.00085 | +0.00015 | -0.00007 | 0 of 4 |
| 0.2 | -0.00324 | -0.00068 | -0.00084 | 0 of 4 |

The fitting itself works. On synthetic data with light shrinkage, the logistic recovers the injected
back-to-back coefficient (0.43 vs 0.47 true). The low recovery has two causes: inner-validation
shrinkage, and the fact that one season of ~1,300 games carries an SE of about 0.0025 on its own.
**An edge worth less than about 0.003 log-loss would usually go unseen at this sample size.**

## Leakage and caveats

* **Features.** These are the engine's leak-free pre-game features, and the box score is dropped
  explicitly. Audit: the features-only model is 0.022 worse than the market, and no feature
  correlates with the market residual above 0.035 overall. A leak would make the features-only model
  look *better* than the market.
* **Opening-line models.**
  * They never see closing-time inputs (spread, total, closing vig, line move).
  * The close appears only as a training target from earlier seasons (F), or in the
    spread-to-probability slope fitted on training rows (D-open).
  * The real timing risk is that the ESPN "open" may predate the previous night's results (see the
    timing check). That would *flatter* the opening-line result, not hide an edge.
* **Training prices before 2022-23** come from other books and snapshots. Their residuals do not
  carry over to the 2023-26 close. The 2021+ robustness run does not do better either.
* **The 2022-23 close** is labelled "ESPN BET" in the feed, before that book launched. It is the
  noisiest market season (0.623), and it is where every model "wins" most. Without 2022-23 the
  closing-line picture is even flatter.
* **Variants added after the first run:** C1o/C2o, F and G. The first run's results did not include
  them. All are counted in the Bonferroni denominators: 18 for the close (14 main + robustness), 14
  for the open.
* **Power:** with 2 test seasons for the open and 4 for the close, only edges of roughly 0.002-0.003
  log-loss or more are detectable.

## Files

* `h5data.py`: dataset build (cached)
* `h5models.py`: features, LightGBM with offset, penalised logistic (recalibration and offset specs), ridge move model
* `run.py`: walk-forward, bootstrap, betting, importances, power check; it also runs the two diagnostics below
* `open_diagnostics.py`: decomposition and timing check of the opening-line gain
* `decay.py`: residual structure 2018-23 vs 2023-26 with a permutation null
* `results/`:
  * `walkforward_close.csv`, `walkforward_open.csv`, `open_models_vs_close.csv`
  * `per_season.csv`, `betting.csv`, `folds.csv`
  * `importance_*.csv`, `logreg_coefs.csv`, `power.csv`
  * `predictions_close.csv`, `predictions_open.csv`
  * `open_diagnostics*.{txt,csv}`, `decay*.{txt,csv}`, `run_log.txt`
