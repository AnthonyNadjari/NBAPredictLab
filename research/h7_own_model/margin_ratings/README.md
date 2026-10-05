# H7 / margin_ratings: our own win probabilities from point-margin team ratings

**Question.** Can we publish our own pre-game home-win probabilities, with **no bookmaker odds as
input**, that are at least as good as the current stats fallback, and how far are they from the
market?

**Answer.** Yes, they beat the fallback, but they are still far from the closing line.
* **Best variant: `margin_full_avail`** (margin ratings + schedule/context terms + an absence
  proxy from player box scores). Pooled over 5,197 test games (2022-23 to 2025-26):
  log-loss **0.6088**, accuracy **66.5%**.
  * Current fallback: 0.6146 / 65.5%. Plain Elo: 0.6277 / 65.0%.
  * Closing line: 0.5916 / 68.6%.
* **Against the fallback:** -0.0058 log-loss, 95% CI [-0.0098, -0.0017]. Better in **4 of 4**
  test seasons, and also in the 2021-22 validation season (0.6313 vs 0.6348).
* **Against the closing line:** +0.0173 [+0.0116, +0.0230]. Clearly worse.
* **Against the opening line** (3,929 games, 2023-24 to 2025-26): +0.0050 [-0.0012, +0.0110].
  This is not significantly worse, but the point estimate is still worse.
  * Opening line: 0.5932 / 68.1%. Our model: 0.5981 / 67.4%.
  * In 2025-26 alone the model is close to the open: 0.5898 vs 0.5863, accuracy 68.9% vs 67.9%.
* **Team-only variant `margin_full`** (no player data): 0.6107 / 65.6%. Against the fallback
  -0.0040 [-0.0072, -0.0006].

Run: `python research/h7_own_model/margin_ratings/run.py`. It takes about 4.5 minutes and needs no
network: every input is a local cache.
Outputs:
* `preds.csv`: the best variant on the 5,197 evaluation games.
* `preds_team_only.csv`: `margin_full`.
* `results.json`: every metric, CI, chosen hyper-parameter and coefficient.
* `grid.csv`: training-season score of every hyper-parameter combination, per test season.

## Results (same 5,197 games for every predictor)

Evaluation set: the canonical game set of `research/upsets.py load()`. That is every 2022-23..2025-26
game (regular season, play-in, playoffs) that has an ESPN closing moneyline, de-vigged. 72
test-season games without a close are excluded for everyone.

Log-loss by test season, then pooled log-loss, Brier and accuracy:

| predictor | 2022-23 | 2023-24 | 2024-25 | 2025-26 | **pooled** | Brier | acc |
|---|---|---|---|---|---|---|---|
| market close (de-vigged) | 0.6228 | 0.5829 | 0.5856 | 0.5763 | **0.5916** | 0.2032 | 68.6% |
| **margin_full_avail (best)** | 0.6422 | 0.6072 | 0.5976 | 0.5898 | **0.6088** | 0.2108 | 66.5% |
| margin_full (team data only) | 0.6414 | 0.6085 | 0.6025 | 0.5916 | 0.6107 | 0.2117 | 65.6% |
| od_full (offense/defense split) | 0.6411 | 0.6090 | 0.6010 | 0.5923 | 0.6105 | 0.2116 | 66.0% |
| margin_luck_sched | 0.6387 | 0.6078 | 0.6041 | 0.5991 | 0.6121 | 0.2122 | 65.7% |
| margin_luck | 0.6425 | 0.6120 | 0.6058 | 0.5957 | 0.6137 | 0.2129 | 65.8% |
| margin_decay | 0.6445 | 0.6138 | 0.6076 | 0.6004 | 0.6163 | 0.2140 | 66.0% |
| srs_season (plain season-to-date SRS) | 0.6540 | 0.6199 | 0.6156 | 0.6035 | 0.6229 | 0.2168 | 65.3% |
| current fallback (`backtest.py` logit on FEATS) | 0.6464 | 0.6095 | 0.6063 | 0.5977 | 0.6146 | 0.2134 | 65.5% |
| plain Elo (`features.py`) | 0.6610 | 0.6223 | 0.6217 | 0.6072 | 0.6277 | 0.2187 | 65.0% |

Open-line subset (3,929 games with an ESPN opening moneyline, 2023-24..2025-26; 2022-23 has none).
Log-loss by season, then pooled log-loss, Brier and accuracy:

| predictor | 2023-24 | 2024-25 | 2025-26 | **pooled** | Brier | acc |
|---|---|---|---|---|---|---|
| market close | 0.5833 | 0.5856 | 0.5763 | **0.5817** | 0.1992 | 69.0% |
| market open | 0.5955 | 0.5978 | 0.5863 | **0.5932** | 0.2038 | 68.1% |
| **margin_full_avail** | 0.6072 | 0.5976 | 0.5898 | **0.5981** | 0.2061 | 67.4% |
| margin_full | 0.6088 | 0.6025 | 0.5916 | 0.6009 | 0.2075 | 66.4% |
| current fallback | 0.6096 | 0.6063 | 0.5977 | 0.6045 | 0.2089 | 66.3% |
| plain Elo | 0.6233 | 0.6217 | 0.6072 | 0.6174 | 0.2143 | 65.9% |

**Paired bootstrap** (10,000 resamples of games). The difference is the model's mean log-loss
minus the other predictor's; negative means our model is better.

| model | vs fallback | vs market close | vs market open (open subset) | vs Elo |
|---|---|---|---|---|
| margin_full_avail | **-0.0058 [-0.0098, -0.0017]** | +0.0173 [+0.0116, +0.0230] | +0.0050 [-0.0012, +0.0110] | -0.0189 [-0.0254, -0.0124] |
| margin_full | -0.0040 [-0.0072, -0.0006] | +0.0191 [+0.0134, +0.0249] | +0.0077 [+0.0016, +0.0136] | -0.0170 [-0.0233, -0.0108] |
| od_full | -0.0041 [-0.0074, -0.0007] | +0.0189 [+0.0133, +0.0248] | +0.0077 [+0.0015, +0.0136] | -0.0172 [-0.0234, -0.0109] |

* `margin_full_avail` vs the fallback, by season: -0.0041 / -0.0023 / -0.0087 / -0.0079 (4/4).
* Availability term alone (`margin_full_avail` vs `margin_full`): -0.0018 [-0.0042, +0.0005].
* Validation season 2021-22 (walk-forward like the tests; never looked at while designing):
  * best variant 0.6313, margin_full 0.6338, fallback 0.6348, Elo 0.6502.
  * Same ordering as on the test seasons.

## Method

**Data.**
* `research/data/games_features_odds.csv`: 10,218 games from 2018-19, with the leak-free pre-game
  features of `research/features.py`. Its odds columns are dropped on load.
* `team_logs.csv`: 3-point makes.
* The H1 cache `rotation_player_games.csv`: availability, used only by the extension.
* Static arena coordinates from `research/h2_travel_schedule/arenas.py`.
* Odds (`upsets_dataset.csv`: `mkt_close`, `mkt_open`) are read only after all predictions
  exist, to score the market.

**Rating engine (`rate`).** For every date D, a ridge least-squares (Massey/SRS) fit of
`margin = r_home - r_away + HCA x (not neutral)` on games of dates < D. The games of date D are
added only after they are predicted. A perturbation test confirmed this: scrambling every result
from a date on leaves all predictions up to that date unchanged.
* **Recency:** past games are weighted `0.5^(age_days / hl)` within the season.
* **Memory:** at the start of a season the data is reset. Ratings are shrunk toward a prior
  = `carry` x last season's final rating (centred), worth `k` fresh games. HCA's prior is last
  season's HCA, worth 60 games.
* **Blow-outs:** margins are capped at +-`cap` points.
* **3-point luck:** `alpha` x (3PM - 3PA x league 3P%) of each side is removed from past margins.
  The league 3P% is the running rate over earlier dates. Opponents' 3P% is mostly noise.
* **Possessions:** an option fits per-100 ratings and rescales them with a team pace model.
* **Neutral floors:** the 2019-20 Orlando bubble and the listed neutral-site games
  (Mexico City, Paris, Berlin, London, NBA Cup in Las Vegas) get no HCA.
* **Link:** P(home win) = Phi(margin_hat / sigma). sigma is fit by maximum likelihood on the
  training seasons.

**Walk-forward selection.** For each test season S, everything is chosen on seasons
[2019-20, S) only, with log-loss of pre-game predictions as the criterion. 2018-19 is the burn-in
season (no prior).
* The grid has 810 combinations:
  * half-life 30-130 days or infinite;
  * carry 0.55/0.7/0.85;
  * k 5/8/12;
  * cap none/25/18;
  * alpha 0/0.2/0.35;
  * per100 yes/no.
* Chosen settings, very stable across seasons: half-life **45 days**, carry **0.7** (0.55 for
  2022-23), **k = 8 games**, **cap 18**, **alpha 0.35**, **no possession adjustment**.
  * Possession adjustment never wins.
  * The O/D split with a separate luck weight for defence (`od_full`, alpha_off 0.15,
    alpha_def 0.5) adds nothing.

**Schedule/context terms.** These are fit by OLS on the training seasons' out-of-sample
residuals. They are removed from the targets of the ratings (each one is known before its game),
then added back to the prediction. Coefficients for 2025-26, in points:
* back-to-back: home -2.1, away +2.8;
* rest difference (capped 1-4 days): ~0;
* games in the last 5 days: -0.46 per game of difference;
* travel: -0.22 per 1,000 km of difference;
* low-altitude visitor in Denver/Utah/Mexico City: +1.1 for the home side;
* playoff/play-in home court: +2.7 on top of HCA (the fitted in-season HCA is 1.3-2.1 points in the test seasons, on capped margins);
* late regular season (schedule share played >= cut, cut chosen by leave-one-season-out on the
  training seasons, 0.6 from 2023-24):
  * team below .400: -5.1 at home, -4.1 on the road;
  * team above .650: +1.6 at home, about 0 on the road.
  * This is tanking and resting: the margin ratings cannot see it.

**Extension: availability, previous-game proxy.** This uses H1's leak-free rotation table:
players with >= 15 min over the team's last 10 games, each valued by prior Game Score above
replacement x expected minutes.
* `miss_prev` = value of rotation players with no box-score line in the team's previous game.
* The feature is home minus away.
* The coefficient is fit on the training seasons' residuals (2020-21+ only, the seasons with
  player logs): -0.15 to -0.21 points per unit.
* Using the true absences of past games to clean the rating targets was tried and did not help.

## What drives the gain

Pooled log-loss on the 5,197 games, adding one piece at a time:

| step | log-loss | change |
|---|---|---|
| plain season SRS | 0.6229 | |
| + recency half-life, prior from last season, margin cap | 0.6163 | -0.0066 (largest) |
| + 3-point luck removal | 0.6137 | -0.0026 |
| + schedule terms (b2b, density, travel, altitude, playoff HCA) | 0.6121 | -0.0016 (hurt in 2025-26 and 2021-22) |
| + late-season standings (tanking/resting) | 0.6107 | -0.0015 (hurt in 2022-23) |
| + previous-game absence proxy | 0.6088 | -0.0018 |

**Where the remaining gap to the close sits** (best variant minus close, log-loss):

| period | gap to close |
|---|---|
| Oct-Dec | +0.011 |
| Jan-Feb | +0.021 |
| Mar-Apr | +0.025 |
| play-in + playoffs | +0.004 |

The model is closest when the market knows least about absences and motivation (early season,
playoffs). It is furthest in the second half of the season, which is load management, injuries,
trades and tanking: who actually plays tonight.

**Calibration.** The decile bins of the best variant match observed home-win rates within ~4
points (`results.json` > `calibration_best`). The exception is the 20-game bottom bin.

## What production would need every day

The model needs no odds and can run any time after the previous night's games are final.
1. **Final score of every completed game** this season and last season: date, home, away,
   points. The ESPN scoreboard is already stored by `src/engine/history.py`.
2. **Team 3PM and 3PA of each completed game**, for the luck adjustment. The ESPN summary team box
   score is already parsed (`espn.boxscore`: `fg3m`, `fg3a`). Possessions (FGA/OREB/TOV/FTA) are
   **not** needed, because the per-100 version never won.
3. **Tonight's games**:
   * home, away, date;
   * season type (regular / play-in / playoffs);
   * neutral-site flag, plus the venue for neutral games.
   The ESPN scoreboard has all of these (`season.type`, `neutralSite`, venue).
4. **Derived from 1 and 3**, nothing new to fetch:
   * each team's rest days, back-to-back, games in the last 5 days;
   * previous venue, giving travel km with a static arena table (copy
     `h2_travel_schedule/arenas.py`) and the altitude flag;
   * games played and win% to date, for the late-season terms;
   * scheduled season length (82).
5. **The extension only: player box-score lines** of every game of each team this season, plus
   enough history for each player's last 82 appearances. That means minutes, PTS, FGM, FGA, FTM,
   FTA, OREB, DREB, STL, AST, BLK, PF, TOV, and who had no line (DNP / inactive).
   * The ESPN game summary contains these player lines.
   * Production only parses team totals today, so it needs a player-line parser.
   * The cached nba_api player logs (2020-21..2025-26) can seed the history.
6. **ESPN injury report:** **not needed** for the backtested model. The proxy is "sat out the
   team's last game".
   * The report is already read (`espn.game_context`: out/doubtful/questionable) and could replace
     the proxy with tonight's actual statuses. That is better information: H1's oracle bound is
     about 2.5x the proxy's gain.
   * But its coefficient cannot be fit until the reports are **logged daily** (no history exists).
     Start logging them now. Meanwhile use the proxy, or treat "Out" like "absent last game".

## Caveats (honest)

* **This does not beat the market.**
  * We are 0.017 log-loss and ~2 accuracy points behind the close (about 27 fewer correct picks
    per 1,300-game season), and the CI excludes zero.
  * Against the open, the CI includes zero only narrowly, and the point estimate is still worse.
  * The account would publish a probability that is honestly our own but less accurate than the
    price.
* **Design was not pre-registered.** Hyper-parameters, sigma and every coefficient are strictly
  walk-forward. But the menu of components was chosen after looking at pooled test-season
  results of prototypes: the luck adjustment, late-season terms, availability, schedule set and
  the 0.40/0.65 win% thresholds.
  * The 2021-22 validation season, never inspected during design, shows the same ordering. Still,
    expect the edge over the fallback to be somewhat smaller on new seasons.
* **Some components are unstable from season to season:**
  * Schedule terms hurt 2025-26 (+0.0034): the away back-to-back and playoff home-court effects
    were weaker than in the training seasons.
  * Late-season terms hurt 2022-23.
  * Tanking behaviour depends on league rules (lottery odds, 65-game award rule since 2023-24)
    and can drift.
* **Grid edges:** alpha = 0.35 and cap = 18 sit at the edge of the grid. Prototype runs with
  alpha 0.5 and cap 12 were slightly worse on the training seasons, so the optimum is close.
* **Availability:**
  * It is only a previous-game proxy.
  * The player value is crude (Game Score).
  * It depends on H1's cache (player logs from 2020-21). Training seasons before that have the
    feature at 0, so 2022-23 has only 2020-21/2021-22 to fit its coefficient, and those seasons
    have COVID-protocol absences.
* **Neutral sites before 2021-22:** only the Orlando bubble is flagged. A handful of
  Mexico City/London/Paris games in 2018-19 and 2019-20 are treated as home games (training data
  only).
* **Bootstrap method:** games are resampled independently, ignoring same-night correlation, so the
  CIs are slightly narrow.
* **Fallback version:** the fallback reproduced here is `research/backtest.py` (LogisticRegression
  C=1 on FEATS, trained on 2019-20..S-1). The live fallback in `src/` may differ in detail.
