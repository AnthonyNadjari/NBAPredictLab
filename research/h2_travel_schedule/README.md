# H2: travel and schedule fatigue (beyond simple rest days)

**Question.** Does schedule information (travel distance, time-zone changes, altitude, road trips,
home stands, schedule density, body clock at tip-off) predict NBA game winners *beyond the betting
price*?

**Answer: no.** None of the 35 model variants beats the closing line or the opening line under the
protocol. No betting rule has a 95% CI above zero after vig. Schedule fatigue clearly affects results:
teams with more games in the last 7 days lose more, by about 1 point of margin per SD. But the closing line already
prices this. Against the closing line, the travel features carry no more information than randomly
shuffled features.

Run end-to-end (about 7 minutes; no network calls, everything comes from local files):

```
python research/h2_travel_schedule/run.py
```

## Data

* Schedule: `research/data/team_logs.csv` (nba_api team game logs), seasons 2021-22 to 2025-26,
  including play-in and playoffs. The three NBA Cup finals (Dec 2023, 2024, 2025) are missing from the
  logs because they do not count in the standings. They are added back because the teams still played
  and travelled.
* Venues (`arenas.py`): hardcoded latitude/longitude, IANA time zone (so DST and Arizona's lack of DST
  are handled) and elevation for the 30 arenas, including the Clippers' Intuit Dome from 2024-25.
  19 neutral-site games are also hardcoded: Mexico City (2,240 m), Paris, Berlin, London, and the NBA
  Cup knockout games in Las Vegas. ESPN does not flag the 2023-24 Cup semifinals as neutral, but they
  were played in Las Vegas.
* Odds and tip-off times: `research/data/espn_<season>.csv`. The evaluation set is the same
  **5,197 games** (2022-23 to 2025-26) as `research/upsets.py` (`load()`), plus 1,323 games from
  **2021-22 used only for training**. ESPN has no explicit close for 2021-22, so the stored "current"
  DraftKings moneyline is used as a stand-in. In later seasons that field equals the close in 99.6% of
  games.

## Features (`h2feat.py`, per team, using only the schedule before the game)

| feature | definition |
|---|---|
| `travel_km` | great-circle km from the previous game's venue to this venue (first game of a season starts from the home arena) |
| `tz_shift` | UTC offset of this venue minus that of the previous venue (h; + = travelled east), split into east and west |
| `visiting_altitude` | game at a venue above 1,000 m (DEN, UTA, Mexico City) for a team based below 1,000 m |
| `after_altitude` | previous game (at most 2 days earlier) at altitude, this game low |
| `road_trip_len` / `home_stand_len` | consecutive games away from / in the team's own arena, including this one |
| `three_in_four`, `four_in_six` | this game is the 3rd in 4 nights / 4th in 6 nights |
| `b2b_travel` | second night of a back-to-back after more than 50 km of travel |
| `games_last7` | games in the 7 days before |
| `travel_7d` | km travelled over days d-6 to d |
| `days_since_home` | days since the last game in the team's own arena (capped at 30) |
| `return_from_trip` | home game right after a road trip: the trip's length |
| `body_clock` | scheduled tip-off hour in the team's home time zone |

Games are modelled as away minus home differences (positive = away team more tired). Results never
enter the features. The only inputs are dates and venues of earlier games plus this game's venue,
date and scheduled tip-off.

## Method (shared protocol)

* **Walk-forward by season.** Closing line: test 2022-23, 2023-24, 2024-25, 2025-26, each fitted on
  all earlier seasons, with 2021-22 as training only. Opening line: opening odds exist only from
  2023-24, so the tests are 2024-25 and 2025-26.
* **Two model specifications**, each using the market probability as an input:
  * **A (protocol):** `logit p = a + b*logit(mkt) + w.X`. The market is re-calibrated.
  * **B (offset):** `logit p = logit(mkt) + w.X`. Only the feature weights are fitted. This is
    the sharper test of "does X add information beyond the price", because spec A pays a price just
    for re-estimating `a` and `b` (`market_recal` alone = +0.00072).
* **Variants:** 15 single features in each spec. Combined models: all 34 per-team columns with an L2
  penalty or a smooth L1 penalty, with strength chosen by inner leave-one-season-out CV on the
  training seasons, plus (spec B only) a LightGBM model with `init_score = logit(mkt)` and rounds
  chosen by inner CV. That is 17 + 18 = **35 variants per baseline**, so the Bonferroni CI level is
  99.857%.
* **Uncertainty:** paired bootstrap over games (4,000 resamples) of the per-game log-loss difference.
  **Survival bar:** Bonferroni CI entirely below 0 AND negative in at least 3 of 4 seasons (2 of 2 for
  the opening line).
* **Betting:** flat 1-unit bets when model probability > the vig-inclusive implied probability
  (1/decimal odds) + threshold (0, 2, 4 pp). Bets are settled at the actual book's closing moneyline
  (average overround 4.5%). Opening-line models are also settled at opening odds. ROI CIs are
  bootstrapped over bets.

## Results

### Closing line: market log-loss 0.59156 on 5,197 games (0.58162 over 2023-24 to 2025-26, matching the established baseline)

Delta = model minus market log-loss (negative = better than market).

| spec | variant | delta | 95% CI | 22-23 | 23-24 | 24-25 | 25-26 | neg |
|---|---|---|---|---|---|---|---|---|
| B | home_return_from_trip | **-0.00037** | [-0.00119, +0.00043] | -0.00090 | -0.00046 | +0.00074 | -0.00089 | 3/4 |
| B | tz_abs_diff | -0.00024 | [-0.00053, +0.00006] | -0.00002 | -0.00021 | +0.00014 | -0.00088 | 3/4 |
| B | games_last7 | -0.00012 | [-0.00066, +0.00042] | -0.00007 | -0.00022 | -0.00021 | +0.00004 | 3/4 |
| B | tz_east_west | -0.00010 | [-0.00053, +0.00033] | -0.00000 | -0.00004 | +0.00058 | -0.00092 | 3/4 |
| B | home_stand_len | -0.00003 | [-0.00101, +0.00099] | -0.00071 | -0.00018 | +0.00060 | +0.00014 | 2/4 |
| B | three_in_four | +0.00000 | [-0.00026, +0.00025] | -0.00005 | -0.00020 | +0.00029 | -0.00003 | 3/4 |
| B | four_in_six | +0.00007 | [-0.00015, +0.00029] | +0.00026 | +0.00001 | +0.00005 | -0.00002 | 1/4 |
| B | body_clock_diff | +0.00010 | [-0.00020, +0.00039] | +0.00035 | +0.00004 | +0.00003 | -0.00001 | 1/4 |
| B | b2b_with_travel | +0.00011 | [-0.00037, +0.00059] | -0.00019 | -0.00017 | +0.00002 | +0.00075 | 2/4 |
| B | travel_7d | +0.00017 | [-0.00085, +0.00122] | +0.00184 | -0.00012 | -0.00003 | -0.00095 | 3/4 |
| B | road_trip_len | +0.00018 | [-0.00024, +0.00061] | +0.00062 | -0.00003 | +0.00018 | -0.00004 | 2/4 |
| B | travel_km_diff | +0.00020 | [-0.00057, +0.00099] | +0.00144 | -0.00061 | +0.00079 | -0.00079 | 2/4 |
| B | days_since_home | +0.00022 | [-0.00027, +0.00072] | +0.00101 | -0.00001 | +0.00004 | -0.00012 | 2/4 |
| B | combined_gbm | +0.00033 | [-0.00008, +0.00069] | +0.00135 | -0.00000 | +0.00000 | -0.00000 | 2/4 |
| B | after_altitude | +0.00052 | [-0.00039, +0.00144] | +0.00068 | +0.00011 | +0.00072 | +0.00058 | 0/4 |
| B | visiting_altitude | +0.00057 | [+0.00002, +0.00111] | +0.00232 | -0.00010 | +0.00018 | -0.00004 | 2/4 |
| B | combined_l2 | +0.00061 | [-0.00067, +0.00185] | +0.00350 | -0.00073 | +0.00022 | -0.00042 | 2/4 |
| B | combined_l1 | +0.00073 | [-0.00048, +0.00190] | +0.00331 | -0.00008 | +0.00009 | -0.00027 | 2/4 |
| A | market_recal (reference) | +0.00072 | [+0.00004, +0.00137] | +0.00265 | +0.00058 | -0.00019 | -0.00005 | 2/4 |
| A | 15 singles | +0.00035 to +0.00146 | all include 0 or are > 0 | | | | | 0-2/4 |
| A | combined_l2 / l1 | +0.00166 / +0.00162 | both > 0 | +0.0068 | | | | 1-2/4 |

The full table with Bonferroni CIs is in `results/walkforward_close.csv`.
**0/35 survive. 0/35 have even a plain 95% CI below 0.** The median variant is +0.00057. Every
spec-A variant is worse than the raw closing line, and `visiting_altitude` (spec A) is significantly
worse even at the Bonferroni level. The 2022-23 column is noisy because it is trained on a single
season with the stand-in closing line. Restricting to the 2023-24 to 2025-26 window does not change
the picture (best spec-B variant -0.00037, `travel_7d`).

### Opening line: market log-loss 0.59204 on 2,630 games (2024-25, 2025-26)

Best variant: B `combined_gbm` -0.00060 [-0.00155, +0.00031], with 1 of 2 seasons negative (it is
exactly the market in 2024-25, where 0 boosting rounds were chosen). Next: B `travel_7d` -0.00044
[-0.00106, +0.00016] (2/2), B `tz_abs_diff` -0.00040 [-0.00124, +0.00041] (1/2). Median variant
+0.00031. **0/35 survive**, and 0/35 have a 95% CI below 0 (`results/walkforward_open.csv`).

### Betting (vig-inclusive moneylines)

| model, odds | rules with at least 50 bets | ROI > 0 | 95% CI above 0 | median ROI | best (cherry-picked) |
|---|---|---|---|---|---|
| closing-line models, closing odds | 65 | 9 | **0** | -7.1% | B travel_7d, thr 4 pp: +13.5% [-23.9%, +53.7%], n=54 |
| opening-line models, opening odds | 41 | 7 | **0** | -4.6% | A tz_east_west, thr 2 pp: +14.7% [-4.4%, +33.5%], n=102 |
| opening-line models, closing odds | 108 | 0 | **0** | -9.5% | -3.9% |

In total, 214 rules have at least 50 bets, and none has a CI above zero. The few positive ROIs are
what selection over 214 noisy rules is expected to produce (`results/betting.csv`).

### Why: the market already prices fatigue (`results/descriptive.csv`, `results/line_move.csv`)

In-sample, pooled 6,520 games, per 1 SD of the feature (away minus home):

| feature | alone: win z | margin (pts) | beyond closing line: z |
|---|---|---|---|
| games_last7 | **4.08** | +0.96 | 1.31 |
| three_in_four | **3.50** | +0.92 | 0.56 |
| b2b_with_travel | **3.31** | +0.73 | 0.33 |
| four_in_six | **2.44** | +0.72 | 0.22 |
| travel_km_diff | 0.98 | +0.01 | 2.00 |
| tz_abs_diff | 0.99 | +0.11 | 2.11 |
| travel_7d | 1.56 | +0.15 | 2.41 |
| home_return_from_trip | -1.70 | -0.28 | -2.40 |

* Schedule density predicts results strongly on its own, and the closing price absorbs it
  (beyond-market z drops to 0.2-1.3).
* Pure travel and time-zone variables barely predict results on their own. They show |z| of about 2
  beyond the price in-sample, but the largest of 16 such z-scores reaching 2.4 is what chance alone
  produces (Bonferroni p ≈ 0.26), and none replicates out-of-sample.
* The line moves toward the more-rested team between open and close (`b2b_with_travel` t = 5.0,
  `three_in_four` t = 2.9, `games_last7` t = 2.8). This is plausibly driven by game-day rest and injury
  news, not by the schedule itself. Even so, the move is about 0.6 pp per SD, too small to give the
  opening-line models a measurable gain.

### Robustness

* **Permutation test of a deliberately leaky fit.** The 34-feature offset model is fitted in-sample
  on the four test seasons themselves. It "improves" log-loss by -0.00246 and shows in-sample ROI of
  +3% / +10% / +22% at thresholds 0 / 2 / 4 pp. The same fit on randomly permuted feature rows (40
  repetitions) gives -0.00310 on average, and **78% of permutations fit at least as well as the real
  features.** About 34/(2n) = 0.0033 of in-sample improvement is pure overfitting. The real travel
  features carry no more information beyond the close than noise does
  (`results/lookahead_*.csv`).
* **Regular season only** (4,841 games): same conclusion. Best is `tz_abs_diff` -0.00032
  [-0.00078, +0.00014], and the combined models are worse than the market
  (`results/regular_season_only.csv`).
* **Spread residual (ATS) walk-forward** (more statistical power than win/loss): no variant improves
  the mean squared cover error significantly (best combined ridge -0.08 pts² [-0.34, +0.20]). Pick
  ROIs at an assumed -110 all have CIs containing 0 or below it (`results/ats_walkforward.csv`).
* **Power.** Outcomes were simulated from the closing line plus a known effect, and the same test was
  rerun 40 times per setting (95% CI below 0 and at least 3 of 4 seasons):

  | true effect per SD (pp at a 50/50 game) | travel_km | tz_abs | home_return | three_in_four |
  |---|---|---|---|---|
  | 0.05 (1.2 pp) | 8% | 5% | 12% | 0% |
  | 0.10 (2.5 pp) | 15% | 12% | 25% | 28% |
  | 0.15 (3.7 pp) | 58% | 65% | 62% | 50% |

  Effects of 1-2 pp per SD cannot be ruled out. They would also be economically useless: at a 50/50
  game, even a 2-SD situation would move the true probability by 2.5-5 pp, against about 2.2 pp of
  vig per side, before estimation noise.

## Caveats

* 2021-22 uses the "current" DraftKings moneyline as a stand-in for the close. This affects
  training for the 2022-23 test only. The 2023-24 to 2025-26 window, which has no stand-in, is
  reported too.
* The real itinerary is unknown. Travel is measured from the previous venue to this venue, so a team
  that flies home between two road games is credited with the direct leg.
* Odds come from a single book per game (ESPN BET or DraftKings). Line shopping could add about 1-2%
  ROI, but that is a price edge, not an information edge.
* The opening-line test has only 2 seasons. The exact time when ESPN's "open" was captured is unknown.
* Multiple testing: 35 log-loss variants per baseline and 214 betting rules. Best variants are shown
  for transparency but are selection-biased.

## Files

* `arenas.py`: arena coordinates, time zones, elevation, neutral sites, Cup finals
* `h2feat.py`: per-team schedule features, cached to `research/data/h2_travel_schedule/team_schedule_features.csv`
* `h2data.py`: evaluation set (odds, outcome, features), cached to `research/data/h2_travel_schedule/eval_dataset.csv`
* `h2eval.py`: penalised offset/recalibrated logit, inner-CV, LightGBM with market offset, bootstrap, betting, ATS
* `supplementary.py`: regular-season-only run, look-ahead fit with permutation null, power simulation
* `run.py`: end-to-end driver. Tables go to `results/`; per-game predictions go to
  `research/data/h2_travel_schedule/preds_close.csv`
