# H7 / player impact: our own win probabilities, with no odds as input

**Question.** Today the account republishes the de-vigged market price. Can we publish our own
pre-game home-win probability instead? It would be built from player ratings and the players
expected to play, with no bookmaker number as input. How far is it from the market?

**Answer (5,197 games, 2022-23 .. 2025-26, walk-forward).**

* **Best deployable variant, `pregame_plus_team`:** log-loss **0.6067**, accuracy **66.9%**.
  * Current fallback: 0.6146 / 65.5%. The difference is **-0.0079**, 95% CI [-0.0128, -0.0029],
    and the model is better in **4 of 4 seasons**.
  * Closing line: 0.5916 / 68.6%. The difference is **+0.0152**, CI [+0.0100, +0.0203]. The close
    stays clearly better.
  * Opening line, on the 3,929 games where it exists: 0.5964 vs 0.5932. The difference is +0.0032,
    CI [-0.0022, +0.0089], so it cannot be told apart from the open.
* **The pure player model without team features (`pregame`) is almost the same:** 0.6077,
  -0.0069 vs the fallback, CI [-0.0121, -0.0017].
* **Knowing exactly who plays (`oracle_roster`, i.e. a perfect injury report)** gives 0.5978. That
  beats the opening line (-0.0058, CI [-0.0117, -0.0001]) but not the close (+0.0063,
  CI [+0.0019, +0.0107]).
* **What drives it** is a player rating learned from game margins and minutes: a dynamic adjusted
  plus-minus, run as a Kalman filter. Box-score composites add nothing on top of it. About two
  thirds of the gain over the fallback comes in March-April, when stars sit and rotations change.
  Those changes are visible player by player but not in team form.

```
python research/h7_own_model/player_impact/run.py          # ~1.5-2 min -> preds.csv, results.json
python research/h7_own_model/player_impact/run.py --tune   # ~5 min, the 2021-22 grid used for 3 Kalman settings
python research/h7_own_model/player_impact/run.py --check  # leakage checks (rating state vs same-day games)
```

Files: `run.py` (whole pipeline), `results.json` (metrics per season, pooled, on the opening-line
subset and by season phase, bootstrap CIs, calibration slopes, last refit coefficients, P(play)
table, hyperparameters), and `preds.csv` (`GAME_ID,season,date,home,away,home_win,p`, where `p` is
the best variant's pre-game probability that the home team wins).

## Data and rules

* **Player logs.** nba_api box score lines for 2020-21..2025-26: regular season, play-in and
  playoffs. That is 165,987 player-games and 1,096 players: minutes, box score and plus-minus.
  Team logs supply possessions (FGA - OREB + TOV + 0.44 FTA, averaged over the two teams) and
  margins. 2020-21 is **burn-in only**, because there are no earlier player logs.
* **Schedule and team features.** `games_features_odds.csv`: rest, back-to-back, games in 5 days,
  Elo and net rating. They are built by `research/features.py` from earlier games only. No odds
  column is ever read as a feature.
* **Evaluation set.** The shared protocol set from `upsets_dataset.csv` (5,197 games), unchanged and
  identical for every predictor. The market probabilities in it are used only as baselines. Opening
  lines exist only from 2023-24 (3,929 games).
* **No look-ahead.** A rating used for a game on date *d* comes from games with date < *d*. Same-day
  games are excluded, and so is the game itself. `--check` verifies this two ways:
  * Changing the margins, points or plus-minus of every game on 2024-01-15 leaves all ratings for
    2024-01-15 unchanged (difference 0.0). They move only from the next day.
  * A filter run on data cut before that date gives the same ratings (difference 2e-8).
* **Walk-forward.** For each test season, the rating weights and the logistic are fitted only on
  games before the refit date, starting from 2021-22. They are refitted on the 1st of every month,
  so earlier games of the same season are used. The rating states update every day.
* **Baselines.**
  * `fallback`: exactly `research/backtest.py`, a logistic on FEATS trained on 2019-20..S-1, once per
    season.
  * `elo`: the `elo_prob` from `features.py`.
  * `team_logit_monthly`: the fallback's features with the same monthly refits and the same
    training window as the player models. It is a control: it shows that the refit schedule alone
    does not explain the gain (it gives +0.0005).
  * `market_close` and `market_open`: de-vigged ESPN lines.

## Method

**1. Player ratings, updated daily.**

* **Kalman APM: the part that matters.** This is a dynamic game-level adjusted plus-minus.
  * Model: home margin per 100 possessions = home court + Σ_home s_i·r_i − Σ_away s_j·r_j + noise.
    Here s_i = 5 × the player's share of team minutes in that game.
  * Each player rating r_i follows a random walk: variance +0.012 per day and +0.5 at each new
    season. The filter keeps the full covariance between players. Starters who always share the
    floor stay correlated, so the filter does not pretend to separate them.
  * A newcomer (rookie, or a player not seen before) enters 4 points per 100 below the current
    average established player, with variance 4.
  * Game noise variance is 150, i.e. a standard deviation of 12.2 per 100 possessions.
  * Three settings were picked by the filter's own one-step-ahead margin error on **2021-22 only**,
    before any test season: newcomer offset, prior variance and daily drift. Grid (`--tune`):
    -4/4/0.012 gives MSE 193.8, the a-priori -1/4/0.004 gives 197.3, and 0/2/0.004 gives 202.1.
    Every other setting was fixed a priori.
* **Rate stats (they turned out not to help).**
  * Box-score counts per 100 on-court possessions (PTS, FGA, FTA, 3PM, 3PA, OREB, DREB, AST, STL,
    BLK, TOV, PF), on-court +/- and on/off.
  * Each is an exponentially weighted sum over the player's appearances (half-life 60 appearances
    for box stats, 120 for +/-).
  * Each is shrunk toward replacement level: the fringe players of 2020-21, using 1,500 to 4,000
    pseudo-possessions.

**2. Expected lineup for each team-game.** A team composite is Σ (5 × minute share) × player
ratings, over the lineup.

| lineup | who plays | minutes | deployable? |
|---|---|---|---|
| `oracle` | players who actually played | actual minutes | no: uses the game itself, and garbage-time minutes depend on the score |
| `oracle_roster` | players who actually played | projected from earlier games | no: this is what a *perfect* injury report would give |
| `pregame` | every player seen for the team in its last 10 games, if his last game was for this team | mean of his last 5 appearances × P(plays) | **yes** |
| `pregame_hard` | seen in the last 5 games, out if absent from both of the last 2 | mean minutes in those games | yes |

P(plays) depends on how many team games in a row the player has just missed and on his projected
minutes (bucketed). It is fitted on 2020-21 and 2021-22 only. If he played the last game, P is 0.93
for a 24+ minute player and 0.58 for a player under 12 minutes. After 1 missed game it is 0.40-0.47,
after 2 it is about 0.30, after 3-4 it is about 0.22, and after 5 or more it is about 0.13. Minutes
are rescaled so that each team has 240.

**3-4. From ratings to a probability.**

* A ridge regression of the game margin per 100 possessions on the home-minus-away composites
  gives the player rating formula (the weights). Its prediction is the **rating gap**.
* A logistic on the gap and the rest, back-to-back and games-in-5-days differences then gives the
  probability. The intercept is home court.
* `*_plus_team` adds the fallback's Elo, net-rating and form differences to the logistic.
* `*_kalman_*` uses only the Kalman rating, without the box/+- rates.

At the last refit of `pregame` (June 2026), the logistic coefficient on the gap is 0.13 per point
per 100 possessions. A back-to-back costs -0.25 in log-odds, and the home-court intercept is +0.25.

## Results

Pooled over all 5,197 games. `market_open` is on its own 3,929 games.

| predictor | log-loss | Brier | accuracy |
|---|---|---|---|
| **pregame_plus_team** (best deployable) | **0.6067** | 0.2097 | 66.9% |
| pregame_kalman_plus_team | 0.6069 | 0.2099 | 66.8% |
| pregame (pure player model) | 0.6077 | 0.2101 | 66.8% |
| pregame_kalman_only | 0.6078 | 0.2103 | 67.1% |
| pregame_hard | 0.6108 | 0.2116 | 66.5% |
| pregame_box_pm_only (no Kalman) | 0.6171 | 0.2141 | 66.6% |
| oracle_roster | 0.5978 | 0.2058 | 67.6% |
| oracle_roster_plus_team | 0.5977 | 0.2057 | 67.9% |
| oracle | 0.5949 | 0.2045 | 68.1% |
| fallback (production today) | 0.6146 | 0.2134 | 65.5% |
| team_logit_monthly (control) | 0.6151 | 0.2136 | 65.6% |
| elo | 0.6277 | 0.2187 | 65.0% |
| market close | 0.5916 | 0.2033 | 68.6% |
| market open (3,929 games) | 0.5932 | 0.2038 | 68.1% |

On the 3,929 games with an opening line:

| predictor | log-loss | accuracy |
|---|---|---|
| pregame_plus_team | 0.5964 | 67.6% |
| pregame | 0.5976 | 67.5% |
| oracle_roster | 0.5874 | 68.2% |
| oracle | 0.5830 | 68.6% |
| fallback | 0.6045 | 66.3% |
| market open | 0.5932 | 68.1% |
| market close | 0.5817 | 69.0% |

Per season, log-loss and accuracy:

| | 2022-23 | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|---|
| pregame_plus_team | 0.6393 / 65.0% | 0.5990 / 67.1% | 0.5975 / 68.1% | 0.5926 / 67.5% |
| pregame | 0.6399 / 64.5% | 0.5990 / 67.0% | 0.5983 / 68.1% | 0.5952 / 67.3% |
| oracle_roster | 0.6311 / 65.5% | 0.5906 / 68.3% | 0.5895 / 68.0% | 0.5816 / 68.4% |
| oracle | 0.6321 / 66.5% | 0.5890 / 68.5% | 0.5833 / 68.1% | 0.5767 / 69.3% |
| fallback | 0.6464 / 63.1% | 0.6095 / 65.2% | 0.6063 / 65.8% | 0.5977 / 67.8% |
| elo | 0.6610 / 62.2% | 0.6223 / 64.4% | 0.6217 / 65.7% | 0.6072 / 67.5% |
| market close | 0.6228 / 67.2% | 0.5830 / 69.1% | 0.5856 / 69.4% | 0.5763 / 68.5% |
| market open | - | 0.5955 / 68.2% (1,299 g.) | 0.5978 / 68.1% | 0.5863 / 67.9% |

Paired bootstrap over games: 2,000 resamples, mean log-loss difference, model minus reference.
Negative means the model is better.

| variant | vs fallback | vs market close | vs market open (open subset) | seasons < fallback |
|---|---|---|---|---|
| pregame_plus_team | -0.0079 [-0.0128, -0.0029] | +0.0152 [+0.0100, +0.0203] | +0.0032 [-0.0022, +0.0089] | 4/4 |
| pregame_kalman_plus_team | -0.0078 [-0.0117, -0.0038] | +0.0153 [+0.0099, +0.0207] | +0.0030 [-0.0028, +0.0085] | 4/4 |
| pregame | -0.0069 [-0.0121, -0.0017] | +0.0162 [+0.0106, +0.0216] | +0.0044 [-0.0014, +0.0102] | 4/4 |
| pregame_kalman_only | -0.0068 [-0.0111, -0.0023] | +0.0162 [+0.0105, +0.0216] | +0.0040 [-0.0022, +0.0100] | 4/4 |
| pregame_hard | -0.0038 [-0.0089, +0.0014] | +0.0193 [+0.0138, +0.0246] | +0.0074 [+0.0013, +0.0135] | 3/4 |
| pregame_box_pm_only | +0.0025 [-0.0034, +0.0085] | +0.0256 [+0.0197, +0.0314] | +0.0167 [+0.0106, +0.0228] | 2/4 |
| oracle_roster | -0.0168 [-0.0230, -0.0106] | +0.0063 [+0.0019, +0.0107] | -0.0058 [-0.0117, -0.0001] | 4/4 |
| oracle | -0.0198 [-0.0263, -0.0134] | +0.0033 [-0.0015, +0.0078] | -0.0102 [-0.0165, -0.0037] | 4/4 |
| elo | +0.0131 [+0.0083, +0.0183] | +0.0361 [+0.0291, +0.0437] | +0.0242 [+0.0169, +0.0318] | 0/4 |
| market close | -0.0230 [-0.0285, -0.0177] | - | -0.0115 [-0.0154, -0.0073] | 4/4 |

Log-loss by season phase:

| phase (games) | pregame_plus_team | pregame | oracle_roster | fallback | close |
|---|---|---|---|---|---|
| Oct-Nov (1,167) | 0.6094 | 0.6115 | 0.6052 | 0.6161 | 0.5985 |
| Dec-Feb (2,365) | 0.6260 | 0.6258 | 0.6144 | 0.6273 | 0.6065 |
| Mar-Apr (1,309) | 0.5593 | 0.5607 | 0.5471 | 0.5799 | 0.5431 |
| play-in + playoffs (356) | 0.6445 | 0.6484 | 0.6499 | 0.6530 | 0.6479 |

Of the -0.0079 pooled gain over the fallback, March-April contributes -0.0052, Oct-Nov -0.0015,
Dec-Feb -0.0006 and the postseason -0.0006.

## What drives it

* **The Kalman APM is the engine.** Without it (`pregame_box_pm_only`) the player model is no
  better than the fallback (+0.0025). With only the Kalman rating it gains -0.0068. Adding the box
  stats on top gives 0.6077 vs 0.6078, i.e. nothing. It also makes the probabilities slightly
  overconfident: the calibration slope is 0.955 with the box stats vs 1.005 without.
* **Team features add very little.** Elo, net rating and form improve the result by 0.0010 (0.6077
  to 0.6067), within noise.
* **Knowing who plays is worth about 0.010 log-loss.** The pre-game lineup (0.6077) vs the perfect
  roster (0.5978) differs by 0.0099. For scale, the market's own open-to-close improvement on the
  same games is 0.0115. The soft P(plays) beats the hard "out if absent twice" rule by 0.003.
* **Accuracy.** The player model picks the same favourite as the closing line in 87.8% of games;
  the fallback does so in 85.5%. Mean |p - close| is 0.067 vs 0.075.
* **Sanity check: top-rated players at the end of 2025-26** (whole formula, Kalman part in
  brackets): Gilgeous-Alexander 8.5 (4.4), Jokic 8.3 (3.2), Kawhi Leonard 7.8 (5.4),
  Wembanyama 7.2 (2.7), Derrick White 6.4, Kevin Huerter 6.3, Paul Reed 6.2, Curry 6.1, Booker 6.1,
  Embiid 6.1... The stars are there. Huerter, Reed and Kornet are typical adjusted plus-minus noise
  without play-by-play.

## How this was developed (forking paths, stated plainly)

1. **First version: rates only (box + +/- + on/off).** It was scored on the test seasons and was
   *worse* than the fallback (pregame 0.6172).
2. **I then added the Kalman APM, a design decision taken after seeing step 1's test results.** It
   started with a-priori settings and no newcomer offset: pregame 0.6085, pregame_plus_team 0.6076.
3. **Newcomer offset, prior variance and daily drift were set on 2021-22 only**, using the `--tune`
   grid: pregame 0.6077, pregame_plus_team 0.6067.
4. **`pregame_kalman_plus_team` was added after seeing that the box stats add nothing.** The "best
   variant" is the lowest pooled log-loss among the deployable variants, so it was picked on the
   test seasons. The top four deployable variants are within 0.0011 of each other, so the choice
   hardly matters.

The conclusion is the same across every Kalman version:

* -0.006 to -0.008 vs the fallback;
* +0.015 to +0.017 vs the close;
* +0.003 to +0.006 vs the open, with a CI that includes 0.

## Caveats

* **`oracle` reads the game's own minutes**, and minutes depend on the score (blowout benches).
  `oracle_roster` (who plays, with projected minutes) is the clean upper bound for an injury report.
* **Season openers are the weak spot.** At a team's first game the pre-game lineup comes from last
  season's last 10 games. Those include stars rested or injured at the end of the season, and they
  miss offseason trades. Example: 2022-23 UTA-DEN, model 0.54 vs close 0.30 after the Gobert and
  Mitchell trades. On the 61 such games log-loss is 0.675 vs 0.606 for the close (0.663 for the
  fallback). Production should take the season-start lineup from the ESPN roster.
* **Short history.** Player logs start in 2020-21, so 2022-23 ratings rest on at most 2 seasons of
  data. That is the season furthest from the market: 0.639 vs 0.623.
* **Game-level, not stint-level.** Without play-by-play the filter cannot separate players who
  always share the floor. Ratings of role players on good teams are noisy.
* **Bootstrap.** It resamples games independently and ignores within-season dependence, so the CIs
  are a little optimistic. The 4/4 seasons result is the more robust statement.
* **Calibration.** The best variant has a calibration slope of 0.94, slightly overconfident, against
  1.04 for the close. `pregame_kalman_plus_team` is at 0.995 for the same log-loss.
* **No betting claim.** Nothing here beats the closing line, consistent with H1-H6.

## What production would need, daily

The player IDs here are nba_api IDs. Production reads ESPN, so the rating state must be rebuilt
from ESPN box scores: one ESPN `summary` call per past game since 2020-21, about 7,800 calls once.
Another option is mapping the IDs by name. After that, the update is incremental: one Kalman step
per game day, under a second.

1. **After each game day (ratings).** From the ESPN `summary?event=<id>` endpoint that
   `src/engine/espn.py` already calls:
   * the final score;
   * the team totals FGA, OREB, TO and FTA, for possessions (`espn.boxscore()` already parses
     these);
   * for every player, `boxscore.players[].statistics[0].athletes[]`: the ESPN athlete id, team and
     **MIN**.
   * MIN is the only per-player field the Kalman-only variants need. The full variant also uses
     PTS, FG, 3PT, FT, OREB, DREB, AST, STL, BLK, TO, PF and +/-.
   * Checked on 2026-10-05 with event 401584689: the labels are `MIN, PTS, FG, 3PT, FT, REB, AST,
     TO, STL, BLK, OREB, DREB, PF, +/-`, plus `starter`, `didNotPlay` and `reason`. **ESPN
     provides everything the ratings need.**
2. **Before tip-off (lineup and probability):**
   * the schedule, for rest, back-to-backs and games in 5 days (already in production);
   * each team's minutes per player over its last 10 games (from the stored box scores);
   * Elo, net rating and form (already computed by `src/engine/features.py`), for `_plus_team`;
   * at season start or after a trade, the ESPN team roster.
3. **ESPN injury report (optional, the upgrade from `pregame` toward `oracle_roster`).**
   * Endpoint: `site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries`. The `summary` also
     has an `injuries` block.
   * Use: set P(plays) = 0 for **Out** and an empirical P for **Day-To-Day**. Keep the
     history-based P for players who are not listed.
   * The feed shows only `Out` and `Day-To-Day` (checked 2026-10-05). There is no Doubtful or
     Questionable split, and it misses late scratches and coach's decisions.
   * So it sits **between** `pregame` (0.6077, or 0.6067 with team features) and `oracle_roster`
     (0.5978): a real gain, but only part of the 0.010 ceiling. A gain of half the ceiling would put
     the model at about 0.602, close to the opening line on the same games.
   * The gain cannot be measured from this data, because there is no history of the report. Next
     step: log the ESPN injury report every day at publishing time (the 21:00 UTC refresh). That
     calibrates P(plays | status) and measures the real gain after a few weeks.
