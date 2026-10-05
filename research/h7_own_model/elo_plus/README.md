# H7 / elo_plus: our own win probabilities from a better Elo (no odds as input)

**Question.** Can we publish our own pre-game home-win probability, computed without any
bookmaker price, that is clearly better than the current stats fallback, and how far is it from
the market?

**Answer.** Yes for the fallback, no for the market. The best variant (`elo_plus`) has a pooled
log-loss of **0.6073** and an accuracy of **67.1%** on 5,197 test games (2022-23..2025-26). The
production fallback gets 0.6146 / 65.5%, so the gain is **-7.3e-3** (95% CI [-10.8, -3.8]),
and the model is better in 4 of 4 seasons. The closing line is still well ahead: 0.5916 /
68.6%, a gap of +15.8e-3 [+10.8, +20.9]. On the 3,929 games that have an opening line, the
model scores 0.5972 / 67.9% and the open scores 0.5932 / 68.1%. That gap is +4.1e-3
[-0.9, +9.3], so we cannot tell the model and the opening line apart. Most of the gain comes
from two things: **tuning the Elo** and the **player-availability adjustment**.

```
python research/h7_own_model/elo_plus/run.py            # ~5 min, uses cache/availability_players.csv
python research/h7_own_model/elo_plus/run.py --rebuild  # rebuild that cache from the player logs (+30 s)
```

Outputs: `preds.csv` (best variant, all 5,269 test-season games), `results.json` (metrics per
season and pooled for every variant and baseline, CIs, ladder, segments, tuned parameters
per season), and `run_log.txt` (console output).

## Data and evaluation set

* `research/data/games_features_odds.csv`: 10,218 games from 2018-19 to 2025-26, with results,
  margins and the leak-free rest columns (`rest`, `b2b`, `g5`) built by `research/features.py`.
* Player box scores `research/data/h1_player_availability/player_logs_*.csv`, 2020-21 to
  2025-26 (165,987 player-games).
* Market data, used only for comparison: ESPN main-book moneylines (`espn_*.csv`), de-vigged
  proportionally exactly as in `research/upsets.py`.
* **Evaluation set:** the 5,197 test-season games with a closing line, across all game types.
  This is the same set as the shared `upsets_dataset.csv`, and every predictor is scored on
  it. Opening lines exist only from 2023-24 onward, on 3,929 of these games, so every
  comparison with the open uses that subset.
* Baselines:
  * `market_close` and `market_open`.
  * `fallback_logit`: the production fallback. It is the `research/backtest.py` logit on
    `FEATS`, with C=1, trained on 2019-20..S-1 and re-implemented line for line.
  * `elo_prod`: the plain Elo already in `features.py` (K=20, home advantage 70, carry 0.75,
    MOV multiplier).
  * `elo_538`: the untuned FiveThirtyEight NBA Elo (K=20, home advantage 100, carry 0.75).

## Method

**Elo core.** Before the game, `p = 1 / (1 + 10^(-d/400))`, where
`d = R_home - R_away + HCA + rest terms + availability terms + talent term`.

After the game, both ratings move by `K * (|MOV|+3)^alpha / (7.5 + 0.006 * d_winner) * (y - p_u)`.
This is FiveThirtyEight's margin-of-victory multiplier, including its autocorrelation
correction.

The core has these settings:
* At its first game of a new season, a team's rating is regressed with
  `R = 1500 + carry * (R - 1500)`.
* Early in the season K is boosted by `1 + k_early * exp(-games played / 10)`.
* There is a separate extra home advantage for play-in and playoff games.
* Home advantage is 0 for the 2019-20 Orlando bubble games.

**Rest.** The model adds Elo-point terms for a back-to-back, for 3 or more days of rest, and
for 3 or more games in the previous 5 days.

**Availability.** This is built from the player box scores in one chronological pass. All
features for date D come from the state after date D-1, and only then is D ingested.
* A **regular** of team T is a player who has appeared for T this season, whose most recent
  appearance anywhere was for T, and whose average over his last 10 appearances for T is at
  least 12 minutes.
* Each regular gets three values, all computed from games before the date:
  * expected minutes;
  * GameScore per minute above replacement × minutes, with the per-minute rate taken from his
    last 82 appearances, shrunk with 300 min, replacement 0.287/min from 2020-21 fringe
    players;
  * on/off +/- per 48 × minutes / 48, shrunk with 2,000 min.
* The **expected missing value** is the sum over regulars of value × P(out today | number of
  consecutive team games he has just missed). P is estimated from earlier seasons only. It is
  about 0.09 when he played the last game, 0.52 after 1 game missed, 0.69 after 2, 0.76 after
  3-4, 0.87 after 5-9, 0.94 after 10-19 and 0.97 after 20 or more.
* This expected missing value moves the team's rating before the game. This is the
  "missed the previous game(s)" proxy, and it needs no injury report.

**Post-game update (`avail_post`, `elo_plus`).** After a game, the expectation `p_u` used in
the rating update is computed with who *actually* played in that finished game. The base
rating therefore stays a full-strength rating and is not dragged down by games played
without the star. This only affects later dates.

**Roster talent (`elo_plus`).** This term adds the difference between the two teams' summed
regular values (GameScore and on/off). It carries information about trades, signings and
returns that the results-based rating has not absorbed yet. It is 0 until both teams have
played a game in the season.

**Tuning.** All parameters are tuned walk-forward. For test season S, L-BFGS-B minimises
the log-loss of the pre-game probabilities on the seasons from 2019-20 to S-1 (2018-19 is
the burn-in). Ratings then run on game by game through S. The parameters of
`elo_plus` for 2025-26 are: K 8.1, alpha 0.90, carry 0.52, home advantage 40 Elo (+23 in the
playoffs), k_early 1.34, back-to-back -51 Elo, rest of 3+ days +11, 3 games in 5 days -1.6,
missing value 3.1/GameScore unit + 0.63/minute + 8.1/on-off unit, and talent 2.9 and 6.3
per unit. All seasons are in `results.json`.

**Leak checks (run once, not part of `run.py`).**
1. Flipping every result and margin on 2024-01-15, and the actual absences that day, changes
   no probability on that date or before it (max |Δp| = 0). Later dates do change.
2. Rebuilding the availability table with that date's box scores deleted leaves every
   pre-game row (values, gaps) on or before that date identical, to 1e-14. The only missing
   rows are the 261 player rows of that date itself, which need its box scores.

## Results (evaluation set, log-loss / accuracy)

| predictor | 2022-23 | 2023-24 | 2024-25 | 2025-26 | pooled | Brier | open subset (n=3,929) |
|---|---|---|---|---|---|---|---|
| market close | 0.6228 / .672 | 0.5829 / .691 | 0.5856 / .694 | 0.5763 / .685 | **0.5916 / .686** | 0.2032 | 0.5817 / .690 |
| market open | - | 0.5955 / .682 | 0.5978 / .681 | 0.5863 / .679 | - | - | 0.5932 / .681 |
| fallback_logit (production) | 0.6464 / .631 | 0.6095 / .652 | 0.6063 / .658 | 0.5977 / .678 | 0.6146 / .655 | 0.2134 | 0.6045 / .663 |
| elo_prod (plain Elo) | 0.6610 / .622 | 0.6223 / .644 | 0.6217 / .657 | 0.6072 / .675 | 0.6277 / .650 | 0.2187 | 0.6174 / .659 |
| elo_538 (untuned) | 0.6522 / .633 | 0.6242 / .643 | 0.6257 / .650 | 0.6119 / .674 | 0.6283 / .650 | 0.2189 | 0.6208 |
| mov_tuned | 0.6470 / .632 | 0.6128 / .649 | 0.6099 / .656 | 0.5982 / .683 | 0.6166 / .655 | 0.2142 | 0.6070 |
| mov_rest (+ rest, playoff home adv.) | 0.6447 / .628 | 0.6091 / .665 | 0.6056 / .656 | 0.5983 / .684 | 0.6141 / .659 | 0.2130 | 0.6043 |
| avail_prev (+ availability) | 0.6402 / .646 | 0.6043 / .671 | 0.5992 / .671 | 0.5900 / .681 | 0.6080 / .668 | 0.2104 | 0.5977 / .675 |
| avail_post (+ post-game update) | 0.6401 / .647 | 0.6046 / .670 | 0.6005 / .668 | 0.5896 / .681 | 0.6083 / .667 | 0.2106 | 0.5982 |
| **elo_plus (+ roster talent)** | 0.6386 / .648 | 0.6007 / .681 | 0.5996 / .675 | 0.5917 / .679 | **0.6073 / .671** | 0.2101 | **0.5972 / .679** |
| oracle_bound (NOT leak-free) | 0.6333 / .648 | 0.5967 / .674 | 0.5907 / .681 | 0.5856 / .671 | 0.6012 / .669 | 0.2075 | 0.5911 / .676 |

`oracle_bound` is `elo_plus` given who actually played in the game being predicted. It is
not a predictor. It is an upper bound on what a perfect injury report or inactive list
could add.

### Paired bootstrap, mean log-loss difference (2,000 resamples over games; resampling game-days gives the same CIs)

| comparison | delta | 95% CI | seasons better |
|---|---|---|---|
| elo_plus - fallback_logit | **-7.28e-3** | [-10.77, -3.80] | 4/4 |
| avail_prev - fallback_logit | -6.56e-3 | [-9.56, -3.42] | 4/4 |
| elo_prod - fallback_logit | +13.08e-3 | [+8.15, +17.92] | 0/4 |
| elo_plus - market close | **+15.76e-3** | [+10.78, +20.94] | 0/4 |
| fallback_logit - market close | +23.04e-3 | [+17.31, +28.69] | 0/4 |
| oracle_bound - market close | +9.64e-3 | [+6.19, +12.95] | 0/4 |
| elo_plus - market open (n=3,929) | **+4.07e-3** | [-0.94, +9.27] | 0/3 |
| fallback_logit - market open | +11.35e-3 | [+5.69, +17.17] | 0/3 |
| oracle_bound - market open | -2.11e-3 | [-7.61, +3.54] | 2/3 |

### What drives it (each step vs the previous one)

| step | delta | 95% CI | seasons better |
|---|---|---|---|
| tune K, alpha, carry, home adv., early K (mov_tuned - elo_538) | -11.63e-3 | [-16.26, -7.10] | 4/4 |
| rest / back-to-back / playoff home adv. (mov_rest - mov_tuned) | -2.57e-3 | [-4.37, -0.60] | 3/4 |
| availability, "missed previous game(s)" (avail_prev - mov_rest) | **-6.01e-3** | [-8.81, -3.11] | 4/4 |
| post-game update with actual absences (avail_post - avail_prev) | +0.30e-3 | [-0.41, +0.97] | 2/4 |
| roster talent (elo_plus - avail_post) | -1.02e-3 | [-2.84, +0.87] | 3/4 |

1. **Tuning the Elo** gives the largest step. Home advantage falls from 100 to about 34-44
   Elo (a 55-56% home win rate for equal teams, which matches 2019-26), K falls to about
   8-16, the season-start regression gets stronger (carry about 0.5-0.6), and K rises early
   in the season.
2. **Rest** matters. A back-to-back is worth about -43 to -51 Elo, roughly 1.5-1.8 points.
3. **Availability** is the largest real-information gain, about 6e-3. The talent term and
   the post-game update are not significant on their own.

### Where the gap to the market sits (log-loss: close / fallback / elo_plus)

| segment | n | close | fallback | elo_plus |
|---|---|---|---|---|
| regular season | 4,841 | 0.5874 | 0.6118 | 0.6041 |
| play-in + playoffs | 356 | 0.6479 | 0.6530 | 0.6515 |
| min games played < 10 | 614 | 0.6066 | 0.6281 | 0.6201 |
| 10-29 | 1,190 | 0.6067 | 0.6193 | 0.6188 |
| 30-59 | 1,779 | 0.5961 | 0.6218 | 0.6160 |
| 60+ (end of season) | 1,614 | 0.5697 | 0.5981 | 0.5844 |

In the playoffs we are close to the market: 0.6515 vs 0.6479 on only 356 games. The gap is
in the regular season, and it is largest late in the season (tanking, resting, motivation: see H3) and in the middle of it.
Calibration on the test games is fine: logistic slope 1.04, intercept +0.02.

## Production: what it would need every day

The backtested model uses **no injury report**. Before the first game of the day it needs:

1. **Final scores** of every completed game: date, home, away, points, and regular season /
   play-in / playoffs. This feeds the Elo update. The ESPN scoreboard already provides it
   (`src/engine/espn.py`, `SCOREBOARD`).
2. **Today's schedule** (home, away, date) and each team's previous game dates. These feed
   the back-to-back, rest and 3-in-5 terms. ESPN scoreboard.
3. **Player box scores** of every completed game. For each player it needs team, minutes,
   PTS, FGM, FGA, FTM, FTA, OREB, DREB, AST, STL, BLK, TOV, PF and +/-. A player with no box
   score line did not play. ESPN's `summary?event=` endpoint, already called by
   `src/engine/espn.py` for team stats, returns these lines per player under
   `boxscore.players`, including +/-. Only the parser is missing. Check +/- coverage on a
   live call before relying on it.
4. **Persisted state**:
   * the 30 ratings;
   * each player's rolling sums over his last 82 appearances;
   * each team's minutes over the last 10 appearances of each regular;
   * the consecutive-games-missed counters.

   This needs a one-off backfill. Either replay the nba_api logs already here, which needs an
   NBA-to-ESPN player-id map, or fetch about one season of ESPN summaries (about 1,300 calls).
   Re-tune the parameters once per offseason with this script.

**ESPN box scores are sufficient. The injury report is optional.** The ESPN injury report
(the `injuries` block of the same summary, already read by `game_context()`) could replace
the P(out | games missed) estimate with today's status (Out / Doubtful / ...). The most it
could add is `oracle_bound`: -6.1e-3 more, which would bring us level with or slightly
better than the opening line, but still +9.6e-3 behind the close. That bound includes
coach's-decision DNPs that no report announces, so the real gain is smaller. ESPN only
serves the current report, so the gain cannot be backtested here. Log the report daily
before trusting it.

## Caveats

* **Selection.** The ladder up to `avail_post` was fixed before any test score was seen. The
  first full run's primary used GameScore + minutes only and scored 0.6080 pooled. The on/off
  values and the roster-talent term were added after that run, chosen on 2022-23 and
  2023-24. On the two later seasons the talent term was mixed: -0.9e-3 in 2024-25 and
  +2.1e-3 in 2025-26. `elo_plus - avail_prev` is -0.72e-3 [-2.62, +1.26]. In practice
  `avail_prev` is equally good and simpler: no post-game re-computation and no talent term.
  Also tried and dropped, because they gave nothing on 2022-24:
  * an online, self-updating home advantage (its tuned rate was 0 in every season, and it was
    removed because it would mix same-day results across games);
  * a separate calibration temperature;
  * a free autocorrelation coefficient;
  * recency-weighted tuning (weight 0.6 per season back).
* The 2022-23 test season is tuned on 2019-20..2021-22. Those are COVID seasons: the bubble
  is flagged neutral, but the empty arenas of 2020-21 are not. Player-based terms have only
  2 prior seasons there (player logs start in 2020-21).
* Availability edge cases:
  * a traded player counts as absent for his old team until he appears for the new one;
  * the first game of each season has no regulars, so no availability or talent term;
  * a regular benched by the coach keeps counting as absent with his old minutes;
  * player value (GameScore, noisy on/off) is crude.
* NBA Cup knockout games in Las Vegas are not flagged as neutral. The Cup final is missing
  from the game table but is used in the player state.
* The opening-line comparison covers 3 seasons, and the time of the ESPN open is unknown
  (see H1 and H6). "Level with the open" is a statement about accuracy, not a betting edge.
  No ROI was tested.
* The bootstrap resamples games. Resampling whole game-days gives the same intervals.
* The closing line remains about 1.5 percentage points more accurate (68.6% vs 67.1%), about
  20 more correct picks per season. Publishing our own number means publishing a less
  accurate number than the market's.

## Files

* `run.py`: the whole pipeline (data, availability build, Elo engine, walk-forward tuning,
  baselines, evaluation).
* `cache/availability_players.csv`: 183,690 rows, one per (team-game, regular), with
  pre-game values, the consecutive-games-missed count and the actual absence. It is
  regenerated by `--rebuild` and can be deleted.
* `preds.csv`: `GAME_ID, season, date, home, away, home_win, p` for `elo_plus`, covering all
  5,269 test-season games. The metrics use the 5,197 that have a closing line.
* `results.json`, `run_log.txt`.
