# H3: Motivation and competitive context vs the betting market

**Question.** Does motivation and competitive context add information about the winner beyond the
betting price? Context here means standings races, eliminated, clinched or locked teams, tanking,
the last week of the season, and the playoff series state (game number, series lead, elimination
games, game 7, 0-2 holes, the "zig-zag" bounce-back after a loss).

**Answer: no edge.**

* **0 of 33 model variants** beat the closing line under the protocol, and none has even a plain
  95% CI below 0. The opening line (2024-25 and 2025-26) gives the same result.
* **0 of 8 pre-registered folk betting rules** and **0 of 244 model betting rules** have a moneyline
  ROI whose CI lies above zero after vig.
* **The "playoff favourites are overpriced" hint shrinks on new data.** On the 2018-19 to 2021-22
  playoffs, which played no part in finding it (337 games, 60 series), favourites won 2.3 pts less
  than priced (z = -0.9, series-clustered CI -5.9 to +1.4). In the 2022-26 sample where the hint was
  found, the gap was 4.8 pts. Pooled over 8 seasons and 669 games it is -3.5 pts (z = -2.0). That is
  below the bar for 38 pre-listed slices (|z| > 3.21), and the 2022-26 half is selection-biased.
  Favourites cover the spread at a fair rate (51.3%). Betting every playoff underdog at the close
  returned +9.6% [-1.6%, +21.3%], so it is not significant.
* **The tests have little power.** Even a true playoff-favourite effect of 4.4 pp, the size of the
  hint, would pass the protocol only about 7% of the time with the 353 playoff games available for
  testing (`power.py`). "No edge" here means "no exploitable edge can be shown", not "proven zero".

Run end-to-end (about 2.5 minutes, no network once the odds are cached; the power simulation takes a few
minutes more):

```
python research/h3_motivation_context/fetch_older_odds.py   # once: ESPN 2018-19..2020-21 odds (~25 min, cached, resumable)
python research/h3_motivation_context/run.py                # all tables -> results/, full log -> results/run_log.txt
python research/h3_motivation_context/power.py              # power simulation -> results/power.csv
```

## Data

* **Standings and series** come from `research/data/team_logs.csv` (nba_api team game logs,
  2018-19 to 2025-26, regular season, play-in and playoffs).
* **Evaluation set** (`h3data.py`):
  * **2022-23 to 2025-26:** exactly the **5,197 games** of `research/upsets.py` (`load()`), with
    closing odds and, from 2023-24, opening odds. The closing-line baseline matches: 0.58162 over
    2023-24 to 2025-26.
  * **2021-22:** training and playoff hold-out only. ESPN has no open/close split for this season, so
    the stored "current" DraftKings moneyline stands in for the close. In later seasons that field
    equals the close in 99.6% of games.
  * **2018-19 to 2020-21:** training and playoff hold-out only. `fetch_older_odds.py` re-downloads
    ESPN's stored last pre-game line of every provider, politely (under 4 requests/s, cached to
    `research/data/h3_motivation_context/`). The book is Caesars (otherwise Westgate, Wynn, Unibet or
    the consensus line), kept only if it is within 6 pp of the median of the real books. Projection
    sites (numberfire, teamrankings, accuscore) are ignored, and so is the "Caesars Sportsbook" feed,
    which sometimes stores spread juice as a moneyline. Checks: the vig is 3.7%, the moneyline and
    spread favourite agree in 99.6% of games, and the log-loss is about 0.596.
* All probabilities are de-vigged proportionally (`h/(h+a)`). Bets are settled at the
  **vig-inclusive** odds of the same book.

## Features (`h3feat.py`): as of the morning of the game

Standings use only regular-season games on dates **strictly before** the game date. The only
forward-looking inputs are schedule facts published before the season: the planned number of games
(82; 72 in 2020-21; for 2019-20, 82 until the bubble schedule was announced) and the date of the last
regular-season day.

| feature | definition |
|---|---|
| `conf_rank` | conference rank, with approximate NBA tiebreakers: win% > head-to-head among the tied teams > conference win% > point differential |
| `margin_b` | signed games relative to the seed boundary b/b+1 (b = 1, 4, 6, 8, 10): positive = cushion, negative = games back |
| `best_seed`, `worst_seed` | best and worst seeds still reachable, by counting: best = 1 + #teams whose current wins exceed my maximum possible wins; worst = 1 + #teams whose maximum possible wins reach my current wins (ties count against me) |
| `elim_post` | eliminated from the postseason: best seed > 10 (> 8 before the play-in era, 2018-19 and 2019-20) |
| `seed_locked` | postseason team whose best seed equals its worst seed: nothing left to play for, so likely to rest starters |
| `no_stakes` | `elim_post` or `seed_locked` |
| `in_race` | a boundary that matters (1/2 top seed, 4/5 home court, 6/7 direct playoffs, 8/9, 10/11) is still reachable on both sides and is within 2 games, with at most 20 games left |
| `tank` | one of the 6 worst records in the league with at most 25 games left (since 2019 the bottom-3 lottery odds are flat; places 4-6 still gain odds by losing) |
| `last_week`, `last_game` | game within 7 days of the last regular-season day; the team's final regular-season game |
| `post_margin`, `race_close`, `lottery` | continuous versions, used in the combined models only (at most 25 games left): margin to the postseason line, closeness to the nearest live boundary, lottery position |
| series state | `game_no`, series wins of each side before the game, `home_lost_prev` (+1 if the home team lost the previous game of the series, -1 if the away team did; the zig-zag variable), previous-game margin, `home_elim` / `away_elim` (facing elimination), `game7`, `home_down02` (+1 home team trails 0-2, -1 leads 2-0) |
| play-in | 9v10 game and the second-round game: both sides face elimination; 7v8 game: nobody does (seeds from the standings on that date) |

The clinch and elimination flags are **approximations**. They ignore remaining head-to-head games
(two teams that still play each other cannot both win them) and treat ties pessimistically, so they
can trigger a few days after the official clinch or elimination. Spot checks look right: in 2023-24
DET is eliminated from mid-March and BOS has the 1 seed locked from 27 March, and on the last day
PHI, IND, ORL and MIA are flagged as in a race for 6/7. Series game numbers match the official
game-ID game number in 100% of playoff games.

## Method (shared protocol)

* **Walk-forward by season.**
  * Closing line: test seasons 2022-23 to 2025-26, each fitted on **all earlier seasons with odds**
    (from 2018-19). The robustness run trains from 2021-22 only, as in H1 and H2.
  * Opening line: test seasons 2024-25 and 2025-26, fitted on 2023-24 onwards (opening odds start
    in 2023-24).
* **Specs.**
  * **A (recal):** `logit p = a + b*logit(mkt) + w.X`.
  * **B (offset):** `logit p = logit(mkt) + w.X`, with no intercept. A game where every feature is 0
    gets exactly the market price, so a sparse context feature can only change the games it
    describes.
* **Variants: 17 per spec.**
  * Regular season: 6 single features, each home minus away: `elim_diff`, `locked_diff`,
    `nostakes_diff`, `tank_diff`, `race_diff`, `lw_nostakes_diff` (no stakes, last week).
  * Playoffs: 7 single features.
    * `po_fav`: the sign of `logit(mkt)` in the postseason. This tests the hint "playoff favourites
      are overpriced" directly.
    * `po_slope`: `logit(mkt)` in the postseason (a playoff-specific calibration slope).
    * `zigzag`, `elim_po`, `game7`, `series_lead`, `down02`.
  * Combined L2 models (penalty chosen by inner leave-one-season-out CV) for the regular season
    (18 per-side columns), the playoffs (8 columns) and everything together. Also a LightGBM model
    with `init_score = logit(mkt)`, with its number of rounds chosen by inner CV (spec B only).
  * Total: 17 + 16 = **33 variants per baseline**, so the Bonferroni CI level is 99.85%.
* **Uncertainty.** A paired bootstrap over games (4,000 resamples) of the per-game log-loss
  difference. **Survival bar:** the Bonferroni CI lies entirely below 0 AND at least 3 of 4 seasons
  are negative (2 of 2 for the opening line). The delta restricted to the games a feature touches is
  reported too ("active").
* **Descriptive slices.** 38 pre-listed slices compare the team of interest's win rate with the
  de-vigged close (z-score), its ATS cover and its ROI. Bonferroni threshold: |z| > 3.21. Playoff
  slices use a **series-clustered bootstrap**, because games of the same series are not independent.
* **Hint re-test.** The hint "playoff favourites underperform" came from 2022-23 to 2025-26. The
  2018-19 to 2021-22 playoffs were never looked at before, so they serve as an **independent
  hold-out**.
* **Betting.**
  * Every model: flat 1u when the model probability exceeds the vig-inclusive implied probability
    plus a threshold (0, 2, 4, 6 pp), at closing odds. Opening-line models are settled at both
    opening and closing odds.
  * 8 pre-registered folk rules (direction fixed by the folk theory, not by the data): moneyline at
    the close, and ATS at an assumed -110.
* **Line movement** (2023-24 onwards): does the price move from open to close in the direction of
  the context features?
* **Power** (`power.py`): outcomes re-simulated from the close plus a known effect, then the same
  test is re-run.

## Results

### 1. Walk-forward vs the closing line (5,197 games; market log-loss 0.59156; 0.58162 over 2023-24 to 2025-26)

Delta = model log-loss minus market log-loss (negative = better than the market). "Active" = games
where the feature is non-zero, and the delta restricted to them. Full table with Bonferroni CIs:
`results/walkforward_close.csv`.

| spec B (offset) variant | delta | 95% CI | 22-23 | 23-24 | 24-25 | 25-26 | neg | active games | delta on active |
|---|---|---|---|---|---|---|---|---|---|
| `po_slope` (playoff calibration slope) | **-0.00022** | [-0.00070, +0.00022] | -0.00047 | +0.00025 | -0.00023 | -0.00044 | 3/4 | 353 | -0.0032 |
| `po_fav` (playoff favourite shift) | -0.00019 | [-0.00043, +0.00004] | -0.00008 | +0.00001 | -0.00019 | -0.00048 | 3/4 | 353 | -0.0027 |
| `game7` | -0.00013 | [-0.00068, +0.00040] | -0.00039 | -0.00032 | +0.00060 | -0.00043 | 3/4 | 15 | -0.045 |
| `zigzag` | -0.00009 | [-0.00043, +0.00025] | -0.00005 | -0.00011 | -0.00055 | +0.00036 | 3/4 | 272 | -0.0016 |
| `locked_diff` | -0.00003 | [-0.00070, +0.00063] | -0.00064 | -0.00006 | +0.00040 | +0.00016 | 2/4 | 58 | -0.0024 |
| `elim_po` | +0.00000 | [-0.00006, +0.00006] | | | | | 0/4 | 81 | +0.0003 |
| `po_combined_l2` | +0.00002 | [-0.00042, +0.00045] | | | | | 3/4 | 355 | +0.0003 |
| `nostakes_diff` | +0.00003 | [-0.00017, +0.00024] | | | | | 2/4 | 333 | +0.0004 |
| `all_combined_l2` | +0.00004 | [-0.00001, +0.00010] | | | | | 1/4 | 1,871 | +0.0001 |
| `rs_combined_l2` | +0.00005 | [-0.00001, +0.00010] | | | | | 0/4 | 1,516 | +0.0002 |
| `lw_nostakes_diff` | +0.00005 | [-0.00056, +0.00073] | | | | | 2/4 | 130 | +0.0021 |
| `tank_diff` | +0.00007 | [-0.00011, +0.00025] | | | | | 1/4 | 481 | +0.0008 |
| `down02` | +0.00009 | [-0.00006, +0.00025] | | | | | 0/4 | 32 | +0.015 |
| `elim_diff` | +0.00011 | [-0.00005, +0.00026] | | | | | 1/4 | 307 | +0.0019 |
| `series_lead` | +0.00012 | [-0.00017, +0.00041] | | | | | 0/4 | 215 | +0.0029 |
| `race_diff` | +0.00013 | [-0.00010, +0.00038] | | | | | 2/4 | 631 | +0.0011 |
| `all_combined_gbm` | +0.00081 | [-0.00009, +0.00169] | +0.00292 | | | | 0/4 | 1,871 | +0.0016 |

* **Spec A (recalibrated a, b):** `market_recal` alone costs +0.00040. All 16 spec-A variants land
  between +0.00014 and +0.00055 (worse than the raw close).
* **0/33 survive. 0/33 have a plain 95% CI below 0.** The median variant is +0.00014. The best
  (`po_slope`) has Bonferroni CI [-0.00095, +0.00050].
* The fitted playoff weights point the expected way in every training window. `po_fav` gets
  w = -0.05 to -0.08 logit, i.e. favourites shaded down by about 1.5 pp. The gain is too small and
  too noisy to pass. On the 356 postseason games alone, `po_fav` improves log-loss by -0.0027
  [-0.0063, +0.0007].
* `locked_diff` learns a large weight: about -0.5 logit, i.e. teams with their seed locked win about
  12 pp less than priced. But it fires on only 58 test games, so its out-of-sample effect is pure
  noise.

### 2. Walk-forward vs the opening line (2,630 games, 2024-25 and 2025-26; market log-loss 0.59204)

0/33 survive and 0/33 have a 95% CI below 0 (median +0.00062).

| variant | delta | 95% CI | seasons negative |
|---|---|---|---|
| B `race_diff` (best) | -0.00037 | [-0.00155, +0.00082] | 2/2 |
| B `lw_nostakes_diff` | -0.00014 | [-0.00041, +0.00017] | 2/2 |
| B `po_fav` | **+0.00075** | [-0.00023, +0.00169] | 0/2 |
| B `down02` | +0.00057 | [+0.00003, +0.00117] (significantly worse) | 0/2 |

The playoff-favourite shading does not help against the opening line either. In 2024-25, trained on
2023-24 only, its weight even has the opposite sign. Full table: `results/walkforward_open.csv`.

### 3. The playoff-favourite hint, re-tested (`results/playoff_favourites.csv`)

Favourites' win rate vs the de-vigged closing price. CIs resample whole series. Calibration slope
from `y ~ a + b*logit(p)`; b < 1 means favourites are overpriced (regular-season reference:
b = 1.008, se 0.028).

| sample | games | series | priced | won | gap (pts) | z | 95% CI (cluster) | slope (se) | fav ATS cover | bet every dog, ML ROI |
|---|---|---|---|---|---|---|---|---|---|---|
| 2018-19 | 82 | 15 | 67.8% | 63.4% | -4.4 | -0.88 | | 0.97 (0.34) | 48.8% | +18.7% |
| 2019-20 (bubble) | 83 | 15 | 68.3% | 63.9% | -4.5 | -0.90 | | 0.83 (0.27) | 51.2% | +12.1% |
| 2020-21 | 85 | 15 | 64.5% | 63.5% | -1.0 | -0.20 | | 0.74 (0.36) | 58.3% | +4.5% |
| 2021-22 | 87 | 15 | 63.8% | 64.4% | +0.6 | +0.11 | | 0.89 (0.40) | 54.0% | -2.4% |
| 2022-23 | 81 | 15 | 66.4% | 64.2% | -2.2 | -0.43 | | 0.49 (0.34) | 51.9% | +13.0% |
| 2023-24 | 82 | 15 | 65.4% | 63.4% | -2.0 | -0.39 | | 1.12 (0.40) | 46.3% | +0.8% |
| 2024-25 | 84 | 15 | 68.1% | 61.9% | -6.2 | -1.24 | | 0.77 (0.30) | 50.0% | +11.9% |
| 2025-26 | 85 | 15 | 67.5% | 58.8% | -8.7 | -1.75 | | 0.74 (0.29) | 49.4% | +19.0% |
| **2018-22 hold-out** | 337 | 60 | 66.1% | 63.8% | **-2.3** | **-0.91** | [-5.9, +1.4] | 0.86 (0.16) | 53.2% | +8.0% [-6.4%, +23.1%] |
| 2022-26 (where the hint was found) | 332 | 60 | 66.9% | 62.1% | -4.8 | -1.91 | [-9.9, +0.7] | 0.77 (0.16) | 49.4% | +11.2% [-6.5%, +29.2%] |
| all 8 seasons | 669 | 120 | 66.5% | 62.9% | -3.5 | -1.98 | [-6.6, -0.4] | 0.82 (0.11) | 51.3% | +9.6% [-1.6%, +21.3%] |
| all, excluding the 2019-20 bubble | 586 | 105 | 66.2% | 62.8% | -3.4 | -1.78 | [-6.7, -0.0] | 0.80 (0.13) | 51.3% | +9.3% |

What to make of it:
* The direction is consistent: the gap is negative in 7 of 8 seasons.
* The size is not. On data that played no part in finding the hint, it is -2.3 ± 2.5 pts, compatible
  with zero. The 2022-26 estimate is inflated because that slice was picked for looking unusual.
* The pooled z = -2.0 does not survive the 38-slice Bonferroni bar (3.21), and the
  pooled underdog ROI CI (+9.6%, [-1.6%, +21.3%]) includes zero. With a Bonferroni correction over
  the 8 folk rules it is [-6.1%, +25.3%].
* **The effect is a moneyline effect, not a spread effect.** Playoff favourites cover the closing
  spread 51.3% of the time (-0.35 pts ATS), and betting every playoff underdog ATS at -110 loses
  -6.9% [-13.7%, -0.4%].
* *Post-hoc diagnostic:* playoff margins scatter more around the closing spread than
  regular-season margins do (SD 14.5 vs 13.3 pts, bootstrap CI on the difference [+0.3, +2.0]). With
  the playoff SD, the spread implies a 64.9% favourite win rate instead of 66.1%. That accounts for
  roughly a third of the moneyline gap. A plausible mechanism, then, is that moneylines in the
  playoffs are set from spreads with regular-season variance, which would be a small,
  favourite-longshot-type pricing quirk rather than "motivation". It is not strong enough to bet.

### 4. Series state and regular-season context: 38 pre-listed slices (`results/slices.csv`)

Team of interest vs the de-vigged close, all 8 seasons. No slice reaches the Bonferroni threshold
|z| > 3.21. The largest is "favourite in rounds 2-4" at -6.3 pts, z = -2.43, which is the same
playoff-favourite effect again.

| slice (side = team of interest) | n | priced | won | gap | z | ATS cover | ML ROI at close |
|---|---|---|---|---|---|---|---|
| zig-zag: loser of the previous series game | 549 | 47.2% | 49.9% | +2.7 | +1.39 | 52.4% | +6.6% [-5.4%, +19.2%] |
| zig-zag, previous loser is the underdog | 302 | 33.0% | 38.1% | +5.0 | +1.91 | 51.3% | +15.2% |
| zig-zag, previous loser is the favourite | 247 | 64.4% | 64.4% | -0.0 | -0.02 | 53.7% | -4.0% |
| team facing elimination (not game 7) | 156 | 41.1% | 41.0% | -0.1 | -0.02 | 51.3% | +2.1% |
| game 7 home team | 28 | 63.2% | 46.4% | -16.8 | -1.87 | 35.7% | -32.1% |
| game 3 home team down 0-2 | 52 | 48.6% | 48.1% | -0.5 | -0.08 | 48.1% | -0.9% |
| game 1 home team | 120 | 68.6% | 63.3% | -5.3 | -1.28 | 55.5% | -10.4% |
| team with stakes vs no-stakes opponent | 584 | 75.4% | 77.2% | +1.8 | +1.11 | 54.7% | +0.5% |
| ... same, last week of the season | 239 | 75.4% | 80.3% | +4.9 | +1.94 | 58.6% | +5.2% |
| seed-locked team, last week | 87 | 53.1% | 43.7% | -9.4 | -2.02 | 47.1% | -26.2% |
| tank team (bottom 6, at most 25 games left) | 900 | 24.4% | 24.1% | -0.3 | -0.22 | 48.0% | -8.6% |
| eliminated team vs non-eliminated | 545 | 21.6% | 21.1% | -0.5 | -0.31 | 45.2% | -6.9% |
| team in a seeding race vs opponent not in a race | 1,080 | 63.8% | 65.0% | +1.2 | +0.94 | 52.1% | -2.8% |
| no-stakes team in its final game | 73 | 27.7% | 19.2% | -8.5 | -1.78 | 44.4% | -41.2% |

* **Zig-zag.** The bounce-back after a loss is priced. The ATS edge seen in 2022-26 (55.9% cover)
  does not appear in 2018-22 (48.9% cover).
* **Elimination games, 0-2 holes and game 7.** These are priced, or too rare to tell.
* **Tanking and eliminated teams.** The market prices them almost exactly (gaps of about 0.3-0.5 pts
  over 900 and 545 games).
* **Seed-locked teams in the last week.** They underperform by about 9 pts, consistent with
  load management beyond the closing price. But there are only 87 games: z = -2.0 and the betting CI
  includes 0.
* **"Stakes vs no stakes" in the last week.** It is directionally positive in both periods (ML +7.4%
  in 2018-22, +3.3% in 2022-26), but with CIs spanning 0.

### 5. Betting (vig-inclusive moneylines)

| what | rules | ROI > 0 | 95% CI above 0 | median ROI | best (cherry-picked) |
|---|---|---|---|---|---|
| closing-line models, closing odds, at least 30 bets | 45 | 22 | **0** | -3.0% | A `zigzag` thr 2 pp: +41.6% [-19.6%, +110%], n=32 |
| opening-line models, opening odds | 63 | 18 | **0** | -4.8% | A `zigzag` thr 4 pp: +18.6% [-12.1%, +51.8%], n=69 |
| opening-line models, closing odds | 136 | 0 | **0** | -10.3% | -4.3% |
| 8 pre-registered folk rules, moneyline at the close (all seasons) | 8 | 5 | **0** (Bonferroni: 0) | | bet every playoff dog +9.6% [-1.6%, +21.3%], n=669; bet against seed-locked team in last week +20.1% [-10.1%, +53.2%], n=87 |
| 8 folk rules, ATS at an assumed -110 | 8 | | 1 (Bonferroni: 0) | | last week, team with stakes vs no-stakes: +11.9% [+0.3%, +24.2%], n=239, but only +4.3% [-11.9%, +20.4%] in 2022-26 |

The closing-line playoff models bet mostly underdogs: `po_slope` at threshold 0 made 250 bets for
+16.1% [-4.5%, +37.6%], and `po_fav` made 232 bets for +8.9% [-14.0%, +30.8%]. This is the same
underdog tilt as above, and it is not significant.

### 6. Is the context already in the price? Line movement open to close (2023-24 onwards, `results/line_move.csv`)

* The line moves toward the team in a seeding race: +0.9 pp, t = 2.4.
* It moves toward a home team down 0-2: +3.1 pp, t = 2.7, n = 26.
* It moves against seed-locked teams: -2.8 pp, t = -1.4.

So the market does react to motivation news between open and close (mostly rest and lineup news on
the day of the game). The moves are small, and the opening-line models gain nothing from them.

### 7. Power (`results/power.csv`, 60 simulations per cell)

| feature | true effect (logit) | mean shift on active games | games | P(pass 95% bar) | P(pass Bonferroni bar) |
|---|---|---|---|---|---|
| `po_fav` | -0.1 / -0.2 / -0.3 | 2.2 / 4.4 / 6.7 pp | 700 | 3% / 7% / 25% | 0% / 0% / 8% |
| `locked_diff` | -0.2 / -0.4 / -0.6 | 3.8 / 7.6 / 11.4 pp | 103 | 0% / 3% / 7% | 0% |
| `tank_diff` | -0.1 / -0.2 / -0.3 | 1.6 / 3.2 / 4.7 pp | 900 | 0% / 5% / 25% | 0% / 0% / 3% |
| `nostakes_diff` | -0.1 / -0.2 / -0.3 | 1.5 / 3.0 / 4.4 pp | 584 | 3% / 13% / 7% | 0% / 0% / 2% |

Context situations are too rare for the log-loss protocol to confirm effects of a few pp. An effect
the size of the playoff-favourite hint would almost never pass. The real constraint is economic,
though: a 2-4 pp shift on 50-350 games a season, against about 2 pp of vig per side, would barely
break even even if it were real and known exactly.

### 8. Robustness

* **Training from 2021-22 only**, as in H1 and H2: 0/33 survive. Best is B `game7` at -0.00007
  [-0.00082, +0.00066], median +0.00065 (`results/walkforward_close_train2021.csv`). The extra
  2018-21 training seasons help a little, but do not change the verdict.
* **Restricted views.** On regular-season games alone, every regular-season variant is ≥ -0.00003.
  On the 356 postseason games alone, the best is `po_slope` at -0.0032 [-0.0100, +0.0033]
  (`results/run_log.txt`, section 4b).
* **LightGBM** chose 400 boosting rounds in the 2022-23 fold and overfitted (+0.0029 that season).
  It chose 50 rounds in the other folds, and 0 rounds against the opening line.

## Caveats

* **The clinch, elimination and locked flags are counting approximations.** They ignore remaining
  head-to-head games and the full tiebreaker rules, so they can fire a few days late. Seed-locked
  games are rare as a result (103 in 8 seasons). Exact NBA clinch scenarios would add a few more
  late-season games, but not enough power to change the conclusion.
* **The closing lines are proxies before 2022-23.** 2018-19 to 2020-21 use ESPN's stored last
  pre-game Caesars line; 2021-22 uses the "current" DraftKings line. These seasons are used only for
  training and for the hold-out descriptive tests, never as a scored test season.
* **2019-20** includes the bubble playoffs (neutral site; ESPN does not flag them as neutral). Results
  excluding them are shown and are the same.
* **Tanking is measured by record, not by incentives.** Pick protections and swaps (a team whose pick
  is owed has no reason to tank) are not modelled. Neither is the 2023-24 player participation policy,
  which limits star rest. The NBA Cup group stage, another motivation context, is not modelled either.
* **Spreads are the book's closing spreads, and ATS ROI assumes -110** (spread prices are not stored).
* **Multiple testing.** 33 log-loss variants per baseline, 38 descriptive slices, 8 folk rules and
  244 model betting rules. Best values are shown for transparency and are selection-biased. The
  playoff-favourite hint was itself chosen from 22 segments, which is why the 2018-22 hold-out is the
  fair test of it.

## Files

* `fetch_older_odds.py`: ESPN events and every provider's pre-game moneyline for 2018-19 to 2020-21.
  Polite and cached in `research/data/h3_motivation_context/` (`espn_<season>.csv`, plus resumable
  `_odds_*.jsonl` and `_events_*.csv`).
* `h3feat.py`: standings as of the morning of each game, clinch/elimination/race/tank flags, playoff
  series state. Cached as `context_features.csv` and `series_state.csv`.
* `h3data.py`: the evaluation set (odds, outcome, context for both sides, series state). Cached as
  `eval_dataset.csv`.
* `h3eval.py`: game-level features, offset and recalibrated logit, inner-CV L2, LightGBM with the
  market as init score, bootstrap (including series-clustered), betting.
* `run.py`: the end-to-end driver. Tables go to `results/` (`slices.csv`, `playoff_favourites.csv`,
  `rule_bets.csv`, `walkforward_close.csv`, `walkforward_open.csv`,
  `walkforward_close_train2021.csv`, `betting_models.csv`, `line_move.csv`), the full console output
  to `results/run_log.txt`, and per-game closing-line predictions to
  `research/data/h3_motivation_context/preds_close.csv`.
* `power.py`: the power simulation, written to `results/power.csv`.
