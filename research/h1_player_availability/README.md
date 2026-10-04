# H1 - Player availability vs the betting market

**Question.** Upsets happen when key players sit. Does a player-level absence signal add
information beyond the betting price?

**Answer.** Not beyond the closing line. Even with perfect knowledge of who sits ("oracle"),
nothing beats the close. Absences are the main thing that moves the line between open and
close (they explain ~21-27% of the variance of open-to-close moves), and the oracle signal
beats the **opening** line. The realistic proxy (out in the previous game, so probably out
again) adds little or nothing, even against the open.

```
python research/h1_player_availability/run.py            # end to end, uses caches
python research/h1_player_availability/run.py --rebuild  # recompute features from cached logs
```

Files: `fetch_players.py` (nba_api download, cached), `build.py` (games + odds + availability
features), `run.py` (all tests). Caches are in `research/data/h1_player_availability/`
(player logs, `rotation_player_games.csv`, `h1_games.csv`). Result tables are in `results/`
(`residual_tests.csv`, `roi.csv`, `line_moves.csv`, `segments.csv`, `upsets.json`,
`no_market_sanity.csv`, `run_log.txt`). Full run is about 1 minute (plus about 1 minute with `--rebuild`).

## Data

* Player game logs, 2020-21..2025-26, regular season + play-in + playoffs (nba_api
  `LeagueGameLog`, player mode, 18 calls, 165,987 player-games). A player who has no row
  in a game's box score did not play.
* Evaluation set: the shared `upsets` dataset (5,197 games, 2022-23..2025-26), unchanged.
  To make 2022-23 testable walk-forward, I added a **training-only** 2021-22 block (1,323
  games) from `team_logs.csv` + `espn_2021-22.csv`. For 2021-22 ESPN has no open/close
  split, so I use its line for the completed game. In 2022-23 that field equals
  `home_ml_close` in 100% of games.
* Opening lines exist only from 2023-24.
* ROI uses the actual ESPN-book American odds, vig included (`home_ml_close` / `home_ml_open`).

## Features (all leak-free except the explicit oracle)

For each team-game *j*, using only games strictly before *j*:

* **Expected rotation.** Players who (a) appeared in at least one of the team's previous 10
  games this season, (b) averaged >= 15 min over their last 10 appearances for the team this
  season, and (c) whose most recent appearance anywhere was for this team (so traded or
  released players drop out). The first game of each season has no rotation, so its
  feature is 0. On average a team has 9.7 rotation players.
* **Player value**, from the player's last 82 appearances (any team, any season) before the game date:
  * `min` = expected minutes
  * `gs` = (Game Score per minute, shrunk with 300 min toward replacement 0.287/min) x expected
    minutes. The replacement rate comes from 2020-21 fringe players only, before any
    training or test season.
  * `oo` = on/off plus-minus per 48 (on-court +/- vs team margin while off, shrunk with
    2,000 min) x expected minutes / 48
* **Absent, oracle.** A rotation player with no box-score row in game *j*. In production
  this would need the official inactive list or a final injury report. That is
  available shortly before tip-off, but it includes coach's DNPs that nobody knows in
  advance. Treat it as an **upper bound**.
* **Absent, previous-game proxy (realistic).** A rotation player with no row in the
  team's previous game. Known before tip-off. P(out today | out last game) = 0.69, versus
  0.08 for players who played last game.
* Feature = missing value (home) - missing value (away), for `min`, `gs` and `oo`.
  This gives 3 metrics x {oracle, prev} = **6 pre-registered variants**.

## Protocol

* Walk-forward by season (fit on seasons < S). Closing line: test 2022-23..2025-26.
  Opening line: only 2024-25 and 2025-26 are walk-forward-trainable, because 2023-24 has
  no earlier season with opening odds. For that reason I also report leave-one-season-out
  (LOSO) on 2023-24..2025-26, labelled as such.
* Two model specs, both using the market as input:
  * `lr` = logistic regression on [logit(market), standardized feature] (the protocol spec)
  * `offset` = logit(p) = logit(market) + b·z, with only *b* fitted. With no feature this is
    exactly the market. It is the cleanest "does X shift the price" test.
* 6 features x 2 specs = **12 primary tests per baseline**. The Bonferroni CI uses
  alpha = 0.05/12. Paired bootstrap over games, 2,000 resamples. 3 exploratory feature
  sets (all-3 oracle, top-2 indicator, gs decomposed into carried-over/new/return) bring
  the total to 18 feature x spec variants and 54 walk-forward/LOSO evaluations.
* Pass bar: 95% CI below 0 **and** negative in >= 3 of 4 seasons (or in all seasons
  when fewer are available).

## Results

### 1. Versus the CLOSING line (walk-forward 2022-23..2025-26, n = 5,197, market log-loss 0.59156)

Delta = model - market, in 1e-3 log-loss units (negative = better than the market).

| spec | variant | delta | 95% CI | Bonferroni CI | per season 22-23 / 23-24 / 24-25 / 25-26 | neg seasons |
|---|---|---|---|---|---|---|
| lr | market refit only (control) | +0.72 | [+0.07, +1.38] | [-0.28, +1.59] | +0.0026 / +0.0006 / -0.0002 / -0.0001 | 2/4 |
| lr | oracle_min | +1.05 | [+0.33, +1.80] | [-0.07, +2.14] | +0.0032 / +0.0004 / +0.0007 / +0.0000 | 0/4 |
| lr | oracle_gs | +1.22 | [+0.27, +2.19] | [-0.06, +2.51] | +0.0041 / +0.0010 / -0.0003 / +0.0002 | 1/4 |
| lr | oracle_oo | +1.89 | [+0.49, +3.32] | [-0.25, +4.18] | +0.0061 / +0.0008 / -0.0003 / +0.0011 | 1/4 |
| lr | prev_min | +0.97 | [+0.23, +1.71] | [-0.15, +1.99] | +0.0037 / +0.0005 / +0.0004 / -0.0006 | 1/4 |
| lr | prev_gs | +0.79 | [-0.04, +1.62] | [-0.48, +1.94] | +0.0031 / +0.0007 / -0.0006 / +0.0000 | 1/4 |
| lr | prev_oo | +1.23 | [-0.01, +2.46] | [-0.62, +2.93] | +0.0024 / -0.0002 / +0.0004 / +0.0024 | 1/4 |
| offset | oracle_min | +0.35 | [+0.02, +0.67] | [-0.14, +0.84] | +0.0010 / +0.0001 / +0.0005 / -0.0002 | 1/4 |
| offset | oracle_gs | +0.52 | [-0.20, +1.24] | [-0.58, +1.54] | +0.0020 / +0.0001 / -0.0002 / +0.0002 | 1/4 |
| offset | oracle_oo | +1.13 | [-0.11, +2.33] | [-0.60, +2.97] | +0.0034 / +0.0002 / -0.0001 / +0.0011 | 1/4 |
| offset | prev_min | +0.24 | [-0.17, +0.65] | [-0.35, +0.85] | +0.0014 / +0.0001 / +0.0004 / -0.0008 | 1/4 |
| offset | **prev_gs (best)** | **+0.06** | [-0.52, +0.64] | [-0.76, +0.90] | +0.0006 / +0.0001 / -0.0004 / +0.0000 | 1/4 |
| offset | prev_oo | +0.50 | [-0.51, +1.57] | [-1.03, +2.11] | -0.0004 / -0.0006 / +0.0006 / +0.0024 | 2/4 |

Not one variant is better than the close. Best realistic is +0.06e-3; best oracle is
+0.35e-3; the median of the 12 is +0.88e-3. The fitted coefficients are close to zero, and
for `min` the sign is even slightly positive. The closing line prices absences fully and
does not under-react to them. The three exploratory sets are also all worse
(+0.5 to +2.4e-3).

### 2. Versus the OPENING line (n = 2,630 walk-forward / 3,929 LOSO)

| spec | variant | WF delta (24-25, 25-26) | WF 95% CI | WF Bonferroni CI | LOSO delta | LOSO 95% CI | LOSO per season 23-24 / 24-25 / 25-26 |
|---|---|---|---|---|---|---|---|
| offset | oracle_min | **-4.05** | [-6.35, -1.81] | [-7.38, -0.71] | -3.10 | [-5.55, -0.50] | -0.0006 / -0.0039 / -0.0047 |
| offset | oracle_gs | -2.37 | [-5.32, +0.68] | | -2.08 | [-4.38, +0.37] | -0.0020 / -0.0042 / +0.0000 |
| lr | oracle_min | -3.39 | [-5.95, -0.74] | [-7.37, +0.10] | -2.78 | [-5.36, +0.06] | +0.0002 / -0.0041 / -0.0044 |
| offset | **prev_min (realistic)** | -2.05 | [-3.36, -0.80] | [-3.83, -0.13] | -1.32 | [-3.24, +0.45] | +0.0010 / -0.0021 / -0.0028 |
| offset | prev_gs | -0.25 | [-1.57, +1.07] | | -0.13 | [-1.16, +0.90] | +0.0000 / -0.0008 / +0.0004 |
| lr | prev_min | -1.03 | [-3.02, +0.95] | | -0.94 | [-2.93, +1.08] | +0.0014 / -0.0019 / -0.0023 |
| offset/lr | prev_oo | +1.9 / +2.8 | worse | | +1.6 / +1.8 | worse | |

* **Oracle vs the open: yes.** `oracle_min` (offset) beats the opening line and passes
  Bonferroni on the 2 walk-forward seasons. Under LOSO it is negative in 3/3 seasons
  (passes uncorrected, not Bonferroni). Median oracle variant: about -2.0e-3.
* **Realistic proxy vs the open: weak.** `prev_min` (offset) is -2.05e-3 on the 2
  walk-forward seasons (passes even Bonferroni there), but under LOSO it is worse in
  2023-24 and its CI includes 0. Median realistic variant: about 0. The `gs`/`oo` versions
  show nothing. The effect is not concentrated in back-to-backs (-2.3e-3 on B2B vs
  -1.9e-3 otherwise). So the result is not explained by an opening line posted before the
  previous game was played.

### 3. How much of the open-to-close line move do absences explain? (2023-24..2025-26, n = 3,889)

Target = logit(close) - logit(open) (sd 0.29). Linear regression; LOSO R² is out of sample.

| features | R² in-sample | R² LOSO |
|---|---|---|
| oracle gs | 0.217 | **0.209** |
| oracle min | 0.133 | 0.132 |
| oracle oo | 0.093 | 0.090 |
| oracle all 3 | 0.234 | 0.225 |
| oracle gs split: carried-over / NEW absence / RETURN | 0.282 | **0.274** (coefs -0.011 / -0.028 / +0.019) |
| prev-game proxy gs | 0.004 | 0.003 |
| prev-game proxy all 3 | 0.007 | 0.002 |

The oracle absence signal explains about **21-27%** of line-move variance. In the 805
games with a large new absence (>= 5 Game Score units), the line moved against the
depleted team 79% of the time. Absences that were already visible in the previous game
explain **~0%** of the move, because the opening line has already priced them. Moves come
from *new* absences and from *returns*. That is news which, by construction, the
previous-game proxy cannot see.

### 4. Calibration in absence segments (team perspective, test seasons)

| segment | n | won | priced close | gap close (pts, z) | priced open | gap open (pts, z) | move open->close |
|---|---|---|---|---|---|---|---|
| team missing a top-2 player (oracle), opp. not | 2106 | 36.8% | 37.2% | -0.4 (z -0.4) | 39.8% | **-4.2 (z -3.8)** | -3.2 |
| team missing its top-1 player (oracle), opp. not | 1613 | 34.2% | 35.4% | -1.2 (z -1.1) | 38.4% | **-5.1 (z -4.0)** | -3.5 |
| team top-2 NEW absence (played last game) | 1187 | 41.4% | 41.4% | 0.0 (z 0.0) | 46.5% | **-5.7 (z -3.8)** | -4.5 |
| team top-2 out last game AND today | 2308 | 40.5% | 40.9% | -0.4 (z -0.4) | 41.6% | -1.6 (z -1.5) | -1.1 |
| team top-2 out last game, RETURNS today | 1190 | 50.4% | 51.0% | -0.6 (z -0.4) | 48.3% | -0.3 (z -0.2) | +1.7 |
| team missing top-2 per prev-game proxy, opp. not | 2055 | 40.5% | 41.3% | -0.8 (z -0.8) | 40.4% | -1.8 (z -1.6) | -0.3 |

(The open columns are computed on the 2023-24+ subset that has opening odds.)
The opening line overrates teams that will be missing a star by 4-6 points. The market
then moves about 3.5-4.5 points, and the closing gap is statistically zero.

### 5. Upsets (favourite by closing line lost) and a favourite missing a top-2 player

Top-2 is ranked by prior `gs` value within the rotation. Test seasons, 5,136 games with
a rotation, 1,628 upsets (31.7%).

| | oracle | prev-game proxy |
|---|---|---|
| share of ALL games where the favourite misses a top-2 player | 23.2% | 25.0% |
| share of UPSETS where the favourite missed a top-2 player | **24.3%** | 26.2% |
| share of favourite WINS where it missed a top-2 player | 22.7% | 24.5% |
| favourite missing top-2: won vs priced at close | 66.8% vs 65.8% (z +0.7) | 66.9% vs 66.9% (z 0.0) |
| underdog missing top-2: won (by the fav) vs priced at close | 71.8% vs 70.9% (z +1.0) | 71.2% vs 70.3% (z +0.8) |

Missing stars barely show up among upsets: 24.3% versus a 23.2% base rate (1.05x). The
favourite is the favourite *after* the market has accounted for the absence, and when it
loses, it loses at the priced rate.

### 6. Betting view (flat 1u at the real book odds incl. vig; bet when model - implied > thr, thr in {0, 0.02, 0.05})

* **At the closing odds:** 36 rules (6 features x 2 specs x 3 thresholds). 25 of them
  have at least 20 bets, and none of those has a CI above 0. The median ROI is about -9%.
  The best realistic rule (lr prev_oo, thr .05) is +42% on only **39 bets**, CI
  [-1%, +90%]: one lucky cell among 13 realistic rules. The control "market refit only"
  already makes -10.6% (551 bets). That comes from the intercept/slope refit (home/away
  bias learned on 2021-22), not from information.
* **At the opening odds, oracle (upper bound, not actionable):** offset oracle_min thr .02
  makes +19.3% [+6.0, +33.8] on 308 bets (WF) and +13.3% [+5.7, +21.3] on 859 bets (LOSO).
  oracle_gs thr .02 makes +13.1% [+2.6, +24.2] on 560 bets (WF).
* **At the opening odds, realistic:** the best is offset prev_min thr .02, +26% on 41 bets
  [-11%, +60%] (WF) and +6.3% [-3.5%, +16.4%] on 436 bets (LOSO). The median realistic
  rule is about -4% to -6%. Nothing is significant.

### 7. Sanity: the signal is real, the market already has it

Without the market, Elo + absence beats Elo alone (WF 2023-24..2025-26): Elo
0.6130 -> **0.6042** with oracle_min (-8.8e-3, CI [-11.7, -6.1]) and 0.6096 with prev_min
(-3.4e-3, CI [-4.8, -2.1]). The closing line on the same seasons is **0.5816**. So even
an Elo model that perfectly knows who sits stays about 0.022 log-loss behind the close.

## Verdict

* **Closing line: no edge.** 0/12 primary variants beat it. The best realistic variant is
  +0.06e-3 (CI [-0.52, +0.64]); the best oracle variant is +0.35e-3. Because the oracle is
  an upper bound on any availability information (injury reports, news), this result
  holds for *any* injury-feed product, given this valuation of players.
* **Opening line: the oracle beats it.** Who actually plays explains a large part of
  open-to-close moves, and an oracle bettor at the opening odds would have made +13-19%.
  The realistic previous-game proxy adds at most about -1 to -2e-3 log-loss, which is not
  robust and gives no significant ROI.
* The hypothesis expectation ("absences beat the open, the close prices them") is confirmed.

## Caveats

* The **oracle** uses the game's own box score. It also counts coach's-decision DNPs and
  in-game situations. It is an upper bound, not a production feature.
* **The time of the ESPN opening line is unknown.** "Beats the open" is actionable only
  if you can bet the opening price *after* the injury news and *before* the move. Testing
  that needs time-stamped injury reports (NBA PDFs, available from 2021-22) together with
  intraday odds. With only open/close snapshots it cannot be tested. For the same reason I
  did not parse the NBA injury-report PDFs: any such feature is bounded by the oracle,
  which already fails against the close.
* Opening-line tests have only 2 walk-forward seasons. LOSO uses future seasons to fit 1-2
  parameters, so it is labelled separately.
* The 2022-23 test season is trained on 2021-22 only. That season had COVID-protocol
  absences, which give the oracle coefficient its largest value. This hurts 2022-23 for
  every refit model, including the control.
* Player value is crude (Game Score, noisy on/off, minutes). A better player model (RAPM,
  EPM, LEBRON) could change the size of the vs-open effect. It cannot help vs the close,
  where even perfect knowledge of who plays, valued three different ways, adds nothing.
* There are edge cases in the rotation definition: traded players count as absent until
  they appear for their new team (capped by the 10-game window); long-term injuries drop
  out after 10 missed games; the first game of each season has no rotation (1.2% of games).
* The bootstrap is over games and ignores same-day correlation. ROI CIs are wide; many
  rules were tested, so judge the median rule, not the best one.

## Production recommendation

Do not add an availability feature in the hope of beating the closing line. It is fully
priced. For the public prediction account: if the model publishes picks **before** lines
settle, that is when the market is beatable (the opening line misses about 4-6 points on
teams that will lose a star). The only practical way to use this is to publish or bet very
early with a real-time injury feed. Proving that it works requires time-stamped odds and
injury reports, which this dataset does not have. Otherwise, use the closing market
probability as the prediction. Absence information mainly helps a non-market model
(about -0.009 log-loss with perfect knowledge, -0.003 with the previous-game proxy), and
it still stays far behind the market.
