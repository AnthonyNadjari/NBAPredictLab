# H7 / ensemble: stacking the three odds-free families (Oct 2026)

```
python research/h7_own_model/ensemble/run.py          # ~15 s: reads the 3 families' preds.csv
python research/h7_own_model/ensemble/run.py --rerun  # re-runs the 3 family scripts first (~6 min)
python research/h7_own_model/ensemble/leak_test.py    # black-box leak test of the 3 families (~8 min)
```

Outputs: `preds.csv` (primary variant `stack3`, 5,197 test games, pre-game P(home win)),
`results.json` (metrics per season + pooled, bootstrap CIs, stack weights, content view),
`leak_test_result.txt`.

## 1. Audit of the three families (before trusting any number)

**Code read line by line** (`player_impact/run.py`, `margin_ratings/run.py`, `elo_plus/run.py`,
plus the H1 builder `h1_player_availability/build.py` that margin_ratings reads and
`research/features.py` that everyone reads):

| Check | player_impact | margin_ratings | elo_plus |
|---|---|---|---|
| Ratings use only earlier dates | Kalman state saved before each date is absorbed; box/PM rates by `merge_asof(allow_exact_matches=False)` | a date's games are predicted, then added to the ridge | Elo moves only with a team's own games; a team plays once a day |
| Same-day games | whole day absorbed together after prediction | same | same |
| Expected lineup / absences | previous 10 team games only (`j-WIN..j-1`) | H1 `absent_prev` = no row in the team's previous game; values `merge_asof` strictly before | availability rows computed before the date is ingested |
| Season-level aggregates | replacement level from 2020-21 only; P(plays) table fit before 2022-10-01 | league 3P% = cumulative over earlier dates | replacement level from 2020-21; P(absent) table from seasons < S |
| Hyper-parameters | Kalman settings picked on 2021-22; ridge+logit refit monthly on earlier games | grid, sigma, schedule/late/availability coefficients on seasons < S | L-BFGS tuning on seasons < S |
| Odds as input | no (upsets_dataset only for the eval set) | no (`mkt/spread/book` dropped at load) | no (ESPN odds merged only to score) |
| Oracle variant reported as best? | no: `oracle*` kept as upper bounds, best = `pregame_plus_team` | n/a | no: `oracle_bound` separate, best = `elo_plus` |

`research/data/games_features_odds.csv` (fallback features, Elo, rest, games played, win%) was
re-derived from `team_logs.csv` with `research/features.py`: identical to 1e-14.

**Black-box test** (`leak_test.py`): every box score, player line and outcome dated on or after
2024-01-15 is replaced by another game's (15% of player rows also deleted, games features rebuilt
from the scrambled logs, H1 rotation table and elo_plus cache rebuilt), and **every odds column in
every file is replaced by random numbers**. Each family's own `run.py` is re-run untouched on that
copy (only its data path changes). Result (`leak_test_result.txt`):

| Family | Games dated <= D: max change | Games after D: mean change | |
|---|---|---|---|
| margin_ratings | 0.0 (bitwise identical) | 0.150 | pass |
| elo_plus | 0.0 (bitwise identical) | 0.162 | pass |
| player_impact | 3.1e-4 | 0.149 | pass, numerical noise |

player_impact's 3e-4 was chased down. Its Kalman ratings, player rates, P(plays) table and team
features are bitwise identical before D. Its player composites differ by at most 4e-14, from
summation order. The same script re-run on the *unscrambled* data, with
`games_features_odds.csv` merely re-derived (values equal to 1e-14), already moves its
probabilities by up to 5e-4. So the 3e-4 is optimiser-tolerance noise from the monthly
logistic refits, not information. A clean re-run of player_impact reproduces its published
`preds.csv` exactly. margin_ratings and elo_plus reproduce exactly on every game before D.

**Verdict: no leak found.** What remains is *selection* bias, not leakage: each family chose
some components and its reported "best" variant after seeing test-season scores (all three say
so). Within each family the top variants are 0.001-0.002 apart, and margin_ratings' untouched 2021-22
season shows a smaller gain over the fallback (-0.0035 vs -0.0058 on the test seasons). Expect
the real edge over the fallback on a new season to be roughly 2/3 of what is reported here.

## 2. Ensemble

Inputs: the three deployable bests (player_impact `pregame_plus_team`, margin_ratings
`margin_full_avail`, elo_plus `elo_plus`). Their logits correlate 0.89-0.94 with each other.

* `avg3`: mean of the three logits. Nothing fitted.
* `stack3` (**primary, declared before scoring**): logistic regression on the three logits,
  refitted at the start of each test season on the earlier test seasons' out-of-sample
  predictions (2022-23 has none, so it uses avg3).
* `stack3_monthly`: same, refitted on the 1st of each month on all earlier games.
* `stack4`: stack3 plus the fallback logit as a 4th input.

## 3. Results (5,197 games 2022-23..2025-26; market open: 3,929 games 2023-24..2025-26)

| Predictor | Log-loss | Brier | Accuracy | Log-loss, open subset |
|---|---|---|---|---|
| **stack3 (primary)** | **0.6037** | **0.2085** | **67.2%** | 0.5941 |
| avg3 (no fitted weights) | 0.6028 | 0.2081 | 67.3% | 0.5929 |
| stack3_monthly | 0.6042 | 0.2087 | 67.4% | 0.5936 |
| stack4 | 0.6038 | 0.2086 | 67.2% | 0.5943 |
| player_impact | 0.6067 | 0.2097 | 66.9% | 0.5964 |
| elo_plus | 0.6073 | 0.2101 | 67.1% | 0.5972 |
| margin_ratings | 0.6088 | 0.2108 | 66.5% | 0.5982 |
| margin team-only | 0.6107 | 0.2117 | 65.6% | 0.6009 |
| current fallback (backtest.py logit) | 0.6146 | 0.2134 | 65.5% | 0.6045 |
| plain Elo (features.py) | 0.6277 | 0.2187 | 65.0% | 0.6174 |
| market open (de-vigged) | n/a | n/a | n/a | 0.5932 (68.1%) |
| market close (de-vigged) | 0.5916 | 0.2033 | 68.6% | 0.5817 |

Per season (log-loss): stack3 0.6338 / 0.6021 / 0.5940 / 0.5865; fallback 0.6464 / 0.6095 /
0.6063 / 0.5977; close 0.6228 / 0.5830 / 0.5856 / 0.5763; open - / 0.5955 / 0.5978 / 0.5863.

Paired bootstrap, mean log-loss difference, 95% CI (over games; over game-days is the same to
±0.0003):

| | vs fallback | vs market close | vs market open |
|---|---|---|---|
| stack3 | **-0.0109 [-0.0143, -0.0073]**, 4/4 seasons | +0.0122 [+0.0077, +0.0168], worse 4/4 | +0.0010 [-0.0037, +0.0060] |
| avg3 | -0.0118 [-0.0150, -0.0084] | +0.0112 [+0.0066, +0.0159] | -0.0003 [-0.0052, +0.0045] |
| player_impact | -0.0079 [-0.0128, -0.0029] | +0.0152 [+0.0100, +0.0207] | +0.0032 [-0.0023, +0.0091] |
| elo_plus | -0.0073 [-0.0109, -0.0037] | +0.0158 [+0.0110, +0.0205] | +0.0041 [-0.0009, +0.0088] |
| margin_ratings | -0.0058 [-0.0099, -0.0017] | +0.0173 [+0.0117, +0.0229] | +0.0050 [-0.0013, +0.0110] |

stack3 beats each family alone: vs player_impact -0.0030 [-0.0057, -0.0003], vs elo_plus
-0.0036 [-0.0061, -0.0011], vs margin_ratings -0.0051 [-0.0083, -0.0018]. The fitted stack does
not beat the plain average (avg3 - stack3 = -0.0009 [-0.0020, +0.0002]): with three predictors
this correlated, equal weights are as good as learned ones. Fitted weights drift toward
player_impact (0.47 / 0.35 / 0.23 for 2025-26) with intercept ~0 (calibrated inputs).

Where the gap to the close sits (log-loss stack3 / fallback / close): Oct-Dec 0.616 / 0.623 /
0.608; Jan-Feb 0.616 / 0.624 / 0.599; Mar-Apr 0.560 / 0.580 / 0.543; play-in + playoffs 0.646 /
0.653 / 0.648 (level with the close on 356 playoff games). The rest of the gap is who plays
tonight (injury news, load management, tanking) that the market prices by tip-off.

## 4. Content view: when we disagree with the closing line

Regular season, ~48 games a week:

* |stack3 - close| > 5 points of probability: **2,413 of 5,197 games (46%), ~22 per week**.
  > 10 points: 876 games (17%), ~8 per week.
* Who is right? **The market, clearly.** When we rate the home team higher (1,343 games) we say
  53.7%, the close says 43.6%, home teams won 43.3%. When we rate it lower (1,070 games) we say
  54.9%, the close 65.0%, home teams won 63.3%. The realised rate sits on the market's number in
  both directions; log-loss on these games 0.634 (us) vs 0.610 (close).
* Different winner picked: 580 games, ~5 per week. The market's pick won 325 (56%), ours 255 (44%).

Honest thread line: "Our model (no odds) and the betting market disagree by 5+ points on about
1 game in 2. Over 4 seasons, when they disagree, the market has been right more often." Our
divergences mostly reflect information the market has and we don't (late injury news), not an
edge. Not betting advice.

## 5. Caveats

* Selection bias (section 1): the families' components and bests were chosen after seeing test
  scores; the ensemble primary was declared before scoring it, but after reading the families'
  pooled numbers. Expect ~2/3 of the gain over the fallback on a new season.
* 2022-23 has no earlier out-of-sample season, so stack3 = avg3 there.
* Market open exists from 2023-24 only; "level with the open" is on 3,929 games, CI ±0.005.
* Season openers are the weakest point of every family: off-season trades are invisible until a
  player's first game for his new team.
