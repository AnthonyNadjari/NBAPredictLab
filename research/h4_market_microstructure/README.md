# H4: market microstructure (are the prices themselves inefficient?)

**Question.** Setting team information aside, are there inefficiencies in the NBA moneyline prices
themselves that a bettor could exploit? Five families are tested: (a) favourite-longshot bias,
(b) cross-book disagreement and line shopping, (c) moneyline vs spread inconsistency, (d) open-to-close
line movement, (e) popular-team bias and overreaction to the last result or to streaks. Each family
is tested against the closing line and, where opening prices exist, against the opening line.

**Answer: no edge.** None of the 37 log-loss variants beats the closing line under the protocol: 0 pass
the 95% bar and 0 pass the Bonferroni bar. None of the 70 betting rules has a 95% CI above zero
after vig. The closing moneyline is calibrated across the whole price range (walk-forward Platt slope
0.97-1.03). The consensus of books is no better than the main book. The spread contains nothing
the moneyline lacks. Once you hold the close, the open-to-close move adds nothing. Popular teams are
not overpriced. Teams coming off blowouts or on streaks are priced correctly. The only real lever
found is **line shopping**: across ~10 books the best price leaves ~1-2% overround instead of
~4.3%, which makes betting favourites roughly break-even (-0.1% ROI over 3,827 bets). That cuts the
cost of betting; it is not an edge.

Run end-to-end:

```
python research/h4_market_microstructure/run.py            # first run downloads ~6.6k ESPN odds pages (~35 min, cached)
python research/h4_market_microstructure/run.py --no-fetch # reruns: about 1 minute, no network
```

## Data

* `research/data/espn_<season>.csv`: one row per ESPN event for 2021-22 to 2025-26 (regular season,
  play-in, playoffs). Main-book moneyline (open/close) and, in column `raw`, the spread and moneyline
  of every book in the feed.
* `fetch_books.py` downloads the same ESPN odds endpoint again and keeps more fields per book: spread
  juice (`spreadOdds`), total, and open/close moneyline and spread where ESPN has them. Cache:
  `research/data/h4_market_microstructure/books/<event_id>.json`, at 4 requests/second at most.
* Outcomes come from ESPN final scores. Previous-game margin and the streak entering each game come
  from `research/data/team_logs.csv`, using only games strictly before the current one.
* Main-book closing price: the explicit `close` moneyline, de-vigged multiplicatively. 2021-22 has no
  `close` field, so the post-game `current` line (DraftKings) is used instead. In later seasons that
  field equals the close in more than 99% of games.

What the feed actually contains limits what can be tested:

| season | main book | other real books | opening ML | spread juice |
|---|---|---|---|---|
| 2021-22 | DraftKings | ~10 | no | yes |
| 2022-23 | ESPN BET | ~11 | no | yes |
| 2023-24 | ESPN BET | ~9 | yes (all books) | yes |
| 2024-25 | ESPN BET | **none** | yes | yes |
| 2025-26 | DraftKings / ESPN BET | **none** | yes | yes |

So **(b) cross-book** can only be tested walk-forward on 2022-23 and 2023-24. **(d) line movement** and
every **opening-line** test can only be run on 2024-25 and 2025-26. In both cases the protocol's
"negative in 3 of 4 seasons" rule cannot be met by construction.

Data hygiene in the book table: live-odds feeds and non-bookmaker sources (consensus, teamrankings,
accuscore, betegy) are removed. **MGM is excluded**: its two sides were recorded inconsistently, with
negative vig in 33% of 2021-22 rows and vig below 2% in 72% of them. Rows with vig outside (0, 10%)
are dropped. A stored line more than 0.75 logit from the main close is treated as an in-play
snapshot and dropped (about 0.5% of rows).

## Protocol

* Walk-forward by season: fit on seasons < S, predict S, for S = 2022-23, 2023-24, 2024-25, 2025-26
  (or the subset where the data exist).
* Models are logistic regressions on `logit(market price)` plus the candidate features (C = 1e4,
  effectively unpenalised). The question is always "does X add information beyond the price?".
* Two deltas are reported. `delta` is model log-loss minus raw market log-loss (the protocol).
  `delta_vs_platt` is the same model minus a refit on `logit(price)` alone, which isolates the
  feature from the recalibration.
* Uncertainty: paired bootstrap over games (2,000 resamples) of the per-game log-loss difference
  gives the 95% CI. The Bonferroni CI uses K = number of variants tested, with a normal
  approximation on the bootstrap SE. A variant "passes" only if its 95% CI is below 0 **and** it is
  negative in at least 3 of 4 seasons.
* Betting: flat 1-unit bets at the **actual closing moneyline incl. vig** (or the opening moneyline,
  or the best price across books for (b)). Bootstrap 95% CI and a Bonferroni CI over all betting
  rules tested.

## Results

Closing-line baseline (de-vigged main book, test seasons): log-loss **0.59127** over 5,217 games.
By season: 2022-23 0.62295 (an upset-heavy season), 2023-24 0.58163, 2024-25 0.58528, 2025-26 0.57675.
The mean of the last three seasons, 0.58122, matches the 0.58162 reported in `research/upsets.py`
(the game set differs slightly).

### Every log-loss variant (model minus market, negative = better than the market)

`vs Platt` = the same model minus a walk-forward refit on `logit(price)` alone. Part `open` is
measured against the opening line. The Bonferroni CI uses K = 37.

| part | variant | n | ll_base | delta | 95% CI | Bonf. CI | neg seasons | vs Platt | 2022-23 | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a | devig_additive | 5217 | 0.59127 | -0.00012 | [-0.0008, +0.0005] | [-0.0012, +0.0010] | 3/4 |  | +0.0012 | -0.0007 | -0.0002 | -0.0006 |
| a | devig_power | 5217 | 0.59127 | +0.00010 | [-0.0010, +0.0012] | [-0.0017, +0.0019] | 3/4 |  | +0.0024 | -0.0009 | -0.0001 | -0.0009 |
| a | devig_shin | 5217 | 0.59127 | -0.00012 | [-0.0008, +0.0005] | [-0.0012, +0.0010] | 3/4 |  | +0.0012 | -0.0007 | -0.0002 | -0.0006 |
| a | platt_recalibration | 5217 | 0.59127 | +0.00068 | [+0.0001, +0.0013] | [-0.0003, +0.0017] | 2/4 |  | +0.0025 | +0.0006 | -0.0002 | -0.0000 |
| a | platt_plus_\|logit\| | 5217 | 0.59127 | +0.00061 | [-0.0002, +0.0016] | [-0.0009, +0.0021] | 2/4 |  | +0.0025 | +0.0005 | -0.0002 | -0.0002 |
| a | platt_plus_vig_interaction | 5217 | 0.59127 | +0.00170 | [+0.0002, +0.0031] | [-0.0007, +0.0041] | 0/4 | +0.00102 | +0.0052 | +0.0005 | +0.0012 | +0.0001 |
| a | isotonic_recalibration | 5217 | 0.59127 | +0.00246 | [+0.0005, +0.0046] | [-0.0009, +0.0058] | 1/4 |  | +0.0065 | +0.0032 | +0.0009 | -0.0006 |
| b | consensus_median_as_price | 2506 | 0.60186 | -0.00009 | [-0.0010, +0.0009] | [-0.0016, +0.0015] | 2/2 |  | -0.0001 | -0.0001 |  |  |
| b | outlier_book_as_price | 2506 | 0.60186 | +0.00075 | [-0.0013, +0.0029] | [-0.0027, +0.0042] | 1/2 |  | -0.0010 | +0.0025 |  |  |
| b | main+consensus_gap | 2506 | 0.60186 | +0.00147 | [+0.0002, +0.0027] | [-0.0006, +0.0035] | 0/2 | -0.00009 | +0.0023 | +0.0006 |  |  |
| b | main+outlier_dev | 2506 | 0.60186 | +0.00156 | [+0.0003, +0.0028] | [-0.0005, +0.0036] | 0/2 | +0.00001 | +0.0022 | +0.0010 |  |  |
| b | main+dispersion_shrink | 2506 | 0.60186 | +0.00168 | [+0.0005, +0.0029] | [-0.0003, +0.0036] | 0/2 | +0.00012 | +0.0025 | +0.0008 |  |  |
| b | main+consensus_gap+dispersion | 2506 | 0.60186 | +0.00159 | [+0.0002, +0.0030] | [-0.0007, +0.0038] | 0/2 | +0.00004 | +0.0023 | +0.0009 |  |  |
| c | spread_only_price (fitted map) | 5216 | 0.59112 | +0.00166 | [+0.0002, +0.0031] | [-0.0007, +0.0041] | 1/4 |  | +0.0023 | +0.0021 | -0.0001 | +0.0023 |
| c | ml+spread_raw | 5216 | 0.59112 | +0.00085 | [-0.0003, +0.0021] | [-0.0011, +0.0028] | 1/4 | +0.00017 | +0.0012 | +0.0010 | -0.0003 | +0.0015 |
| c | ml+spread_eff | 5216 | 0.59112 | +0.00098 | [-0.0000, +0.0020] | [-0.0007, +0.0027] | 1/4 | +0.00030 | +0.0014 | +0.0021 | -0.0004 | +0.0008 |
| c | ml+spread_eff+spread_x_total | 5216 | 0.59112 | +0.00123 | [+0.0001, +0.0024] | [-0.0007, +0.0031] | 1/4 | +0.00055 | +0.0024 | +0.0021 | -0.0003 | +0.0008 |
| c | ml+spread_eff_x_\|ml\| | 5216 | 0.59112 | +0.00106 | [-0.0000, +0.0022] | [-0.0008, +0.0029] | 1/4 | +0.00038 | +0.0019 | +0.0020 | -0.0001 | +0.0004 |
| d | close+move | 2644 | 0.58111 | +0.00031 | [-0.0016, +0.0021] | [-0.0028, +0.0034] | 0/2 | -0.00008 |  |  | +0.0006 | +0.0000 |
| d | close+move+steam(\|move\|>0.2) | 2644 | 0.58111 | +0.00084 | [-0.0013, +0.0029] | [-0.0026, +0.0043] | 0/2 | +0.00045 |  |  | +0.0013 | +0.0003 |
| d | close+move_toward_fav | 2644 | 0.58111 | +0.00124 | [-0.0008, +0.0033] | [-0.0021, +0.0045] | 0/2 | +0.00085 |  |  | +0.0019 | +0.0006 |
| d | close+spread_move | 2644 | 0.58111 | +0.00021 | [-0.0017, +0.0020] | [-0.0028, +0.0033] | 0/2 | -0.00018 |  |  | +0.0003 | +0.0001 |
| d | close+move+spread_move | 2644 | 0.58111 | +0.00038 | [-0.0016, +0.0023] | [-0.0028, +0.0035] | 0/2 | -0.00001 |  |  | +0.0007 | +0.0001 |
| e | pop4_diff | 5217 | 0.59127 | +0.00081 | [+0.0001, +0.0015] | [-0.0003, +0.0019] | 2/4 | +0.00013 | +0.0030 | +0.0007 | -0.0002 | -0.0001 |
| e | pop10_diff | 5217 | 0.59127 | +0.00043 | [-0.0004, +0.0013] | [-0.0009, +0.0018] | 2/4 | -0.00025 | +0.0024 | +0.0001 | -0.0006 | -0.0001 |
| e | pop10_diff+pop10_x_fav | 5217 | 0.59127 | +0.00077 | [-0.0005, +0.0020] | [-0.0013, +0.0028] | 1/4 | +0.00009 | +0.0024 | +0.0007 | +0.0003 | -0.0003 |
| e | prev_game_blowout(20+) win/loss | 5217 | 0.59127 | +0.00079 | [-0.0003, +0.0019] | [-0.0011, +0.0026] | 1/4 | +0.00011 | +0.0023 | +0.0012 | +0.0006 | -0.0008 |
| e | prev_game_margin | 5217 | 0.59127 | +0.00090 | [+0.0000, +0.0018] | [-0.0005, +0.0024] | 0/4 | +0.00022 | +0.0024 | +0.0005 | +0.0001 | +0.0007 |
| e | streak_5+_hot/cold | 5217 | 0.59127 | +0.00110 | [+0.0003, +0.0019] | [-0.0002, +0.0024] | 0/4 | +0.00042 | +0.0040 | +0.0005 | +0.0001 | +0.0000 |
| e | streak_continuous | 5217 | 0.59127 | +0.00092 | [+0.0001, +0.0018] | [-0.0004, +0.0023] | 1/4 | +0.00024 | +0.0034 | +0.0004 | -0.0002 | +0.0002 |
| open | open: platt | 2644 | 0.59222 | +0.00059 | [-0.0011, +0.0022] | [-0.0021, +0.0033] | 0/2 |  |  |  | +0.0008 | +0.0004 |
| open | open: +\|logit\| (FLB) | 2644 | 0.59222 | +0.00073 | [-0.0010, +0.0025] | [-0.0021, +0.0036] | 0/2 |  |  |  | +0.0011 | +0.0003 |
| open | open: +pop10_diff+pop10_x_fav | 2644 | 0.59222 | +0.00104 | [-0.0018, +0.0038] | [-0.0036, +0.0057] | 0/2 |  |  |  | +0.0018 | +0.0003 |
| open | open: +prev_game_blowout | 2644 | 0.59222 | +0.00059 | [-0.0011, +0.0022] | [-0.0022, +0.0033] | 0/2 |  |  |  | +0.0008 | +0.0004 |
| open | open: +streak_5+ | 2644 | 0.59222 | +0.00085 | [-0.0009, +0.0026] | [-0.0020, +0.0037] | 0/2 |  |  |  | +0.0014 | +0.0003 |
| open | open: +open_spread_eff | 2643 | 0.59197 | +0.00149 | [-0.0011, +0.0042] | [-0.0028, +0.0058] | 0/2 |  |  |  | +0.0013 | +0.0017 |
| all | ALL 4-season features together | 5216 | 0.59112 | +0.00169 | [-0.0002, +0.0036] | [-0.0015, +0.0048] | 1/4 | +0.00101 | +0.0038 | +0.0022 | +0.0013 | -0.0005 |

Best variant: Shin/additive de-vig of the closing ML, -0.00012 (95% CI [-0.0008, +0.0005]), negative in
3 of 4 seasons. It fails because the CI includes 0. (For a two-outcome market, Shin's method and
additive de-vig give the same probabilities.) The median variant is +0.00085: most variants make
the price worse out of sample, because refitting adds noise to an already calibrated number.

### (a) Favourite-longshot bias

* Calibration of the de-vigged close by favourite price band (all 5 seasons, 6,538 games): the gaps
  are -2.3 to +4.6 points with |z| < 2 everywhere, and they do not grow with the price. The
  walk-forward Platt slope is 1.03, 0.97, 1.03, 1.03 by test season (slope > 1 would mean favourites
  are underpriced).
* Recalibrating does not help: Platt +0.00068 and isotonic +0.00246 (both worse). Power de-vig
  +0.00010, Shin/additive -0.00012 (not significant).
* The classic bias is visible in ROI terms, but only weakly. Betting every team at the close in the
  <20% band returns -9.9% [-22.7%, +3.7%], against -4.4% [-6.6%, -2.2%] for >80% favourites. With
  multiplicative de-vig the longshot side carries a little more of the margin, but the win rates
  match the de-vigged prices.

| price band (all teams, all seasons) | bets | win rate | priced | ROI at close | 95% CI |
|---|---|---|---|---|---|
| 0.00-0.20 | 1130 | 14.6% | 14.7% | -9.9% | [-22.7%, +3.7%] |
| 0.20-0.35 | 2427 | 27.4% | 28.0% | -6.6% | [-12.8%, -0.5%] |
| 0.35-0.50 | 2913 | 41.5% | 42.2% | -5.5% | [-9.7%, -1.2%] |
| 0.50-0.65 | 3049 | 58.1% | 57.5% | -3.1% | [-6.2%, -0.2%] |
| 0.65-0.80 | 2427 | 72.6% | 72.0% | -3.4% | [-5.9%, -1.0%] |
| 0.80-1.00 | 1130 | 85.4% | 85.3% | -4.4% | [-6.6%, -2.2%] |

### (b) Cross-book dispersion (2021-22 to 2023-24 only; test seasons 2022-23 and 2023-24)

* About 10 real books per game. The average disagreement is 0.047 logit, about 1.2 points of
  probability.
* Log-loss over 2021-24: consensus median 0.6033, main book 0.6034, **the outlier book 0.6043 (worse)**.
  No book is sharper than the consensus on the same games: every book is within +/-0.001 of it.
  Westgate and PointsBet are the best at -0.0006 and -0.0004.
* Adding the consensus gap, the outlier book's deviation or the dispersion to the main close:
  +0.0015 to +0.0017 against the raw market, and about 0 against Platt.
* Leak check: for each book and season, a regression of the result on its deviation from the main
  close shows no predictive deviation (max |z| = 1.98 over 33 book-seasons). Nothing suggests
  in-play contamination after the filters.
* **Line shopping.** The best price across books leaves an overround of 0.9% / 1.2% / 1.8% by
  season, against 4.2% / 5.1% / 4.3% at the main book. But the best home and best away prices form
  an "arbitrage" in **17% / 13% / 8%** of games. That is implausible at a real close, so the book
  snapshots were taken at different times and some best prices are stale.
  * All favourites at the best price: **-0.1%** ROI [-2.3%, +2.2%] (3,827 bets). All underdogs: -1.3%.
  * Bet when consensus probability x best price > 1 + threshold: between -6.5% and +3.7% ROI, every
    CI about +/-10-15%. The EV these rules claim (5-13%) never shows up in the results: realised
    minus claimed is -3% to -20%. The outlier quotes are mostly stale or noise, not mispricings.
  * Best rule: EV>2% restricted to games without an arbitrage, +8.2% on 278 bets, CI [-17%, +35%].
    It comes from a single season: +22% in 2022-23, -3% in 2023-24, -4.8% in 2021-22.

### (c) Moneyline vs spread

The spread is mapped walk-forward to a win probability. The juice is folded in: the de-vigged cover
probability is converted to points at 31.3 points per unit of probability (sd 12.5). A total
interaction is optional.

* The spread-only price is worse than the ML: +0.0017.
* Adding the spread to the ML: +0.0009 to +0.0012 (worse) in every version.
* When they disagree, neither is consistently right (quintile table in `results/c_gap_quintiles.csv`).
* Betting the ML side the spread favours loses: -6.6% (|gap| > 0.1 logit, 2,058 bets) and -10.2%
  (> 0.2).

### (d) Line movement (opening lines from 2023-24; test seasons 2024-25 and 2025-26)

* The close beats the open: log-loss 0.5811 vs 0.5922 (delta -0.0111, CI [-0.0159, -0.0063]).
  Accuracy over 2023-26 is 69.0% at the close vs 68.1% at the open. The average move is 4 points of
  probability.
* Once you hold the close, the move adds nothing (+0.0003). Neither do steam nonlinearity, moves
  toward the favourite, or the spread move.
* Weak sign of under-reaction: teams the line moved toward beat their closing price by 1.0 / 1.7 /
  2.5 points for moves > 0.1 / 0.2 / 0.3 logit (z = 1.1 / 1.5 / 1.7). That is not enough to beat
  the vig: following moves > 0.3 at the close returns +1.0% [-6.3%, +8.2%]. Fading them returns
  -14%.
* **Oracle, not a strategy:** betting at the open on the side the line will later move toward
  returns **+13.3%** [+8%, +19%]. All the value is in anticipating the move before it happens (CLV),
  none in reacting to it.

### (e) Popular teams and overreaction

* Popular teams (LAL/GSW/BOS/NYK, or an approximate social-media top 10: GSW, LAL, CHI, MIA, BOS,
  CLE, HOU, SAS, OKC, NYK) are **not overpriced**. As favourites against non-popular teams, they won
  72.9% vs 70.4% priced (+2.5 points, z = 2.3, 1,612 games), which is the opposite of the
  hypothesis. That does not survive multiple testing, and backing them at the close still returns
  -0.5% (post-hoc). Fading them returns -14.4%.
* Overreaction to the last game or to a streak: no. Backing teams after a 20+ point loss returns
  -2.5%. Fading teams after a 20+ point win returns -8.8%. Fading 5+ game win streaks returns
  -11.2%, and backing 5+ game losing streaks -18.7%. All the log-loss variants are worse than the
  market (+0.0004 to +0.0011).

### Opening line as the baseline (test seasons 2024-25 and 2025-26)

No feature improves the opening price either. Platt on the open gives +0.0006. Adding |logit|,
popularity, blowouts, streaks or the opening spread gives +0.0006 to +0.0015. No rule that bets at
the opening price has a positive CI.

### Betting rules (flat 1 unit at actual odds including vig)

The `[test]` / `[all3]` / `[23-26]` tags give the seasons. Rules without fitted parameters may use
every season with data. The Bonferroni CI uses K = 70.

<details><summary>All 70 rules plus the oracle</summary>

| part | rule | bets | ROI | 95% CI | Bonf. CI | 2021-22 | 2022-23 | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|---|---|---|---|---|---|---|
| a | platt edge>0.0 | 450 | -11.6% | [-26.5%, +4.9%] | [-38.8%, +15.5%] |  | -11.0% | -13.1% |  |  |
| a | isotonic edge>0.0 | 2274 | -6.2% | [-11.5%, -0.6%] | [-15.6%, +3.1%] |  | -6.7% | -12.5% | -5.4% | +0.8% |
| a | isotonic edge>0.02 | 755 | -9.4% | [-19.2%, +0.5%] | [-26.3%, +7.6%] |  | -5.1% | -19.7% | -13.7% | +2.8% |
| a | isotonic edge>0.04 | 153 | -0.3% | [-21.5%, +24.4%] | [-40.2%, +39.6%] |  | -1.2% | -0.7% | +3.6% | +5.9% |
| a | all favourites >= 75% | 1497 | -4.4% | [-6.5%, -2.2%] | [-8.2%, -0.7%] |  | -8.3% | -1.4% | -4.5% | -4.7% |
| a | all favourites >= 85% | 473 | -1.9% | [-4.9%, +0.9%] | [-6.9%, +3.1%] |  | -11.3% | -4.2% | +0.4% | +1.6% |
| b | best price EV>0% vs consensus [test] | 1006 | +0.4% | [-10.9%, +12.3%] | [-19.4%, +20.2%] |  | +1.2% | -0.5% |  |  |
| b | best price EV>1% vs consensus [test] | 723 | +3.2% | [-10.0%, +16.6%] | [-19.9%, +26.4%] |  | +2.7% | +3.8% |  |  |
| b | best price EV>2% vs consensus [test] | 548 | +2.7% | [-13.0%, +18.4%] | [-24.4%, +29.7%] |  | +4.2% | +1.1% |  |  |
| b | best price EV>3% vs consensus [test] | 418 | +3.4% | [-14.3%, +22.0%] | [-27.3%, +34.1%] |  | +0.9% | +6.3% |  |  |
| b | best price EV>5% vs consensus [test] | 262 | -6.5% | [-27.0%, +16.0%] | [-43.8%, +30.8%] |  | -7.1% | -5.9% |  |  |
| b | best price EV>2% vs consensus, no-arb games only [test] | 278 | +8.2% | [-17.1%, +35.2%] | [-37.8%, +54.3%] |  | +22.0% | -3.0% |  |  |
| b | all favourites at best price [test] | 2506 | -0.7% | [-3.6%, +2.0%] | [-5.5%, +4.2%] |  | -0.9% | -0.5% |  |  |
| b | all underdogs at best price [test] | 2506 | -1.4% | [-7.8%, +5.2%] | [-12.4%, +9.6%] |  | +2.7% | -5.6% |  |  |
| b | best price EV>0% vs consensus [all3] | 1629 | +1.1% | [-7.8%, +9.5%] | [-13.9%, +16.1%] | +2.3% | +1.2% | -0.5% |  |  |
| b | best price EV>1% vs consensus [all3] | 1200 | +0.5% | [-9.9%, +10.4%] | [-16.8%, +17.8%] | -3.7% | +2.7% | +3.8% |  |  |
| b | best price EV>2% vs consensus [all3] | 918 | +2.8% | [-8.9%, +14.8%] | [-17.7%, +23.3%] | +3.0% | +4.2% | +1.1% |  |  |
| b | best price EV>3% vs consensus [all3] | 727 | +3.7% | [-8.9%, +17.2%] | [-18.6%, +26.1%] | +4.2% | +0.9% | +6.3% |  |  |
| b | best price EV>5% vs consensus [all3] | 483 | -1.0% | [-16.9%, +14.9%] | [-28.1%, +26.1%] | +5.6% | -7.1% | -5.9% |  |  |
| b | best price EV>2% vs consensus, no-arb games only [all3] | 400 | +4.2% | [-17.1%, +27.0%] | [-34.2%, +42.7%] | -4.8% | +22.0% | -3.0% |  |  |
| b | all favourites at best price [all3] | 3827 | -0.1% | [-2.3%, +2.2%] | [-3.9%, +3.7%] | +1.0% | -0.9% | -0.5% |  |  |
| b | all underdogs at best price [all3] | 3827 | -1.3% | [-6.4%, +3.8%] | [-10.0%, +7.4%] | -1.1% | +2.7% | -5.6% |  |  |
| b | consensus-gap model EV>0% at best price [test] | 1967 | -2.2% | [-9.3%, +5.1%] | [-14.5%, +10.0%] |  | -1.2% | -3.7% |  |  |
| b | consensus-gap model EV>2% at best price [test] | 1449 | -0.4% | [-9.6%, +8.3%] | [-16.1%, +15.3%] |  | +0.5% | -1.9% |  |  |
| c | ML+spread model edge>0.0 | 1058 | -2.7% | [-10.8%, +6.1%] | [-17.4%, +12.0%] |  | +1.2% | -6.8% | +0.4% | -1.6% |
| c | ML+spread model edge>0.02 | 175 | -15.4% | [-32.4%, +4.4%] | [-47.7%, +17.0%] |  | -10.1% | -26.2% | -100.0% |  |
| c | ML+spread model edge>0.04 | 12 | -44.2% | [-85.7%, +0.3%] | [-120.1%, +31.8%] |  | +0.2% | -71.8% | -100.0% |  |
| c | bet ML side favoured by spread, \|gap\|>0.1 logit | 2058 | -6.6% | [-11.4%, -1.4%] | [-15.1%, +1.9%] |  | -4.2% | -7.8% | -6.1% | -9.0% |
| c | bet ML side favoured by spread, \|gap\|>0.2 logit | 454 | -10.2% | [-18.7%, -1.6%] | [-25.1%, +4.7%] |  | -5.3% | -41.8% | -21.4% | -4.8% |
| c | bet ML side favoured by spread, \|gap\|>0.3 logit | 70 | -17.0% | [-38.6%, +8.0%] | [-57.6%, +23.7%] |  | -2.2% | -53.3% | -100.0% | -26.4% |
| d | close+move model edge>0.0 | 579 | -2.6% | [-9.1%, +3.6%] | [-13.5%, +8.2%] |  |  |  | -4.8% | +3.8% |
| d | close+move model edge>0.02 | 1 | -100.0% | [-100.0%, -100.0%] | [-100.0%, -100.0%] |  |  |  |  | -100.0% |
| d | follow move \|logit\|>0.1 at close [23-26] | 2559 | -2.4% | [-6.6%, +1.6%] | [-9.7%, +4.8%] |  |  | -0.7% | -6.5% | +0.1% |
| d | fade move \|logit\|>0.1 at close [23-26] | 2559 | -9.4% | [-14.4%, -4.7%] | [-17.9%, -1.0%] |  |  | -11.2% | -6.8% | -10.4% |
| d | follow move \|logit\|>0.2 at close [23-26] | 1498 | -0.7% | [-6.0%, +5.0%] | [-10.2%, +8.7%] |  |  | -2.5% | -3.6% | +5.1% |
| d | fade move \|logit\|>0.2 at close [23-26] | 1498 | -11.5% | [-18.1%, -4.6%] | [-23.1%, +0.1%] |  |  | -9.3% | -9.4% | -17.1% |
| d | follow move \|logit\|>0.3 at close [23-26] | 867 | +1.0% | [-6.3%, +8.2%] | [-11.5%, +13.5%] |  |  | -0.2% | -1.0% | +5.6% |
| d | fade move \|logit\|>0.3 at close [23-26] | 867 | -14.0% | [-22.9%, -5.0%] | [-29.7%, +1.6%] |  |  | -12.5% | -13.5% | -17.0% |
| d | ORACLE: bet at open the side the line moves to (\|move\|>0.1) | 2559 | +13.3% | [+8.0%, +18.6%] | [+4.1%, +22.5%] |  |  | +16.9% | +9.4% | +13.8% |
| e | pop10_diff+pop10_x_fav model edge>0.0 | 1307 | -3.2% | [-11.1%, +4.4%] | [-16.5%, +10.1%] |  | -5.4% | -6.6% | +0.3% | +17.1% |
| e | pop10_diff+pop10_x_fav model edge>0.02 | 366 | -12.0% | [-23.2%, +0.5%] | [-32.2%, +8.2%] |  | -7.9% | -32.7% | -19.7% | +132.5% |
| e | prev_game_blowout(20+) win/loss model edge>0.0 | 1025 | -4.2% | [-12.6%, +4.7%] | [-19.3%, +10.8%] |  | -5.2% | -3.6% | -7.3% | +5.7% |
| e | prev_game_blowout(20+) win/loss model edge>0.02 | 310 | -3.6% | [-19.6%, +12.8%] | [-31.8%, +24.6%] |  | -6.2% | -0.2% | -4.9% |  |
| e | fade popular (top10) favourite vs non-popular dog | 1612 | -14.4% | [-22.2%, -6.5%] | [-28.0%, -0.8%] |  | -5.7% | -30.4% | -14.1% | -8.0% |
| e | back popular (top10) underdog vs non-popular fav | 850 | -1.3% | [-11.1%, +8.3%] | [-17.9%, +15.2%] |  | -1.5% | -8.4% | -8.7% | +21.6% |
| e | fade LAL/GSW/BOS/NYK favourite | 931 | -7.7% | [-18.0%, +3.2%] | [-26.0%, +10.6%] |  | +9.4% | -29.1% | +1.7% | -12.0% |
| e | back LAL/GSW/BOS/NYK underdog | 400 | +0.1% | [-12.5%, +12.2%] | [-21.3%, +21.4%] |  | +2.3% | -11.4% | -6.1% | +12.8% |
| e | back team coming off a 20+ pt loss | 909 | -2.5% | [-12.3%, +7.7%] | [-19.9%, +15.0%] |  | +13.7% | -1.9% | -12.4% | -4.9% |
| e | fade team coming off a 20+ pt win | 948 | -8.8% | [-17.8%, +0.6%] | [-24.9%, +7.3%] |  | -1.7% | -0.6% | -5.0% | -25.0% |
| e | fade team on 5+ win streak | 550 | -11.2% | [-22.0%, +0.5%] | [-31.0%, +8.6%] |  | -8.0% | -2.6% | -18.3% | -14.8% |
| e | back team on 5+ losing streak | 593 | -18.7% | [-31.2%, -5.6%] | [-41.2%, +3.9%] |  | -1.6% | -20.1% | -19.5% | -27.9% |
| e | POST-HOC mirror: back popular (top10) favourite vs non-popular dog | 1612 | -0.5% | [-3.5%, +2.7%] | [-5.7%, +4.8%] |  | -0.4% | +6.0% | +0.4% | -6.1% |
| open | open: platt edge>0.0 @open | 430 | -4.5% | [-11.6%, +2.8%] | [-17.1%, +8.1%] |  |  |  | -4.2% | -16.9% |
| open | open: +\|logit\| (FLB) edge>0.0 @open | 810 | -3.8% | [-12.0%, +4.3%] | [-18.2%, +10.6%] |  |  |  | -3.8% | -3.8% |
| open | open: +pop10_diff+pop10_x_fav edge>0.0 @open | 900 | -1.9% | [-8.1%, +4.3%] | [-12.6%, +8.8%] |  |  |  | -4.3% | +2.8% |
| open | open: +pop10_diff+pop10_x_fav edge>0.02 @open | 502 | -2.0% | [-10.0%, +6.5%] | [-16.4%, +12.4%] |  |  |  | -2.2% | -1.8% |
| open | open: +prev_game_blowout edge>0.0 @open | 549 | -4.5% | [-11.8%, +2.5%] | [-16.9%, +7.8%] |  |  |  | -4.5% | -4.8% |
| open | open: +prev_game_blowout edge>0.02 @open | 18 | +9.9% | [-23.5%, +37.5%] | [-44.0%, +63.8%] |  |  |  | +9.9% |  |
| open | open: +streak_5+ edge>0.0 @open | 497 | -1.9% | [-9.6%, +5.5%] | [-15.2%, +11.5%] |  |  |  | -2.8% | +4.4% |
| open | open: +streak_5+ edge>0.02 @open | 65 | -13.8% | [-34.3%, +6.7%] | [-49.8%, +22.3%] |  |  |  | -13.8% |  |
| open | open: +open_spread_eff edge>0.0 @open | 956 | -2.5% | [-8.8%, +4.5%] | [-14.2%, +9.2%] |  |  |  | -2.3% | -2.8% |
| open | open: +open_spread_eff edge>0.02 @open | 488 | -2.2% | [-10.2%, +5.7%] | [-16.0%, +11.5%] |  |  |  | -1.4% | -4.6% |
| all | ALL-features model edge>0.0 | 2163 | -4.1% | [-9.6%, +1.3%] | [-13.4%, +5.1%] |  | -1.9% | -9.2% | -4.7% | +2.2% |
| all | ALL-features model edge>0.02 | 772 | -0.4% | [-9.2%, +8.8%] | [-15.7%, +14.9%] |  | -0.9% | -1.3% | -0.5% | +14.3% |

</details>

## Leakage checks and caveats

* **Per-book lines are "current" values read after the game.** If a book's line had been captured
  in-play, it would leak the result. Two guards: lines more than 0.75 logit from the main close are
  dropped, and a per book-season regression of the result on the deviation shows nothing
  predictive. Two failure modes remain possible: stale lines (more likely) and partially in-play
  lines (no evidence of these).
* **Asynchronous snapshots across books** (an arbitrage in 8-17% of games): the line-shopping ROIs
  are optimistic upper bounds. Those prices may never have been available at the same moment.
* 2021-22 has no explicit closing field. The post-game `current` DraftKings line is used, which
  affects training for the 2022-23 fold and the no-fit rules tagged [all3].
* The opening-line timestamp is unknown. For back-to-backs, the previous result may not have been
  known when the line opened. That would help the open-line models, and they still found nothing.
* ESPN's `close.pointSpread` field holds odds rather than points in 2022-23. It is not used: the
  spread move is measured as the main-book closing spread minus the opening spread.
* MGM is excluded because its quotes are corrupt (sides recorded at different times).
* The popular-team lists are hardcoded from general knowledge of fan bases, not from results. They
  are approximate.
* The main book changes over time: DraftKings, then ESPN BET, then mostly DraftKings again. Platt
  fits pool across books.
* Power: for four test seasons the log-loss CI half-width is about 0.0007-0.0012. ROI CIs are about
  +/-2-3% for 2,500-3,800 bets on favourites and +/-10-15% for 500-1,000 bets heavy on underdogs. A
  systematic mispricing of 1-2 points on a large segment would have been visible. Smaller ones
  could not be.

## Recommendation

Do not build a betting product on any of these price-structure signals. Use the de-vigged closing
(or latest) moneyline as the reference probability. Multiplicative de-vig is fine: Shin/additive is
only 0.0001 better, which is not significant. If the account bets or publishes picks, the practical
levers are:

1. Shop prices across books. This cuts the effective vig from ~4% to ~1-2%.
2. Judge picks by closing-line value. Taking the side the market later moves toward is worth ~13%
   ROI at the open, while reacting to the move after the fact is worth nothing.

## Files

| file | role |
|---|---|
| `run.py` | end-to-end runner (fetch -> tables -> parts a-e + open -> `results/`) |
| `fetch_books.py` | cached re-download of per-book ESPN odds (juice, totals, open/close where present) |
| `common.py` | game and book tables, walk-forward LR, paired bootstrap, Bonferroni, betting ROI |
| `a_longshot.py`, `b_books.py`, `c_ml_vs_spread.py`, `d_line_move.py`, `e_popular.py`, `o_open.py` | the hypothesis families |
| `results/logloss_variants.csv`, `results/betting_rules.csv`, `results/summary.json` | all numbers above |
| `results/a_*.csv` ... `results/d_*.csv` | calibration, ROI by band, book sharpness, leak check, best-price overround, gap quintiles, move bins |
| `research/data/h4_market_microstructure/` | cache: `books/*.json`, `games.csv`, `books.csv` (gitignored) |

