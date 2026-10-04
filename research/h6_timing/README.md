# H6: timing (when should the bot publish, and how much accuracy is at stake?)

**Question.** The bot publishes the de-vigged ESPN market price (see `src/engine/model.py`). Its main run
is at 09:00 UTC, which is close to the opening line, and it refreshes at 21:00 UTC. Tip-offs are mostly
23:00-03:00 UTC. How much more accurate is the market later in the day, and when should the
prediction (and the thread) go out?

**Answer.**
* **ESPN keeps no intraday odds history.** The `history/0/movement` endpoint is empty for every
  past game we tried (0 of 319 provider feeds, 40 events, all 5 seasons). We used **Kalshi**
  instead. Its public market-data API keeps hourly and 1-minute bid/ask candles for its NBA
  game-winner markets: 1,402 games (2024-25 play-in and playoffs, all of 2025-26). Right before tip,
  Kalshi's price is as good as ESPN's closing line: log-loss 0.5799 vs 0.5805, difference -0.0006,
  95% CI [-0.0021, +0.0009].
* **The 09:00 UTC price is about as good as the opening line.** ESPN's "open" is set about 24 h
  before tip. The price then barely improves until about **17:00 UTC (1 pm ET)**, when game-day
  injury reports arrive. More than half (54%) of the price movement between 09:00 UTC and tip happens
  after 19:00 UTC, and a third (32%) after 22:00 UTC.
* **How much is at stake:** 09:00 UTC to tip is worth **0.006 log-loss** (1%), 0.0025 Brier, and
  about +1 pt of pick accuracy. The bot's pick flips in 4.8% of games, and the later pick wins 61%
  of those flips. **The existing 21:00 UTC refresh captures about 80% of that** (-0.0048 log-loss,
  95% CI [-0.0080, -0.0013]), but only for games whose thread is published *after* the refresh.
  Published games are frozen.
* **Nothing beats the close.** Kalshi adds no information beyond the ESPN closing line (8
  log-loss variants), and no betting rule at the ESPN closing moneyline makes money (6 rules, all
  ROI < 0). Timing improves calibration. It does not create an edge.

Run end-to-end:

```
python research/h6_timing/run.py              # first run downloads (cached): ~50 min Kalshi, ~25 min NBA PDFs
python research/h6_timing/run.py --no-fetch   # reruns from cache, ~3 min
```

## 1. What ESPN exposes (task item 1 and item 3)

| What | Finding |
|---|---|
| `core .../odds/{provider}/history/{0,1}/movement?limit=100` | `count: 0` for every provider on every past event probed (40 events, 8 per season, 319 provider feeds). `odds/{pid}/history` and `/history/0` return 404. No intraday history for past NBA games. Live preseason games had no odds posted yet (4 Oct 2026), so the live behaviour could not be checked. |
| Odds fields per event | `open`, `close`, `current` only. From the Kalshi match (section B2), ESPN's **open looks like the market about 24 h before tip** (mean distance 2.5 pts, the minimum is at T-24h / T-18h). The **close is the price at tip** (minimum at T-1 min, 1.0 pt). |
| Books in the feed | **One real book since 2024-25**: ESPN BET (2024-25, and 2025-26 until 1 Dec 2025), then **DraftKings** (from 2 Dec 2025). 2021-22 to 2023-24 had about 10-12 books. The bot's "consensus" (`espn.odds`) is therefore one book's price. |
| `site summary?event=ID` → `injuries` | Per-player status (Out / Day-To-Day ...), `fantasyStatus` (GTD), `returnDate`, and a timestamp (`date`) of the last note. For **past** games it returns **today's** injury list (all 40 probed events showed notes dated 2026-07..2026-10, none on or before the game). So it cannot be backtested. It is usable live, but only if the bot logs it at each run. |
| `summary` → `pickcenter` | Same single book (DraftKings), no history. |
| Polymarket (alternative intraday source) | DNS-blocked from this connection (French ANJ block); not used, and the block was not circumvented. |

The authoritative injury timeline is the **NBA official injury report**. Since about 22 Dec 2025 it
is published every 15 minutes and every version stays online
(`ak-static.cms.nba.com/referee/injury/Injury-Report_<date>_<hh>_<mm><AM|PM>.pdf`; hourly
`_<hh><AM|PM>.pdf` before that). Section 4.6 uses it.

## 2. Data

* **Kalshi series `KXNBAGAME`** (public `trade-api/v2`, no key): 2,900 team markets / 1,450 events,
  of which **1,406 matched to ESPN games** (ET date + team pair) and **1,402** have a valid pre-tip
  quote. Games: 2024-25 play-in 6 + playoffs 80; 2025-26 regular season 1,229 + play-in 6 +
  playoffs 81. For each team market: hourly candles from tip-60 h to tip, and 1-minute candles over
  the last 3 h. Settled markets live under `/historical/markets/{ticker}/candlesticks`. That endpoint
  answers 429 above about 2 requests/s, so the fetcher runs at 2/s.
* **Price at time t** = mid of best bid / best ask at the last candle closing at or before t
  (minute candles within 3 h of tip, otherwise hourly; quotes with a spread > 10 cents are discarded).
  Home-win probability = average of the home market's mid and 1 - the away market's mid. Kalshi mids
  carry no vig.
* **Snapshots**: hours before *scheduled* tip (ESPN `date_utc`): 48, 36, 24, 18, 12, 9, 6, 4, 3, 2,
  1, 0.5, 0.25 h and T-1 min ("close"). UTC clock times on the game's US date: 09:00 ... 02:00 (+1),
  and **09:00 UTC the day before** (the bot also predicts tomorrow's games at 09:00 UTC).
* **ESPN** (`research/data/espn_<season>.csv`): outcome, main-book open/close moneylines
  (multiplicative de-vig, as the bot does), actual closing American odds for the betting test.
* **NBA official injury reports**: every 4th game day of the 2025-26 regular season (42 days, 258
  games, 838 report versions). Each day gets the report at the bot's 09:00 UTC run plus 22 ET clock
  times from 08:00 to 22:00. Player minutes weights come from `research/data/h1_player_availability`
  2025-26 player logs, using only games *before* that date.

## 3. Protocol

* All comparisons are **paired on the same games**, with a paired bootstrap over games (2,000
  resamples) of the per-game log-loss difference → 95% CI. A **Bonferroni** CI is also reported
  over the number of variants in each family.
* Seasons: Kalshi exists only for 2024-25 (play-in and playoffs, 62-86 games) and 2025-26. The
  "negative in 3 of 4 seasons" rule therefore **cannot be met by construction** for anything that
  uses Kalshi. The ESPN open-vs-close comparison covers 3 seasons (2023-24 to 2025-26). For the
  "beyond the close" test we also ran a monthly expanding-window walk-forward inside the data.
* "Policy" = what the bot would publish if it ran at the listed UTC times. Each game gets the price
  from the latest run that finishes **at least 30 min before tip** (otherwise it keeps the 09:00 UTC
  price).

## 4. Results

### 4.1 Kalshi is a faithful proxy for the market (section B)

| price (n = 1,402) | log-loss | Brier | accuracy |
|---|---|---|---|
| Kalshi T-1 min | 0.57988 | 0.19904 | 68.7% |
| ESPN close (main book) | 0.58051 | 0.19908 | 68.3% |
| ESPN open (n = 1,401) | 0.58930 | 0.20283 | 67.5% |

Kalshi close minus ESPN close: **-0.00063 [-0.00218, +0.00086]**; against an ESPN close de-vigged
with the power method: +0.00000 [-0.0015, +0.0016]. Correlation of logits 0.998, mean absolute gap
0.97 pts. No game has a gap > 15 pts, so there is no sign of in-play prices leaking into the T-1
min snapshot.

### 4.2 Accuracy vs hours before tip (section C, balanced panel n = 1,338)

| snapshot | log-loss | Δ vs close [95% CI] | accuracy | pick differs from close |
|---|---|---|---|---|
| 09:00 UTC day before* | 0.5898 | +0.0131 [+0.0059, +0.0203] | 67.8% | 7.5% |
| T-24 h | 0.5873 | +0.0085 [+0.0026, +0.0140] | 67.9% | 5.8% |
| T-12 h | 0.5841 | +0.0053 [+0.0009, +0.0095] | 67.6% | 4.5% |
| T-6 h | 0.5825 | +0.0037 [+0.0001, +0.0071] | 67.7% | 3.6% |
| T-4 h | 0.5804 | +0.0016 [-0.0012, +0.0042] | 68.0% | 2.8% |
| T-2 h | 0.5802 | +0.0014 [-0.0012, +0.0038] | 67.9% | 2.3% |
| T-1 h | 0.5791 | +0.0002 [-0.0015, +0.0020] | 68.1% | 1.5% |
| T-30 min | 0.5791 | +0.0003 [-0.0008, +0.0015] | 68.3% | 0.8% |
| T-1 min (close) | 0.5788 | 0 | 68.8% | 0 |

\* all games with a quote (n = 1,276). Accuracy differences below about 1.3 pts are within binomial
noise (SE on 1,338 games ≈ 1.3 pts); log-loss and Brier are the meaningful measures.

### 4.3 Publish policies (section D, n = 1,377, Δ vs the current 09:00 UTC price)

| policy | log-loss | Δ log-loss [95% CI] | Bonferroni CI (K = 18) | share of max gain | picks changed | later pick right | net correct picks / 1,230 games |
|---|---|---|---|---|---|---|---|
| 09:00 UTC **day before** (tomorrow's games) | 0.5905 | +0.0059 [-0.0005, +0.0115] | | worse | 5.9% | 48% | -3 |
| **09:00 UTC only** | 0.5866 | 0 | | 0% | 0 | | 0 |
| 09 + 12 UTC | 0.5868 | +0.0002 [-0.0006, +0.0010] | | -4% | 0.5% | | -1 |
| 09 + 15 UTC | 0.5856 | -0.0010 [-0.0026, +0.0005] | | 17% | 1.6% | | -4 |
| 09 + 17 UTC | 0.5844 | -0.0022 [-0.0042, -0.0001] | [-0.0053, +0.0010] | 36% | 1.7% | 54% | +2 |
| 09 + 18 UTC | 0.5839 | -0.0027 [-0.0051, -0.0002] | [-0.0065, +0.0011] | 45% | 1.9% | 54% | +2 |
| 09 + 19 UTC | 0.5839 | -0.0027 [-0.0056, +0.0002] | | 45% | 2.8% | 53% | +2 |
| 09 + 20 UTC | 0.5821 | -0.0045 [-0.0077, -0.0011] | [-0.0095, +0.0006] | 75% | 3.1% | 56% | +4 |
| **09 + 21 UTC (current refresh)** | 0.5818 | **-0.0048 [-0.0080, -0.0013]** | [-0.0100, +0.0004] | **80%** | 3.5% | 56% | +5 |
| 09 + 22 UTC | 0.5818 | -0.0048 [-0.0082, -0.0010] | [-0.0102, +0.0006] | 80% | 3.3% | 52% | +2 |
| 09 + 23 UTC | 0.5825 | -0.0040 [-0.0076, -0.0004] | | 68% | 3.2% | 52% | +2 |
| 09 + 21 + 23 UTC | 0.5812 | -0.0054 [-0.0091, -0.0015] | [-0.0112, +0.0004] | 90% | 3.9% | 55% | +4 |
| hourly runs 09..02 UTC | 0.5814 | -0.0052 [-0.0092, -0.0010] | | 87% | 4.5% | 55% | +5 |
| per-game thread at T-1 h | 0.5812 | -0.0054 [-0.0092, -0.0013] | | 90% | 4.7% | 53% | +4 |
| per-game thread at T-30 min | 0.5812 | -0.0054 [-0.0096, -0.0009] | | 90% | 4.7% | 55% | +5 |
| T-1 min (upper bound, not publishable) | 0.5806 | -0.0060 [-0.0104, -0.0013] | [-0.0130, +0.0010] | 100% | 4.8% | 61% | +13 |

By season (Δ vs 09:00 UTC): 2025-26 (n = 1,315): 09+21 -0.0049, T-30 min -0.0056, close -0.0059.
2024-25 playoffs (n = 62): 09+21 -0.0031, close -0.0069. A run at 22:00 or 23:00 UTC alone is no
better than 21:00, because it starts missing the earlier tips (about 10% of games tip at or before
22:00 UTC: weekend matinees and early East Coast games). Scheduled tip hours (UTC): 23:00 15%,
00:00 33%, 01:00 21%, 02:00 13%, 03:00 8%, before 23:00 10%.

### 4.4 Sportsbook bounds: ESPN open vs close (section E, all games)

| sample | n | log-loss open | log-loss close | Δ close - open [95% CI] | accuracy open → close | picks flipped |
|---|---|---|---|---|---|---|
| 2024-25 + 2025-26 | 2,644 | 0.59222 | 0.58111 | **-0.0111 [-0.0159, -0.0062]** | 67.8% → 68.8% | 7.4% |
| 2023-24 | 1,303 | 0.59536 | 0.58192 | -0.0134 | 68.1% → 69.0% | 7.9% |
| 2024-25 | 1,323 | 0.59763 | 0.58528 | -0.0124 | 68.0% → 69.0% | 7.0% |
| 2025-26 | 1,321 | 0.58679 | 0.57694 | -0.0099 | 67.7% → 68.5% | 7.9% |

The close beats the open in **3 of 3 seasons**. On the Kalshi-matched games (E2, n = 1,376):
Kalshi at 09:00 UTC vs ESPN open = -0.0035 [-0.0079, +0.0014], so **the bot's 09:00 UTC price is
about opening-line quality**. The **09 + 21 UTC policy vs ESPN open = -0.0083 [-0.0137, -0.0022]**,
93% of the close's -0.0089.

### 4.5 When does the price move? (section F, evening tips 23:00-02:59 UTC, n ≈ 1,130)

Share of the 09:00 UTC → tip movement (mean squared logit change) still to come:

| UTC | 09:00 | 12:00 | 15:00 | 17:00 | 18:00 | 19:00 | 20:00 | 21:00 | 22:00 | 23:00 | T-1 h | T-30 min | T-15 min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| remaining | 100% | 97% | 84% | 75% | 69% | 54% | 40% | 36% | 32% | 25% | 14% | 7% | 3% |

The mean absolute move from 09:00 UTC to tip is 2.6 pts of probability; from 21:00 UTC it is 1.4 pts.

### 4.6 When does injury information arrive? (NBA official reports, 2025-26, 231 evening games)

Players counted: anyone listed Questionable / Doubtful / Probable on game day, or whose play/sit
status changes during the day (late scratches, late activations). Long-term Outs that never change
are excluded. A team marked NOT YET SUBMITTED counts as unknown. Two-way / G League assignments are
excluded.

* Per evening game, **3.9 players (85 player-minutes)** carry game-day status uncertainty. Only
  0.01 per game is still Questionable in the last report before tip.
* At the bot's 09:00 UTC run, **27% of teams have not yet submitted** their game-day report, and
  **87% of games** still have at least one unresolved status (3.0 players / 64 player-minutes per
  game).

| report time (UTC) | 09:00 | 15:00 | 17:00 | 18:00 | 19:00 | 20:00 | 21:00 | 22:00 | 23:00 | 00:00 | 01:00 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| teams not yet submitted | 27% | 27% | 22% | 13% | 8% | 4.5% | 1% | 2% | 1% | 0% | 0% |
| games with ≥ 1 unresolved status | 87% | 86% | 83% | 81% | 78% | 75% | 71% | 65% | 49% | 33% | 29% |
| share of 09:00 uncertainty still unresolved (player-minutes) | 100% | 101% | 89% | 78% | 72% | 63% | 59% | 51% | 37% | 26% | 21% |

Teams submit their game-day reports between about 17:00 and 20:00 UTC (noon-3 pm ET in winter,
1-4 pm ET in October/April). Statuses keep
resolving until about 30 min before tip. The market follows the same clock (4.5).

### 4.7 Protocol test: does Kalshi / the intraday path add anything beyond the close? (section G)

| variant (K = 8) | scheme | n | Δ vs ESPN close [95% CI] | months negative |
|---|---|---|---|---|
| LR(logit ESPN close, logit Kalshi close) | fit 2024-25 → test 2025-26 | 1,315 | +0.0216 [+0.0124, +0.0312] | 2/9 |
| same | monthly expanding | 1,016 | +0.0047 [+0.0018, +0.0076] | 2/7 |
| LR(logit ESPN close, Kalshi move 09 UTC → tip) | season | 1,315 | +0.0292 [+0.0176, +0.0413] | 3/9 |
| same | monthly | 1,016 | +0.0072 [+0.0030, +0.0112] | 1/7 |
| LR(logit Kalshi close) | season | 1,315 | +0.0225 [+0.0132, +0.0321] | 2/9 |
| same | monthly | 1,016 | +0.0026 [-0.0002, +0.0053] | 2/7 |
| Kalshi close, raw (no fit) | all games | 1,377 | -0.0006 [-0.0021, +0.0010] | 4/12 |
| mean of the two logits (no fit) | all games | 1,377 | -0.0004 [-0.0012, +0.0004] | 5/12 |

Fitted models lose (the 2024-25 training set is 62 playoff games, and the two logits are nearly
collinear). The unfitted ones are indistinguishable from the close. Per season, the raw Kalshi
close vs ESPN close is +0.0020 (2024-25, n = 62) and -0.0007 (2025-26).

Betting at the **actual ESPN closing moneyline (incl. vig)** when Kalshi's price exceeds the
implied probability by a threshold:

| signal | threshold | bets | ROI [95% CI] |
|---|---|---|---|
| Kalshi T-1 min | 0 / 2% / 4% | 46 / 1 / 0 | -27.0% [-57.5%, +8.2%] / n.a. / n.a. |
| Kalshi 09:00 UTC (stale) | 0 | 593 | -4.8% [-15.5%, +5.9%] |
| Kalshi 09:00 UTC | 2% | 274 | -4.6% [-19.0%, +10.7%] |
| Kalshi 09:00 UTC | 4% | 146 | -14.5% [-32.7%, +4.5%] |

No rule is positive. At the close, the two markets almost never disagree by more than the ~4.4% vig.

## 5. Recommendation for the bot

1. **Publish after the evening refresh, not after the 09:00 UTC run.** The 09:00 UTC price is
   opening-line quality. The 21:00 UTC refresh already exists and captures about 80% of the
   achievable gain, but only for games still unpublished at that time (published games are frozen
   by `published_game_keys`). 21:00 UTC is 22:00 Paris in winter (CET) and 23:00 in October/April
   (CEST). That is still 2 h or more before the first evening tip-off (23:00 UTC in October/April,
   00:00 UTC in winter).
2. If the thread must go out earlier for a Paris audience: 17:00-18:00 UTC (18:00-20:00 Paris)
   captures about 40% of the gain, and 20:00 UTC about 75%. Before 15:00 UTC, re-running brings
   nothing (prices barely move before the game-day injury reports).
3. **Optional second refresh at 23:00 UTC** for the late West Coast games (tips 02:00-04:00 UTC,
   about 22% of games). 09 + 21 + 23 reaches about 90% of the gain (-0.0054). Moving the refresh
   alone to 22:00 or 23:00 UTC is *not* better than 21:00, because it misses the earlier tips.
4. **Never publish tomorrow's games from the day-before 09:00 UTC run.** That price is 0.006
   log-loss worse than same-day 09:00 UTC, and 0.013 worse than the close.
5. **Expectations for the public record:** the gain is mostly calibration (log-loss / Brier). The
   pick changes in about 3.5% of games between 09:00 and 21:00 UTC, and the later pick wins about
   56% of those. That is about **+5 correct picks per 1,230-game season (+0.3-0.4 pt)**. The
   theoretical maximum (publish at tip) is about +1 pt (67.6% → 68.6% on the same games), matching
   ESPN's open → close (+0.8 to +1.1 pt per season, 3 seasons). This is within one season's noise. Timing will not make the
   account beat the market; it only closes the gap to the closing line.
6. **Logging for the future** (optional, needs a change under `src/`, not done here): store the
   ESPN odds and the `injuries` block at every run. Kalshi's public API (no key, not geo-blocked
   here) could serve as a second price source or a cross-check. On its own it is not better than
   DraftKings.

## 6. Caveats and leakage checks

* **Proxy.** Kalshi is not the DraftKings feed the bot reads. They agree at the close (4.1), and
  ESPN's open matches Kalshi at T-24 h. Between those points we assume DraftKings moves like Kalshi.
  The ESPN open → close result (3 of 3 seasons) confirms the direction and size independently.
* **One season.** Almost all Kalshi data is 2025-26 (+86 playoff games of 2024-25). The 3-of-4
  seasons rule cannot be applied. With Bonferroni over the 18 policies, every policy CI touches 0
  (09+21: [-0.0100, +0.0004]). The policies are nested and strongly correlated, so Bonferroni is
  conservative here. The monotone pattern in 4.2/4.5 and the 3-season ESPN result are the stronger
  evidence.
* **Granularity.** Snapshots more than 3 h before tip use the last hourly candle at or before the
  time (up to 59 min stale). Inside 3 h, minute candles are used. Tip = ESPN *scheduled* start; the
  actual tip is 5-15 min later, so T-1 min is pre-game (no in-play price: 0 games with a Kalshi-ESPN
  gap > 15 pts).
* **Quote filter.** Mids with a bid-ask spread > 10 cents are dropped. Coverage: 98% at 09:00 UTC,
  91% at 09:00 UTC the day before (markets open about 2-3 days ahead), 100% from T-4 h.
* **Injury reports.** Every 4th game day only (258 games). Before 22 Dec 2025 the archive has hourly
  versions only, so :30 slots are missing for those days. Names are matched to nba_api logs after
  accent/punctuation normalisation; unmatched players get weight 0. The "uncertain player"
  definition is ours. The table compares each time slot with the 09:00 UTC report on the same games.
* **Accuracy** differences under about 1.3 pts are noise. Rely on log-loss and Brier.

## 7. Files

| file | purpose |
|---|---|
| `run.py` | end-to-end runner (`--no-fetch` to reuse cache); log → `results/run_log.txt` |
| `probe_espn.py` | ESPN movement / injuries / pickcenter probe (cache `research/data/h6_timing/espn_probe.json`) |
| `fetch_kalshi.py` | Kalshi markets + candles (cache `research/data/h6_timing/kalshi/`), 2 req/s |
| `common.py` | loaders, snapshot builder, bootstrap, metrics |
| `analyze.py` | sections A-G → `results/*.csv`, `results/summary.json` |
| `injury_timeline.py` | NBA injury-report download (cache `research/data/h6_timing/injury_pdfs/`), parsing, timeline |
