# H6 timing: adversarial verification notes (2026-10-04)

**Verdict: not refuted.** The claim reproduces exactly. No leakage was found. The direction ("later
price is more accurate than the 09:00 UTC / opening price, and nothing beats the close") is robust.
Several secondary numbers are less precise than the README suggests (see "Problems").

## 1. Rerun

`python research/h6_timing/run.py --no-fetch` finished in 1 min 36 s, exit 0. Every file in `results/`
(`summary.json` and all 13 CSVs) is **byte-identical** to the committed version.
`kalshi_snapshots.csv`, rebuilt from the raw candles (`--rebuild`), has the same md5.
`injury_timeline_by_game.csv` came out with a different md5 (float summation order over a Python
`set`). It does not affect any result: `injury_timeline_by_utc.csv` is identical. The original file
was restored.

I also re-implemented the 09+21 policy independently. It gives the same prices and the same delta:
-0.00480 [-0.00811, -0.00137].

## 2. Leakage checks

* **Candle timestamps.** For 300 random markets, the hourly candle close at `end_period_ts = T`
  equals the minute candle at `T` in 100% of cases. It matches the minute candle at T+59 min or T+60 min
  in only about 59% of cases. So hourly snapshots do not read one hour ahead.
* **In-play contamination of the Kalshi "close".** Kalshi `close_time` is 2.3-2.8 h after ESPN's tip
  (the end of the game), and the snapshot is taken at the scheduled tip minus 1 min. |T-1min - T-15min|
  has a median of 0.25 pts and a 99th percentile of 2 pts. Only one game moves more than 5 pts, and
  it ends near the ESPN close. Two Kalshi events have an `occurrence_datetime` that is a 20:00 UTC
  placeholder, but their prices are normal and pre-game.
* **Liquidity artefact.** The Kalshi spread at 09:00 UTC has a median of 1 c and a 90th percentile of
  2 c (it is 1-2 c at 21:00 too). The 09→21 gain is therefore not the mid getting more precise.
* **Proxy.** Kalshi at T-24h vs the ESPN open: -0.0004 [-0.0049, +0.0041]. Kalshi at T-1min vs the
  ESPN close: -0.0006. The proxy matches at both ends.
* Policy lead time of 30, 60, 90 or 120 min: 09+21 vs 09 stays between -0.0047 and -0.0048. The result
  is not sensitive to this choice.
* The betting rules use the actual ESPN closing American odds (vig included), and every signal is
  known at or before the close. All 6 rules have ROI < 0, so there is no hindsight issue that matters.
* Fitted variants (section G) train only on earlier seasons or earlier months, and all of them lose to the close.

## 3. Protocol and robustness

* The bootstrap is paired by game (`boot_diff`), as claimed.
* Kalshi covers only 2024-25 playoffs (62 games) and 2025-26. The opening delta (09+21 vs the ESPN
  open) is **+0.0072 in 2024-25** (n = 62, wrong sign) and -0.0090 in 2025-26. That is 1 of 2
  seasons, so the 3-of-4-seasons rule cannot be met, as the README itself says.
* Independent multi-season evidence. ESPN open→close is significant in **each** season taken
  separately: 2023-24 -0.0134 [-0.0216, -0.0057], 2024-25 -0.0124 [-0.0195, -0.0053],
  2025-26 -0.0099 [-0.0160, -0.0036].
* Within 2025-26, 09→09+21 is negative in 9 of 12 months. The first half of the season is
  -0.0029 [-0.0078, +0.0019] and the second half -0.0069 [-0.0121, -0.0016].

## 4. Multiple testing

The headline comparison is the bot's **existing** 21:00 UTC refresh, chosen before looking at the
data. The best-looking policy in the grid (09+18+22+00, -0.0055) is not the one recommended.

The 09+21 vs 09 result does not survive Bonferroni over K = 18 policies ([-0.0100, +0.0004]). Even
the T-1min upper bound does not survive it ([-0.0130, +0.0010]): one season is underpowered for a
Bonferroni correction. The opening delta does survive Bonferroni at K = 18 once 5 corrupt ESPN opens
are dropped ([-0.0173, -0.0010]). So does the 3-season ESPN open→close result.

## 5. Problems found (none overturns the verdict)

1. **"~80% of the achievable gain" has a bootstrap CI of [37%, 170%].** The ranking 21 vs 22 vs 21+23
   ("80% vs 90%") cannot be resolved with one season of data.
2. **"+5 correct picks per season" is ±12** (48 flips, 27 of them right). 09→close is +12.5 ±14.
   Neither is distinguishable from 0. The README does call this noise, but the summary JSON states it
   as a gain.
3. **Recommendation 4** ("the day-before price is 0.006 worse than the same-day 09:00 price"): the CI
   is [-0.0005, +0.0115] and includes 0. The day-before price vs the close (+0.0131) is significant.
4. **ESPN data errors.**
   * Three 2023-24 games have "ESPN Bet - Live Odds" as their main book, and their closes are in-play
     (0.96-0.97, home team won). This inflates the 2023-24 open→close delta.
   * Two opening lines are exact home/away swaps of the close: 401585456 and 401704632.
   * Some Kalshi-matched games have an ESPN open that looks swapped: 401810615 (open 0.29 vs Kalshi
     0.72-0.75 all day) and 401810646 (0.33 vs 0.66-0.72).

   Excluding games where the ESPN open and Kalshi T-24h differ by more than 20 pts (5 games):
   * 09+21 vs ESPN open: -0.0092 [-0.0146, -0.0037]
   * Kalshi 09:00 vs ESPN open: **-0.0045 [-0.0084, -0.0005]**. The 09:00 price is then slightly
     *better* than the open, not "opening-line quality".

   Excluding the in-play and big-jump games, ESPN open→close per season becomes 2023-24 -0.0113,
   2024-25 -0.0098 and 2025-26 -0.0105, all with CIs below 0.
5. **The opening delta flatters the bot's actual gain.** The bot's current price is the 09:00 price,
   not the ESPN open. The real gain from publishing after 21:00 is 09+21 vs 09, which is -0.0048, not
   -0.0083.
6. **Two caveats on the production recommendation.** The intraday freshness of ESPN's DraftKings feed
   is still unverified: Kalshi is only a proxy between the open and the close. GitHub cron jobs often
   start 5-30 min late. With a 30-120 min lead the delay barely matters, but a thread published right
   after "21:00" may in practice go out at about 21:30.

The verification scripts are in the session scratchpad (`verify1.py`, `verify2.py`, `verify3.py`).
None of them writes to the repo.
