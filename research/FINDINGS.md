# Can anything beat the betting market? (Oct 2026)

Question from the owner: the closing-line favourite loses ~32% of NBA games; what explains
those misses, and can the bot predict them?

Data: 5,197 games 2021-22..2025-26 (regular season, play-in, playoffs) with ESPN open/close
moneylines, leak-free team features, player logs, schedules, Kalshi intraday prices.
Protocol for every test: walk-forward by season (train on earlier seasons only), the market
probability as an input, paired bootstrap CI on log-loss, >= 3 of 4 seasons better, multiple
testing accounted for, ROI at real odds with vig. Each hypothesis was then re-run and
attacked by an independent reviewer (`verify_notes.md` in each folder).

## 1. The 32% is priced, not missed (`upsets.py`)
Favourites won 68.3% when the closing price said 67.9%. Calibration holds in every price band.
Median price of a beaten favourite: 62%. 35% of upsets were decided by 5 points or fewer.

## 2. Six hypotheses, all verified

| | Hypothesis | Variants | Beats closing line? | Notes |
|---|---|---|---|---|
| H1 | Player availability (who sits) | 18 | No, even with perfect knowledge | Absences explain 21-27% of open-to-close moves: books price them by tip-off |
| H2 | Travel, time zones, altitude, schedule density | 35 | No | Density predicts results on its own (z=4.1) but is in the price |
| H3 | Motivation: standings, tanking, playoff series state | 33 | No | "Playoff favourites overpriced" shrinks to noise on independent seasons |
| H4 | Price structure: longshot bias, book dispersion, spread vs ML, line moves, popular teams | 37 | No | Line shopping cuts vig from ~4.3% to 1-2%: a cost saving, not an edge |
| H5 | ML on everything (LightGBM from the market, interactions, margin model) | 18 | No | Median variant is slightly worse than the market |
| H6 | Timing: how accurate is the price at 09:00 vs later | 8 | No (nothing beats the close) | **Actionable:** 09:00 UTC = opening-line quality; publishing after the 21:00 UTC refresh captures ~80% of the gain (~+5 correct picks/season), 21:00+23:00 ~90% |

No betting rule had a positive ROI with a confidence interval above zero after vig.

## 3. What changed in production
- Published probability = de-vigged market consensus (stats model only as a fallback).
- Refreshes at 21:00 and 23:00 UTC; the panel shows when each probability was last updated
  and advises publishing after the evening refresh. Published games are frozen.
- Thread content uses these facts honestly: "the risk" tweet quotes the real upset rate at
  the price.

Each folder has a README (method, tables, caveats), `run.py` to reproduce, and the reviewer's
notes. Raw downloads go to `research/data/` (not versioned; the scripts re-download them).
