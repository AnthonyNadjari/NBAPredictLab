# H7b: can the bookmakers' price improve our own model (and the reverse)?

`run.py` blends our odds-free model (ensemble preds) with the market in logit space,
`logit p = w·logit(own) + (1-w)·logit(market)`, w picked walk-forward on earlier seasons only.
`roi.py` bets 1 unit when `blend_prob × opening_decimal_odds - 1 > threshold` at the main book's real
opening moneyline (vig included). `verify_data/` and `verify_stats/` are two independent adversarial
checks (join, leakage, closing line value, executable prices).

## Verdict: no bettable edge

| | Result |
|---|---|
| Blend vs **closing** line | w(own) 0-0.2; log-loss +0.0005 [-0.0001, +0.0012]: the close already has everything |
| Blend vs **ESPN opening** line | w(own) 0.25-0.45; log-loss -0.0032 [-0.0053, -0.0010]: better than the open |
| Value bets at the ESPN open (2024-25, 2025-26) | +9.7% ROI over 1,062 bets [-0.5%, +20.6%]; always backing the opening favourite -5.1% |
| ...why | the ESPN "open" is posted ~24 h before tip, **before the previous night's games end**; our model uses those results. Two thirds of the profit comes from teams that played the night before (+20%); other games +5% [-7%, +18%] |
| At a price you can actually take after our 09:00 UTC run (Kalshi, 2025-26, fee and spread included) | about 0% ROI; blend vs the 09:00 price +0.0009 [-0.0039, +0.0054] |
| Closing line value of the bets | +2.1% [+0.7, +3.6] with multiplicative de-vig, almost all in 2024-25; 2025-26 +0.1%; vs Kalshi close -2.7% |
| Multiple testing | best of 9 thresholds: family-wise p = 0.11 |

Our model does carry information the opening line lacks (shuffling it within a month gives -5.0%,
0 of 200 shuffles reach +9.7%), but it is information the market prices overnight: by the time we
can publish, it is gone. Expected ROI on a new season at executable prices: about 0 to +2%, not
bettable, and not to be promoted as such.

## What it is good for

- **Our probability, improved by the market:** the walk-forward blend with the evening price is as
  accurate as the market (68.8% vs 69.0% on 3,942 games, log-loss difference not significant) and
  still carries our own view (w(own) ~0.1-0.2). It can be published as "our probability" without
  losing accuracy.
- **Our raw model** (no odds) is a legitimate second opinion for content (67.3%, opening-line level),
  shown next to the market, with the honest note that when the two disagree the market is right
  more often (56%).
