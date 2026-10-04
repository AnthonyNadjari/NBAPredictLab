# H1 adversarial verification (2026-10-04)

Scripts and their raw outputs are in `verify/`. They import `build.py` and `run.py` and do not
write anything in the repo except `verify/perm_null_open_wf.csv`. Run them from the repo root:
`python research/h1_player_availability/verify/<script>.py`.

## Bottom line

* **Reproduction: OK.** `run.py --rebuild` reproduces every number. `h1_games.csv` is
  byte-identical; result tables differ by at most 1.6e-8. The B2B split quoted in the README is
  not computed by `run.py`, but `verify/b2b_checks.py` reproduces it: -2.32e-3 on B2B vs -1.94e-3 otherwise.
* **Closing line: no edge. Confirmed.** 0/12 primary variants beat it. Day-clustered bootstrap
  CIs barely change. Changing the rotation definition (min 10/15/20, window 5/10/20) still gives
  no closing-line pass.
* **The `beats_opening_only` label is not supported for any usable signal. Corrected verdict: no_edge.**
  * The oracle beats the open only through look-ahead. It uses the game's own box score, which is
    information revealed after the open (late scratches, rest days, coach's DNPs). Beating a
    price with information revealed after that price is not an edge.
  * The realistic proxy (`offset prev_min`) does beat the open on the 2 walk-forward seasons.
    A permutation test confirms it: family-wise p < 1/300 over the 6 realistic variants. But:
    1. It is **not specific to the open.** If the closing-line model uses the same training
       window as the opening model (seasons >= 2023-24), the same feature also beats the CLOSE on
       the same 2 seasons: -1.11e-3 [-1.91, -0.35] vs -2.05e-3 vs the open. The README contrasts
       "beats open, not close". Part of that gap comes from the training windows: the closing-line
       walk-forward also trains on 2021-22 and 2022-23, where the coefficient has the opposite sign.
    2. It is **season-dependent.** In-season offset z-scores of `prev_min` vs the close are
       -0.9, +1.6, -0.1, -2.1, -2.9 for 2021-22..2025-26. Vs the open they are -0.8, -2.3, -3.1
       for 2023-24..2025-26. Under LOSO, 2023-24 is worse than the market in 4 of 5 rotation
       definitions. It fails the ">= 3 seasons negative" rule.
    3. The sibling variants disagree: `prev_gs` is about 0, and `prev_oo` is significantly
       *worse* than the open (+1.9e-3, CI above 0). No opening-odds ROI rule is significant.
    4. The opening-line timestamp is unknown, so the result cannot be acted on in any case.

## Checks

1. **Leakage (`verify/leak_truncation.py`).** I truncated the player logs at 4 cutoff dates D,
   keeping only rows dated before D, and rebuilt all features. For every game before D, all
   features are identical (max diff 6e-14). For games on D, all `*_prev`, `n_rot` and
   `rot_gs_total` features are identical (max diff 1e-14), and only the oracle features change.
   So the realistic features use only information from before the game. Rotation, player value,
   the traded-player filter (merge_asof, strict) and the replacement rate (2020-21 only) are
   clean. Standardization uses training-set statistics only. Both specs include the market as
   an input.
2. **Protocol.** Walk-forward by season (`season < S`) is correct. The bootstrap is paired
   per game. Day-clustered bootstrap CIs are almost identical (`verify/stats_out.txt`).
   The opening line has only 2 walk-forward seasons, so ">= 3 of 4 seasons" cannot be met there.
3. **Multiple testing.** 18 feature x spec variants x 3 setups gives 54 evaluations, plus 108
   ROI rules, 7 line-move sets and 6 segments. The README reports this honestly. Nothing passes
   against the close, so correction does not matter there. The opening-line passes survive a
   permutation family-wise test. They fail on robustness and timing, not on chance.
4. **ROI.** It uses the real ESPN American odds with vig, and the bet rule is fixed in advance
   (model minus implied probability > thr). The numbers match `roi.csv`: 0/25 closing rules with
   >= 20 bets have a CI above 0. At opening odds, the significant rules are oracle-only (look-ahead).

## Lead worth a pre-registered test (not an edge)

In 2024-25 and 2025-26, "rotation minutes missing in the previous game" was underpriced by
both the open and the close. Vs the close, b is about -0.14 to -0.20 per standard deviation.
It holds in every phase of the regular season (Oct-Dec, Jan-Feb, Mar-Apr; z about -1.9 to -2.1
in each, see `split_checks.py`). In 2022-23 and 2023-24 it was zero or reversed. The minutes
missing per team-game also rose from about 38 to 47 in 2024-25 and 2025-26. If this is a
regime change (for example in rest or availability rules), the honest test is to freeze
`offset prev_min` with b fitted on 2024-25 and 2025-26 and score it on 2026-27 against the
close, with no other variant.
