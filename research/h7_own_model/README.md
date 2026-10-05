# H7: our own NBA win probabilities, without any bookmaker odds (Oct 2026)

**En bref (FR)**
1. On peut publier nos propres probabilités sans aucune cote en entrée : la moyenne de trois modèles (Elo amélioré, ratings de marge, impact joueurs) fait 0,603-0,604 de log-loss et 67,2 % de bons pronostics, contre 0,615 / 65,5 % pour le modèle de secours actuel.
2. Le gain sur le secours est net et stable : -0,011 [IC 95 % -0,014 ; -0,007], meilleur sur les 4 saisons testées. Aucune fuite de données trouvée (audit du code + test « on brouille le futur et les cotes, rien ne bouge »).
3. On est au niveau de la cote d'ouverture (+0,001 [-0,004 ; +0,006]) mais toujours derrière la cote de clôture (+0,012 [+0,008 ; +0,017]), soit environ 18 bons pronostics de moins par saison.
4. L'écart restant, c'est « qui joue ce soir » (blessures de dernière minute, repos, tanking) : quand notre modèle et le marché divergent de plus de 5 points (~22 matchs par semaine), c'est le marché qui a raison en moyenne.
5. À livrer : la moyenne simple des 3 modèles, calculée chaque matin dans le workflow GitHub Actions existant à partir des box-scores ESPN (scores + lignes joueurs). Le rapport de blessures ESPN n'est pas nécessaire, mais il faut commencer à l'archiver chaque jour.

Folders: `elo_plus/`, `margin_ratings/`, `player_impact/` (one family each, own README),
`ensemble/` (audit, leak test, stacking, content view: start there). Nothing under `src/` was
changed.

## Verdict

An odds-free model is feasible and clearly better than the current stats fallback, about as
good as the **opening** line, and clearly worse than the **closing** line. Combining the three
families helps a little (each family is 0.89-0.94 correlated with the others), and equal weights
work as well as fitted ones.

## Main table (5,197 games, 2022-23..2025-26, same games for every row)

| Predictor | Log-loss | Brier | Accuracy | vs fallback, Δ log-loss [95% CI] | vs close | vs open (3,929 games) |
|---|---|---|---|---|---|---|
| **Ensemble `stack3`** (primary) | **0.6037** | **0.2085** | **67.2%** | **-0.0109 [-0.0143, -0.0073]** | +0.0122 [+0.0077, +0.0168] | +0.0010 [-0.0037, +0.0060] |
| Ensemble `avg3` (equal weights, to ship) | 0.6028 | 0.2081 | 67.3% | -0.0118 [-0.0150, -0.0084] | +0.0112 [+0.0066, +0.0159] | -0.0003 [-0.0052, +0.0045] |
| player_impact (`pregame_plus_team`) | 0.6067 | 0.2097 | 66.9% | -0.0079 [-0.0128, -0.0029] | +0.0152 [+0.0100, +0.0207] | +0.0032 [-0.0023, +0.0091] |
| elo_plus | 0.6073 | 0.2101 | 67.1% | -0.0073 [-0.0109, -0.0037] | +0.0158 [+0.0110, +0.0205] | +0.0041 [-0.0009, +0.0088] |
| margin_ratings (`margin_full_avail`) | 0.6088 | 0.2108 | 66.5% | -0.0058 [-0.0099, -0.0017] | +0.0173 [+0.0117, +0.0229] | +0.0050 [-0.0013, +0.0110] |
| Current fallback (`backtest.py` logit) | 0.6146 | 0.2134 | 65.5% | - | +0.0230 | |
| Plain Elo (`features.py`) | 0.6277 | 0.2187 | 65.0% | +0.0131 [+0.0081, +0.0177] | +0.0361 | +0.0242 |
| Market open (de-vigged, 3,929 games) | 0.5932 | 0.2038 | 68.1% | | | |
| Market close (de-vigged) | 0.5916 | 0.2033 | 68.6% | -0.0230 [-0.0286, -0.0177] | - | |

On the 3,929 games with an opening line: avg3 0.5929, stack3 0.5941, open 0.5932, close 0.5817.
`stack3` was declared the primary before the ensemble was scored; `avg3` is recommended for
production because it is simpler, has no fitted weights, and is not significantly different
(avg3 - stack3 = -0.0009 [-0.0020, +0.0002]). Details, per-season numbers, the audit and the
content view are in `ensemble/README.md` and `ensemble/results.json`.

## What drives it

1. A better team rating than the fallback's: tuned margin-of-victory Elo (home edge ~40 Elo,
   not 100; stronger regression at season start) and recency-weighted margin ratings with
   3-point luck removed.
2. Who played recently: players missing from the previous game(s), weighted by value and by how
   often such absences continue (largest single gain, about -0.006).
3. Player ratings (Kalman adjusted plus-minus), so a roster change moves the team at once.
4. Schedule: back-to-backs (~45 Elo), rest, extra playoff home edge, late-season tanking.

## Production plan

**Ship:** `avg3` = mean of the logits of elo_plus, margin_ratings and player_impact, as "our
own probability", computed every run next to the market price. Roll out in two steps: first
elo_plus alone (cheapest, most of the gain: 0.6073), then add the two others. Each family keeps
its parameters frozen for the season (re-tuned once before the season on earlier seasons only, as
in the backtest).

**Daily inputs** (all from ESPN, no odds):
- scoreboard: schedule, season type, neutral-site flag, final scores (already used);
- game summary, team box score: FGA, OREB, TOV, FTA, FG3M, FG3A (already parsed by
  `src/engine/espn.py:boxscore`);
- game summary, player box score: MIN, PTS, FGM/FGA, FTM/FTA, FG3M/FG3A, OREB/DREB, AST, STL,
  BLK, TOV, PF, +/- for every player of every finished game (same `summary` call, under
  `boxscore.players`; +/- present on a live check of 2026-10-05). **A parser is the one new piece
  of code needed.** Players who have no line count as "did not play";
- a static arena table (travel, altitude: `research/h2_travel_schedule/arenas.py`);
- no injury report. ESPN's feed only has Out / Day-To-Day and no history, so it can't be
  backtested. Log it daily from now on (`espn.game_context` already reads it); the families
  estimate it is worth at most +0.006-0.010 of log-loss.

**One-off backfill:** the research used nba_api player IDs. Production needs the player history
in ESPN IDs: about 2,700 `summary` calls for two seasons (enough for the Elo and margin states and
the 82-appearance player values), about 7,800 for the full six seasons the Kalman ratings used.
Run it once, rate-limited, in a manual workflow, and commit the result as
`data/player_games.csv`.

**Compute:** recompute all three states from the history at every run instead of storing
incremental states. That is idempotent and cannot drift. Cost, from the research scripts: Elo
pass < 1 s, margin ratings a few seconds, Kalman player ratings about 30 s; under 2 minutes in
total on a GitHub runner. Pre-season re-tuning (Elo L-BFGS, margin grid) takes about 10 minutes
in a separate `workflow_dispatch` job that commits a `params.json`.

**Where it plugs in** (`.github/workflows/daily_predictions.yml`, runs at 09:00 / 21:00 /
23:00 UTC): after `refresh_history`, add a step `update player_games.csv from ESPN summaries`,
then a new `src/engine/own_model.py` that reads `games_history.csv` + `player_games.csv` +
`params.json` and writes `own_prob` for today's games. `model.final_probability` keeps returning
the market price unless the owner chooses to publish `own_prob`. In either case, store both and
show both on the panel. The current logit fallback is replaced by `own_prob` where no market
price exists.

**Risks**
- It is less accurate than the market we publish today: about 18 fewer correct picks per
  season and +0.012 log-loss. If we publish our own numbers, the thread should say so.
- Optimism: components and family "bests" were picked after seeing the test seasons. On the one
  untouched season (2021-22, margin family) the gain over the fallback was about 60% of the
  test-season gain. Expect about -0.007 rather than -0.011 against the fallback.
- Data plumbing: a broken or renamed player box-score field silently removes the player terms
  (fall back to team-only: margin team-only is 0.6107, still better than today). Alert when a
  finished game has no player lines.
- Season openers and trades: a player's move is invisible until his first game for the new team.
  The first two weeks are the weakest period.
- ESPN ID backfill: a wrong player-ID map would corrupt the player ratings. Validate it by
  re-scoring 2025-26 from the ESPN-sourced history against this backtest.
