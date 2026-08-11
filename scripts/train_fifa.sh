#!/usr/bin/env bash
#
# Reproducible training run for the FIFA World Cup 2026 player-performance dataset.
#
# Task: predict a player's `position` (Goalkeeper / Defender / Midfielder / Forward)
# from a single match's performance record.
#
# Three things in this invocation are load-bearing:
#
#   --group-col player_id
#       The grain is player-match: 1,248 players contribute ~44 rows each. A plain
#       stratified split puts the same player on both sides, and the player-constant
#       columns then act as an identity fingerprint. Measured: `preferred_foot`
#       scores 0.846 under a random row split but 0.712 under a grouped split —
#       below its 0.745 majority baseline. The random-split number was memorisation.
#
#   --max-rows 15000
#       SVC scales O(n^2)-O(n^3) and `probability=True` adds internal 5-fold Platt
#       calibration. The RBF fit, not the data, is the runtime constraint.
#
#   --drop ...
#       Three separate categories, listed individually below.
#
set -euo pipefail
cd "$(dirname "$0")/.."

python model/train.py \
  --data data/fifa_world_cup_2026_player_performance.csv \
  --target position \
  --group-col player_id \
  --max-rows 15000 \
  --test-size 0.25 \
  --max-test-rows 2000 \
  --extra-models svm \
  `# Counts, not labels. The default heuristic sends any integer column with <=10
   # distinct values to the one-hot branch, which is right for an integer-coded
   # month but wrong here: 3 tackles really is more than 1, and discarding that
   # ordering costs kNN and GaussianNB in particular.` \
  --force-numeric \
    goals assists shots_on_target key_passes successful_dribbles crosses \
    successful_crosses tackles interceptions blocks aerial_duels_won \
    aerial_duels_lost fouls_committed fouls_suffered offsides saves punches \
    goals_conceded \
  --drop \
    `# 1. Identity / metadata — no predictive content, and several uniquely
     # fingerprint a player or a fixture.` \
    player_name match_id match_date jersey_number club_name \
    team nationality opponent_team stadium city market_value_eur \
    \
    `# 2. Tournament-level rollups — these aggregate the whole competition,
     # including matches that fall in the test split. Using them to predict a
     # single match is target leakage across time.` \
    total_goals_tournament total_assists_tournament total_minutes_tournament \
    player_of_match_awards tournament_rating \
    \
    `# 3. Team-match outcome — properties of the fixture, not the player. Verified
     # to carry no signal: RF on match_result scores 0.380 vs a 0.369 majority
     # baseline, i.e. the generator sampled these independently of everything else.` \
    match_result goals_team goals_opponent tournament_stage

echo
echo "Done. Next: review reports/metrics.md, then 'streamlit run app.py'."
