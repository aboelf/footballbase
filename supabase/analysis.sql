-- Analysis: William Hill final_win = 1.44 matches

-- 1. Basic: Find matches where William Hill final_win = 1.44
SELECT
  match_id,
  league_name,
  match_time,
  home_team_name,
  away_team_name,
  final_score,
  final_win,
  final_draw,
  final_lose,
  -- Parse score to determine result
  SPLIT_PART(final_score, '-', 1)::INT as home_goals,
  SPLIT_PART(final_score, '-', 2)::INT as away_goals,
  CASE
    WHEN SPLIT_PART(final_score, '-', 1)::INT > SPLIT_PART(final_score, '-', 2)::INT THEN '主胜'
    WHEN SPLIT_PART(final_score, '-', 1)::INT < SPLIT_PART(final_score, '-', 2)::INT THEN '主负'
    ELSE '平'
  END as actual_result
FROM v_match_odds
WHERE bookmaker_name = 'William Hill'
  AND final_win = 1.44
ORDER BY match_time DESC;

-- 2. Summary: Win rate for William Hill final_win = 1.44
WITH wh_matches AS (
  SELECT
    match_id,
    final_score,
    CASE
      WHEN SPLIT_PART(final_score, '-', 1)::INT > SPLIT_PART(final_score, '-', 2)::INT THEN '主胜'
      WHEN SPLIT_PART(final_score, '-', 1)::INT < SPLIT_PART(final_score, '-', 2)::INT THEN '主负'
      ELSE '平'
    END as actual_result
  FROM v_match_odds
  WHERE bookmaker_name = 'William Hill'
    AND final_win = 1.44
)
SELECT
  actual_result,
  COUNT(*) as matches,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM wh_matches
GROUP BY actual_result
ORDER BY matches DESC;

-- 3. Range analysis: William Hill final_win around 1.44 (1.40-1.48)
WITH wh_matches AS (
  SELECT
    match_id,
    league_name,
    final_score,
    final_win,
    CASE
      WHEN SPLIT_PART(final_score, '-', 1)::INT > SPLIT_PART(final_score, '-', 2)::INT THEN '主胜'
      WHEN SPLIT_PART(final_score, '-', 1)::INT < SPLIT_PART(final_score, '-', 2)::INT THEN '主负'
      ELSE '平'
    END as actual_result
  FROM v_match_odds
  WHERE bookmaker_name = 'William Hill'
    AND final_win BETWEEN 1.40 AND 1.48
)
SELECT
  actual_result,
  COUNT(*) as matches,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage,
  ROUND(AVG(final_win), 3) as avg_odds
FROM wh_matches
GROUP BY actual_result
ORDER BY matches DESC;

-- 4. Detailed comparison: William Hill vs Bet 365 when WH final_win = 1.44
SELECT
  t1.match_id,
  t1.league_name,
  t1.match_time,
  t1.home_team_name,
  t1.away_team_name,
  t1.final_score,
  t1.final_win as wh_win,
  t1.final_draw as wh_draw,
  t1.final_lose as wh_lose,
  t2.final_win as bet365_win,
  t2.final_draw as bet365_draw,
  t2.final_lose as bet365_lose,
  CASE
    WHEN SPLIT_PART(t1.final_score, '-', 1)::INT > SPLIT_PART(t1.final_score, '-', 2)::INT THEN '主胜'
    WHEN SPLIT_PART(t1.final_score, '-', 1)::INT < SPLIT_PART(t1.final_score, '-', 2)::INT THEN '主负'
    ELSE '平'
  END as actual_result
FROM v_match_odds t1
LEFT JOIN v_match_odds t2
  ON t1.match_id = t2.match_id AND t2.bookmaker_name = 'Bet 365'
WHERE t1.bookmaker_name = 'William Hill'
  AND t1.final_win = 1.44
ORDER BY t1.match_time DESC;

-- 5. Statistical summary: Overall home win rate by William Hill odds range
SELECT
  CASE
    WHEN final_win < 1.5 THEN '1.0-1.49 (Strong Home)'
    WHEN final_win < 2.0 THEN '1.50-1.99 (Moderate Home)'
    WHEN final_win < 2.5 THEN '2.00-2.49 (Slight Home)'
    WHEN final_win < 3.0 THEN '2.50-2.99 (Neutral)'
    ELSE '3.0+ (Away可能的)'
  END as odds_range,
  COUNT(*) as total_matches,
  SUM(CASE
    WHEN SPLIT_PART(final_score, '-', 1)::INT > SPLIT_PART(final_score, '-', 2)::INT THEN 1 ELSE 0
  END) as home_wins,
  ROUND(SUM(CASE
    WHEN SPLIT_PART(final_score, '-', 1)::INT > SPLIT_PART(final_score, '-', 2)::INT THEN 1 ELSE 0
  END) * 100.0 / COUNT(*), 2) as home_win_pct,
  ROUND(AVG(final_win), 3) as avg_odds
FROM v_match_odds
WHERE bookmaker_name = 'William Hill'
  AND final_score IS NOT NULL
GROUP BY
  CASE
    WHEN final_win < 1.5 THEN '1.0-1.49 (Strong Home)'
    WHEN final_win < 2.0 THEN '1.50-1.99 (Moderate Home)'
    WHEN final_win < 2.5 THEN '2.00-2.49 (Slight Home)'
    WHEN final_win < 3.0 THEN '2.50-2.99 (Neutral)'
    ELSE '3.0+ (Away可能的)'
  END
ORDER BY odds_range;
