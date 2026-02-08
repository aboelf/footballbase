CREATE TABLE match_odds AS
  SELECT
    lm.match_id,
    lm.league_name,
    lm.match_time,
    lm.home_team_name,
    lm.away_team_name,
    lm.final_score,
    CASE 
      WHEN SPLIT_PART(lm.final_score, '-', 1)::INT > SPLIT_PART(lm.final_score, '-', 2)::INT THEN '主胜'
      WHEN SPLIT_PART(lm.final_score, '-', 1)::INT < SPLIT_PART(lm.final_score, '-', 2)::INT THEN '客胜'
      ELSE '平局'
    END AS result,
    bo.bookmaker_name,
    bo.init_win,
    bo.init_draw,
    bo.init_lose,
    bo.init_return,
    bo.final_win,
    bo.final_draw,
    bo.final_lose,
    bo.final_return,
    bo.kelly_win,
    bo.kelly_draw,
    bo.kelly_lose
  FROM league_matches lm
  INNER JOIN betting_odds bo ON lm.match_id = bo.schedule_id
  ORDER BY lm.match_time DESC, bo.bookmaker_name;