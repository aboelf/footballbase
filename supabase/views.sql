CREATE VIEW v_match_odds AS
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

CREATE VIEW public.v_match_odds_sax AS
SELECT
  bo.schedule_id AS match_id,
  bo.bookmaker_name,
  lm.league_name,
  lm.match_time,
  lm.home_team_name,
  lm.away_team_name,
  lm.final_score,
  bo.init_win,
  bo.init_draw,
  bo.init_lose,
  bo.init_return,
  bo.final_win,
  bo.final_draw,
  bo.final_lose,
  bo.final_return,
  bs.sax_interleaved,
  bs.sax_delta,
  bs.home_mean,
  bs.draw_mean,
  bs.away_mean,
  lm.season,
  lm.round
FROM public.betting_odds bo
INNER JOIN public.sax_encoding bs
  ON bo.schedule_id = bs.match_id
  AND bo.bookmaker_name = bs.bookmaker
LEFT JOIN public.league_matches lm
  ON bo.schedule_id = lm.match_id;

CREATE VIEW public.v_match_odds_sax_handicap AS
SELECT
  bo.schedule_id AS match_id,
  bo.bookmaker_name,
  lm.league_name,
  lm.match_time,
  lm.home_team_name,
  lm.away_team_name,
  lm.final_score,
  bo.init_win,
  bo.init_draw,
  bo.init_lose,
  bo.init_return,
  bo.final_win,
  bo.final_draw,
  bo.final_lose,
  bo.final_return,
  bs.sax_interleaved,
  bs.sax_delta,
  bs.home_mean,
  bs.draw_mean,
  bs.away_mean,
  lm.season,
  lm.round,
  th.handicap AS init_handicap,
  th.init_odds_home,
  th.init_odds_away,
  th.final_handicap,
  th.final_odds_home,
  th.final_odds_away,
  th.odds_detail
FROM public.betting_odds bo
INNER JOIN public.sax_encoding bs
  ON bo.schedule_id = bs.match_id
  AND bo.bookmaker_name = bs.bookmaker
LEFT JOIN public.league_matches lm
  ON bo.schedule_id = lm.match_id
LEFT JOIN public.total_handicap th
  ON bo.schedule_id = th.match_id
  AND bo.bookmaker_name = CASE
    WHEN th.bookmaker = 'bet365' THEN 'Bet 365'
    WHEN th.bookmaker = 'easybets' THEN 'Easybets'
    ELSE th.bookmaker
  END