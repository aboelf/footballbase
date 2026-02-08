-- WARNING: This schema is for context only and is not meant to be run.
-- Table order and constraints may not be valid for execution.

-- Bet 365 SAX 编码结果表
CREATE TABLE IF NOT EXISTS public.bet365sax (
  id BIGSERIAL PRIMARY KEY,
  match_id BIGINT NOT NULL,
  sax_interleaved VARCHAR(50),
  sax_delta VARCHAR(50),
  home_mean NUMERIC(6, 3),
  draw_mean NUMERIC(6, 3),
  away_mean NUMERIC(6, 3),
  created_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(match_id)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_bet365sax_match_id ON bet365sax(match_id);

-- 可以与 league_matches 表关联查询
-- 示例: SELECT l.*, b.sax_interleaved, b.sax_delta FROM league_matches l
--       LEFT JOIN bet365sax b ON l.match_id = b.match_id WHERE l.season = '2024-2025';

CREATE TABLE public.betting_odds (
  id bigint NOT NULL DEFAULT nextval('betting_odds_id_seq'::regclass),
  schedule_id bigint NOT NULL,
  bookmaker_name character varying NOT NULL,
  init_win numeric,
  init_draw numeric,
  init_lose numeric,
  init_return numeric,
  final_win numeric,
  final_draw numeric,
  final_lose numeric,
  final_return numeric,
  kelly_win numeric,
  kelly_draw numeric,
  kelly_lose numeric,
  created_at timestamp without time zone DEFAULT now(),
  CONSTRAINT betting_odds_pkey PRIMARY KEY (id)
);
CREATE TABLE public.league_matches (
  id bigint NOT NULL DEFAULT nextval('league_matches_id_seq'::regclass),
  match_id bigint NOT NULL,
  league_id integer,
  league_name character varying,
  season character varying,
  round integer,
  match_time timestamp without time zone,
  home_team_id integer,
  home_team_name character varying,
  away_team_id integer,
  away_team_name character varying,
  final_score character varying,
  half_score character varying,
  created_at timestamp without time zone DEFAULT now(),
  CONSTRAINT league_matches_pkey PRIMARY KEY (match_id)
);

-- 联合数据表：从 betting_odds、bet365sax、league_matches 聚合数据
-- 执行此 SQL 会创建表并填充数据
DROP TABLE IF EXISTS public.match_odds_sax;

CREATE TABLE public.match_odds_sax AS
SELECT
  bo.schedule_id AS match_id,
  bo.bookmaker_name,
  bo.init_win, bo.init_draw, bo.init_lose, bo.init_return,
  bo.final_win, bo.final_draw, bo.final_lose, bo.final_return,
  bo.kelly_win, bo.kelly_draw, bo.kelly_lose,
  bs.sax_interleaved,
  bs.sax_delta,
  bs.home_mean,
  bs.draw_mean,
  bs.away_mean,
  lm.league_name,
  lm.match_time,
  lm.home_team_name,
  lm.away_team_name,
  lm.final_score,
  lm.season,
  lm.round,
  lm.half_score,
  bo.created_at AS odds_created_at,
  bs.created_at AS sax_created_at
FROM public.betting_odds bo
LEFT JOIN public.bet365sax bs ON bo.schedule_id = bs.match_id
LEFT JOIN public.league_matches lm ON bo.schedule_id = lm.match_id
WHERE bo.bookmaker_name = 'Bet 365';

-- 添加主键和索引
ALTER TABLE public.match_odds_sax ADD PRIMARY KEY (match_id);

CREATE INDEX IF NOT EXISTS idx_match_odds_sax_sax ON match_odds_sax(sax_interleaved, sax_delta);
CREATE INDEX IF NOT EXISTS idx_match_odds_sax_season ON match_odds_sax(season);
CREATE INDEX IF NOT EXISTS idx_match_odds_sax_match_time ON match_odds_sax(match_time);

-- 示例查询
-- 1. 查询带 SAX 编码的赔率变化
-- SELECT * FROM v_match_odds_sax WHERE sax_interleaved IS NOT NULL LIMIT 10;

-- 2. 按 SAX 模式分组统计
-- SELECT sax_interleaved, COUNT(*) as cnt, AVG(final_win) as avg_win
-- FROM v_match_odds_sax WHERE sax_interleaved IS NOT NULL
-- GROUP BY sax_interleaved ORDER BY cnt DESC;

-- 3. 查询特定 SAX 模式的后续胜率
-- WITH target_matches AS (
--   SELECT match_id, final_win, final_draw, final_lose
--   FROM v_match_odds_sax
--   WHERE sax_interleaved = 'aaabbbcc'
-- )
-- SELECT
--   COUNT(*) as total,
--   SUM(CASE WHEN final_result = 'win' THEN 1 ELSE 0 END) as wins,
--   SUM(CASE WHEN final_result = 'draw' THEN 1 ELSE 0 END) as draws,
--   SUM(CASE WHEN final_result = 'lose' THEN 1 ELSE 0 END) as losses
-- FROM target_matches;
