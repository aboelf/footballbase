
-- 总亚盘赔率表
CREATE TABLE IF NOT EXISTS public.total_handicap (
  id BIGSERIAL PRIMARY KEY,
  match_id BIGINT NOT NULL,
  bookmaker VARCHAR(50) NOT NULL,
  init_odds_home NUMERIC(6, 3),
  handicap VARCHAR(50),
  init_odds_away NUMERIC(6, 3),
  final_handicap VARCHAR(50),
  final_odds_home NUMERIC(6, 3),
  final_odds_away NUMERIC(6, 3),
  odds_detail JSONB,
  created_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(match_id, bookmaker)
);

CREATE INDEX IF NOT EXISTS idx_total_handicap_match_id ON total_handicap(match_id);
CREATE INDEX IF NOT EXISTS idx_total_handicap_bookmaker ON total_handicap(bookmaker);
-- ============================================================
-- 联合数据表：整合 betting_odds、league_matches、sax_encoding
-- 连接条件：match_id = schedule_id AND sax_encoding.bookmaker = betting_odds.bookmaker_name
-- ============================================================

-- 联合视图：支持多庄家的赔率与 SAX 编码关联查询
DROP TABLE IF EXISTS public.v_match_odds_sax;

CREATE TABLE public.v_match_odds_sax AS
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

-- 创建索引加速查询
CREATE INDEX IF NOT EXISTS idx_v_match_odds_sax_match_id ON v_match_odds_sax(match_id);
CREATE INDEX IF NOT EXISTS idx_v_match_odds_sax_bookmaker ON v_match_odds_sax(bookmaker_name);
CREATE INDEX IF NOT EXISTS idx_v_match_odds_sax_sax ON v_match_odds_sax(sax_interleaved, sax_delta);
CREATE INDEX IF NOT EXISTS idx_v_match_odds_sax_season ON v_match_odds_sax(season);
CREATE INDEX IF NOT EXISTS idx_v_match_odds_sax_league ON v_match_odds_sax(league_name);

-- ============================================================
-- 实体表：SAX 编码结果
-- ============================================================
CREATE TABLE IF NOT EXISTS public.sax_encoding (
  id BIGSERIAL PRIMARY KEY,
  bookmaker VARCHAR(50) NOT NULL DEFAULT 'Bet 365',
  match_id BIGINT NOT NULL,
  hometeam VARCHAR(100),
  guestteam VARCHAR(100),
  season VARCHAR(20),
  sax_interleaved VARCHAR(100),
  sax_delta VARCHAR(100),
  home_mean NUMERIC(6, 3),
  draw_mean NUMERIC(6, 3),
  away_mean NUMERIC(6, 3),
  running_odds_count INTEGER,
  created_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(bookmaker, match_id)
);

CREATE INDEX IF NOT EXISTS idx_sax_encoding_bookmaker ON sax_encoding(bookmaker);
CREATE INDEX IF NOT EXISTS idx_sax_encoding_match_id ON sax_encoding(match_id);
CREATE INDEX IF NOT EXISTS idx_sax_encoding_season ON sax_encoding(season);

-- ============================================================
-- 实体表：投注赔率
-- ============================================================
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

-- ============================================================
-- 实体表：联赛比赛
-- ============================================================
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

-- ============================================================
-- 旧版联合表（已废弃，保留兼容）
-- ============================================================
-- DROP TABLE IF EXISTS public.match_odds_sax;
-- CREATE TABLE public.match_odds_sax AS
-- SELECT ... FROM public.betting_odds bo
-- LEFT JOIN public.bet365sax bs ON bo.schedule_id = bs.match_id
-- LEFT JOIN public.league_matches lm ON bo.schedule_id = lm.match_id
-- WHERE bo.bookmaker_name = 'Bet 365';

-- ============================================================
-- 示例查询
-- ============================================================

-- 1. 查询带 SAX 编码的赔率变化（支持所有庄家）
-- SELECT * FROM v_match_odds_sax WHERE sax_interleaved IS NOT NULL LIMIT 10;

-- 2. 按 SAX 模式分组统计（特定庄家）
-- SELECT sax_interleaved, COUNT(*) as cnt, AVG(final_win) as avg_win
-- FROM v_match_odds_sax
-- WHERE bookmaker_name = 'Bet 365' AND sax_interleaved IS NOT NULL
-- GROUP BY sax_interleaved ORDER BY cnt DESC;

-- 3. 比较不同庄家的 SAX 模式分布
-- SELECT bookmaker_name, sax_interleaved, COUNT(*) as cnt
-- FROM v_match_odds_sax
-- WHERE sax_interleaved IS NOT NULL
-- GROUP BY bookmaker_name, sax_interleaved
-- ORDER BY bookmaker_name, cnt DESC;

-- 4. 查询特定 SAX 模式的后续胜率
-- WITH target_matches AS (
--   SELECT match_id, final_win, final_draw, final_lose
--   FROM v_match_odds_sax
--   WHERE sax_interleaved = 'aaabbbcc' AND bookmaker_name = 'Bet 365'
-- )
-- SELECT
--   COUNT(*) as total,
--   AVG(final_win) as avg_win_odds,
--   AVG(final_draw) as avg_draw_odds,
--   AVG(final_lose) as avg_lose_odds
-- FROM target_matches;
