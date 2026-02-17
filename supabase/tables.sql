
-- 总亚盘赔率表
create table public.total_handicap (
  id bigserial not null,
  match_id bigint not null,
  bookmaker character varying(50) not null,
  init_odds_home numeric(6, 3) null,
  handicap character varying(50) null,
  init_odds_away numeric(6, 3) null,
  final_handicap character varying(50) null,
  final_odds_home numeric(6, 3) null,
  final_odds_away numeric(6, 3) null,
  odds_detail jsonb null,
  created_at timestamp without time zone null default now(),
  constraint total_handicap_pkey primary key (id),
  constraint total_handicap_match_id_bookmaker_key unique (match_id, bookmaker)
) TABLESPACE pg_default;

create index IF not exists idx_total_handicap_match_id on public.total_handicap using btree (match_id) TABLESPACE pg_default;

create index IF not exists idx_total_handicap_bookmaker on public.total_handicap using btree (bookmaker) TABLESPACE pg_default;

-- ============================================================
-- 实体表：SAX 编码结果
-- ============================================================
create table public.sax_encoding (
  id bigserial not null,
  bookmaker character varying(50) not null,
  match_id bigint not null,
  hometeam character varying(100) null,
  guestteam character varying(100) null,
  season character varying(20) null,
  sax_interleaved character varying(100) null,
  sax_delta character varying(100) null,
  home_mean numeric(6, 3) null,
  draw_mean numeric(6, 3) null,
  away_mean numeric(6, 3) null,
  running_odds_count integer null,
  created_at timestamp without time zone null default now(),
  constraint sax_encoding_pkey primary key (id),
  constraint sax_encoding_bookmaker_match_id_key unique (bookmaker, match_id)
) TABLESPACE pg_default;

create index IF not exists idx_sax_encoding_bookmaker on public.sax_encoding using btree (bookmaker) TABLESPACE pg_default;

create index IF not exists idx_sax_encoding_match_id on public.sax_encoding using btree (match_id) TABLESPACE pg_default;

create index IF not exists idx_sax_encoding_season on public.sax_encoding using btree (season) TABLESPACE pg_default;

-- ============================================================
-- 实体表：投注赔率
-- ============================================================
create table public.betting_odds (
  id bigserial not null,
  schedule_id bigint not null,
  bookmaker_name character varying(50) not null,
  init_win numeric(5, 2) null,
  init_draw numeric(5, 2) null,
  init_lose numeric(5, 2) null,
  init_return numeric(5, 2) null,
  final_win numeric(5, 2) null,
  final_draw numeric(5, 2) null,
  final_lose numeric(5, 2) null,
  final_return numeric(5, 2) null,
  kelly_win numeric(5, 2) null,
  kelly_draw numeric(5, 2) null,
  kelly_lose numeric(5, 2) null,
  created_at timestamp without time zone null default now(),
  constraint betting_odds_pkey primary key (id),
  constraint betting_odds_schedule_id_bookmaker_name_key unique (schedule_id, bookmaker_name)
) TABLESPACE pg_default;

create index IF not exists idx_betting_odds_schedule on public.betting_odds using btree (schedule_id) TABLESPACE pg_default;

create index IF not exists idx_betting_odds_bookmaker on public.betting_odds using btree (bookmaker_name) TABLESPACE pg_default;

-- ============================================================
-- 实体表：联赛比赛
-- ============================================================
create table public.league_matches (
  id bigserial not null,
  match_id bigint not null,
  league_id integer null,
  league_name character varying(50) null,
  season character varying(10) null,
  round integer null,
  match_time timestamp without time zone null,
  home_team_id integer null,
  home_team_name character varying(100) null,
  away_team_id integer null,
  away_team_name character varying(100) null,
  final_score character varying(10) null,
  half_score character varying(10) null,
  created_at timestamp without time zone null default now(),
  constraint league_matches_pkey primary key (match_id),
  constraint league_matches_match_id_league_id_key unique (match_id, league_id)
) TABLESPACE pg_default;

create index IF not exists idx_league_matches_league_season on public.league_matches using btree (league_id, season) TABLESPACE pg_default;

create index IF not exists idx_league_matches_round on public.league_matches using btree (league_id, season, round) TABLESPACE pg_default;

create index IF not exists idx_league_matches_match_time on public.league_matches using btree (match_time) TABLESPACE pg_default;
