-- WARNING: This schema is for context only and is not meant to be run.
-- Table order and constraints may not be valid for execution.

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