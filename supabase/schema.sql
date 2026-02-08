-- Unified League Matches Table Schema
-- Supports: Premier League, Serie A, La Liga, Bundesliga, Ligue 1
-- Run this in your Supabase SQL Editor

-- Create unified matches table
CREATE TABLE IF NOT EXISTS league_matches (
  id BIGSERIAL PRIMARY KEY,
  match_id BIGINT NOT NULL,
  league_id INTEGER,
  league_name VARCHAR(50),
  season VARCHAR(10),
  round INTEGER,
  match_time TIMESTAMP,
  home_team_id INTEGER,
  home_team_name VARCHAR(100),
  away_team_id INTEGER,
  away_team_name VARCHAR(100),
  final_score VARCHAR(10),
  half_score VARCHAR(10),
  created_at TIMESTAMP DEFAULT NOW(),
  UNIQUE(match_id, league_id)
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_league_matches_league_season
  ON league_matches(league_id, season);
CREATE INDEX IF NOT EXISTS idx_league_matches_round
  ON league_matches(league_id, season, round);
CREATE INDEX IF NOT EXISTS idx_league_matches_match_time
  ON league_matches(match_time);

-- Example queries:

-- Get all matches for Premier League (league_id: 36) in a season
-- SELECT * FROM league_matches WHERE league_id = 36 AND season = '2024-2025' ORDER BY round, match_time;

-- Get all matches for Serie A (league_id: 34) in a season
-- SELECT * FROM league_matches WHERE league_id = 34 AND season = '2024-2025' ORDER BY round, match_time;

-- Get match results by team across all leagues
-- SELECT * FROM league_matches WHERE home_team_name = '阿森纳' OR away_team_name = '阿森纳' ORDER BY match_time DESC;

-- Get season statistics by league
-- SELECT league_name, season, COUNT(*) as matches FROM league_matches GROUP BY league_name, season ORDER BY season DESC;

-- List available leagues
-- SELECT DISTINCT league_id, league_name FROM league_matches ORDER BY league_name;
