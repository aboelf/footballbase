-- Betting Odds Table Schema
-- Stores odds data from various bookmakers for league matches
-- Run this in your Supabase SQL Editor

-- Create betting odds table (with composite primary key)
CREATE TABLE IF NOT EXISTS betting_odds (
  schedule_id BIGINT NOT NULL,
  bookmaker_name VARCHAR(50) NOT NULL,
  -- Initial odds
  init_win DECIMAL(5,2),
  init_draw DECIMAL(5,2),
  init_lose DECIMAL(5,2),
  init_return DECIMAL(5,2),
  -- Final odds
  final_win DECIMAL(5,2),
  final_draw DECIMAL(5,2),
  final_lose DECIMAL(5,2),
  final_return DECIMAL(5,2),
  -- Kelly indices
  kelly_win DECIMAL(5,2),
  kelly_draw DECIMAL(5,2),
  kelly_lose DECIMAL(5,2),
  created_at TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (schedule_id, bookmaker_name)
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_betting_odds_bookmaker
  ON betting_odds(bookmaker_name);

-- Example queries:

-- Get all odds for a specific match
-- SELECT * FROM betting_odds WHERE schedule_id = 1903804 ORDER BY bookmaker_name;

-- Get odds history for a bookmaker
-- SELECT * FROM betting_odds WHERE bookmaker_name = 'Bet 365' ORDER BY schedule_id;

-- Compare odds across bookmakers for a match
-- SELECT bookmaker_name, final_win, final_draw, final_lose, final_return
-- FROM betting_odds WHERE schedule_id = 1903804 ORDER BY final_return;

-- Calculate average odds by bookmaker
-- SELECT bookmaker_name,
--        AVG(final_win) as avg_win,
--        AVG(final_draw) as avg_draw,
--        AVG(final_lose) as avg_lose,
--        COUNT(*) as matches
-- FROM betting_odds GROUP BY bookmaker_name ORDER BY matches DESC;

-- Count odds per bookmaker
-- SELECT bookmaker_name, COUNT(*) as count FROM betting_odds GROUP BY bookmaker_name ORDER BY count DESC;
