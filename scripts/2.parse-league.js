/**
 * Parse league data files (s36.js, s34_2948.js, etc.) and store to Supabase
 *
 * Supported files:
 * - Premier League: s36.js (league_id: 36)
 * - Serie A: s34_2948.js (league_id: 34)
 */

const fs = require('fs');
const path = require('path');

// Load environment variables from .env file
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const { createClient } = require('@supabase/supabase-js');

// Supabase configuration from .env
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_KEY;

let supabase;
if (supabaseUrl && supabaseKey) {
  supabase = createClient(supabaseUrl, supabaseKey);
} else {
  console.warn('Supabase credentials not configured, data will be logged only');
}

/**
 * Extract arrTeam array from file content
 */
function extractTeamArray(content) {
  const match = content.match(/var arrTeam\s*=\s*(\[.+?\]);/s);
  if (!match) {
    throw new Error('Cannot find arrTeam in file');
  }
  return eval(match[1]);
}

/**
 * Extract all rounds from file content
 */
function extractRounds(content) {
  const rounds = {};
  const roundRegex = /jh\s*\[\s*["']R_(\d+)["']\s*\]\s*=\s*(\[.+?\]);/gs;
  let match;

  while ((match = roundRegex.exec(content)) !== null) {
    const roundNum = parseInt(match[1], 10);
    const matches = eval(match[2]);
    rounds[`R_${roundNum}`] = matches;
  }

  return rounds;
}

/**
 * Build team ID to name mapping
 */
function buildTeamMap(arrTeam) {
  const teamMap = new Map();
  for (const team of arrTeam) {
    const [id, nameCN] = team;
    teamMap.set(id, nameCN);
  }
  return teamMap;
}

/**
 * Extract league info from file
 */
function extractLeagueInfo(content) {
  const leagueMatch = content.match(/var arrLeague\s*=\s*\[(\d+),['"]([^'"]+)['"]/);
  if (leagueMatch) {
    return {
      leagueId: parseInt(leagueMatch[1], 10),
      leagueName: leagueMatch[2]
    };
  }
  return { leagueId: null, leagueName: null };
}

/**
 * Parse a single data file
 */
function parseFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf-8');

  // Extract season from path - support both "2021" and "2021-2022" formats
  const seasonMatch = filePath.match(/(\d{4})(?:-\d{4})?[\\/]/);
  let season = null;
  if (seasonMatch) {
    season = seasonMatch[1]; // Keep as single year: "2021"
  }

  // Extract league info
  const leagueInfo = extractLeagueInfo(content);

  // Parse data
  const arrTeam = extractTeamArray(content);
  const rounds = extractRounds(content);
  const teamMap = buildTeamMap(arrTeam);

  // Collect all matches
  const matches = [];

  for (const [roundKey, roundMatches] of Object.entries(rounds)) {
    const roundNum = parseInt(roundKey.replace('R_', ''), 10);

    for (const match of roundMatches) {
      const [
        matchId,
        leagueId,
        unknown,
        matchTime,
        homeTeamId,
        awayTeamId,
        finalScore,
        halfScore,
        ...rest
      ] = match;

      const homeTeamName = teamMap.get(homeTeamId) || `Unknown (${homeTeamId})`;
      const awayTeamName = teamMap.get(awayTeamId) || `Unknown (${awayTeamId})`;

      matches.push({
        match_id: matchId,
        league_id: leagueId || leagueInfo.leagueId,
        league_name: leagueInfo.leagueName,
        season,
        round: roundNum,
        match_time: matchTime,
        home_team_id: homeTeamId,
        home_team_name: homeTeamName,
        away_team_id: awayTeamId,
        away_team_name: awayTeamName,
        final_score: finalScore,
        half_score: halfScore
      });
    }
  }

  return matches;
}

/**
 * Get all data file paths (s36.js, s34_2948.js, etc.)
 */
function getDataFiles(baseDir, pattern = null) {
  const results = [];

  function walkDir(dir) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);

      if (entry.isDirectory()) {
        walkDir(fullPath);
      } else if (entry.isFile()) {
        // Match specific pattern or default patterns
        if (pattern) {
          if (entry.name.includes(pattern) || entry.name.match(new RegExp(pattern))) {
            results.push(fullPath);
          }
        } else {
          // Default: match common data files
          if (entry.name.match(/^s\d+.*\.js$/) || entry.name.match(/^s\d+_\d+\.js$/)) {
            results.push(fullPath);
          }
        }
      }
    }
  }

  walkDir(baseDir);
  return results;
}

/**
 * Create table if not exists
 */
async function ensureTableExists() {
  if (!supabase) return;

  const createTableSQL = `
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

    CREATE INDEX IF NOT EXISTS idx_league_matches_league_season
      ON league_matches(league_id, season);
    CREATE INDEX IF NOT EXISTS idx_league_matches_round
      ON league_matches(league_id, season, round);
  `;

  // Note: Using raw query instead of RPC since exec_sql may not be available
  console.log('Please run supabase/schema.sql in your Supabase SQL Editor to create tables');
}

/**
 * Insert matches into Supabase
 */
async function insertMatches(matches) {
  if (!supabase) {
    console.log('Supabase not configured, skipping insert');
    return { inserted: 0, skipped: 0 };
  }

  let inserted = 0;
  let skipped = 0;

  // Process in batches
  const batchSize = 100;
  for (let i = 0; i < matches.length; i += batchSize) {
    const batch = matches.slice(i, i + batchSize);

    const { data, error } = await supabase
      .from('league_matches')
      .upsert(
        batch.map(m => ({
          match_id: m.match_id,
          league_id: m.league_id,
          league_name: m.league_name,
          season: m.season,
          round: m.round,
          match_time: m.match_time,
          home_team_id: m.home_team_id,
          home_team_name: m.home_team_name,
          away_team_id: m.away_team_id,
          away_team_name: m.away_team_name,
          final_score: m.final_score,
          half_score: m.half_score
        })),
        { onConflict: 'match_id,league_id' }
      );

    if (error) {
      console.error('Insert error:', error.message);
      skipped += batch.length;
    } else {
      inserted += data?.length || batch.length;
    }
  }

  return { inserted, skipped };
}

/**
 * Main function
 */
async function main() {
  const targetDir = process.argv[2] || __dirname;
  const pattern = process.argv[3] || null;

  console.log(`Scanning for data files in: ${targetDir}`);
  if (pattern) {
    console.log(`Pattern: ${pattern}`);
  }

  const files = getDataFiles(targetDir, pattern);
  console.log(`Found ${files.length} data files`);

  if (files.length === 0) {
    console.log('No data files found');
    return;
  }

  // Parse all files
  const allMatches = [];
  const byLeague = {};

  for (const file of files) {
    console.log(`Parsing: ${file}`);
    try {
      const matches = parseFile(file);
      console.log(`  - Found ${matches.length} matches`);

      // Group by league for summary
      for (const match of matches) {
        const key = match.league_name || match.league_id || 'Unknown';
        byLeague[key] = (byLeague[key] || 0) + 1;
      }

      allMatches.push(...matches);
    } catch (error) {
      console.error(`  - Error: ${error.message}`);
    }
  }

  // Summary by season
  const bySeason = {};
  for (const match of allMatches) {
    if (match.season) {
      const key = `${match.league_name || 'Unknown'} ${match.season}`;
      bySeason[key] = (bySeason[key] || 0) + 1;
    }
  }

  console.log('\n=== Summary ===');
  console.log(`Total matches: ${allMatches.length}`);
  console.log('By league/season:');
  for (const [key, count] of Object.entries(bySeason)) {
    console.log(`  ${key}: ${count} matches`);
  }

  // Show sample data
  console.log('\n=== Sample Data ===');
  console.log(JSON.stringify(allMatches.slice(0, 2), null, 2));

  // Insert to Supabase if configured
  if (supabase) {
    console.log('\nInserting to Supabase...');
    const result = await insertMatches(allMatches);
    console.log(`Inserted: ${result.inserted}, Skipped: ${result.skipped}`);
  } else {
    console.log('\nSupabase not configured. Please configure .env file:');
    console.log('  SUPABASE_URL=https://your-project.supabase.co');
    console.log('  SUPABASE_KEY=your-anon-key');
    console.log('\nSee .env.example for reference');
  }
}

// Run if called directly
if (require.main === module) {
  main().catch(console.error);
}

module.exports = {
  parseFile,
  getDataFiles,
  extractTeamArray,
  extractRounds,
  buildTeamMap
};
