#!/usr/bin/env node
/**
 * Export all bookmaker odds for matches where William Hill final_win = 1.44
 *
 * Step 1: Find all match_ids where WH final_win = 1.44
 * Step 2: Get all odds for those matches from v_match_odds
 * Step 3: Export to CSV/JSON
 */

const { createClient } = require('@supabase/supabase-js');
require('dotenv').config();

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_KEY;

if (!supabaseUrl || !supabaseKey) {
  console.error('Error: SUPABASE_URL and SUPABASE_KEY must be set in .env');
  process.exit(1);
}

const supabase = createClient(supabaseUrl, supabaseKey);

async function exportMatches(options = {}) {
  const { format = 'json', limit = 10000 } = options;

  console.log('Step 1: Finding matches where William Hill final_win = 1.44...\n');

  // Get match_ids where WH final_win = 1.44
  const { data: whMatches, error: whError } = await supabase
    .from('v_match_odds')
    .select('match_id')
    .eq('bookmaker_name', 'William Hill')
    .eq('final_win', 1.44);

  if (whError) {
    console.error('Error fetching WH matches:', whError);
    process.exit(1);
  }

  // Extract unique match_ids
  const matchIds = [...new Set(whMatches.map(m => m.match_id))];
  console.log(`Found ${matchIds.length} unique matches with WH final_win = 1.44`);

  if (matchIds.length === 0) {
    console.log('No matches found.');
    return;
  }

  console.log('\nStep 2: Fetching all odds for these matches...\n');

  // Fetch all odds for these matches
  const { data: allOdds, error: oddsError } = await supabase
    .from('v_match_odds')
    .select('*')
    .in('match_id', matchIds.slice(0, limit))
    .order('match_id')
    .order('bookmaker_name');

  if (oddsError) {
    console.error('Error fetching odds:', oddsError);
    process.exit(1);
  }

  console.log(`Fetched ${allOdds.length} odds records from ${allOdds.length / 5} bookmakers per match`);

  // Group by match for summary
  const matchesGrouped = {};
  allOdds.forEach(odd => {
    if (!matchesGrouped[odd.match_id]) {
      matchesGrouped[odd.match_id] = {
        match_info: {
          match_id: odd.match_id,
          league_name: odd.league_name,
          match_time: odd.match_time,
          home_team: odd.home_team_name,
          away_team: odd.away_team_name,
          final_score: odd.final_score
        },
        bookmakers: {}
      };
    }
    matchesGrouped[odd.match_id].bookmakers[odd.bookmaker_name] = {
      init_win: odd.init_win,
      init_draw: odd.init_draw,
      init_lose: odd.init_lose,
      final_win: odd.final_win,
      final_draw: odd.final_draw,
      final_lose: odd.final_lose
    };
  });

  console.log('\nStep 3: Exporting data...\n');

  if (format === 'csv') {
    // Export as CSV
    const headers = [
      'match_id', 'league_name', 'match_time', 'home_team', 'away_team', 'final_score',
      'bookmaker', 'init_win', 'init_draw', 'init_lose', 'final_win', 'final_draw', 'final_lose'
    ];
    console.log(headers.join(','));

    Object.values(matchesGrouped).forEach(match => {
      Object.entries(match.bookmakers).forEach(([bookmaker, odds]) => {
        const row = [
          match.match_info.match_id,
          match.match_info.league_name,
          match.match_info.match_time,
          `"${match.match_info.home_team}"`,
          `"${match.match_info.away_team}"`,
          match.match_info.final_score,
          bookmaker,
          odds.init_win,
          odds.init_draw,
          odds.init_lose,
          odds.final_win,
          odds.final_draw,
          odds.final_lose
        ];
        console.log(row.join(','));
      });
    });
  } else {
    // Export as JSON
    const exportData = {
      exported_at: new Date().toISOString(),
      filter: {
        bookmaker: 'William Hill',
        final_win: 1.44
      },
      summary: {
        total_matches: matchIds.length,
        total_odds_records: allOdds.length,
        bookmakers: [...new Set(allOdds.map(o => o.bookmaker_name))]
      },
      matches: Object.values(matchesGrouped)
    };
    console.log(JSON.stringify(exportData, null, 2));
  }

  console.log(`\nExport complete! ${matchIds.length} matches exported.`);
}

async function main() {
  const args = process.argv.slice(2);
  const format = args.includes('--csv') ? 'csv' : 'json';

  await exportMatches({ format });
}

main().catch(console.error);
