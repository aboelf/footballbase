/**
 * Parse match betting odds files
 *
 * Supports:
 * - Parse all leagues
 * - Parse specific league (e.g., 德甲)
 * - Automatically iterate through all seasons
 *
 * Usage:
 *   node scripts/parse-odds.js                    # Parse all
 *   node scripts/parse-odds.js 德甲                 # Parse 德甲, all seasons
 *   node scripts/parse-odds.js 德甲 2024-2025      # Parse 德甲, specific season
 */

const fs = require('fs');
const path = require('path');

// Load environment variables from .env file
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const { createClient } = require('@supabase/supabase-js');

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_KEY;

let supabase;
if (supabaseUrl && supabaseKey) {
  supabase = createClient(supabaseUrl, supabaseKey);
} else {
  console.warn('Supabase credentials not configured, data will be logged only');
}

const LEAGUES = ['英超', '意甲', '西甲', '德甲', '法甲', '日职联'];
const SEASONS = ['2020-2021', '2021-2022', '2022-2023', '2023-2024', '2024-2025', '2021', '2022', '2023', '2024', '2025'];

// Target bookmakers
const TARGET_BOOKMAKERS = [
  'William Hill',
  'Bet 365',
  'Easybets',
  'Lottery Official',
  'Macauslot'
];

/**
 * Parse game array string into individual entries
 */
function parseGameArray(content) {
  const match = content.match(/game\s*=\s*Array\(\s*["'](.+?)["']\s*\)\s*;?/s);
  if (!match) return [];

  const gameStr = match[1];
  const entries = gameStr.split(/"\s*,\s*"/);
  return entries.map(e => e.replace(/^"|"$/g, '').trim()).filter(Boolean);
}

/**
 * Parse a single game entry
 */
function parseGameEntry(entry) {
  const parts = entry.split('|');
  if (parts.length < 20) return null;

  return {
    bookmaker_name: parts[2],
    init_win: parseFloat(parts[3]) || null,
    init_draw: parseFloat(parts[4]) || null,
    init_lose: parseFloat(parts[5]) || null,
    init_return: parseFloat(parts[9]) || null,
    final_win: parseFloat(parts[10]) || null,
    final_draw: parseFloat(parts[11]) || null,
    final_lose: parseFloat(parts[12]) || null,
    final_return: parseFloat(parts[16]) || null,
    kelly_win: parseFloat(parts[17]) || null,
    kelly_draw: parseFloat(parts[18]) || null,
    kelly_lose: parseFloat(parts[19]) || null
  };
}

/**
 * Parse a single match file
 */
function parseMatchFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf-8');
  const fileName = path.basename(filePath, '.js');
  const scheduleId = parseInt(fileName, 10);

  const gameEntries = parseGameArray(content);
  const oddsData = [];

  for (const entry of gameEntries) {
    const parsed = parseGameEntry(entry);
    if (!parsed) continue;

    if (TARGET_BOOKMAKERS.includes(parsed.bookmaker_name)) {
      oddsData.push({
        schedule_id: scheduleId,
        bookmaker_name: parsed.bookmaker_name,
        init_win: parsed.init_win,
        init_draw: parsed.init_draw,
        init_lose: parsed.init_lose,
        init_return: parsed.init_return,
        final_win: parsed.final_win,
        final_draw: parsed.final_draw,
        final_lose: parsed.final_lose,
        final_return: parsed.final_return,
        kelly_win: parsed.kelly_win,
        kelly_draw: parsed.kelly_draw,
        kelly_lose: parsed.kelly_lose
      });
    }
  }

  return oddsData;
}

/**
 * Get all match files from league/season directory
 */
function getMatchFiles(leagueDir, season) {
  const matchesDir = path.join(__dirname, '..', leagueDir, season, 'matches');

  if (!fs.existsSync(matchesDir)) {
    return [];
  }

  return fs.readdirSync(matchesDir)
    .filter(f => f.endsWith('.js'))
    .map(f => path.join(matchesDir, f));
}

/**
 * Insert odds data into Supabase
 */
async function insertOddsData(oddsData) {
  if (!supabase || oddsData.length === 0) {
    return { inserted: 0, skipped: oddsData.length };
  }

  let inserted = 0;
  let skipped = 0;
  const batchSize = 100;

  for (let i = 0; i < oddsData.length; i += batchSize) {
    const batch = oddsData.slice(i, i + batchSize);

    const { data, error } = await supabase
      .from('betting_odds')
      .upsert(
        batch.map(odd => ({
          schedule_id: odd.schedule_id,
          bookmaker_name: odd.bookmaker_name,
          init_win: odd.init_win,
          init_draw: odd.init_draw,
          init_lose: odd.init_lose,
          init_return: odd.init_return,
          final_win: odd.final_win,
          final_draw: odd.final_draw,
          final_lose: odd.final_lose,
          final_return: odd.final_return,
          kelly_win: odd.kelly_win,
          kelly_draw: odd.kelly_draw,
          kelly_lose: odd.kelly_lose
        })),
        { onConflict: 'schedule_id,bookmaker_name' }
      );

    if (error) {
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
  // Parse arguments
  let leagueArg = null;
  let seasonArg = null;

  for (let i = 2; i < process.argv.length; i++) {
    if (process.argv[i] === '--league' && process.argv[i + 1]) {
      leagueArg = process.argv[i + 1];
    }
    if (process.argv[i] === '--season' && process.argv[i + 1]) {
      seasonArg = process.argv[i + 1];
    }
  }

  // Also support positional arguments
  if (!leagueArg && process.argv[2]) {
    leagueArg = process.argv[2];
  }
  if (!seasonArg && process.argv[3]) {
    seasonArg = process.argv[3];
  }

  if (leagueArg && !LEAGUES.includes(leagueArg)) {
    console.log('Usage:');
    console.log('  node scripts/parse-odds.js                    # Parse all leagues');
    console.log('  node scripts/parse-odds.js 德甲              # Parse 德甲, all seasons');
    console.log('  node scripts/parse-odds.js 德甲 2024-2025   # Parse 德甲, specific season');
    console.log('');
    console.log('Available leagues:', LEAGUES.join(', '));
    process.exit(1);
  }

  const targetLeagues = leagueArg ? [leagueArg] : LEAGUES;
  const targetSeasons = seasonArg ? [seasonArg] : SEASONS;

  console.log('=== Odds Parser ===');
  console.log(`Leagues: ${targetLeagues.join(', ')}`);
  console.log(`Seasons: ${targetSeasons.join(', ')}`);
  console.log('');

  let totalFiles = 0;
  let totalOdds = 0;
  const byBookmaker = {};
  const byLeague = {};
  const results = [];

  // Process each league and season
  for (const league of targetLeagues) {
    const leaguePath = path.join(__dirname, '..', league);

    if (!fs.existsSync(leaguePath)) {
      console.log(`[SKIP] ${league} - Directory not found`);
      continue;
    }

    byLeague[league] = {};

    for (const season of targetSeasons) {
      const files = getMatchFiles(league, season);

      if (files.length === 0) {
        continue;
      }

      console.log(`[${league}] ${season}: ${files.length} files`);

      let seasonOdds = [];
      let seasonFiles = 0;

      for (const file of files) {
        try {
          const oddsData = parseMatchFile(file);
          seasonOdds.push(...oddsData);
          seasonFiles++;

          for (const odd of oddsData) {
            byBookmaker[odd.bookmaker_name] = (byBookmaker[odd.bookmaker_name] || 0) + 1;
          }
        } catch (err) {
          console.error(`  Error: ${path.basename(file)} - ${err.message}`);
        }
      }

      // Insert season data
      if (seasonOdds.length > 0 && supabase) {
        const insertResult = await insertOddsData(seasonOdds);
        results.push({
          league,
          season,
          files: seasonFiles,
          odds: seasonOdds.length,
          inserted: insertResult.inserted,
          skipped: insertResult.skipped
        });
      }

      totalFiles += seasonFiles;
      totalOdds += seasonOdds.length;
      byLeague[league][season] = {
        files: seasonFiles,
        odds: seasonOdds.length
      };
    }
  }

  // Summary
  console.log('\n=== Summary ===');
  console.log(`Total files processed: ${totalFiles}`);
  console.log(`Total odds records: ${totalOdds}`);

  console.log('\nBy bookmaker:');
  for (const [bm, count] of Object.entries(byBookmaker)) {
    console.log(`  ${bm}: ${count}`);
  }

  console.log('\nBy league:');
  for (const [league, seasons] of Object.entries(byLeague)) {
    const totalFiles = Object.values(seasons).reduce((a, b) => a + b.files, 0);
    const totalOdds = Object.values(seasons).reduce((a, b) => a + b.odds, 0);
    console.log(`  ${league}: ${totalFiles} files, ${totalOdds} odds`);
  }

  console.log('\nInsert results:');
  for (const r of results) {
    console.log(`  ${r.league} ${r.season}: ${r.inserted} inserted, ${r.skipped} skipped`);
  }

  // Sample data
  if (totalOdds > 0) {
    console.log('\n=== Sample Data ===');
    const sample = results
      .flatMap(r => {
        const leaguePath = path.join(__dirname, '..', r.league, r.season, 'matches');
        const files = fs.readdirSync(leaguePath).filter(f => f.endsWith('.js'));
        if (files.length > 0) {
          const sampleFile = path.join(leaguePath, files[0]);
          return parseMatchFile(sampleFile).slice(0, 2);
        }
        return [];
      })
      .slice(0, 3)
      .map(o => ({
        schedule_id: o.schedule_id,
        bookmaker: o.bookmaker_name,
        init: `${o.init_win}/${o.init_draw}/${o.init_lose}`,
        final: `${o.final_win}/${o.final_draw}/${o.final_lose}`,
        return: o.final_return
      }));
    console.log(JSON.stringify(sample, null, 2));
  }
}

// Run if called directly
if (require.main === module) {
  main().catch(console.error);
}

module.exports = {
  parseMatchFile,
  getMatchFiles,
  parseGameArray,
  parseGameEntry
};
