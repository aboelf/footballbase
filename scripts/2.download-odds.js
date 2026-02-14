/**
 * Download match odds files and parse them
 *
 * Usage:
 *   node scripts/download-odds.js <league_dir> <season>
 *   node scripts/download-odds.js 意甲 2024-2025
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const { pipeline } = require('stream/promises');

// Load environment variables
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const { createClient } = require('@supabase/supabase-js');

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_KEY;

let supabase;
if (supabaseUrl && supabaseKey) {
  supabase = createClient(supabaseUrl, supabaseKey);
} else {
  console.warn('Supabase credentials not configured');
}

const BASE_URL = 'https://1x2d.titan007.com';
const QUERY_PARAM = '?r=007134148238774303818';

// Target bookmakers
const TARGET_BOOKMAKERS = [
  'William Hill',
  'Bet 365',
  'Easybets',
  'Lottery Official',
  'Macauslot'
];

// Cache directory
const CACHE_DIR = path.join(__dirname, '..');

/**
 * Extract match IDs from sXX.js file
 */
function extractMatchIds(filePath) {
  const content = fs.readFileSync(filePath, 'utf-8');
  const matchIds = [];

  // Match jh["R_N"] = [[id, ...], [id, ...], ...]
  const roundRegex = /jh\s*\[\s*["']R_(\d+)["']\s*\]\s*=\s*(\[.+?\]);/gs;
  let match;

  while ((match = roundRegex.exec(content)) !== null) {
    const roundMatches = eval(match[2]);
    for (const m of roundMatches) {
      const matchId = m[0];
      if (matchId && typeof matchId === 'number') {
        matchIds.push(matchId);
      }
    }
  }

  return matchIds;
}

/**
 * Download a file
 */
function downloadFile(url, destPath) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(destPath);

    https.get(url, (response) => {
      if (response.statusCode === 302 || response.statusCode === 301) {
        // Handle redirect
        downloadFile(response.headers.location, destPath)
          .then(resolve)
          .catch(reject);
        return;
      }

      if (response.statusCode !== 200) {
        reject(new Error(`HTTP ${response.statusCode}`));
        return;
      }

      pipeline(response, file)
        .then(() => resolve())
        .catch(reject);
    }).on('error', reject);
  });
}

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
    bookmaker_id: parts[0],
    some_id: parts[1],
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
 * Extract match metadata from file
 */
function extractMeta(content) {
  const meta = {};

  const scheduleMatch = content.match(/ScheduleID\s*=\s*(\d+)/);
  if (scheduleMatch) meta.ScheduleID = parseInt(scheduleMatch[1], 10);

  return meta;
}

/**
 * Parse a downloaded match file
 */
function parseMatchFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf-8');
  const meta = extractMeta(content);
  const gameEntries = parseGameArray(content);

  const oddsData = [];

  for (const entry of gameEntries) {
    const parsed = parseGameEntry(entry);
    if (!parsed) continue;

    if (TARGET_BOOKMAKERS.includes(parsed.bookmaker_name)) {
      oddsData.push({
        schedule_id: meta.ScheduleID,
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
 * Insert odds data into Supabase
 */
async function insertOdds(oddsData) {
  if (!supabase || oddsData.length === 0) return { inserted: 0, skipped: oddsData.length };

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
  // Parse arguments: node script.js <league_dir> <season>
  // Allow: npm run download:odds -- --league 德甲 --season 2024-2025
  // Or: node scripts/download-odds.js 德甲 2024-2025

  let leagueDir = process.argv[2];
  let season = process.argv[3];

  // Handle npm run extra arguments (--league, --season)
  for (let i = 2; i < process.argv.length; i++) {
    if (process.argv[i] === '--league' && process.argv[i + 1]) {
      leagueDir = process.argv[i + 1];
    }
    if (process.argv[i] === '--season' && process.argv[i + 1]) {
      season = process.argv[i + 1];
    }
  }

  if (!leagueDir || !season) {
    console.log('Usage:');
    console.log('  node scripts/download-odds.js <league_dir> <season>');
    console.log('  npm run download:odds -- --league 意甲 --season 2024-2025');
    console.log('');
    console.log('Examples:');
    console.log('  node scripts/download-odds.js 德甲 2024-2025');
    console.log('  npm run download:odds -- --league 西甲 --season 2023-2024');
    process.exit(1);
  }

  const baseDir = path.join(__dirname, '..', leagueDir, season);
  const sxxFile = fs.readdirSync(baseDir).find(f => f.match(/^s\d+.*\.js$/));

  if (!sxxFile) {
    console.error('No sXX.js file found in', baseDir);
    process.exit(1);
  }

  const sxxPath = path.join(baseDir, sxxFile);
  console.log(`Reading: ${sxxPath}`);

  const matchIds = extractMatchIds(sxxPath);
  console.log(`Found ${matchIds.length} matches\n`);

  // Ensure cache directory exists: ./[league]/[season]/matches/
  const leagueCacheDir = path.join(CACHE_DIR, leagueDir, season, 'matches');
  fs.mkdirSync(leagueCacheDir, { recursive: true });

  // Process in batches
  const batchSize = 50;
  const delay = 200; // ms between requests

  let totalDownloaded = 0;
  let totalSkipped = 0;
  let totalErrors = 0;
  const byBookmaker = {};

  for (let i = 0; i < matchIds.length; i += batchSize) {
    const batch = matchIds.slice(i, i + batchSize);
    const batchNum = Math.floor(i / batchSize) + 1;
    const totalBatches = Math.ceil(matchIds.length / batchSize);

    console.log(`Batch ${batchNum}/${totalBatches}...`);

    const batchPromises = batch.map(async (matchId) => {
      const cachePath = path.join(leagueCacheDir, `${matchId}.js`);
      const url = `${BASE_URL}/${matchId}.js${QUERY_PARAM}`;

      try {
        // Check cache first
        if (fs.existsSync(cachePath)) {
          // console.log(`  [CACHED] ${matchId}`);
        } else {
          await downloadFile(url, cachePath);
          await new Promise(r => setTimeout(r, delay));
        }

        // Parse the file
        if (fs.existsSync(cachePath)) {
          const oddsData = parseMatchFile(cachePath);

          if (oddsData.length > 0) {
            // Insert to Supabase
            const result = await insertOdds(oddsData);

            for (const odd of oddsData) {
              byBookmaker[odd.bookmaker_name] = (byBookmaker[odd.bookmaker_name] || 0) + 1;
            }

            return { downloaded: 1, inserted: result.inserted, skipped: result.skipped, error: 0 };
          }
        }

        return { downloaded: 1, inserted: 0, skipped: 1, error: 0 };
      } catch (err) {
        totalErrors++;
        console.error(`  Error ${matchId}: ${err.message}`);
        return { downloaded: 0, inserted: 0, skipped: 1, error: 1 };
      }
    });

    const results = await Promise.all(batchPromises);

    for (const r of results) {
      totalDownloaded += r.downloaded;
      totalSkipped += r.skipped;
      totalErrors += r.error;
    }

    console.log(`  Progress: ${totalDownloaded}/${matchIds.length} downloaded`);
  }

  console.log('\n=== Summary ===');
  console.log(`Total matches: ${matchIds.length}`);
  console.log(`Downloaded: ${totalDownloaded}`);
  console.log(`Skipped: ${totalSkipped}`);
  console.log(`Errors: ${totalErrors}`);
  console.log('\nBy bookmaker:');
  for (const [bm, count] of Object.entries(byBookmaker)) {
    console.log(`  ${bm}: ${count}`);
  }
}

main().catch(console.error);
