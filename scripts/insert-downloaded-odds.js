/**
 * Parse all downloaded match files and insert into Supabase
 */

const fs = require('fs');
const path = require('path');

// Load environment variables
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const { createClient } = require('@supabase/supabase-js');

const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_KEY;

let supabase;
if (supabaseUrl && supabaseKey) {
  supabase = createClient(supabaseUrl, supabaseKey);
} else {
  console.error('Supabase credentials not configured');
  process.exit(1);
}

const TARGET_BOOKMAKERS = [
  'William Hill',
  'Bet 365',
  'Easybets',
  'Lottery Official',
  'Macauslot'
];

/**
 * Parse game array string
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
 * Extract schedule ID from file
 */
function extractScheduleId(filePath) {
  const fileName = path.basename(filePath, '.js');
  return parseInt(fileName, 10);
}

/**
 * Parse a downloaded match file
 */
function parseMatchFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf-8');
  const scheduleId = extractScheduleId(filePath);
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
 * Insert odds data into Supabase
 */
async function insertOdds(oddsData) {
  if (oddsData.length === 0) return { inserted: 0, skipped: oddsData.length };

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
 * Get all downloaded files
 */
function getDownloadedFiles(baseDir) {
  const results = [];

  function walkDir(dir) {
    if (!fs.existsSync(dir)) return;

    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);

      if (entry.isDirectory()) {
        walkDir(fullPath);
      } else if (entry.isFile() && entry.name.endsWith('.js')) {
        results.push(fullPath);
      }
    }
  }

  walkDir(baseDir);
  return results;
}

async function main() {
  const downloadsDir = process.argv[2] || path.join(__dirname, '..');

  console.log(`Scanning: ${downloadsDir}`);

  const files = getDownloadedFiles(downloadsDir);
  console.log(`Found ${files.length} files\n`);

  if (files.length === 0) {
    console.log('No files found');
    return;
  }

  let totalInserted = 0;
  let totalSkipped = 0;
  let totalErrors = 0;
  const byBookmaker = {};

  // Process in batches of 100 files
  const fileBatchSize = 100;
  const insertBatchSize = 100;

  for (let i = 0; i < files.length; i += fileBatchSize) {
    const batch = files.slice(i, i + fileBatchSize);
    const batchNum = Math.floor(i / fileBatchSize) + 1;
    const totalBatches = Math.ceil(files.length / fileBatchSize);

    console.log(`Processing batch ${batchNum}/${totalBatches}...`);

    const allOddsData = [];

    for (const file of batch) {
      try {
        const oddsData = parseMatchFile(file);
        allOddsData.push(...oddsData);

        for (const odd of oddsData) {
          byBookmaker[odd.bookmaker_name] = (byBookmaker[odd.bookmaker_name] || 0) + 1;
        }
      } catch (err) {
        totalErrors++;
      }
    }

    // Insert this batch
    for (let j = 0; j < allOddsData.length; j += insertBatchSize) {
      const insertBatch = allOddsData.slice(j, j + insertBatchSize);
      const result = await insertOdds(insertBatch);
      totalInserted += result.inserted;
      totalSkipped += result.skipped;
    }

    console.log(`  Progress: ${Math.min(i + fileBatchSize, files.length)}/${files.length}`);
  }

  console.log('\n=== Summary ===');
  console.log(`Total files: ${files.length}`);
  console.log(`Inserted: ${totalInserted}`);
  console.log(`Skipped: ${totalSkipped}`);
  console.log(`Errors: ${totalErrors}`);
  console.log('\nBy bookmaker:');
  for (const [bm, count] of Object.entries(byBookmaker)) {
    console.log(`  ${bm}: ${count}`);
  }
}

main().catch(console.error);
