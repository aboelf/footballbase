const fs = require('fs');
const path = require('path');
const https = require('https');
const { pipeline } = require('stream/promises');

const BASE_URL = 'https://1x2d.titan007.com';
const QUERY_PARAM = '?r=007134148238774303818';

const TARGET_BOOKMAKERS = [
  'William Hill',
  'Bet 365',
  'Easybets',
  'Lottery Official',
  'Macauslot'
];

require('dotenv').config({ path: path.join(__dirname, '..', '.env') });
const { createClient } = require('@supabase/supabase-js');
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_KEY;
const supabase = createClient(supabaseUrl, supabaseKey);

function downloadFile(url, destPath) {
  return new Promise((resolve, reject) => {
    const file = fs.createWriteStream(destPath);
    https.get(url, (response) => {
      if (response.statusCode === 302 || response.statusCode === 301) {
        downloadFile(response.headers.location, destPath).then(resolve).catch(reject);
        return;
      }
      if (response.statusCode !== 200) {
        reject(new Error('HTTP ' + response.statusCode));
        return;
      }
      pipeline(response, file).then(() => resolve()).catch(reject);
    }).on('error', reject);
  });
}

function parseGameArray(content) {
  const match = content.match(/game\s*=\s*Array\(\s*["'](.+?)["']\s*\)\s*;?/s);
  if (!match) return [];
  const gameStr = match[1];
  const entries = gameStr.split(/"\s*,\s*"/);
  return entries.map(e => e.replace(/^"|"$/g, '').trim()).filter(Boolean);
}

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

function extractMeta(content) {
  const scheduleMatch = content.match(/ScheduleID\s*=\s*(\d+)/);
  return scheduleMatch ? { ScheduleID: parseInt(scheduleMatch[1], 10) } : {};
}

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

async function insertOdds(oddsData) {
  if (!supabase || oddsData.length === 0) return { inserted: 0 };
  const { error } = await supabase
    .from('betting_odds')
    .upsert(
      oddsData.map(odd => ({
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
  return { error };
}

async function main() {
  const missing = [2538896];
  console.log('Re-downloading missing matches:', missing);

  for (const matchId of missing) {
    const cachePath = path.join(__dirname, '..', '日职联', '2024', 'matches', matchId + '.js');
    const url = BASE_URL + '/' + matchId + '.js' + QUERY_PARAM;

    console.log('Downloading:', url);
    await downloadFile(url, cachePath);
    
    const stats = fs.statSync(cachePath);
    console.log('Saved:', cachePath, 'Size:', stats.size);

    if (stats.size > 0) {
      const oddsData = parseMatchFile(cachePath);
      console.log('Parsed odds:', oddsData.length);
      if (oddsData.length > 0) {
        const result = await insertOdds(oddsData);
        console.log('Inserted to DB:', result.error ? 'ERROR: ' + result.error.message : 'OK');
      }
    }
  }
}

main().catch(console.error);
