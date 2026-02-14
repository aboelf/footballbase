const fs = require('fs');
const path = require('path');

const BASE_URL = 'https://zq.titan007.com/jsData/matchResult';
const VERSION = '2026021412';
const LEAGUE_NAME = '日职联';
const CODE = 's25_943';
const SEASONS = ['2021', '2022', '2023', '2024', '2025'];

const OUTPUT_DIR = path.join(__dirname, '..', 'rawdata', LEAGUE_NAME);

async function downloadSeason(season) {
  const url = `${BASE_URL}/${season}/${CODE}.js?version=${VERSION}`;
  const outputPath = path.join(OUTPUT_DIR, season, `${CODE}.js`);

  console.log(`Downloading ${LEAGUE_NAME} ${season}...`);
  console.log(`  URL: ${url}`);

  try {
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const content = await response.text();

    // Ensure directory exists
    const dir = path.dirname(outputPath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    fs.writeFileSync(outputPath, content, 'utf-8');
    console.log(`  Saved: ${outputPath}`);
    console.log(`  Size: ${content.length} bytes`);
    
    return true;
  } catch (error) {
    console.error(`  Error: ${error.message}`);
    return false;
  }
}

async function main() {
  console.log(`=== Download ${LEAGUE_NAME} Data ===\n`);
  console.log(`Output directory: ${OUTPUT_DIR}\n`);

  // Ensure base output directory exists
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  const results = [];
  
  for (const season of SEASONS) {
    const success = await downloadSeason(season);
    results.push({ season, success });
    console.log('');
  }

  // Summary
  console.log('=== Summary ===');
  const successCount = results.filter(r => r.success).length;
  console.log(`Total: ${results.length}, Success: ${successCount}, Failed: ${results.length - successCount}`);
  
  results.forEach(r => {
    console.log(`  ${r.season}: ${r.success ? 'OK' : 'FAILED'}`);
  });
}

main().catch(console.error);