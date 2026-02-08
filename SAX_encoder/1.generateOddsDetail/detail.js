/**
 * Bet 365 比赛详情数据提取脚本
 *
 * 使用方法:
 *   node detail.js <matches目录>
 *
 * 示例:
 *   node detail.js ../英超/2024-2025/matches
 *   node detail.js ../英超/2020-2021/matches
 *
 * 输出:
 *   结果保存到 SAX encoder/bet365_details/bet365_details.json
 *
 * 数据说明:
 *   - 从 [联赛]/[赛季]/matches/ 目录下的 JS 文件提取 Bet 365 赔率
 *   - game 数组第二个元素是庄家编号，第三个是庄家名称
 *   - gameDetail 数组中以 ^ 分割，第一个子值是庄家编号
 */

const fs = require('fs');
const path = require('path');

function parseMatchFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf-8');

  // 提取基本信息
  const scheduleIdMatch = content.match(/var ScheduleID=(\d+);/);
  const hometeamMatch = content.match(/var hometeam="([^"]+)";/);
  const guestteamMatch = content.match(/var guestteam="([^"]+)";/);
  const matchTimeMatch = content.match(/var MatchTime="([^"]+)";/);

  // 提取 game 数据 - 使用 [\s\S]*? 匹配整个数组内容
  const gameMatch = content.match(/var game=Array\(([\s\S]*?)\);/);
  if (!gameMatch) return null;

  // 按 "," 分割获取每个庄家的数据
  const gameItems = gameMatch[1].split('","');

  // 找到 Bet 365 的庄家ID
  let bet365Id = null;
  for (const item of gameItems) {
    const parts = item.split('|');
    if (parts[2] === 'Bet 365') {
      bet365Id = parts[1];
      break;
    }
  }

  if (!bet365Id) return null;

  // 提取 gameDetail 数据
  const gameDetailMatch = content.match(/var gameDetail=Array\(([\s\S]*?)\);/);
  if (!gameDetailMatch) return null;

  // 分割详细数据
  const detailItems = gameDetailMatch[1].split('","');
  let bet365Detail = null;
  for (const item of detailItems) {
    if (item.startsWith(bet365Id + '^')) {
      bet365Detail = item;
      break;
    }
  }

  if (!bet365Detail) return null;

  // 解析 Bet 365 的详细赔率
  const oddsRecords = bet365Detail.split('^')[1].split(';').filter(Boolean);
  const runningOdds = oddsRecords.map(record => {
    const parts = record.split('|');
    return {
      home: parts[0],
      draw: parts[1],
      away: parts[2],
      time: parts[3],
      returnRate: {
        home: parts[4],
        draw: parts[5],
        away: parts[6],
      },
    };
  });

  // 获取 Bet 365 的初始赔率（game 中的数据）
  const bet365GameData = gameItems.find(item => item.includes('|Bet 365|'));
  let initialOdds = { home: '', draw: '', away: '' };
  if (bet365GameData) {
    // 移除首尾引号
    const cleanData = bet365GameData.replace(/^"|"$/g, '');
    const parts = cleanData.split('|');
    initialOdds = {
      home: parts[3],
      draw: parts[4],
      away: parts[5],
    };
  }

  return {
    scheduleId: scheduleIdMatch ? parseInt(scheduleIdMatch[1]) : 0,
    hometeam: hometeamMatch ? hometeamMatch[1] : '',
    guestteam: guestteamMatch ? guestteamMatch[1] : '',
    matchTime: matchTimeMatch ? matchTimeMatch[1] : '',
    season: '',
    bookmaker: 'Bet 365',
    bookmakerId: bet365Id,
    initialOdds,
    runningOdds,
  };
}

function extractAllMatchDetails(matchesDir, season = '') {
  const results = [];

  if (!fs.existsSync(matchesDir)) {
    console.log(`Directory not found: ${matchesDir}`);
    return results;
  }

  const files = fs.readdirSync(matchesDir).filter(f => f.endsWith('.js'));

  for (const file of files) {
    const filePath = path.join(matchesDir, file);
    try {
      const matchData = parseMatchFile(filePath);
      if (matchData) {
        matchData.season = season;
        results.push(matchData);
        console.log(`Processed: ${season}/${file}`);
      }
    } catch (error) {
      console.error(`Error processing ${file}:`, error);
    }
  }

  return results;
}

// 递归处理联赛目录下的所有赛季
function extractAllSeasons(leagueDir) {
  const results = [];

  if (!fs.existsSync(leagueDir)) {
    console.log(`Directory not found: ${leagueDir}`);
    return results;
  }

  const entries = fs.readdirSync(leagueDir, { withFileTypes: true });
  const seasonDirs = entries
    .filter(entry => entry.isDirectory() && entry.name.match(/\d{4}-\d{4}/))
    .map(entry => entry.name)
    .sort();

  console.log(`Found seasons: ${seasonDirs.join(', ')}`);

  for (const season of seasonDirs) {
    const matchesDir = path.join(leagueDir, season, 'matches');
    if (fs.existsSync(matchesDir)) {
      const seasonResults = extractAllMatchDetails(matchesDir, season);
      results.push(...seasonResults);
    }
  }

  return results;
}

// 使用示例
if (require.main === module) {
  const args = process.argv.slice(2);

  // 如果没有参数，处理所有5大联赛
  let results = [];

  if (args.length === 0) {
    console.log('Usage: node detail.js [league_directory]');
    console.log('Or: node detail.js --all');
    console.log('\nProcessing all 5 major European leagues: 德甲, 法甲, 西甲, 意甲, 英超');
    console.log('========================================');

    const leagues = ['德甲', '法甲', '西甲', '意甲', '英超'];

    for (const league of leagues) {
      const leagueDir = `../../rawdata/${league}`;
      console.log(`\nProcessing ${league}...`);
      if (fs.existsSync(leagueDir)) {
        const leagueResults = extractAllSeasons(leagueDir);
        results.push(...leagueResults);
        console.log(`✓ ${league}: ${leagueResults.length} matches`);
      } else {
        console.log(`✗ ${league}: Directory not found`);
      }
    }
  } else if (args[0] === '--all') {
    // 处理所有5大联赛
    const leagues = ['德甲', '法甲', '西甲', '意甲', '英超'];

    for (const league of leagues) {
      const leagueDir = `../${league}`;
      console.log(`Processing ${league}...`);
      if (fs.existsSync(leagueDir)) {
        const leagueResults = extractAllSeasons(leagueDir);
        results.push(...leagueResults);
        console.log(`✓ ${league}: ${leagueResults.length} matches`);
      } else {
        console.log(`✗ ${league}: Directory not found`);
      }
    }
  } else {
    const targetPath = args[0];
    // 检查是否是联赛目录（包含多个赛季）
    if (fs.existsSync(targetPath)) {
      const entries = fs.readdirSync(targetPath, { withFileTypes: true });
      const hasSeasons = entries.some(e => e.isDirectory() && e.name.match(/\d{4}-\d{4}/));
      if (hasSeasons) {
        console.log(`Processing all seasons in: ${targetPath}`);
        results = extractAllSeasons(targetPath);
      } else {
        const season = path.basename(path.dirname(targetPath));
        console.log(`Processing single season: ${season}`);
        results = extractAllMatchDetails(targetPath, season);
      }
    } else {
      console.log(`Directory not found: ${targetPath}`);
      process.exit(1);
    }
  }

  // 输出 JSON 结果
  console.log('\n=== Results ===');
  console.log(`Total matches with Bet 365 odds: ${results.length}`);

  // 保存到文件
  const outputDir = 'SAX encoder/bet365_details';
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  const outputPath = path.join(outputDir, 'bet365_details.json');
  fs.writeFileSync(outputPath, JSON.stringify(results, null, 2), 'utf-8');
  console.log(`Results saved to: ${outputPath}`);
}

module.exports = { parseMatchFile, extractAllMatchDetails };
