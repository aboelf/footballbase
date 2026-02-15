/**
 * 多庄家比赛详情数据提取脚本
 *
 * 支持解析多个庄家:
 * - Bet 365
 * - Easybets
 * - William Hill
 * - Ladbrokes
 * - 以及更多...
 *
 * 使用方法:
 *   node detail.js <matches目录> [--bookmakers "Bet 365,Easybets"]
 *
 * 示例:
 *   node detail.js ../英超/2024-2025/matches
 *   node detail.js ../英超/2020-2021/matches --bookmakers "Bet 365,Easybets"
 *
 * 输出:
 *   结果保存到 SAX encoder/bet365_details/bet365_details.json
 *   (或根据庄家名称自动命名，如 easybets_details.json)
 *
 * 数据说明:
 *   - 从 [联赛]/[赛季]/matches/ 目录下的 JS 文件提取赔率
 *   - game 数组格式: ID|InternalID|Name|initialHome|initialDraw|initialAway|...
 *   - gameDetail 数组格式: ID^odds1;odds2;odds3...
 *   - 每个赔率记录: home|draw|away|time|returnHome|returnDraw|returnAway|year
 */

const fs = require('fs');
const path = require('path');

/**
 * 解析单个庄家的详细赔率数据
 * @param {string} detailData - gameDetail 中该庄家的数据
 * @returns {Array} 运行盘赔率数组
 */
function parseRunningOdds(detailData) {
  if (!detailData) return [];

  // 分割: ID^records
  const parts = detailData.split('^');
  if (parts.length < 2) return [];

  const oddsRecords = parts[1].split(';').filter(Boolean);

  return oddsRecords.map(record => {
    const p = record.split('|');
    return {
      home: p[0],
      draw: p[1],
      away: p[2],
      time: p[3],
      returnRate: {
        home: p[4],
        draw: p[5],
        away: p[6],
      },
      year: p[7] || '',
    };
  });
}

/**
 * 解析 game 数组中单个庄家的初始赔率
 * @param {string} gameItem - game 数组中的一项
 * @returns {Object} 初始赔率对象
 */
function parseInitialOdds(gameItem) {
  if (!gameItem) return { home: '', draw: '', away: '' };

  // 移除首尾引号
  const cleanData = gameItem.replace(/^"|"$/g, '');
  const parts = cleanData.split('|');

  // game 数组格式: ID|InternalID|Name|initialHome|initialDraw|initialAway|...
  return {
    home: parts[3] || '',
    draw: parts[4] || '',
    away: parts[5] || '',
  };
}

/**
 * 解析单个庄家的完整数据
 * @param {string} gameItem - game 数组中的一项
 * @param {string} detailItem - gameDetail 中对应的数据
 * @returns {Object|null} 庄家数据对象
 */
function parseBookmakerData(gameItem, detailItem) {
  if (!gameItem) return null;

  // 移除首尾引号
  const cleanData = gameItem.replace(/^"|"$/g, '');
  const parts = cleanData.split('|');

  const bookmakerId = parts[0];
  const internalId = parts[1];
  const bookmakerName = parts[2];

  // game 数组格式: ID|InternalID|Name|initialHome|initialDraw|initialAway|returnHome|returnDraw|returnAway|totalReturn|finalHome|finalDraw|finalAway|...
  // 初始赔率 (fields 4-6: indices 3-5)
  const initialOdds = {
    home: parts[3] || '',
    draw: parts[4] || '',
    away: parts[5] || '',
  };

  // 最终赔率 (fields 11-13: indices 10-12) - 滚球盘最终赔率
  const finalOdds = {
    home: parts[10] || '',
    draw: parts[11] || '',
    away: parts[12] || '',
  };

  // 解析运行盘数据
  const runningOdds = parseRunningOdds(detailItem);

  return {
    bookmakerId,
    internalId,
    bookmakerName,
    initialOdds,
    finalOdds,
    runningOdds,
  };
}

/**
 * 解析单个比赛文件，提取所有庄家数据
 * @param {string} filePath - 比赛文件路径
 * @param {Array<string>} targetBookmakers - 目标庄家名称列表
 * @returns {Object|null} 比赛数据对象
 */
function parseMatchFile(filePath, targetBookmakers = []) {
  const content = fs.readFileSync(filePath, 'utf-8');

  // 提取基本信息
  const scheduleIdMatch = content.match(/var ScheduleID=(\d+);/);
  const hometeamMatch = content.match(/var hometeam="([^"]+)";/);
  const guestteamMatch = content.match(/var guestteam="([^"]+)";/);
  const matchTimeMatch = content.match(/var MatchTime="([^"]+)";/);

  // 提取 game 数据
  const gameMatch = content.match(/var game=Array\(([\s\S]*?)\);/);
  if (!gameMatch) return null;

  // 按 "," 分割获取每个庄家的数据
  const gameItems = gameMatch[1].split('","');

  // 提取 gameDetail 数据
  const gameDetailMatch = content.match(/var gameDetail=Array\(([\s\S]*?)\);/);
  const detailItems = gameDetailMatch
    ? gameDetailMatch[1].split('","')
    : [];

  // 构建庄家ID到gameDetail的映射
  const detailMap = {};
  for (const item of detailItems) {
    if (item.includes('^')) {
      const id = item.split('^')[0];
      detailMap[id] = item;
    }
  }

  // 解析所有庄家数据
  const bookmakersData = [];

  for (const item of gameItems) {
    // 提取庄家名称和ID
    const cleanData = item.replace(/^"|"$/g, '');
    const parts = cleanData.split('|');
    const bookmakerName = parts[2];

    // 检查是否为目标庄家（如果指定了目标列表）
    if (targetBookmakers.length > 0) {
      if (!targetBookmakers.includes(bookmakerName)) {
        continue;
      }
    }

    // gameDetail 中使用的是 InternalID (parts[1]) 来匹配
    const internalId = parts[1];
    const bookmakerData = parseBookmakerData(item, detailMap[internalId]);

    if (bookmakerData) {
      bookmakersData.push(bookmakerData);
    }
  }

  if (bookmakersData.length === 0) return null;

  return {
    scheduleId: scheduleIdMatch ? parseInt(scheduleIdMatch[1]) : 0,
    hometeam: hometeamMatch ? hometeamMatch[1] : '',
    guestteam: guestteamMatch ? guestteamMatch[1] : '',
    matchTime: matchTimeMatch ? matchTimeMatch[1] : '',
    season: '',
    bookmakers: bookmakersData,
  };
}

/**
 * 提取所有比赛的庄家数据
 * @param {string} matchesDir - 比赛目录路径
 * @param {string} season - 赛季名称
 * @param {Array<string>} targetBookmakers - 目标庄家列表
 * @returns {Array} 比赛数据数组
 */
function extractAllMatchDetails(matchesDir, season = '', targetBookmakers = []) {
  const results = [];

  if (!fs.existsSync(matchesDir)) {
    console.log(`Directory not found: ${matchesDir}`);
    return results;
  }

  const files = fs.readdirSync(matchesDir).filter(f => f.endsWith('.js'));

  for (const file of files) {
    const filePath = path.join(matchesDir, file);
    try {
      const matchData = parseMatchFile(filePath, targetBookmakers);
      if (matchData) {
        matchData.season = season;
        results.push(matchData);
        console.log(`Processed: ${season}/${file} (${matchData.bookmakers.length} bookmakers)`);
      }
    } catch (error) {
      console.error(`Error processing ${file}:`, error);
    }
  }

  return results;
}

/**
 * 递归处理联赛目录下的所有赛季
 * @param {string} leagueDir - 联赛目录路径
 * @param {Array<string>} targetBookmakers - 目标庄家列表
 * @returns {Array} 所有赛季的比赛数据
 */
function extractAllSeasons(leagueDir, targetBookmakers = []) {
  const results = [];

  if (!fs.existsSync(leagueDir)) {
    console.log(`Directory not found: ${leagueDir}`);
    return results;
  }

  const entries = fs.readdirSync(leagueDir, { withFileTypes: true });
  // 支持两种格式: 2024-2025 或 2024
  const seasonDirs = entries
    .filter(entry => entry.isDirectory() && (entry.name.match(/^\d{4}$/) || entry.name.match(/^\d{4}-\d{4}$/)))
    .map(entry => entry.name)
    .sort();

  console.log(`Found seasons: ${seasonDirs.join(', ')}`);

  for (const season of seasonDirs) {
    const matchesDir = path.join(leagueDir, season, 'matches');
    if (fs.existsSync(matchesDir)) {
      const seasonResults = extractAllMatchDetails(matchesDir, season, targetBookmakers);
      results.push(...seasonResults);
    }
  }

  return results;
}

// 主函数入口
if (require.main === module) {
  const args = process.argv.slice(2);

  // 解析命令行参数
  let targetPath = null;
  let targetBookmakers = [];
  let allLeagues = false;

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === '--bookmakers' && i + 1 < args.length) {
      targetBookmakers = args[i + 1].split(',').map(b => b.trim());
      i++;
    } else if (arg === '--all') {
      allLeagues = true;
    } else if (!arg.startsWith('--')) {
      targetPath = arg;
    }
  }

  let results = [];
  let outputFileName = 'multi_bookmaker_details';

  // 如果指定了目标庄家，更新输出文件名
  if (targetBookmakers.length > 0) {
    outputFileName = targetBookmakers
      .map(b => b.toLowerCase().replace(/\s+/g, '_'))
      .join('_');
  }

  if (targetPath && !allLeagues) {
    // 处理指定目录
    if (fs.existsSync(targetPath)) {
      const entries = fs.readdirSync(targetPath, { withFileTypes: true });
      // 支持两种赛季格式: 2024-2025 或 2024
      const hasSeasons = entries.some(e => e.isDirectory() && (e.name.match(/^\d{4}-\d{4}$/) || e.name.match(/^\d{4}$/)));
      const isLeagueDir = entries.some(e => e.isDirectory() && !e.name.match(/^\d{4}(-\d{4})?$/));

      if (hasSeasons) {
        // 直接是赛季目录
        console.log(`Processing all seasons in: ${targetPath}`);
        console.log(`Target bookmakers: ${targetBookmakers.join(', ') || 'All'}`);
        results = extractAllSeasons(targetPath, targetBookmakers);
      } else if (isLeagueDir) {
        // 是联赛目录（包含多个赛季）
        console.log(`Processing league directory: ${targetPath}`);
        console.log(`Target bookmakers: ${targetBookmakers.join(', ') || 'All'}`);
        for (const entry of entries) {
          if (entry.isDirectory()) {
            const leaguePath = path.join(targetPath, entry.name);
            console.log(`\nProcessing league: ${entry.name}`);
            const leagueResults = extractAllSeasons(leaguePath, targetBookmakers);
            results.push(...leagueResults);
            console.log(`✓ ${entry.name}: ${leagueResults.length} matches`);
          }
        }
      } else {
        // 单个赛季目录
        const season = path.basename(path.dirname(targetPath));
        console.log(`Processing single season: ${season}`);
        console.log(`Target bookmakers: ${targetBookmakers.join(', ') || 'All'}`);
        results = extractAllMatchDetails(targetPath, season, targetBookmakers);
      }
    } else {
      console.log(`Directory not found: ${targetPath}`);
      process.exit(1);
    }
  } else {
    // 处理所有5大联赛
    console.log('Processing all 5 major European leagues: 德甲, 法甲, 西甲, 意甲, 英超');
    console.log(`Target bookmakers: ${targetBookmakers.join(', ') || 'All'}`);
    console.log('='.repeat(50));

    const leagues = ['德甲', '法甲', '西甲', '意甲', '英超'];

    for (const league of leagues) {
      const leagueDir = `../../rawdata/${league}`;
      console.log(`\nProcessing ${league}...`);
      if (fs.existsSync(leagueDir)) {
        const leagueResults = extractAllSeasons(leagueDir, targetBookmakers);
        results.push(...leagueResults);
        console.log(`✓ ${league}: ${leagueResults.length} matches`);
      } else {
        console.log(`✗ ${league}: Directory not found`);
      }
    }
  }

  // 输出统计信息
  console.log('\n' + '='.repeat(50));
  console.log('=== Results Summary ===');

  // 统计各庄家数据量
  const bookmakerCounts = {};
  for (const match of results) {
    for (const bm of match.bookmakers) {
      bookmakerCounts[bm.bookmakerName] = (bookmakerCounts[bm.bookmakerName] || 0) + 1;
    }
  }

  console.log(`Total matches with data: ${results.length}`);
  console.log('Bookmakers found:');
  for (const [name, count] of Object.entries(bookmakerCounts)) {
    console.log(`  - ${name}: ${count} matches`);
  }

  // 保存到文件
  const outputDir = 'SAX encoder/bookmaker_details';
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  const outputPath = path.join(outputDir, `${outputFileName}_details.json`);
  fs.writeFileSync(outputPath, JSON.stringify(results, null, 2), 'utf-8');
  console.log(`\nResults saved to: ${outputPath}`);
}

module.exports = { parseMatchFile, extractAllMatchDetails };
