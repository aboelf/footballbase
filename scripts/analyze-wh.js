#!/usr/bin/env node
/**
 * Run analysis queries on Supabase
 * Analyzes William Hill final_win = 1.44 matches
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

async function runAnalysis() {
  console.log('='.repeat(60));
  console.log('Analysis: William Hill final_win = 1.44');
  console.log('='.repeat(60));

  // Query 1: Basic matches with WH final_win = 1.44
  console.log('\n--- Query 1: All matches with WH final_win = 1.44 ---\n');

  const { data: matches, error: matchError } = await supabase
    .from('v_match_odds')
    .select(`
      match_id,
      league_name,
      match_time,
      home_team_name,
      away_team_name,
      final_score,
      final_win,
      final_draw,
      final_lose
    `)
    .eq('bookmaker_name', 'William Hill')
    .eq('final_win', 1.44)
    .order('match_time', { ascending: false })
    .limit(20);

  if (matchError) {
    console.error('Error fetching matches:', matchError);
  } else {
    console.log(`Found ${matches.length} matches with WH final_win = 1.44`);
    console.log('\nRecent matches:');
    matches.forEach(m => {
      const homeGoals = m.final_score?.split('-')[0] || 'N/A';
      const awayGoals = m.final_score?.split('-')[1] || 'N/A';
      const result = homeGoals > awayGoals ? '主胜' : homeGoals < awayGoals ? '主负' : '平';
      console.log(`  ${m.match_time?.split('T')[0]} | ${m.home_team_name} vs ${m.away_team_name} | ${m.final_score} | ${result}`);
    });
  }

  // Query 2: Summary statistics
  console.log('\n--- Query 2: Win rate summary ---\n');

  const { data: summary, error: summaryError } = await supabase
    .rpc('analyze_wh_144', {});

  if (summaryError) {
    // If RPC doesn't exist, use raw query
    console.log('Running raw query instead...\n');

    const { data: rawSummary, error: rawError } = await supabase
      .from('v_match_odds')
      .select('final_score')
      .eq('bookmaker_name', 'William Hill')
      .eq('final_win', 1.44)
      .not('final_score', 'is', null);

    if (!rawError && rawSummary) {
      let homeWins = 0;
      let draws = 0;
      let awayWins = 0;

      rawSummary.forEach(m => {
        const [home, away] = m.final_score.split('-').map(Number);
        if (home > away) homeWins++;
        else if (home === away) draws++;
        else awayWins++;
      });

      const total = homeWins + draws + awayWins;
      console.log(`Total matches: ${total}`);
      console.log(`主胜 (Home Win): ${homeWins} (${((homeWins/total)*100).toFixed(2)}%)`);
      console.log(`平 (Draw): ${draws} (${((draws/total)*100).toFixed(2)}%)`);
      console.log(`主负 (Away Win): ${awayWins} (${((awayWins/total)*100).toFixed(2)}%)`);
    }
  } else {
    console.log(JSON.stringify(summary, null, 2));
  }

  // Query 3: Range analysis (1.40-1.48)
  console.log('\n--- Query 3: Range analysis (1.40-1.48) ---\n');

  const { data: rangeData, error: rangeError } = await supabase
    .from('v_match_odds')
    .select('final_score, final_win')
    .eq('bookmaker_name', 'William Hill')
    .gte('final_win', 1.40)
    .lte('final_win', 1.48)
    .not('final_score', 'is', null);

  if (!rangeError && rangeData) {
    let homeWins = 0;
    let draws = 0;
    let awayWins = 0;

    rangeData.forEach(m => {
      const [home, away] = m.final_score.split('-').map(Number);
      if (home > away) homeWins++;
      else if (home === away) draws++;
      else awayWins++;
    });

    const total = homeWins + draws + awayWins;
    console.log(`Odds range: 1.40-1.48`);
    console.log(`Total matches: ${total}`);
    console.log(`主胜 (Home Win): ${homeWins} (${((homeWins/total)*100).toFixed(2)}%)`);
    console.log(`平 (Draw): ${draws} (${((draws/total)*100).toFixed(2)}%)`);
    console.log(`主负 (Away Win): ${awayWins} (${((awayWins/total)*100).toFixed(2)}%)`);
  }

  // Query 4: Overall statistics by odds range
  console.log('\n--- Query 4: Home win rate by William Hill odds range ---\n');

  const oddsRanges = [
    { name: '1.0-1.49 (Strong Home)', min: 0, max: 1.49 },
    { name: '1.50-1.99 (Moderate Home)', min: 1.50, max: 1.99 },
    { name: '2.00-2.49 (Slight Home)', min: 2.00, max: 2.49 },
    { name: '2.50-2.99 (Neutral)', min: 2.50, max: 2.99 },
    { name: '3.0+ (Away possible)', min: 3.00, max: 999 }
  ];

  for (const range of oddsRanges) {
    const { data: rangeData } = await supabase
      .from('v_match_odds')
      .select('final_score, final_win')
      .eq('bookmaker_name', 'William Hill')
      .gte('final_win', range.min)
      .lt('final_win', range.max)
      .not('final_score', 'is', null);

    if (rangeData && rangeData.length > 0) {
      let homeWins = 0;
      rangeData.forEach(m => {
        const [home, away] = m.final_score.split('-').map(Number);
        if (home > away) homeWins++;
      });

      const winRate = ((homeWins / rangeData.length) * 100).toFixed(2);
      const avgOdds = (rangeData.reduce((sum, m) => sum + Number(m.final_win), 0) / rangeData.length).toFixed(3);
      console.log(`${range.name}:`);
      console.log(`  Matches: ${rangeData.length}, Home Win: ${homeWins} (${winRate}%), Avg Odds: ${avgOdds}`);
    }
  }

  console.log('\n' + '='.repeat(60));
  console.log('Analysis complete!');
  console.log('='.repeat(60));
}

runAnalysis().catch(console.error);
