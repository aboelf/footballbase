#!/usr/bin/env node
/**
 * AI Database Analyst Tool
 * Allows AI to run arbitrary SQL queries on Supabase
 * Usage: node scripts/db-analyst.js "SELECT ..."
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

/**
 * Execute a query and return formatted results
 */
async function query(sql, options = {}) {
  const startTime = Date.now();
  const limit = options.limit || 100;

  // If using the view, handle it properly
  if (sql.toLowerCase().includes('v_match_odds')) {
    const { data, error } = await supabase
      .from('v_match_odds')
      .select('*')
      .limit(limit);

    if (error) throw error;
    return { data, count: data.length, time: Date.now() - startTime };
  }

  // For direct SQL, we need to use RPC or raw query
  // First try to use v_match_odds for analysis
  if (sql.toLowerCase().includes('from v_match_odds')) {
    // Parse the query to extract filters
    const queryBuilder = supabase.from('v_match_odds').select('*');

    // Extract bookmaker filter
    const bookmakerMatch = sql.match(/bookmaker_name\s*=\s*['"]([^'"]+)['"]/i);
    if (bookmakerMatch) {
      queryBuilder.eq('bookmaker_name', bookmakerMatch[1]);
    }

    // Extract final_win filter
    const winMatch = sql.match(/final_win\s*=\s*([\d.]+)/i);
    if (winMatch) {
      queryBuilder.eq('final_win', parseFloat(winMatch[1]));
    }

    // Extract BETWEEN filter
    const betweenMatch = sql.match(/final_win\s+BETWEEN\s+([\d.]+)\s+AND\s+([\d.]+)/i);
    if (betweenMatch) {
      queryBuilder.gte('final_win', parseFloat(betweenMatch[1]));
      queryBuilder.lte('final_win', parseFloat(betweenMatch[2]));
    }

    // Extract order
    if (sql.toLowerCase().includes('order by')) {
      const orderMatch = sql.match(/order by\s+(\w+)\s*(desc|asc)?/i);
      if (orderMatch) {
        queryBuilder.order(orderMatch[1], { ascending: !orderMatch[2] || orderMatch[2].toLowerCase() === 'asc' });
      }
    }

    queryBuilder.limit(limit);

    const { data, error } = await queryBuilder;
    if (error) throw error;
    return { data, count: data.length, time: Date.now() - startTime };
  }

  // Fallback: try direct table query
  const { data, error } = await supabase.from('league_matches').select('*').limit(limit);
  if (error) throw error;
  return { data, count: data.length, time: Date.now() - startTime };
}

/**
 * Known tables in the database
 */
const KNOWN_TABLES = ['league_matches', 'betting_odds', 'v_match_odds'];

/**
 * Get table info
 */
async function tables() {
  return KNOWN_TABLES;
}

/**
 * Get table schema - known schemas for our tables
 */
const TABLE_SCHEMAS = {
  league_matches: [
    { column_name: 'match_id', data_type: 'bigint', is_nullable: false },
    { column_name: 'league_name', data_type: 'character varying', is_nullable: true },
    { column_name: 'season', data_type: 'character varying', is_nullable: true },
    { column_name: 'match_time', data_type: 'timestamp without time zone', is_nullable: true },
    { column_name: 'home_team_name', data_type: 'character varying', is_nullable: true },
    { column_name: 'away_team_name', data_type: 'character varying', is_nullable: true },
    { column_name: 'final_score', data_type: 'character varying', is_nullable: true }
  ],
  betting_odds: [
    { column_name: 'schedule_id', data_type: 'bigint', is_nullable: false },
    { column_name: 'bookmaker_name', data_type: 'character varying', is_nullable: false },
    { column_name: 'init_win', data_type: 'numeric', is_nullable: true },
    { column_name: 'init_draw', data_type: 'numeric', is_nullable: true },
    { column_name: 'init_lose', data_type: 'numeric', is_nullable: true },
    { column_name: 'final_win', data_type: 'numeric', is_nullable: true },
    { column_name: 'final_draw', data_type: 'numeric', is_nullable: true },
    { column_name: 'final_lose', data_type: 'numeric', is_nullable: true },
    { column_name: 'kelly_win', data_type: 'numeric', is_nullable: true },
    { column_name: 'kelly_draw', data_type: 'numeric', is_nullable: true },
    { column_name: 'kelly_lose', data_type: 'numeric', is_nullable: true }
  ],
  v_match_odds: [
    { column_name: 'match_id', data_type: 'bigint', is_nullable: false },
    { column_name: 'league_name', data_type: 'character varying', is_nullable: true },
    { column_name: 'match_time', data_type: 'timestamp without time zone', is_nullable: true },
    { column_name: 'home_team_name', data_type: 'character varying', is_nullable: true },
    { column_name: 'away_team_name', data_type: 'character varying', is_nullable: true },
    { column_name: 'final_score', data_type: 'character varying', is_nullable: true },
    { column_name: 'bookmaker_name', data_type: 'character varying', is_nullable: false },
    { column_name: 'init_win', data_type: 'numeric', is_nullable: true },
    { column_name: 'init_draw', data_type: 'numeric', is_nullable: true },
    { column_name: 'init_lose', data_type: 'numeric', is_nullable: true },
    { column_name: 'final_win', data_type: 'numeric', is_nullable: true },
    { column_name: 'final_draw', data_type: 'numeric', is_nullable: true },
    { column_name: 'final_lose', data_type: 'numeric', is_nullable: true },
    { column_name: 'kelly_win', data_type: 'numeric', is_nullable: true },
    { column_name: 'kelly_draw', data_type: 'numeric', is_nullable: true },
    { column_name: 'kelly_lose', data_type: 'numeric', is_nullable: true }
  ]
};

async function schema(tableName) {
  const columns = TABLE_SCHEMAS[tableName] || [];

  // Get row count
  const { count, error } = await supabase
    .from(tableName)
    .select('*', { count: 'exact', head: true });

  if (error) throw error;

  return { columns, count };
}

/**
 * Analyze odds correlation
 */
async function analyze(bookmaker, oddsField = 'final_win', range = null) {
  let query = supabase
    .from('v_match_odds')
    .select('final_score, ' + oddsField)
    .eq('bookmaker_name', bookmaker)
    .not('final_score', 'is', null);

  if (range) {
    query = query
      .gte(oddsField, range[0])
      .lte(oddsField, range[1]);
  }

  const { data, error } = await query;
  if (error) throw error;

  // Calculate results
  let homeWins = 0, draws = 0, awayWins = 0;
  data.forEach(m => {
    const [home, away] = (m.final_score || '0-0').split('-').map(Number);
    if (home > away) homeWins++;
    else if (home === away) draws++;
    else awayWins++;
  });

  const total = data.length;
  return {
    bookmaker,
    odds_field: oddsField,
    range,
    total,
    home_wins: homeWins,
    draws,
    away_wins: awayWins,
    home_win_rate: total ? (homeWins / total * 100).toFixed(2) + '%' : 'N/A',
    draw_rate: total ? (draws / total * 100).toFixed(2) + '%' : 'N/A',
    away_win_rate: total ? (awayWins / total * 100).toFixed(2) + '%' : 'N/A'
  };
}

/**
 * Interactive mode
 */
async function interactive() {
  console.log('\n=== AI Database Analyst ===');
  console.log('Available commands:');
  console.log('  tables               - List all tables');
  console.log('  schema <table>       - Get table schema');
  console.log('  analyze <bookmaker> [odds] [min] [max] - Analyze odds correlation');
  console.log('  query <SQL-like>     - Run query on v_match_odds');
  console.log('  help                 - Show this message');
  console.log('  quit                 - Exit\n');

  const readline = require('readline');
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });

  const ask = () => {
    rl.question('analyst> ', async (input) => {
      const cmd = input.trim().split(/\s+/);
      const command = cmd[0].toLowerCase();

      try {
        switch (command) {
          case 'tables': {
            const tablesList = await tables();
            console.log('\nTables:', tablesList.join(', '));
            break;
          }
          case 'schema': {
            const tableName = cmd[1];
            if (!tableName) { console.log('Usage: schema <table>'); break; }
            const sch = await schema(tableName);
            console.log('\nSchema for', tableName + ':');
            console.table(sch.columns.map(c => ({
              Column: c.column_name,
              Type: c.data_type,
              Nullable: c.is_nullable
            })));
            console.log(`Rows: ${sch.count}`);
            break;
          }
          case 'analyze': {
            const bookmaker = cmd[1];
            const oddsField = cmd[2] || 'final_win';
            const min = cmd[3] ? parseFloat(cmd[3]) : null;
            const max = cmd[4] ? parseFloat(cmd[4]) : null;
            if (!bookmaker) { console.log('Usage: analyze <bookmaker> [odds] [min] [max]'); break; }
            const result = await analyze(bookmaker, oddsField, min !== null ? [min, max] : null);
            console.log('\nAnalysis Result:');
            console.log(JSON.stringify(result, null, 2));
            break;
          }
          case 'query': {
            const q = input.substring(cmd[0].length + 1);
            if (!q) { console.log('Usage: query <SQL-like query>'); break; }
            const result = await query(q);
            console.log(`\nFound ${result.count} rows (${result.time}ms):`);
            console.table(result.data.slice(0, 20));
            if (result.count > 20) console.log(`... and ${result.count - 20} more rows`);
            break;
          }
          case 'help':
            console.log('\nCommands: tables, schema <table>, analyze <bookmaker> [odds] [min] [max], query <SQL-like>, help, quit');
            break;
          case 'quit':
          case 'exit':
            rl.close();
            return;
          default:
            console.log('Unknown command. Type "help" for available commands.');
        }
      } catch (error) {
        console.error('\nError:', error.message);
      }

      console.log('');
      ask();
    });
  };

  ask();
}

/**
 * Main entry point
 */
async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0) {
    await interactive();
    return;
  }

  const command = args[0].toLowerCase();

  try {
    switch (command) {
      case 'tables':
        const tablesList = await tables();
        console.log(JSON.stringify({ tables: tablesList }, null, 2));
        break;

      case 'schema':
        if (!args[1]) { console.error('Usage: schema <table>'); process.exit(1); }
        const sch = await schema(args[1]);
        console.log(JSON.stringify(sch, null, 2));
        break;

      case 'analyze': {
        const bookmaker = args[1];
        const oddsField = args[2] || 'final_win';
        const min = args[3] ? parseFloat(args[3]) : null;
        const max = args[4] ? parseFloat(args[4]) : null;
        const result = await analyze(bookmaker, oddsField, min !== null ? [min, max] : null);
        console.log(JSON.stringify(result, null, 2));
        break;
      }

      case 'query': {
        const queryStr = args.slice(1).join(' ');
        const result = await query(queryStr);
        console.log(JSON.stringify(result, null, 2));
        break;
      }

      case 'help':
        console.log(`
AI Database Analyst - Commands:
  tables               - List all tables
  schema <table>       - Get table schema and row count
  analyze <bookmaker> [odds] [min] [max] - Analyze odds correlation
  query <SQL-like>     - Run query on v_match_odds
  help                 - Show this message
        `);
        break;

      default:
        console.error('Unknown command:', command);
        console.error('Use "help" for available commands');
        process.exit(1);
    }
  } catch (error) {
    console.error('Error:', error.message);
    process.exit(1);
  }
}

main();
