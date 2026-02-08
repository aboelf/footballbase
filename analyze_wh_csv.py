import csv
import sys
from datetime import datetime
from collections import defaultdict

def parse_score(score_str):
    try:
        if not score_str or not isinstance(score_str, str) or '-' not in score_str:
            return None
        parts = score_str.split('-')
        # Handle cases where score might be dates like "2月2日" which are invalid scores
        # The CSV provided seems to have some of these in 'final_score' column for postponed/cancelled matches?
        # Looking at the csv content: "1877368... 2月2日" -> Invalid.
        # "3-0" -> Valid.
        if not parts[0].isdigit() or not parts[1].isdigit():
            return None
        return int(parts[0]), int(parts[1])
    except:
        return None

def get_result(home_score, away_score):
    if home_score > away_score:
        return 'Home Win'
    elif home_score == away_score:
        return 'Draw'
    else:
        return 'Away Win'

def analyze_csv(file_path):
    rows = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Filter valid matches
    valid_matches = []
    for row in rows:
        score = parse_score(row['final_score'])
        if score:
            row['parsed_result'] = get_result(score[0], score[1])
            valid_matches.append(row)
    
    print(f"Total rows with valid scores: {len(valid_matches)}")
    print("-" * 40)

    # 1. Analyze William Hill specifically
    wh_rows = [r for r in valid_matches if r['bookmaker'] == 'William Hill']
    wh_144 = [r for r in wh_rows if r['final_win'] == '1.44']

    if wh_144:
        print(f"Analysis for William Hill with Final Home Win Odds = 1.44")
        print(f"Total Matches: {len(wh_144)}")
        
        counts = defaultdict(int)
        for r in wh_144:
            counts[r['parsed_result']] += 1
            
        total = len(wh_144)
        for res in ['Home Win', 'Draw', 'Away Win']:
            count = counts[res]
            pct = (count / total) * 100
            print(f"  {res}: {count} ({pct:.2f}%)")

        home_wins_count = counts['Home Win']

        # Specific Goal Difference Analysis
        wins_by_more_than_1 = [r for r in wh_144 if parse_score(r['final_score'])[0] - parse_score(r['final_score'])[1] > 1]
        wins_gd2_count = len(wins_by_more_than_1)
        wins_gd2_pct_total = (wins_gd2_count / total) * 100
        wins_gd2_pct_wins = (wins_gd2_count / home_wins_count * 100) if home_wins_count > 0 else 0

        print(f"\n  Home Win by > 1 goal (GD >= 2): {wins_gd2_count}")
        print(f"    % of All 1.44 Matches: {wins_gd2_pct_total:.2f}%")
        print(f"    % of Winning Matches:  {wins_gd2_pct_wins:.2f}%")

        big_wins = [r for r in wh_144 if parse_score(r['final_score'])[0] - parse_score(r['final_score'])[1] > 2]
        big_wins_count = len(big_wins)
        big_wins_pct_total = (big_wins_count / total) * 100
        # Percentage relative to wins
        home_wins_count = counts['Home Win']
        big_wins_pct_wins = (big_wins_count / home_wins_count * 100) if home_wins_count > 0 else 0

        print(f"\n  Home Win by > 2 goals (GD >= 3): {big_wins_count}")
        print(f"    % of All 1.44 Matches: {big_wins_pct_total:.2f}%")
        print(f"    % of Winning Matches:  {big_wins_pct_wins:.2f}%")

        print("\nOdds Movement (Init -> Final) for WH 1.44:")
        moved_up = []
        moved_down = []
        same = []

        for r in wh_144:
            try:
                init = float(r['init_win'])
                final = float(r['final_win'])
                if final > init:
                    moved_up.append(r)
                elif final < init:
                    moved_down.append(r)
                else:
                    same.append(r)
            except:
                continue
        
        print(f"  Odds Drifted Up (Drift): {len(moved_up)}")
        print(f"  Odds Dropped (Steam): {len(moved_down)}")
        print(f"  Odds Unchanged: {len(same)}")

        if moved_down:
            down_counts = defaultdict(int)
            for r in moved_down: down_counts[r['parsed_result']] += 1
            win_pct = (down_counts['Home Win'] / len(moved_down)) * 100
            print(f"  \n  When Odds Dropped to 1.44, Home Win %: {win_pct:.2f}% ({down_counts['Home Win']}/{len(moved_down)})")

        if moved_up:
            up_counts = defaultdict(int)
            for r in moved_up: up_counts[r['parsed_result']] += 1
            win_pct = (up_counts['Home Win'] / len(moved_up)) * 100
            print(f"  When Odds Rose to 1.44, Home Win %: {win_pct:.2f}% ({up_counts['Home Win']}/{len(moved_up)})")
            
        if same:
            same_counts = defaultdict(int)
            for r in same: same_counts[r['parsed_result']] += 1
            win_pct = (same_counts['Home Win'] / len(same)) * 100
            print(f"  When Odds Started at 1.44, Home Win %: {win_pct:.2f}% ({same_counts['Home Win']}/{len(same)})")

    print("-" * 40)

    # 3. League Analysis
    print("Breakdown by League (WH 1.44):")
    league_stats = defaultdict(lambda: defaultdict(int))
    for r in wh_144:
        league_stats[r['league_name']]['total'] += 1
        league_stats[r['league_name']][r['parsed_result']] += 1
    
    # Sort by Win %
    league_list = []
    for league, stats in league_stats.items():
        win_pct = (stats['Home Win'] / stats['total']) * 100
        league_list.append((league, win_pct, stats['total'], stats['Home Win'], stats['Draw'], stats['Away Win']))
    
    league_list.sort(key=lambda x: x[1], reverse=True)
    
    print(f"{'League':<20} | {'Win %':<8} | {'Total':<6} | {'W':<4} | {'D':<4} | {'L':<4}")
    print("-" * 60)
    for l in league_list:
        print(f"{l[0]:<20} | {l[1]:6.2f}%  | {l[2]:<6} | {l[3]:<4} | {l[4]:<4} | {l[5]:<4}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_csv(sys.argv[1])
    else:
        print("Please provide a CSV file path.")