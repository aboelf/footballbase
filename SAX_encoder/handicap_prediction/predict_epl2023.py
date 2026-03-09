#!/usr/bin/env python3
"""
使用训练好的模型预测2023-2024赛季英超
"""

import os
import re
from pathlib import Path
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent.parent
HANDICAP_MAP = {
    '受三球': -3.0, '受二球/二球半': -2.25, '受二球半': -2.5, '受二球半/三球': -2.75,
    '受三球半': -3.5, '受三球半/四球': -3.75, '受四球': -4.0, '受四球半': -4.5,
    '受四球半/五球': -4.75, '受五球': -5.0, '受半球': -0.5, '受半/一': -0.75,
    '受一球': -1.0, '受一球/球半': -1.25, '受球半': -1.5, '受球半/两球': -1.75,
    '受两球': -2.0, '受两球/两球半': -2.25, '平手': 0.0, '平手/半球': 0.25,
    '半球': 0.5, '半球/一球': 0.75, '一球': 1.0, '一球/球半': 1.25, '球半': 1.5,
    '球半/两球': 1.75, '两球': 2.0, '两球/两球半': 2.25, '两球半': 2.5,
    '两球半/三球': 2.75, '三球': 3.0, '三球半': 3.5,
}


def parse_match(filepath):
    try:
        content = open(filepath, 'r', encoding='utf-8').read()
    except:
        return None

    mid = re.search(r'ScheduleID=(\d+);', content)
    if not mid:
        return None

    home = re.search(r'hometeam="([^"]+)"', content)
    away = re.search(r'guestteam="([^"]+)"', content)

    bet365_id = re.search(r'"281\|([0-9]+)\|Bet 365\|', content)
    if not bet365_id:
        return None

    odds = re.search(rf'"281\|{bet365_id.group(1)}\|Bet 365\|([^|]+)\|([^|]+)\|([^|]+)\|', content)
    if not odds:
        return None

    return {
        'match_id': int(mid.group(1)),
        'home_team': home.group(1) if home else None,
        'away_team': away.group(1) if away else None,
        'eu_home': float(odds.group(1)) if odds.group(1) else None,
        'eu_draw': float(odds.group(2)) if odds.group(2) else None,
        'eu_away': float(odds.group(3)) if odds.group(3) else None,
    }


def parse_asian(filepath):
    try:
        html = open(filepath, 'rb').read().decode('gbk', errors='replace')
    except:
        return {}

    soup = BeautifulSoup(html, 'html.parser')
    for row in soup.find_all('tr', bgcolor=['#FFFFFF', '#FAFAFA']):
        cells = row.find_all('td')
        if len(cells) < 6 or not cells[1].get_text(strip=True).startswith('36*'):
            continue
        try:
            handicap = cells[4].get_text(strip=True)
            handicap = handicap.replace('受让', '受')  # 统一"受让"和"受"前缀
            return {
                'asian_home': float(cells[3].get_text(strip=True)),
                'asian_handicap': HANDICAP_MAP.get(handicap),
                'asian_away': float(cells[5].get_text(strip=True)),
            }
        except:
            return {}
    return {}


def parse_analysis_html(filepath):
    try:
        html = open(filepath, 'rb').read().decode('utf-8', errors='replace')
    except:
        return None

    score_match = re.search(r'<span class="row">\((\d+-\d+)\)</span>', html)
    if not score_match:
        return None

    return {'final_score': score_match.group(1)}


def main():
    print("=" * 50)
    print("预测 2023-2024 赛季英超")
    print("=" * 50)

    # 加载模型
    import lightgbm as lgb
    model_path = BASE_DIR / 'SAX_encoder/handicap_prediction/model_epl_2024.txt'
    model = lgb.Booster(model_file=str(model_path))
    print(f"模型已加载: {model_path}")

    # 2023-2024赛季
    matches_dir = BASE_DIR / 'rawdata/英超/2023-2024/matches'
    handicap_dir = BASE_DIR / 'download_module/rawdata/odds/total_handicap'
    analysis_dir = BASE_DIR / 'download_module/rawdata/比赛分析'

    files = list(matches_dir.glob('*.js'))
    print(f"\n[1] 处理 {len(files)} 场比赛 (英超2023-2024)...")

    data = []
    for i, f in enumerate(files):
        match = parse_match(str(f))
        if not match:
            continue

        match_id = match['match_id']

        # 亚盘
        asian_file = handicap_dir / f"{match_id}_total.html"
        if asian_file.exists():
            match.update(parse_asian(str(asian_file)))

        # 比赛结果
        analysis_file = analysis_dir / f"{match_id}.html"
        if analysis_file.exists():
            result = parse_analysis_html(str(analysis_file))
            if result:
                match.update(result)

        if match.get('asian_handicap') and match.get('final_score'):
            data.append(match)

    df = pd.DataFrame(data)
    print(f"有亚盘和比赛结果: {len(df)} 场")

    # 随机选20场
    np.random.seed(42)
    if len(df) > 20:
        df = df.sample(n=20, random_state=42)

    print(f"\n选取 {len(df)} 场比赛进行预测")

    # 计算标签
    def calc_label(r):
        h = r.get('asian_handicap')
        s = r.get('final_score')
        if pd.isna(h) or pd.isna(s):
            return -1
        try:
            hg, ag = map(int, s.split('-'))
        except:
            return -1
        diff = hg - ag - h
        return 1 if diff > 0 else 0

    df['actual_label'] = df.apply(calc_label, axis=1)
    df = df[df['actual_label'].isin([0, 1])]

    # 特征
    df['eu_implied_home'] = 1 / df['eu_home']
    df['eu_implied_away'] = 1 / df['eu_away']
    total = df['eu_implied_home'] + df['eu_implied_away']
    df['eu_implied_home'] /= total
    df['eu_implied_away'] /= total

    features = ['eu_home', 'eu_draw', 'eu_away', 'asian_handicap', 'asian_home', 'asian_away', 'eu_implied_home', 'eu_implied_away']
    X = df[features].fillna(0).values

    # 预测
    y_pred = model.predict(X)
    df['pred_label'] = (y_pred > 0.5).astype(int)
    df['pred_prob'] = y_pred

    # 准确率
    correct = (df['pred_label'] == df['actual_label']).sum()
    accuracy = correct / len(df)

    print(f"\n[2] 预测结果")
    print("=" * 80)
    print(f"{'主队':<20} {'客队':<20} {'比分':<6} {'盘口':<8} {'预测':<6} {'概率':<6} {'实际':<6} {'结果'}")
    print("-" * 80)

    for _, row in df.iterrows():
        home = row['home_team'][:18] if row['home_team'] else ''
        away = row['away_team'][:18] if row['away_team'] else ''
        score = row['final_score']
        handicap = row['asian_handicap']
        pred = '赢盘' if row['pred_label'] == 1 else '输盘'
        prob = f"{row['pred_prob']:.2f}"
        actual = '赢盘' if row['actual_label'] == 1 else '输盘'
        result = '✓' if row['pred_label'] == row['actual_label'] else '✗'

        print(f"{home:<20} {away:<20} {score:<6} {handicap:<8} {pred:<6} {prob:<6} {actual:<6} {result}")

    print("-" * 80)
    print(f"\n预测准确率: {accuracy:.2%} ({correct}/{len(df)})")
    print(f"赢盘预测: {(df['pred_label']==1).sum()}, 输盘预测: {(df['pred_label']==0).sum()}")
    print(f"实际赢盘: {(df['actual_label']==1).sum()}, 实际输盘: {(df['actual_label']==0).sum()}")

    print("\n完成!")


if __name__ == "__main__":
    main()
