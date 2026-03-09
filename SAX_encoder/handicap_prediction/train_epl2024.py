#!/usr/bin/env python3
"""
亚盘预测 - 英超2024-2025赛季训练
"""

import os
import re
from pathlib import Path
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
    """解析比赛JS文件"""
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
    """解析亚盘HTML"""
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
    """从比赛分析页面提取比赛结果"""
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
    print("亚盘预测 - 英超2024-2025赛季")
    print("=" * 50)

    # 英超2024-2025
    matches_dir = BASE_DIR / 'rawdata/英超/2024-2025/matches'
    handicap_dir = BASE_DIR / 'download_module/rawdata/odds/total_handicap'
    analysis_dir = BASE_DIR / 'download_module/rawdata/比赛分析'

    files = list(matches_dir.glob('*.js'))
    print(f"\n[1] 处理 {len(files)} 场比赛 (英超2024-2025)...")

    data = []
    for i, f in enumerate(files):
        if i % 50 == 0:
            print(f"  进度 {i}/{len(files)}...")

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
    print(f"\n有亚盘和比赛结果: {len(df)} 场")

    if len(df) < 50:
        print("数据太少!")
        return

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

    df['label'] = df.apply(calc_label, axis=1)
    df = df[df['label'].isin([0, 1])]

    print(f"\n有效数据: {len(df)} 场")
    print(f"赢盘: {(df['label']==1).sum()}, 输盘: {(df['label']==0).sum()}")

    # 特征
    df['eu_implied_home'] = 1 / df['eu_home']
    df['eu_implied_away'] = 1 / df['eu_away']
    total = df['eu_implied_home'] + df['eu_implied_away']
    df['eu_implied_home'] /= total
    df['eu_implied_away'] /= total

    features = ['eu_home', 'eu_draw', 'eu_away', 'asian_handicap', 'asian_home', 'asian_away', 'eu_implied_home', 'eu_implied_away']
    X = df[features].fillna(0).values
    y = df['label'].values

    # 训练
    print("\n[2] 训练模型...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

    import lightgbm as lgb

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=features)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    model = lgb.train(
        {'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1, 'num_threads': 8},
        train_data,
        num_boost_round=200,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(20)]
    )

    y_pred = model.predict(X_test)
    acc = ((y_pred > 0.5) == y_test).mean()
    print(f"\n测试准确率: {acc:.4f}")

    # 特征重要性
    imp = pd.DataFrame({'feature': features, 'imp': model.feature_importance()})
    print("\n特征重要性:")
    print(imp.sort_values('imp', ascending=False).to_string(index=False))

    # 保存
    model.save_model(str(BASE_DIR / 'SAX_encoder/handicap_prediction/model_epl_2024.txt'))
    print("\n模型已保存!")

    print("\n完成!")


if __name__ == "__main__":
    main()
