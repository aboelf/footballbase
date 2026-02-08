#!/usr/bin/env python3
"""
Bet 365 赔率 SAX 编码主入口

对所有比赛数据进行 SAX 编码：
1. 分别编码 (Individual Encoding) - 三种赔率独立编码
2. 联合编码 (Joint Encoding) - 交错拼接 + 差值编码

输出:
  - bet365_sax_individual.json: 分别编码结果
  - bet365_sax_joint.json: 联合编码结果
"""

import json
import os
import sys
from datetime import datetime
from collections import Counter

from sax_encoder import (
    SAXEncoder,
    extract_odds_series,
    create_joint_series,
    create_delta_series
)

# SAX 编码参数（基于 analyze_odds.py 的分析结果）
WORD_SIZE = 8  # 分段数
ALPHABET_SIZE = 7  # 字母表大小 (a-g)
INTERPOLATE_LEN = 32  # 插值统一长度


def process_match_individual(match, encoder):
    """
    处理比赛 - 分别编码方案

    Returns:
        编码结果 dict 或 None
    """
    home, draw, away = extract_odds_series(match)
    if home is None:
        return None

    return {
        'scheduleId': match.get('scheduleId'),
        'hometeam': match.get('hometeam'),
        'guestteam': match.get('guestteam'),
        'matchTime': match.get('matchTime'),
        'season': match.get('season'),
        'sax_home': encoder.encode(home, INTERPOLATE_LEN),
        'sax_draw': encoder.encode(draw, INTERPOLATE_LEN),
        'sax_away': encoder.encode(away, INTERPOLATE_LEN),
        'stats': {
            'home_mean': round(float(np.mean(home)), 3),
            'draw_mean': round(float(np.mean(draw)), 3),
            'away_mean': round(float(np.mean(away)), 3),
            'home_std': round(float(np.std(home)), 3),
            'draw_std': round(float(np.std(draw)), 3),
            'away_std': round(float(np.std(away)), 3),
            'running_odds_count': len(home)
        }
    }


def process_match_joint(match, encoder):
    """
    处理比赛 - 联合编码方案

    Returns:
        编码结果 dict 或 None
    """
    home, draw, away = extract_odds_series(match)
    if home is None:
        return None

    # 交错拼接序列
    joint_series = create_joint_series(home, draw, away)

    # 差值序列
    delta_series = create_delta_series(home, away)

    return {
        'scheduleId': match.get('scheduleId'),
        'hometeam': match.get('hometeam'),
        'guestteam': match.get('guestteam'),
        'matchTime': match.get('matchTime'),
        'season': match.get('season'),
        'sax_interleaved': encoder.encode(joint_series, INTERPOLATE_LEN * 3),
        'sax_delta': encoder.encode(delta_series, INTERPOLATE_LEN),
        'stats': {
            'home_mean': round(float(np.mean(home)), 3),
            'draw_mean': round(float(np.mean(draw)), 3),
            'away_mean': round(float(np.mean(away)), 3),
            'running_odds_count': len(home)
        }
    }


def find_data_file():
    """查找数据文件"""
    possible_paths = [
        'bet365_details/bet365_details.json',
        '../SAX encoder/bet365_details/bet365_details.json',
        '../SAX encoder/SAX encoder/bet365_details/bet365_details.json',
        'SAX encoder/bet365_details/bet365_details.json',
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    # 递归搜索
    for root, dirs, files in os.walk('.'):
        for f in files:
            if f == 'bet365_details.json':
                return os.path.join(root, f)

    return None


def main():
    print("=" * 60)
    print("Bet 365 赔率 SAX 编码")
    print("=" * 60)

    # 初始化参数
    config_path = 'sax_config.json'
    global WORD_SIZE, ALPHABET_SIZE, INTERPOLATE_LEN
    
    # 初始化编码器
    if os.path.exists(config_path):
        encoder = SAXEncoder(config_path=config_path)
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        WORD_SIZE = config.get("word_size", WORD_SIZE)
        ALPHABET_SIZE = config.get("alphabet_size", ALPHABET_SIZE)
        INTERPOLATE_LEN = config.get("interpolate_len", INTERPOLATE_LEN)
        word_size, alphabet_size, interpolate_len = WORD_SIZE, ALPHABET_SIZE, INTERPOLATE_LEN
    else:
        encoder = SAXEncoder(word_size=WORD_SIZE, alphabet_size=ALPHABET_SIZE)
        word_size, alphabet_size, interpolate_len = WORD_SIZE, ALPHABET_SIZE, INTERPOLATE_LEN

    print(f"\nSAX 参数:")
    print(f"  - word_size: {word_size}")
    print(f"  - alphabet_size: {alphabet_size}")
    print(f"  - interpolate_len: {interpolate_len}")
    print(f"  - alphabet: {encoder.alphabet}")
    print(f"  - breakpoints: {encoder.breakpoints.round(3)}")

    # 查找数据文件
    data_file = find_data_file()
    if not data_file:
        print("\n错误: 找不到 bet365_details.json 文件")
        sys.exit(1)

    print(f"\n数据文件: {data_file}")

    # 加载数据
    print("\n加载数据...")
    with open(data_file, 'r', encoding='utf-8') as f:
        matches = json.load(f)
    print(f"  比赛总数: {len(matches)}")

    # 处理比赛
    print(f"\n处理比赛...")

    results_individual = []
    results_joint = []
    errors = []
    processed = 0

    for i, match in enumerate(matches):
        # 分别编码
        result_ind = process_match_individual(match, encoder)
        if result_ind:
            results_individual.append(result_ind)

        # 联合编码
        result_joint = process_match_joint(match, encoder)
        if result_joint:
            results_joint.append(result_joint)

        processed += 1
        if (i + 1) % 200 == 0:
            print(f"  已处理: {i + 1}/{len(matches)}")

    print(f"  完成! 成功编码: {len(results_individual)}")

    # 统计各类型分布
    print(f"\n编码结果统计:")

    # 分别编码 - 各类型分布
    home_counts = Counter(r['sax_home'] for r in results_individual)
    draw_counts = Counter(r['sax_draw'] for r in results_individual)
    away_counts = Counter(r['sax_away'] for r in results_individual)

    print(f"\n分别编码 - 模式分布 (Top 10):")
    print(f"  主胜 (home): {home_counts.most_common(10)}")
    print(f"  平局 (draw): {draw_counts.most_common(10)}")
    print(f"  客胜 (away): {away_counts.most_common(10)}")

    # 联合编码
    joint_counts = Counter(r['sax_interleaved'] for r in results_joint)
    delta_counts = Counter(r['sax_delta'] for r in results_joint)

    print(f"\n联合编码 - 模式分布 (Top 10):")
    print(f"  交错拼接: {joint_counts.most_common(10)}")
    print(f"  差值编码: {delta_counts.most_common(10)}")

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    output_dir = 'bet365_sax_results'
    os.makedirs(output_dir, exist_ok=True)

    # 保存分别编码结果
    output_individual = os.path.join(output_dir, 'bet365_sax_individual.json')
    with open(output_individual, 'w', encoding='utf-8') as f:
        json.dump(results_individual, f, ensure_ascii=False, indent=2)
    print(f"\n分别编码结果已保存: {output_individual}")

    # 保存联合编码结果
    output_joint = os.path.join(output_dir, 'bet365_sax_joint.json')
    with open(output_joint, 'w', encoding='utf-8') as f:
        json.dump(results_joint, f, ensure_ascii=False, indent=2)
    print(f"联合编码结果已保存: {output_joint}")

    print(f"\n{'=' * 60}")
    print("完成!")
    print("=" * 60)


if __name__ == '__main__':
    import numpy as np
    main()
