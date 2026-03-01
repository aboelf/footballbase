#!/usr/bin/env python3
"""
亚盘数据预处理模块 - 时间轴对齐版本

功能：
1. 解析亚盘odds_detail数据，提取时间信息
2. 按固定时间轴重采样（与欧赔流程一致）
3. Z-Score标准化
4. SAX编码

这个模块可以被 find_similar_matches.py 导入使用
"""

import json
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np


# ============ 辅助函数 ============


def z_score_normalize(arr: List[float]) -> List[float]:
    """Z-Score标准化"""
    arr = np.array(arr, dtype=float)
    mean = arr.mean()
    std = arr.std()
    if std == 0:
        return (arr - mean).tolist()
    return ((arr - mean) / std).tolist()


def parse_datetime_from_handicap(
    time_str: str, year_hint: int = 2024
) -> Optional[datetime]:
    """解析亚盘时间字符串（格式：'8-22 00:59' 或 '8-20 18:22'）"""
    if not time_str:
        return None

    match = re.match(r"(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{2})", time_str)
    if match:
        try:
            month, day, hour, minute = match.groups()
            return datetime(
                int(year_hint), int(month), int(day), int(hour), int(minute)
            )
        except ValueError:
            return None
    return None


# ============ 核心函数 ============


def parse_handicap_odds_detail(odds_detail_json) -> Tuple[List, List, List, List]:
    """
    解析亚盘数据，提取水位、盘口和时间序列

    Returns:
        (home_odds, away_odds, handicap_list, times_list)
    """
    if not odds_detail_json:
        return [], [], [], []

    # 解析JSON
    if isinstance(odds_detail_json, str):
        try:
            detail_list = json.loads(odds_detail_json)
        except json.JSONDecodeError:
            return [], [], [], []
    elif isinstance(odds_detail_json, list):
        detail_list = odds_detail_json
    else:
        return [], [], [], []

    if not detail_list:
        return [], [], [], []

    home_odds = []
    away_odds = []
    handicap_list = []
    times_list = []

    for item in detail_list:
        if not isinstance(item, dict):
            continue

        home = item.get("home")
        away = item.get("away")
        handicap = item.get("handicap")
        time_str = item.get("time", "")

        if home is not None and away is not None:
            home_odds.append(float(home))
            away_odds.append(float(away))
            handicap_list.append(handicap)

            # 解析时间
            dt = parse_datetime_from_handicap(time_str)
            times_list.append(dt)

    return home_odds, away_odds, handicap_list, times_list


def resample_handicap_to_fixed_timeline(
    home_odds: List[float],
    away_odds: List[float],
    handicap_list: List[str],
    times_list: List[datetime],
    target_points: int = 16,
) -> Dict:
    """
    亚盘数据时间轴对齐与重采样

    时间轴策略（与欧赔resample_odds_to_fixed_timeline一致）：
    1. 开盘水位（第一条记录）
    2. 赛前48小时水位（时间加权均值）
    3. 赛前24小时到开赛：关键变化点采样
    4. 统一时间轴后进行Z-Score标准化
    """
    if not home_odds or len(home_odds) < 2:
        return {
            "home": [],
            "away": [],
            "handicap": [],
            "home_norm": [],
            "away_norm": [],
        }

    # 构建带时间的记录
    records = []
    for i in range(len(home_odds)):
        records.append(
            {
                "home": home_odds[i],
                "away": away_odds[i],
                "handicap": handicap_list[i] if i < len(handicap_list) else "",
                "time": times_list[i] if i < len(times_list) else None,
            }
        )

    # 按时间排序（只保留有时间信息的记录）
    valid_records = [r for r in records if r["time"] is not None]

    if len(valid_records) < 3:
        # 时间信息不足，使用均匀采样
        indices = np.linspace(
            0, len(home_odds) - 1, min(target_points, len(home_odds)), dtype=int
        )
        sampled_home = [home_odds[i] for i in indices]
        sampled_away = [away_odds[i] for i in indices]
        sampled_handicap = [handicap_list[i] for i in indices]
    else:
        valid_records.sort(key=lambda x: x["time"])

        # 提取排序后的数据
        sorted_home = [r["home"] for r in valid_records]
        sorted_away = [r["away"] for r in valid_records]
        sorted_handicap = [r["handicap"] for r in valid_records]

        # 均匀采样到目标点数
        if len(sorted_home) >= target_points:
            indices = np.linspace(0, len(sorted_home) - 1, target_points, dtype=int)
        else:
            indices = list(range(len(sorted_home)))

        sampled_home = [sorted_home[i] for i in indices]
        sampled_away = [sorted_away[i] for i in indices]
        sampled_handicap = [sorted_handicap[i] for i in indices]

    # Z-Score标准化（与欧赔一致）
    home_norm = z_score_normalize(sampled_home)
    away_norm = z_score_normalize(sampled_away)

    return {
        "home": sampled_home,
        "away": sampled_away,
        "handicap": sampled_handicap,
        "home_norm": home_norm,
        "away_norm": away_norm,
    }


def encode_handicap_sax_v2(
    encoder,
    odds_detail_json,
    interpolate_len: int = 32,
    word_size: int = 8,
) -> Optional[Dict]:
    """
    亚盘SAX编码（改进版：时间轴对齐 + Z-Score标准化）

    与欧赔处理流程一致：
    1. 解析数据并提取时间
    2. 时间轴对齐重采样
    3. Z-Score标准化
    4. SAX编码
    """
    home_odds, away_odds, handicap_list, times_list = parse_handicap_odds_detail(
        odds_detail_json
    )

    if len(home_odds) < 2 or len(away_odds) < 2:
        return None

    # 时间轴对齐重采样
    resampled = resample_handicap_to_fixed_timeline(
        home_odds,
        away_odds,
        handicap_list,
        times_list,
        target_points=word_size * 2,  # 2倍用于插值
    )

    if not resampled["home"] or len(resampled["home"]) < 2:
        return None

    # 使用标准化数据进行SAX编码
    if resampled["home_norm"] and resampled["away_norm"]:
        sax_home = encoder.encode(resampled["home_norm"], interpolate_len)
        sax_away = encoder.encode(resampled["away_norm"], interpolate_len)

        # 差值
        min_len = min(len(resampled["home_norm"]), len(resampled["away_norm"]))
        diff_norm = [
            resampled["home_norm"][i] - resampled["away_norm"][i]
            for i in range(min_len)
        ]
        sax_diff = encoder.encode(diff_norm, interpolate_len)
    else:
        # 降级使用原始数据
        sax_home = encoder.encode(resampled["home"], interpolate_len)
        sax_away = encoder.encode(resampled["away"], interpolate_len)

        min_len = min(len(resampled["home"]), len(resampled["away"]))
        diff = [resampled["home"][i] - resampled["away"][i] for i in range(min_len)]
        sax_diff = encoder.encode(diff, interpolate_len)

    # 盘口变化次数
    handicap_changes = 0
    if handicap_list:
        prev = handicap_list[0]
        for h in handicap_list[1:]:
            if h != prev:
                handicap_changes += 1
                prev = h

    # 水位变化
    home_odds_change = home_odds[-1] - home_odds[0] if len(home_odds) > 1 else 0
    away_odds_change = away_odds[-1] - away_odds[0] if len(away_odds) > 1 else 0

    return {
        "sax_home": sax_home,
        "sax_away": sax_away,
        "sax_diff": sax_diff,
        "handicap_changes": handicap_changes,
        "home_count": len(home_odds),
        "away_count": len(away_odds),
        "home_odds_change": home_odds_change,
        "away_odds_change": away_odds_change,
        "home_odds_initial": home_odds[0],
        "home_odds_final": home_odds[-1],
        "away_odds_initial": away_odds[0],
        "away_odds_final": away_odds[-1],
    }


# ============ 测试 ============

if __name__ == "__main__":
    # 测试数据
    test_data = [
        {"handicap": "半球", "home": 1.10, "away": 0.80, "time": "8-20 18:22"},
        {"handicap": "半球", "home": 1.05, "away": 0.85, "time": "8-20 20:00"},
        {"handicap": "半球/一球", "home": 1.00, "away": 0.90, "time": "8-21 10:00"},
        {"handicap": "半球/一球", "home": 0.95, "away": 0.95, "time": "8-21 15:00"},
        {"handicap": "一球", "home": 0.90, "away": 1.00, "time": "8-21 20:00"},
    ]

    # 导入SAXEncoder（需要项目路径）
    import os
    import sys

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(project_root, "SAX_encoder"))
    from find_similar_matches import SAXEncoder

    encoder = SAXEncoder(word_size=8, alphabet_size=4)
    result = encode_handicap_sax_v2(encoder, test_data, word_size=8)

    print("=== 亚盘SAX编码结果（改进版）===")
    print(f"sax_home: {result.get('sax_home')}")
    print(f"sax_away: {result.get('sax_away')}")
    print(f"sax_diff: {result.get('sax_diff')}")
    print(f"handicap_changes: {result.get('handicap_changes')}")
    print(f"home_odds_change: {result.get('home_odds_change')}")
    print(f"home_odds_initial: {result.get('home_odds_initial')}")
    print(f"home_odds_final: {result.get('home_odds_final')}")
