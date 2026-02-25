#!/usr/bin/env python3
"""
2.analyze_odds.py - 赔率数据分析

分析赔率数据分布，生成SAX编码配置

功能：
- 分析赔率数据分布
- 计算最优字母表大小
- 生成配置文件

注意：本文件可直接引用 find_similar_matches.py 中的函数
"""

import json
import numpy as np
from collections import Counter
from scipy import stats
import math
import os
import glob
import sys

# 引用 find_similar_matches.py 中的函数
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from find_similar_matches import (
    download_match_data,
    extract_bet365_odds,
    resample_odds_to_fixed_timeline,
    normalize_odds_series,
    get_interleaved_series,
    get_delta_series,
    SAXEncoder,
)


def calculate_entropy(labels):
    """计算序列的熵"""
    if len(labels) == 0:
        return 0
    counts = Counter(labels)
    total = sum(counts.values())
    probs = [count / total for count in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def analyze_match_data(match_id: str, match_time: str = None):
    """
    分析单场比赛的赔率数据

    Args:
        match_id: 比赛ID
        match_time: 比赛时间（可选）

    Returns:
        标准化后的数据统计
    """
    # 下载数据
    content = download_match_data(match_id)
    if not content:
        return None

    # 提取赔率
    match_odds = extract_bet365_odds(content)
    if not match_odds:
        return None

    # 使用比赛时间（如果未提供）
    if match_time is None:
        match_time = match_odds.match_time

    # 预处理（20点固定时间轴）
    resampled = resample_odds_to_fixed_timeline(match_odds.running_odds, match_time)
    if not resampled or len(resampled["home"]) < 2:
        return None

    # Z-Score 标准化
    normalized = normalize_odds_series(resampled)

    # 交错序列
    interleaved = get_interleaved_series(normalized)

    # 差值序列
    delta = get_delta_series(normalized)

    return {
        "match_id": match_id,
        "home_team": match_odds.hometeam,
        "away_team": match_odds.guestteam,
        "match_time": match_time,
        "raw_records": len(match_odds.running_odds),
        "resampled_points": len(resampled["home"]),
        "home_series": resampled["home"],
        "draw_series": resampled["draw"],
        "away_series": resampled["away"],
        "normalized_home": normalized["home"],
        "normalized_draw": normalized["draw"],
        "normalized_away": normalized["away"],
        "interleaved": interleaved,
        "delta": delta,
    }


def analyze_multiple_matches(match_ids: list) -> dict:
    """
    分析多场比赛的赔率数据

    Args:
        match_ids: 比赛ID列表

    Returns:
        统计结果
    """
    all_normalized = []
    all_interleaved = []
    all_delta = []
    success_count = 0
    failed_count = 0

    for mid in match_ids:
        result = analyze_match_data(str(mid))
        if result:
            all_normalized.extend(result["normalized_home"])
            all_normalized.extend(result["normalized_draw"])
            all_normalized.extend(result["normalized_away"])
            all_interleaved.extend(result["interleaved"])
            all_delta.extend(result["delta"])
            success_count += 1
        else:
            failed_count += 1

    if not all_normalized:
        return None

    all_normalized = np.array(all_normalized)
    all_interleaved = np.array(all_interleaved)
    all_delta = np.array(all_delta)

    # 统计
    return {
        "total_matches": len(match_ids),
        "success_count": success_count,
        "failed_count": failed_count,
        "total_data_points": len(all_normalized),
        # 原始数据统计
        "mean": float(np.mean(all_normalized)),
        "std": float(np.std(all_normalized)),
        "min": float(np.min(all_normalized)),
        "max": float(np.max(all_normalized)),
        # 正态性检验
        "skewness": float(stats.skew(all_normalized)),
        "kurtosis": float(stats.kurtosis(all_normalized)),
        # 序列长度统计
        "interleaved_mean": float(np.mean(all_interleaved)),
        "interleaved_std": float(np.std(all_interleaved)),
        "delta_mean": float(np.mean(all_delta)),
        "delta_std": float(np.std(all_delta)),
    }


def optimize_alphabet_size(normalized_data: list) -> dict:
    """
    优化字母表大小

    Args:
        normalized_data: 标准化后的数据

    Returns:
        各字母表大小的分析结果
    """
    normalized = np.array(normalized_data)
    results = {}

    for size in range(3, 11):
        # 使用经验分位数
        bins = np.percentile(normalized, np.linspace(0, 100, size + 1))
        digitized = np.digitize(normalized, bins[1:-1])
        entropy = calculate_entropy(digitized)
        efficiency = entropy / math.log2(size)

        results[size] = {
            "entropy": float(entropy),
            "efficiency": float(efficiency),
            "breakpoints": [float(b) for b in bins[1:-1]],
        }

    return results


def generate_sax_config(
    bookmaker_name: str = "Bet 365", match_ids: list = None
) -> dict:
    """
    生成SAX编码配置文件

    Args:
        bookmaker_name: 庄家名称
        match_ids: 比赛ID列表（用于分析）

    Returns:
        配置字典
    """
    if match_ids is None:
        # 默认测试数据
        match_ids = list(range(2789379, 2789389))

    # 分析数据
    analysis = analyze_multiple_matches(match_ids)
    if not analysis:
        print("错误: 无法分析数据")
        return None

    # 优化字母表大小
    all_data = []
    for mid in match_ids:
        result = analyze_match_data(str(mid))
        if result:
            all_data.extend(result["normalized_home"])
            all_data.extend(result["normalized_draw"])
            all_data.extend(result["normalized_away"])

    alphabet_optimization = optimize_alphabet_size(all_data)

    # 选择最优字母表大小
    best_size = 4  # 默认
    best_efficiency = 0
    for size, data in alphabet_optimization.items():
        if data["efficiency"] > best_efficiency:
            best_efficiency = data["efficiency"]
            best_size = size

    # 生成配置
    config = {
        "bookmaker": bookmaker_name,
        "strategy": "interleaved",  # 使用交错编码
        "word_size": 8,
        "alphabet_size": best_size,
        "interpolate_len": 32,
        "breakpoints_type": "gaussian",  # 使用高斯断点
        "empirical_breakpoints": alphabet_optimization[best_size]["breakpoints"],
        "gaussian_breakpoints": [
            float(stats.norm.ppf(i / best_size)) for i in range(1, best_size)
        ],
        "data_preprocessing": {
            "resample_points": 20,
            "normalization": "z_score",
            "description": "20点固定时间轴 + Z-Score标准化",
        },
        "metadata": {
            "n_matches_analyzed": analysis["success_count"],
            "n_data_points": analysis["total_data_points"],
            "skewness": analysis["skewness"],
            "kurtosis": analysis["kurtosis"],
        },
    }

    return config


def print_analysis_report(analysis: dict):
    """打印分析报告"""
    print("=" * 70)
    print("赔率数据分布分析报告")
    print("=" * 70)

    print(f"\n[1] 数据概况")
    print(f"  - 比赛总数: {analysis['total_matches']}")
    print(f"  - 成功分析: {analysis['success_count']}")
    print(f"  - 失败: {analysis['failed_count']}")
    print(f"  - 数据点数: {analysis['total_data_points']}")

    print(f"\n[2] 标准化后数据统计")
    print(f"  - 均值: {analysis['mean']:.4f}")
    print(f"  - 标准差: {analysis['std']:.4f}")
    print(f"  - 范围: [{analysis['min']:.4f}, {analysis['max']:.4f}]")

    print(f"\n[3] 正态性检验")
    print(f"  - 偏度: {analysis['skewness']:.4f} (越接近0越对称)")
    print(f"  - 峰度: {analysis['kurtosis']:.4f} (越接近0越符合正态)")

    print(f"\n[4] 序列特征")
    print(f"  - 交错序列均值: {analysis['interleaved_mean']:.4f}")
    print(f"  - 交错序列标准差: {analysis['interleaved_std']:.4f}")
    print(f"  - 差值序列均值: {analysis['delta_mean']:.4f}")
    print(f"  - 差值序列标准差: {analysis['delta_std']:.4f}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="赔率数据分析")
    parser.add_argument("--match-id", type=str, help="单个比赛ID")
    parser.add_argument("--match-ids", type=str, help="比赛ID列表(逗号分隔)")
    parser.add_argument(
        "--output", type=str, default="sax_config.json", help="输出配置文件"
    )
    parser.add_argument("--bookmaker", type=str, default="Bet 365", help="庄家名称")
    parser.add_argument("--report", action="store_true", help="打印分析报告")

    args = parser.parse_args()

    # 确定要分析的比赛
    match_ids = []
    if args.match_id:
        match_ids = [args.match_id]
    elif args.match_ids:
        match_ids = [m.strip() for m in args.match_ids.split(",")]
    else:
        # 默认分析10场比赛
        match_ids = list(range(2789379, 2789389))

    # 分析数据
    print(f"分析 {len(match_ids)} 场比赛...")
    analysis = analyze_multiple_matches(match_ids)

    if not analysis:
        print("错误: 无法分析数据")
        return

    # 打印报告
    if args.report:
        print_analysis_report(analysis)

    # 生成配置
    config = generate_sax_config(args.bookmaker, match_ids)
    if config:
        # 保存配置
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"\n配置已保存至: {args.output}")

        print(f"\n推荐参数:")
        print(f"  - word_size: {config['word_size']}")
        print(f"  - alphabet_size: {config['alphabet_size']}")
        print(f"  - interpolate_len: {config['interpolate_len']}")
        print(f"  - strategy: {config['strategy']}")


if __name__ == "__main__":
    main()
