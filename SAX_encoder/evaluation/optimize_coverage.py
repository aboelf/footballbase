#!/usr/bin/env python3
"""
SAX编码覆盖率优化方案

当前问题：
- 覆盖率仅14.5%（目标40%+）
- 只统计Top 20高纯模式
- 最小样本数要求10场
- 高纯阈值70%

优化策略：
1. 降低阈值参数
2. 增加Top N数量
3. 分层编码策略
4. 相似模式聚类
"""

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sax_encoder import SAXEncoder
from run_sax_evaluation import (
    load_odds_from_json,
    load_results_from_supabase,
    MatchData,
    PatternStats,
    encode_match,
)


def evaluate_with_params(
    matches, encoder, encoding_type, min_samples=5, purity_threshold=0.65, top_n=50
):
    """
    使用自定义参数评估覆盖率

    Args:
        min_samples: 最小样本数（降低可增加模式数量）
        purity_threshold: 纯度阈值（降低可纳入更多模式）
        top_n: Top N模式数量（增加可提高覆盖率）
    """
    # 编码所有比赛
    pattern_groups = defaultdict(list)
    for match in matches:
        if not match.get_result():
            continue
        pattern = encode_match(match, encoder, encoding_type)
        if pattern:
            pattern_groups[pattern].append(match)

    # 计算每个模式的统计
    pattern_stats = {}
    for pattern, group in pattern_groups.items():
        if len(group) < min_samples:
            continue

        stats = PatternStats(pattern=pattern, total=len(group))
        for match in group:
            result = match.get_result()
            if result == "home":
                stats.home_wins += 1
            elif result == "draw":
                stats.draws += 1
            elif result == "away":
                stats.away_wins += 1

        stats.compute_purity()
        if stats.purity >= purity_threshold:
            pattern_stats[pattern] = stats

    if not pattern_stats:
        return None

    # 按样本数排序（而非纯度）- 优先大样本模式
    sorted_patterns = sorted(
        pattern_stats.values(),
        key=lambda x: (-x.total, -x.purity),  # 优先样本数，其次纯度
    )

    top_patterns = sorted_patterns[:top_n]

    total_matches = sum(s.total for s in pattern_stats.values())
    covered_matches = sum(s.total for s in top_patterns)
    avg_purity = np.mean([s.purity for s in top_patterns]) if top_patterns else 0

    return {
        "min_samples": min_samples,
        "purity_threshold": purity_threshold,
        "top_n": top_n,
        "high_purity_patterns": len(pattern_stats),
        "coverage": covered_matches / len(matches) if matches else 0,
        "avg_purity": avg_purity,
        "total_covered_matches": covered_matches,
        "top_patterns": top_patterns[:10],
    }


def test_coverage_optimization(matches, bookmaker_name):
    """测试不同参数组合的覆盖率"""

    print(f"\n{'=' * 70}")
    print(f"覆盖率优化测试 - {bookmaker_name}")
    print(f"{'=' * 70}")
    print(f"总比赛数: {len(matches)}")

    # 使用individual策略（已验证为最佳）
    encoder = SAXEncoder(word_size=4, alphabet_size=3)
    encoding_type = "individual"

    # 测试不同参数组合
    test_configs = [
        # (min_samples, purity_threshold, top_n)
        (10, 0.70, 20, "原始参数"),
        (10, 0.65, 20, "降低纯度阈值到65%"),
        (10, 0.60, 20, "降低纯度阈值到60%"),
        (5, 0.70, 20, "降低最小样本到5"),
        (5, 0.65, 20, "双降参数"),
        (10, 0.70, 50, "增加Top N到50"),
        (5, 0.65, 50, "三优化参数"),
        (3, 0.60, 100, "激进参数"),
    ]

    results = []
    for min_samples, purity_threshold, top_n, desc in test_configs:
        print(f"\n测试: {desc}")
        print(
            f"  参数: min_samples={min_samples}, purity={purity_threshold}, top_n={top_n}"
        )

        result = evaluate_with_params(
            matches,
            encoder,
            encoding_type,
            min_samples=min_samples,
            purity_threshold=purity_threshold,
            top_n=top_n,
        )

        if result:
            print(f"  高纯模式数: {result['high_purity_patterns']}")
            print(f"  覆盖率: {result['coverage']:.2%}")
            print(f"  平均纯度: {result['avg_purity']:.2%}")
            print(f"  覆盖比赛数: {result['total_covered_matches']}")
            results.append((desc, result))

    # 打印对比表
    print(f"\n{'=' * 70}")
    print("覆盖率优化对比")
    print(f"{'=' * 70}")
    print(f"{'配置':<25} {'高纯模式':<10} {'覆盖率':<12} {'平均纯度':<12} {'覆盖场数'}")
    print("-" * 70)

    for desc, result in results:
        print(
            f"{desc:<25} {result['high_purity_patterns']:<10} "
            f"{result['coverage']:<12.2%} {result['avg_purity']:<12.2%} "
            f"{result['total_covered_matches']}"
        )

    return results


def hierarchical_encoding_strategy(matches):
    """
    分层编码策略 - 先粗分再细分

    第一层: 粗粒度分类 (4×3) - 区分主胜/平局/客胜大方向
    第二层: 细粒度分类 (6×4) - 在大方向内细分赔率变化模式
    """
    print(f"\n{'=' * 70}")
    print("分层编码策略")
    print(f"{'=' * 70}")

    # 第一层: 粗粒度
    coarse_encoder = SAXEncoder(word_size=4, alphabet_size=3)
    coarse_groups = defaultdict(list)

    for match in matches:
        if not match.get_result():
            continue
        pattern = encode_match(match, coarse_encoder, "individual")
        if pattern:
            coarse_groups[pattern].append(match)

    print(f"第一层粗分: {len(coarse_groups)} 个模式")

    # 统计每个粗模式的纯度
    coarse_stats = []
    for pattern, group in coarse_groups.items():
        if len(group) < 5:
            continue

        home = sum(1 for m in group if m.get_result() == "home")
        draw = sum(1 for m in group if m.get_result() == "draw")
        away = sum(1 for m in group if m.get_result() == "away")
        total = len(group)
        purity = max(home, draw, away) / total
        dominant = max(
            ["home", "draw", "away"],
            key=lambda x: {"home": home, "draw": draw, "away": away}[x],
        )

        coarse_stats.append(
            {
                "pattern": pattern,
                "total": total,
                "purity": purity,
                "dominant": dominant,
                "matches": group,
            }
        )

    # 按样本数排序
    coarse_stats.sort(key=lambda x: -x["total"])

    print(f"\n粗粒度模式 (样本>=5):")
    print(f"{'模式':<20} {'样本':<8} {'纯度':<8} {'主导结果'}")
    print("-" * 50)
    for s in coarse_stats[:10]:
        print(f"{s['pattern']:<20} {s['total']:<8} {s['purity']:<8.2%} {s['dominant']}")

    # 第二层: 对"混合模式"（纯度<70%）进行细分
    fine_encoder = SAXEncoder(word_size=6, alphabet_size=4)

    mixed_patterns = [s for s in coarse_stats if s["purity"] < 0.7]
    print(f"\n需要细分的混合模式: {len(mixed_patterns)} 个")

    total_covered = 0
    high_purity_count = 0

    for coarse in coarse_stats:
        if coarse["purity"] >= 0.7:
            # 已经是高纯模式，直接使用
            total_covered += coarse["total"]
            high_purity_count += 1
        else:
            # 需要细分
            fine_groups = defaultdict(list)
            for match in coarse["matches"]:
                fine_pattern = encode_match(match, fine_encoder, "individual")
                if fine_pattern:
                    fine_groups[fine_pattern].append(match)

            # 统计细分后的纯度
            for pattern, group in fine_groups.items():
                if len(group) < 5:
                    continue
                home = sum(1 for m in group if m.get_result() == "home")
                draw = sum(1 for m in group if m.get_result() == "draw")
                away = sum(1 for m in group if m.get_result() == "away")
                total = len(group)
                purity = max(home, draw, away) / total

                if purity >= 0.65:  # 细分后降低阈值
                    total_covered += total
                    high_purity_count += 1

    coverage = total_covered / len(matches) if matches else 0

    print(f"\n分层策略结果:")
    print(f"  高纯模式总数: {high_purity_count}")
    print(f"  覆盖率: {coverage:.2%}")
    print(f"  覆盖比赛数: {total_covered}")

    return coverage


def multi_bookmaker_fusion(b365_matches, easy_matches):
    """
    多庄家融合策略

    使用两个庄家的数据进行交叉验证
    当两个庄家的预测一致时，提高覆盖率
    """
    print(f"\n{'=' * 70}")
    print("多庄家融合策略")
    print(f"{'=' * 70}")

    encoder = SAXEncoder(word_size=4, alphabet_size=3)

    # 为每个庄家构建模式库
    def build_pattern_library(matches, bookmaker_name):
        pattern_stats = {}
        pattern_groups = defaultdict(list)

        for match in matches:
            if not match.get_result():
                continue
            pattern = encode_match(match, encoder, "individual")
            if pattern:
                pattern_groups[pattern].append(match)

        for pattern, group in pattern_groups.items():
            if len(group) < 5:
                continue

            home = sum(1 for m in group if m.get_result() == "home")
            draw = sum(1 for m in group if m.get_result() == "draw")
            away = sum(1 for m in group if m.get_result() == "away")
            total = len(group)
            purity = max(home, draw, away) / total

            if purity >= 0.65:
                dominant = max(
                    ["home", "draw", "away"],
                    key=lambda x: {"home": home, "draw": draw, "away": away}[x],
                )
                pattern_stats[pattern] = {
                    "prediction": dominant,
                    "purity": purity,
                    "total": total,
                }

        return pattern_stats

    b365_library = build_pattern_library(b365_matches, "Bet 365")
    easy_library = build_pattern_library(easy_matches, "Easybets")

    print(f"Bet 365 高纯模式: {len(b365_library)}")
    print(f"Easybets 高纯模式: {len(easy_library)}")

    # 统计融合覆盖率
    # 找出同时在两个庄家数据中存在的比赛
    common_matches = []
    b365_ids = {m.match_id for m in b365_matches}
    easy_ids = {m.match_id for m in easy_matches}
    common_ids = b365_ids & easy_ids

    print(f"\n共同比赛数: {len(common_ids)}")

    # 统计融合预测
    covered_by_both = 0
    covered_by_either = 0
    consistent_predictions = 0

    for match_id in common_ids:
        # 获取两个庄家的编码
        b365_match = next((m for m in b365_matches if m.match_id == match_id), None)
        easy_match = next((m for m in easy_matches if m.match_id == match_id), None)

        if not b365_match or not easy_match:
            continue

        b365_pattern = encode_match(b365_match, encoder, "individual")
        easy_pattern = encode_match(easy_match, encoder, "individual")

        b365_in_lib = b365_pattern in b365_library
        easy_in_lib = easy_pattern in easy_library

        if b365_in_lib or easy_in_lib:
            covered_by_either += 1

        if b365_in_lib and easy_in_lib:
            covered_by_both += 1
            # 检查预测是否一致
            if (
                b365_library[b365_pattern]["prediction"]
                == easy_library[easy_pattern]["prediction"]
            ):
                consistent_predictions += 1

    either_coverage = covered_by_either / len(common_ids) if common_ids else 0
    both_coverage = covered_by_both / len(common_ids) if common_ids else 0
    consistency = consistent_predictions / covered_by_both if covered_by_both else 0

    print(f"\n融合策略结果:")
    print(
        f"  任一庄家覆盖: {either_coverage:.2%} ({covered_by_either}/{len(common_ids)})"
    )
    print(f"  两庄家都覆盖: {both_coverage:.2%} ({covered_by_both}/{len(common_ids)})")
    print(
        f"  预测一致性: {consistency:.2%} ({consistent_predictions}/{covered_by_both})"
    )

    return either_coverage, both_coverage


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SAX编码覆盖率优化")
    parser.add_argument("--data-dir", required=True, help="数据文件目录")
    parser.add_argument("--bookmaker", default="Bet 365", help="庄家名称")
    args = parser.parse_args()

    # 1. 加载数据
    data_path = os.path.join(
        args.data_dir, f"{args.bookmaker.lower().replace(' ', '_')}_details.json"
    )
    matches = load_odds_from_json(data_path, args.bookmaker)

    if not matches:
        print("错误: 未加载到比赛数据")
        return

    # 从Supabase加载结果
    match_ids = [m.match_id for m in matches]
    results = load_results_from_supabase(match_ids)
    for match in matches:
        if match.match_id in results:
            match.final_score = results[match.match_id]

    with_result = sum(1 for m in matches if m.final_score)
    print(f"有结果的比赛: {with_result}/{len(matches)}")
    matches = [m for m in matches if m.final_score]

    # 2. 测试覆盖率优化
    test_coverage_optimization(matches, args.bookmaker)

    # 3. 测试分层策略
    hierarchical_encoding_strategy(matches)

    # 4. 如果有两个庄家的数据，测试融合策略
    easy_path = os.path.join(args.data_dir, "easybets_details.json")
    if os.path.exists(easy_path) and args.bookmaker == "Bet 365":
        easy_matches = load_odds_from_json(easy_path, "Easybets")
        for match in easy_matches:
            if match.match_id in results:
                match.final_score = results[match.match_id]
        easy_matches = [m for m in easy_matches if m.final_score]

        multi_bookmaker_fusion(matches, easy_matches)


if __name__ == "__main__":
    main()
