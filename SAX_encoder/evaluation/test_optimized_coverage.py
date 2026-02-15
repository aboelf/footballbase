#!/usr/bin/env python3
"""
SAX编码覆盖率优化测试 - 实现篇

测试优化方案：
1. 降低粒度: 4×3 → 3×3
2. 放宽阈值: 70% → 65%
3. 降低样本: 10 → 5
4. 软匹配: 汉明距离≤1
5. 多庄家融合
"""

import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple
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


def hamming_distance(s1: str, s2: str) -> int:
    """计算两个字符串的汉明距离"""
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def build_pattern_library(
    matches: List[MatchData],
    encoder: SAXEncoder,
    encoding_type: str,
    min_samples: int = 5,
    purity_threshold: float = 0.65,
) -> Dict[str, Dict]:
    """构建模式库"""
    pattern_groups = defaultdict(list)

    for match in matches:
        if not match.get_result():
            continue
        pattern = encode_match(match, encoder, encoding_type)
        if pattern:
            pattern_groups[pattern].append(match)

    library = {}
    for pattern, group in pattern_groups.items():
        if len(group) < min_samples:
            continue

        home = sum(1 for m in group if m.get_result() == "home")
        draw = sum(1 for m in group if m.get_result() == "draw")
        away = sum(1 for m in group if m.get_result() == "away")
        total = len(group)
        purity = max(home, draw, away) / total

        if purity >= purity_threshold:
            dominant = max(
                ["home", "draw", "away"],
                key=lambda x: {"home": home, "draw": draw, "away": away}[x],
            )
            library[pattern] = {
                "prediction": dominant,
                "purity": purity,
                "total": total,
                "home": home,
                "draw": draw,
                "away": away,
            }

    return library


def evaluate_coverage(
    matches: List[MatchData],
    library: Dict[str, Dict],
    encoder: SAXEncoder,
    encoding_type: str,
    use_soft_match: bool = False,
    max_distance: int = 1,
) -> Dict:
    """评估覆盖率"""
    total = len(matches)
    covered = 0
    high_confidence = 0
    correct = 0

    for match in matches:
        if not match.get_result():
            continue

        pattern = encode_match(match, encoder, encoding_type)
        if not pattern:
            continue

        # 直接匹配
        if pattern in library:
            covered += 1
            if library[pattern]["purity"] >= 0.70:
                high_confidence += 1
            if library[pattern]["prediction"] == match.get_result():
                correct += 1
        # 软匹配
        elif use_soft_match:
            for lib_pattern, stats in library.items():
                if hamming_distance(pattern, lib_pattern) <= max_distance:
                    covered += 1
                    if stats["prediction"] == match.get_result():
                        correct += 1
                    break

    accuracy = correct / covered if covered > 0 else 0

    return {
        "total": total,
        "covered": covered,
        "coverage": covered / total if total > 0 else 0,
        "high_confidence": high_confidence,
        "accuracy": accuracy,
    }


def test_optimized_params(matches: List[MatchData], bookmaker_name: str):
    """测试优化参数"""
    print(f"\n{'=' * 70}")
    print(f"优化参数测试 - {bookmaker_name}")
    print(f"{'=' * 70}")
    print(f"总比赛数: {len(matches)}")

    results = []

    # 测试配置
    configs = [
        # (word_size, alphabet_size, min_samples, purity_threshold, use_soft_match, name)
        (4, 3, 10, 0.70, False, "原始参数 (4×3)"),
        (3, 3, 10, 0.70, False, "降低粒度 (3×3)"),
        (4, 3, 5, 0.65, False, "放宽阈值 (4×3)"),
        (3, 3, 5, 0.65, False, "双优化 (3×3+65%) ⭐"),
        (3, 3, 3, 0.60, False, "激进参数 (3×3+60%)"),
        (3, 3, 5, 0.65, True, "软匹配 (3×3+65%+软) ⭐⭐"),
    ]

    for (
        word_size,
        alphabet_size,
        min_samples,
        purity_threshold,
        use_soft,
        name,
    ) in configs:
        print(f"\n测试: {name}")
        print(
            f"  参数: w={word_size}, a={alphabet_size}, min={min_samples}, purity={purity_threshold}"
        )

        encoder = SAXEncoder(word_size=word_size, alphabet_size=alphabet_size)
        library = build_pattern_library(
            matches, encoder, "individual", min_samples, purity_threshold
        )

        result = evaluate_coverage(matches, library, encoder, "individual", use_soft)

        print(f"  模式库大小: {len(library)}")
        print(
            f"  覆盖率: {result['coverage']:.2%} ({result['covered']}/{result['total']})"
        )
        print(f"  预测准确率: {result['accuracy']:.2%}")

        results.append(
            {
                "name": name,
                "config": (word_size, alphabet_size, min_samples, purity_threshold),
                "library_size": len(library),
                "coverage": result["coverage"],
                "accuracy": result["accuracy"],
                "covered": result["covered"],
            }
        )

    # 打印对比表
    print(f"\n{'=' * 70}")
    print("优化参数对比")
    print(f"{'=' * 70}")
    print(f"{'配置':<30} {'模式数':<10} {'覆盖率':<12} {'准确率':<10} {'覆盖场数'}")
    print("-" * 70)

    for r in results:
        print(
            f"{r['name']:<30} {r['library_size']:<10} {r['coverage']:<12.2%} "
            f"{r['accuracy']:<10.2%} {r['covered']}"
        )

    return results


def test_multi_bookmaker_fusion(
    b365_matches: List[MatchData], easy_matches: List[MatchData]
):
    """测试多庄家融合"""
    print(f"\n{'=' * 70}")
    print("多庄家融合测试")
    print(f"{'=' * 70}")

    # 使用优化参数
    encoder = SAXEncoder(word_size=3, alphabet_size=3)

    # 构建两个庄家的模式库
    b365_lib = build_pattern_library(b365_matches, encoder, "individual", 5, 0.65)
    easy_lib = build_pattern_library(easy_matches, encoder, "individual", 5, 0.65)

    print(f"Bet 365 模式库: {len(b365_lib)} 个模式")
    print(f"Easybets 模式库: {len(easy_lib)} 个模式")

    # 找出共同比赛
    b365_ids = {m.match_id for m in b365_matches}
    easy_ids = {m.match_id for m in easy_matches}
    common_ids = b365_ids & easy_ids

    print(f"\n共同比赛数: {len(common_ids)}")

    # 创建比赛查找字典
    b365_dict = {m.match_id: m for m in b365_matches}
    easy_dict = {m.match_id: m for m in easy_matches}

    # 统计融合效果
    covered_by_both = 0
    covered_by_either = 0
    consistent_correct = 0
    consistent_total = 0

    results = {
        "both_high": [],  # 两庄家都高纯且一致
        "single_high": [],  # 只有一个高纯
        "both_low": [],  # 都不高纯
    }

    for match_id in common_ids:
        b365_match = b365_dict[match_id]
        easy_match = easy_dict[match_id]

        b365_pattern = encode_match(b365_match, encoder, "individual")
        easy_pattern = encode_match(easy_match, encoder, "individual")

        b365_in = b365_pattern in b365_lib if b365_pattern else False
        easy_in = easy_pattern in easy_lib if easy_pattern else False

        actual_result = b365_match.get_result()

        if b365_in and easy_in:
            covered_by_both += 1
            b365_pred = b365_lib[b365_pattern]["prediction"]
            easy_pred = easy_lib[easy_pattern]["prediction"]

            if b365_pred == easy_pred:
                consistent_total += 1
                if b365_pred == actual_result:
                    consistent_correct += 1
                results["both_high"].append(
                    {
                        "match_id": match_id,
                        "prediction": b365_pred,
                        "actual": actual_result,
                        "correct": b365_pred == actual_result,
                    }
                )
        elif b365_in or easy_in:
            covered_by_either += 1
            pred = (
                b365_lib[b365_pattern]["prediction"]
                if b365_in
                else easy_lib[easy_pattern]["prediction"]
            )
            results["single_high"].append(
                {
                    "match_id": match_id,
                    "prediction": pred,
                    "actual": actual_result,
                    "correct": pred == actual_result,
                }
            )
        else:
            results["both_low"].append(match_id)

    # 计算统计
    both_coverage = covered_by_both / len(common_ids)
    either_coverage = (covered_by_both + covered_by_either) / len(common_ids)
    consistency = consistent_correct / consistent_total if consistent_total > 0 else 0

    # 单庄家高纯的准确率
    single_correct = sum(1 for r in results["single_high"] if r["correct"])
    single_accuracy = (
        single_correct / len(results["single_high"]) if results["single_high"] else 0
    )

    print(f"\n融合统计:")
    print(f"  两庄家都覆盖: {covered_by_both} ({both_coverage:.2%})")
    print(
        f"  任一庄家覆盖: {covered_by_both + covered_by_either} ({either_coverage:.2%})"
    )
    print(f"  两庄家都高纯且一致: {consistent_total}")
    print(
        f"  一致预测准确率: {consistency:.2%} ({consistent_correct}/{consistent_total})"
    )
    print(
        f"  单庄家预测准确率: {single_accuracy:.2%} ({single_correct}/{len(results['single_high'])})"
    )

    # 计算总体可用覆盖率
    usable = len(results["both_high"]) + len(results["single_high"])
    usable_coverage = usable / len(common_ids)

    print(f"\n总体效果:")
    print(f"  可用覆盖率: {usable_coverage:.2%} ({usable}/{len(common_ids)})")
    print(
        f"    - 高置信度(双高): {len(results['both_high'])} ({len(results['both_high']) / len(common_ids):.2%})"
    )
    print(
        f"    - 中置信度(单高): {len(results['single_high'])} ({len(results['single_high']) / len(common_ids):.2%})"
    )

    return {
        "both_coverage": both_coverage,
        "either_coverage": either_coverage,
        "usable_coverage": usable_coverage,
        "consistency": consistency,
        "single_accuracy": single_accuracy,
    }


def print_final_recommendation(
    results_b365: List[Dict], results_easy: List[Dict], fusion_result: Dict
):
    """打印最终推荐"""
    print(f"\n{'=' * 70}")
    print("最终推荐方案")
    print(f"{'=' * 70}")

    # 找出最佳单庄家配置
    best_single = max(results_b365, key=lambda x: x["coverage"] * x["accuracy"])

    print(f"\n[单庄家最佳配置]")
    print(f"  方案: {best_single['name']}")
    print(
        f"  参数: word_size={best_single['config'][0]}, alphabet_size={best_single['config'][1]}"
    )
    print(
        f"  最小样本: {best_single['config'][2]}, 纯度阈值: {best_single['config'][3]}"
    )
    print(f"  覆盖率: {best_single['coverage']:.2%}")
    print(f"  预测准确率: {best_single['accuracy']:.2%}")

    print(f"\n[多庄家融合效果]")
    print(f"  可用覆盖率: {fusion_result['usable_coverage']:.2%}")
    print(f"  双高预测准确率: {fusion_result['consistency']:.2%}")
    print(f"  单高预测准确率: {fusion_result['single_accuracy']:.2%}")

    print(f"\n[对比总结]")
    print(f"  原始参数覆盖率: 14.5%")
    print(
        f"  优化后单庄家: {best_single['coverage']:.2%} (↑{best_single['coverage'] / 0.145:.1f}x)"
    )
    print(
        f"  优化后双庄家: {fusion_result['usable_coverage']:.2%} (↑{fusion_result['usable_coverage'] / 0.145:.1f}x)"
    )

    if fusion_result["usable_coverage"] >= 0.35:
        print(f"\n  ✅ 达成 35%+ 覆盖率目标!")
    elif fusion_result["usable_coverage"] >= 0.25:
        print(
            f"\n  ⚠️ 覆盖率 {fusion_result['usable_coverage']:.1%}，接近目标，建议继续优化"
        )
    else:
        print(f"\n  ❌ 覆盖率不足，需要进一步调整参数")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SAX编码覆盖率优化测试")
    parser.add_argument("--data-dir", required=True, help="数据文件目录")
    args = parser.parse_args()

    # 加载 Bet 365 数据
    b365_path = os.path.join(args.data_dir, "bet_365_details.json")
    b365_matches = load_odds_from_json(b365_path, "Bet 365")

    # 加载 Easybets 数据
    easy_path = os.path.join(args.data_dir, "easybets_details.json")
    easy_matches = load_odds_from_json(easy_path, "Easybets")

    if not b365_matches or not easy_matches:
        print("错误: 未加载到比赛数据")
        return

    # 从Supabase加载结果
    print("\n从Supabase加载比赛结果...")
    all_ids = [m.match_id for m in b365_matches] + [m.match_id for m in easy_matches]
    results = load_results_from_supabase(list(set(all_ids)))

    for match in b365_matches:
        if match.match_id in results:
            match.final_score = results[match.match_id]
    for match in easy_matches:
        if match.match_id in results:
            match.final_score = results[match.match_id]

    # 过滤有结果的比赛
    b365_matches = [m for m in b365_matches if m.final_score]
    easy_matches = [m for m in easy_matches if m.final_score]

    print(f"Bet 365: {len(b365_matches)} 场有结果")
    print(f"Easybets: {len(easy_matches)} 场有结果")

    # 测试优化参数
    results_b365 = test_optimized_params(b365_matches, "Bet 365")
    results_easy = test_optimized_params(easy_matches, "Easybets")

    # 测试多庄家融合
    fusion_result = test_multi_bookmaker_fusion(b365_matches, easy_matches)

    # 打印最终推荐
    print_final_recommendation(results_b365, results_easy, fusion_result)


if __name__ == "__main__":
    main()
