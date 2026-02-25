#!/usr/bin/env python3
"""
2.analyze_odds.py - 赔率数据分析

分析赔率数据分布，生成SAX编码配置

支持两种数据源：
1. 从JSON文件批量分析（支持多庄家）
2. 从titan007下载分析

使用方式:
    # 方式1: 从JSON文件分析
    python 2.analyze_odds.py "./1.generateOddsDetail/SAX encoder/bookmaker_details"
    python 2.analyze_odds.py "./data" --bookmaker "Bet 365"
    
    # 方式2: 从titan007下载分析
    python 2.analyze_odds.py --match-ids 2789379,2789380
    python 2.analyze_odds.py --match-id 2789382 --report
"""

import json
import numpy as np
from collections import Counter
from scipy import stats
import math
import os
import glob
import sys
import argparse

# 尝试导入find_similar_matches中的函数
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from find_similar_matches import (
        download_match_data,
        extract_bet365_odds,
        resample_odds_to_fixed_timeline,
        normalize_odds_series,
        get_interleaved_series,
        get_delta_series,
    )
    HAS_FIND_SIMILAR = True
except ImportError:
    HAS_FIND_SIMILAR = False


# ==================== 通用函数 ====================

def calculate_entropy(labels):
    """计算序列的熵"""
    if len(labels) == 0:
        return 0
    counts = Counter(labels)
    total = sum(counts.values())
    probs = [count / total for count in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def normalize_series(series):
    """Z-score 归一化"""
    series = np.array(series, dtype=float)
    if len(series) < 2:
        return np.zeros_like(series)
    std = np.std(series)
    if std == 0:
        return series - np.mean(series)
    return (series - np.mean(series)) / std


# ==================== JSON文件分析模式 ====================

def extract_running_odds(match, bookmaker_name=None):
    """从比赛数据中提取运行盘赔率"""
    if "bookmakers" in match and isinstance(match["bookmakers"], list):
        if bookmaker_name:
            for bm in match["bookmakers"]:
                if bm.get("bookmakerName") == bookmaker_name:
                    return bm.get("runningOdds", [])
            return []
        else:
            for bm in match["bookmakers"]:
                odds = bm.get("runningOdds", [])
                if odds and len(odds) >= 2:
                    return odds
            return []
    return match.get("runningOdds", [])


def load_match_data(json_file):
    """加载比赛数据"""
    data = []
    if os.path.isfile(json_file):
        files = [json_file]
    elif os.path.isdir(json_file):
        files = glob.glob(os.path.join(json_file, "*.json"))
        files.sort()
    else:
        return data
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                matches = json.load(fp)
                if isinstance(matches, list):
                    data.extend(matches)
                else:
                    data.append(matches)
        except Exception as e:
            print(f"警告: 读取文件 {f} 失败: {e}")
    return data


def safe_float(value, default=None):
    """安全转换为浮点数"""
    if default is None:
        default = np.nan
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def analyze_single_bookmaker(matches, bookmaker_name, output_config):
    """分析单个庄家的数据（从JSON文件）"""
    print("=" * 70)
    print(f"庄家数据分布深度分析报告: {bookmaker_name}")
    print("=" * 70)

    home_series_all = []
    draw_series_all = []
    away_series_all = []
    normalized_all = []
    lengths = []
    processed_count = 0
    skipped_count = 0

    for match in matches:
        running = extract_running_odds(match, bookmaker_name)
        if len(running) >= 2:
            try:
                h = [safe_float(o.get("home")) for o in running]
                d = [safe_float(o.get("draw")) for o in running]
                a = [safe_float(o.get("away")) for o in running]

                if any(np.isnan(h)) or any(np.isnan(d)) or any(np.isnan(a)):
                    skipped_count += 1
                    continue

                processed_count += 1
                lengths.append(len(running))
                home_series_all.extend(h)
                draw_series_all.extend(d)
                away_series_all.extend(a)

                normalized_all.extend(normalize_series(h))
                normalized_all.extend(normalize_series(d))
                normalized_all.extend(normalize_series(a))
            except Exception:
                skipped_count += 1
                continue

    if skipped_count > 0:
        print(f"\n警告: 跳过 {skipped_count} 条无效记录")

    if not lengths:
        print(f"错误: 未找到庄家 '{bookmaker_name}' 的有效赔率序列数据")
        return None

    print(f"\n[1] 数据集概况")
    print(f"  - 比赛总数: {len(matches)}")
    print(f"  - 有效比赛数: {processed_count}")

    # 序列长度分布
    p25, p50, p75, p90 = np.percentile(lengths, [25, 50, 75, 90])
    print(f"\n[2] 序列长度特征")
    print(f"  - 平均长度: {np.mean(lengths):.1f}")
    print(f"  - 长度中位数: {p50:.0f}")
    print(f"  - 75% 分位数: {p75:.0f}")
    print(f"  - 90% 分位数: {p90:.0f}")
    print(f"  - 最大长度: {max(lengths)}")

    # 正态性分析
    skew = stats.skew(normalized_all)
    kurt = stats.kurtosis(normalized_all)
    _, p_val = stats.normaltest(normalized_all)

    print(f"\n[3] 数据分布正态性检查")
    print(f"  - 偏度 (Skewness): {skew:.4f}")
    print(f"  - 峰度 (Kurtosis): {kurt:.4f}")
    print(f"  - K^2 检验 p-value: {p_val:.4e}")

    is_gaussian = p_val > 0.05
    if not is_gaussian:
        print("  - 结论: 数据呈现典型的'厚尾'分布，建议使用经验分位数断点。")
    else:
        print("  - 结论: 数据基本符合正态分布。")

    # 字母表大小优化
    print(f"\n[4] 字母表大小 (alphabet_size) 优化分析")
    print(f"  {'Size':<6} | {'Entropy':<10} | {'Efficiency':<10}")
    print("-" * 35)

    suggested_word_size = 8
    if p50 > 30:
        suggested_word_size = 10
    elif p50 < 15:
        suggested_word_size = 6

    suggested_alphabet_size = 4
    suggested_interpolate_len = int(max(32, p90))

    for size in range(3, 11):
        bins = np.percentile(normalized_all, np.linspace(0, 100, size + 1))
        digitized = np.digitize(normalized_all, bins[1:-1])
        ent = calculate_entropy(digitized)
        efficiency = ent / math.log2(size)
        print(f"  {size:<6} | {ent:<10.4f} | {efficiency:.2%}")

    emp_breakpoints = np.percentile(
        normalized_all, np.linspace(0, 100, suggested_alphabet_size + 1)[1:-1]
    )

    print(f"\n{'=' * 70}")
    print("最终系统配置建议")
    print("=" * 70)
    print(f"  - strategy: interleaved")
    print(f"  - word_size: {suggested_word_size}")
    print(f"  - alphabet_size: {suggested_alphabet_size}")
    print(f"  - interpolate_len: {suggested_interpolate_len}")
    print(f"  - breakpoints_type: {'empirical' if not is_gaussian else 'gaussian'}")
    print("=" * 70)

    config = {
        "bookmaker": bookmaker_name,
        "strategy": "interleaved",
        "word_size": int(suggested_word_size),
        "alphabet_size": int(suggested_alphabet_size),
        "interpolate_len": int(suggested_interpolate_len),
        "breakpoints_type": "empirical" if not is_gaussian else "gaussian",
        "empirical_breakpoints": [float(b) for b in emp_breakpoints],
        "gaussian_breakpoints": [
            float(b)
            for b in stats.norm.ppf(
                np.linspace(0, 1, suggested_alphabet_size + 1)[1:-1]
            )
        ],
        "metadata": {
            "n_matches_total": len(matches),
            "n_matches_valid": processed_count,
            "median_length": float(p50),
            "skewness": float(skew),
            "kurtosis": float(kurt),
        },
    }

    with open(output_config, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"\n配置已保存至: {output_config}")
    return config


def analyze_all_bookmakers(data_dir, output_dir=None):
    """分析目录下所有庄家数据"""
    if output_dir is None:
        output_dir = data_dir

    matches = load_match_data(data_dir)
    if not matches:
        print(f"错误: 未找到有效数据")
        return

    bookmakers_found = set()
    for match in matches:
        if "bookmakers" in match and isinstance(match["bookmakers"], list):
            for bm in match["bookmakers"]:
                name = bm.get("bookmakerName")
                if name:
                    bookmakers_found.add(name)

    if not bookmakers_found:
        print("错误: 数据中未找到庄家信息")
        return

    print(f"发现庄家: {', '.join(sorted(bookmakers_found))}")
    print(f"比赛总数: {len(matches)}")
    print()

    all_configs = {}
    for bm_name in sorted(bookmakers_found):
        safe_name = bm_name.lower().replace(" ", "_").replace("/", "_")
        output_config = os.path.join(output_dir, f"sax_config_{safe_name}.json")
        config = analyze_single_bookmaker(matches, bm_name, output_config)
        if config:
            all_configs[bm_name] = config
        print()

    return all_configs


# ==================== titan007下载分析模式 ====================

def analyze_match_from_titan007(match_id: str, match_time: str = None):
    """从titan007下载并分析单场比赛"""
    if not HAS_FIND_SIMILAR:
        print("错误: 无法导入find_similar_matches模块")
        return None

    content = download_match_data(match_id)
    if not content:
        return None

    match_odds = extract_bet365_odds(content)
    if not match_odds:
        return None

    if match_time is None:
        match_time = match_odds.match_time

    # 预处理
    resampled = resample_odds_to_fixed_timeline(match_odds.running_odds, match_time)
    if not resampled or len(resampled["home"]) < 2:
        return None

    # Z-Score标准化
    normalized = normalize_odds_series(resampled)
    interleaved = get_interleaved_series(normalized)
    delta = get_delta_series(normalized)

    return {
        "match_id": match_id,
        "home_team": match_odds.hometeam,
        "away_team": match_odds.guestteam,
        "match_time": match_time,
        "raw_records": len(match_odds.running_odds),
        "resampled_points": len(resampled["home"]),
        "normalized_home": normalized["home"],
        "normalized_draw": normalized["draw"],
        "normalized_away": normalized["away"],
        "interleaved": interleaved,
        "delta": delta,
    }


def analyze_multiple_matches_titan007(match_ids: list) -> dict:
    """分析多场从titan007下载的比赛"""
    all_normalized = []
    all_interleaved = []
    all_delta = []
    success_count = 0
    failed_count = 0

    for mid in match_ids:
        result = analyze_match_from_titan007(str(mid).strip())
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

    return {
        "total_matches": len(match_ids),
        "success_count": success_count,
        "failed_count": failed_count,
        "total_data_points": len(all_normalized),
        "mean": float(np.mean(all_normalized)),
        "std": float(np.std(all_normalized)),
        "min": float(np.min(all_normalized)),
        "max": float(np.max(all_normalized)),
        "skewness": float(stats.skew(all_normalized)),
        "kurtosis": float(stats.kurtosis(all_normalized)),
        "interleaved_mean": float(np.mean(all_interleaved)),
        "interleaved_std": float(np.std(all_interleaved)),
        "delta_mean": float(np.mean(all_delta)),
        "delta_std": float(np.std(all_delta)),
    }


def generate_sax_config_titan007(bookmaker_name: str, match_ids: list) -> dict:
    """生成SAX配置（从titan007数据）"""
    if not match_ids:
        match_ids = list(range(2789379, 2789389))

    analysis = analyze_multiple_matches_titan007(match_ids)
    if not analysis:
        print("错误: 无法分析数据")
        return None

    all_data = []
    for mid in match_ids:
        result = analyze_match_from_titan007(str(mid))
        if result:
            all_data.extend(result["normalized_home"])
            all_data.extend(result["normalized_draw"])
            all_data.extend(result["normalized_away"])

    # 优化字母表
    normalized = np.array(all_data)
    best_size = 4
    best_efficiency = 0

    for size in range(3, 11):
        bins = np.percentile(normalized, np.linspace(0, 100, size + 1))
        digitized = np.digitize(normalized, bins[1:-1])
        entropy = calculate_entropy(digitized)
        efficiency = entropy / math.log2(size)
        if efficiency > best_efficiency:
            best_efficiency = efficiency
            best_size = size

    emp_breakpoints = np.percentile(
        normalized, np.linspace(0, 100, best_size + 1)[1:-1]
    )

    config = {
        "bookmaker": bookmaker_name,
        "strategy": "interleaved",
        "word_size": 8,
        "alphabet_size": best_size,
        "interpolate_len": 32,
        "breakpoints_type": "gaussian",
        "empirical_breakpoints": [float(b) for b in emp_breakpoints],
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


def print_titan007_report(analysis: dict):
    """打印titan007分析报告"""
    print("=" * 70)
    print("赔率数据分布分析报告 (titan007)")
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
    print(f"  - 偏度: {analysis['skewness']:.4f}")
    print(f"  - 峰度: {analysis['kurtosis']:.4f}")


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description="赔率数据分析")
    
    # JSON文件模式
    parser.add_argument("data_path", nargs="?", help="JSON文件或目录路径")
    parser.add_argument("--bookmaker", type=str, help="指定分析的庄家名称")
    parser.add_argument("--output", type=str, help="输出配置文件路径")
    parser.add_argument("--list", action="store_true", help="列出所有可用的庄家")
    
    # titan007模式
    parser.add_argument("--match-id", type=str, help="单个比赛ID (titan007)")
    parser.add_argument("--match-ids", type=str, help="比赛ID列表 (逗号分隔, titan007)")
    parser.add_argument("--report", action="store_true", help="打印分析报告")
    
    args = parser.parse_args()

    # titan007模式
    if args.match_id or args.match_ids:
        if not HAS_FIND_SIMILAR:
            print("错误: 无法使用titan007模式（find_similar_matches模块不可用）")
            return
        
        match_ids = []
        if args.match_id:
            match_ids = [args.match_id]
        elif args.match_ids:
            match_ids = [m.strip() for m in args.match_ids.split(",")]
        
        bookmaker = args.bookmaker or "Bet 365"
        
        print(f"从titan007分析 {len(match_ids)} 场比赛...")
        analysis = analyze_multiple_matches_titan007(match_ids)
        
        if not analysis:
            print("错误: 无法分析数据")
            return
        
        if args.report:
            print_titan007_report(analysis)
        
        config = generate_sax_config_titan007(bookmaker, match_ids)
        if config:
            output_path = args.output or "sax_config.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"\n配置已保存至: {output_path}")
            print(f"\n推荐参数:")
            print(f"  - word_size: {config['word_size']}")
            print(f"  - alphabet_size: {config['alphabet_size']}")
            print(f"  - interpolate_len: {config['interpolate_len']}")
        
        return

    # JSON文件模式
    if args.list:
        if not args.data_path:
            print("错误: --list需要指定数据路径")
            return
        matches = load_match_data(args.data_path)
        if matches:
            bookmakers = set()
            for m in matches:
                if "bookmakers" in m:
                    for bm in m["bookmakers"]:
                        if bm.get("bookmakerName"):
                            bookmakers.add(bm.get("bookmakerName"))
            print("可用的庄家:")
            for bm in sorted(bookmakers):
                print(f"  - {bm}")
        return

    if not args.data_path:
        # 默认路径
        args.data_path = "./1.generateOddsDetail/SAX encoder/bookmaker_details"

    if not os.path.exists(args.data_path):
        print(f"错误: 找不到数据路径 '{args.data_path}'")
        print("\n使用方式:")
        print("  # 从JSON文件分析:")
        print("  python 2.analyze_odds.py ./data")
        print("  python 2.analyze_odds.py ./data --bookmaker 'Bet 365'")
        print("\n  # 从titan007分析:")
        print("  python 2.analyze_odds.py --match-ids 2789379,2789380")
        return

    # 分析JSON文件
    if os.path.isfile(args.data_path):
        bookmaker = args.bookmaker or "Unknown"
        output_config = args.output or "sax_config.json"
        matches = load_match_data(args.data_path)
        if matches:
            analyze_single_bookmaker(matches, bookmaker, output_config)
    else:
        # 目录模式
        if args.bookmaker:
            output_dir = os.path.dirname(args.output) if args.output else args.data_path
            output_config = args.output or os.path.join(
                output_dir, f"sax_config_{args.bookmaker.lower().replace(' ', '_')}.json"
            )
            matches = load_match_data(args.data_path)
            if matches:
                analyze_single_bookmaker(matches, args.bookmaker, output_config)
        else:
            # 分析所有庄家
            analyze_all_bookmakers(args.data_path, args.output)


if __name__ == "__main__":
    main()
