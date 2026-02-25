#!/usr/bin/env python3
"""
3.run_sax.py - SAX编码主程序

使用预处理和Z-Score标准化后的数据进行SAX编码

功能：
- 从titan007下载赔率数据
- 预处理：20点固定时间轴
- Z-Score标准化
- SAX编码

使用方式:
    python 3.run_sax.py 2789382
    python 3.run_sax.py 2789379,2789380,2789381
    python 3.run_sax.py --batch
"""

import json
import os
import sys
import argparse
import numpy as np
from typing import List, Dict, Optional
from collections import Counter

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


def process_single_match(match_id: str, config: dict = None) -> Optional[dict]:
    """
    处理单场比赛，返回SAX编码结果

    Args:
        match_id: 比赛ID
        config: SAX编码配置

    Returns:
        编码结果字典
    """
    # 加载配置
    if config is None:
        config = {
            "word_size": 8,
            "alphabet_size": 4,
            "interpolate_len": 32,
        }

    # 下载数据
    content = download_match_data(match_id)
    if not content:
        print(f"  错误: 无法下载比赛 {match_id} 的数据")
        return None

    # 提取赔率
    match_odds = extract_bet365_odds(content)
    if not match_odds:
        print(f"  错误: 无法提取比赛 {match_id} 的赔率")
        return None

    # 预处理：20点固定时间轴
    resampled = resample_odds_to_fixed_timeline(
        match_odds.running_odds, match_odds.match_time
    )
    if not resampled or len(resampled["home"]) < 2:
        print(f"  错误: 比赛 {match_id} 预处理失败")
        return None

    # Z-Score 标准化
    normalized = normalize_odds_series(resampled)

    # 创建编码器
    encoder = SAXEncoder(
        word_size=config.get("word_size", 8),
        alphabet_size=config.get("alphabet_size", 4),
    )

    # 生成序列
    interleaved = get_interleaved_series(normalized)
    delta = get_delta_series(normalized)

    # SAX编码
    interpolate_len = config.get("interpolate_len", 32)
    sax_interleaved = encoder.encode(interleaved, interpolate_len * 3)
    sax_delta = encoder.encode(delta, interpolate_len)

    return {
        "match_id": match_id,
        "home_team": match_odds.hometeam,
        "away_team": match_odds.guestteam,
        "match_time": match_odds.match_time,
        "initial_odds": {
            "home": match_odds.initial_odds_home,
            "draw": match_odds.initial_odds_draw,
            "away": match_odds.initial_odds_away,
        },
        "final_odds": {
            "home": match_odds.final_odds_home,
            "draw": match_odds.final_odds_draw,
            "away": match_odds.final_odds_away,
        },
        "raw_records": len(match_odds.running_odds),
        "resampled_points": len(resampled["home"]),
        "sax_interleaved": sax_interleaved,
        "sax_delta": sax_delta,
        "stats": {
            "home_mean": float(np.mean(resampled["home"])),
            "draw_mean": float(np.mean(resampled["draw"])),
            "away_mean": float(np.mean(resampled["away"])),
            "home_final": float(resampled["home"][-1]),
            "draw_final": float(resampled["draw"][-1]),
            "away_final": float(resampled["away"][-1]),
        },
    }


def process_batch_match_ids(match_ids: List[str], config: dict = None) -> List[dict]:
    """
    批量处理比赛ID

    Args:
        match_ids: 比赛ID列表
        config: SAX编码配置

    Returns:
        编码结果列表
    """
    results = []
    for mid in match_ids:
        print(f"处理比赛: {mid}")
        result = process_single_match(str(mid).strip(), config)
        if result:
            results.append(result)
        else:
            print(f"  跳过比赛 {mid}")
    return results


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(results: List[dict], output_path: str):
    """保存结果"""
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存至: {output_path}")


def print_statistics(results: List[dict]):
    """打印统计信息"""
    if not results:
        print("无结果")
        return

    print(f"\n{'=' * 60}")
    print(f"编码结果统计 (共 {len(results)} 场比赛)")
    print("=" * 60)

    # 交错编码分布
    interleaved_codes = [r["sax_interleaved"] for r in results]
    delta_codes = [r["sax_delta"] for r in results]

    print(f"\n交错编码分布 (Top 10):")
    for code, count in Counter(interleaved_codes).most_common(10):
        print(f"  {code}: {count} 场")

    print(f"\n差值编码分布 (Top 10):")
    for code, count in Counter(delta_codes).most_common(10):
        print(f"  {code}: {count} 场")


def main():
    parser = argparse.ArgumentParser(description="SAX编码")
    parser.add_argument("match_ids", nargs="?", help="比赛ID (逗号分隔)")
    parser.add_argument("--config", "-c", help="配置文件路径")
    parser.add_argument("--output", "-o", help="输出文件路径")
    parser.add_argument(
        "--batch", "-b", action="store_true", help="批量模式:处理配置文件中所有比赛"
    )
    parser.add_argument(
        "--word-size", "-w", type=int, default=8, help="word_size (默认8)"
    )
    parser.add_argument(
        "--alphabet-size", "-a", type=int, default=4, help="alphabet_size (默认4)"
    )
    parser.add_argument(
        "--interpolate-len", "-i", type=int, default=32, help="interpolate_len (默认32)"
    )

    args = parser.parse_args()

    # 加载配置
    config = {}
    if args.config:
        config = load_config(args.config)

    # 命令行参数覆盖配置
    config.setdefault("word_size", args.word_size)
    config.setdefault("alphabet_size", args.alphabet_size)
    config.setdefault("interpolate_len", args.interpolate_len)

    print("=" * 60)
    print("SAX 编码 (预处理 + Z-Score标准化)")
    print("=" * 60)
    print(
        f"配置: word_size={config['word_size']}, alphabet_size={config['alphabet_size']}, interpolate_len={config['interpolate_len']}"
    )

    # 处理比赛
    results = []

    if args.batch:
        # 批量模式：从配置文件读取比赛ID
        if args.config:
            config_data = load_config(args.config)
            match_ids = config_data.get("match_ids", [])
        else:
            print("错误: 批量模式需要提供配置文件")
            return
    elif args.match_ids:
        # 指定比赛ID
        match_ids = [m.strip() for m in args.match_ids.split(",")]
    else:
        # 默认测试
        match_ids = ["2789382"]

    print(f"\n处理 {len(match_ids)} 场比赛...")
    results = process_batch_match_ids(match_ids, config)

    # 打印统计
    print_statistics(results)

    # 保存结果
    if args.output:
        save_results(results, args.output)
    elif results:
        # 默认保存
        output_path = (
            f"sax_results_{match_ids[0]}.json"
            if len(match_ids) == 1
            else "sax_results.json"
        )
        save_results(results, output_path)


if __name__ == "__main__":
    main()
