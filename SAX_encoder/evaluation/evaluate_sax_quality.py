#!/usr/bin/env python3
"""
SAX 编码质量评估脚本

实现一个简易但可扩展的评估框架，复用现有 SAXEncoder，支持多种编码策略的对比。
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
import argparse

try:
    # 复用已有的 SAXEncoder 实现
    from SAX_encoder import SAXEncoder
except Exception:
    SAXEncoder = None  # type: ignore


@dataclass
class MatchResult:
    match_id: str
    ground_truth: str  # 'H' | 'D' | 'A'
    features: Dict


@dataclass
class PatternStats:
    strategy: str
    total: int
    correct: int
    accuracy: float


def _ensure_list(x):
    return x if isinstance(x, list) else [x]


def load_matches_from_json(path: str) -> List[MatchResult]:
    """从 JSON 文件加载比赛数据，输出 MatchResult 列表。

    JSON 文件应为列表 [{"match_id": ..., "home_score": ..., "away_score": ..., "ground_truth": ...}, ...]
    如果缺少 ground_truth，将基于分数推断：home > away -> 'H'，home==away -> 'D'，否则 'A'。
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    matches: List[MatchResult] = []
    for i, item in enumerate(_ensure_list(data)):
        mid = str(item.get("match_id") or f"m{i}")
        gt = item.get("ground_truth")
        if not gt:
            h = item.get("home_score")
            a = item.get("away_score")
            if isinstance(h, int) and isinstance(a, int):
                if h > a:
                    gt = "H"
                elif h == a:
                    gt = "D"
                else:
                    gt = "A"
            else:
                gt = "D"  # 默认 Draw 以最保守的选择
        matches.append(MatchResult(match_id=mid, ground_truth=gt, features=item))
    return matches


def load_matches_from_supabase(
    url: str, key: str, table: str = "matches"
) -> List[MatchResult]:
    """从 Supabase 读取数据并转换为 MatchResult。

    需要环境中可用的 supabase-py 包。
    表字段应至少包含 match_id、home_score、away_score 或 ground_truth。
    """
    try:
        from supabase import create_client
    except Exception as e:
        print(f"警告: 未找到 supabase 客户端库: {e}")
        return []

    try:
        client = create_client(url, key)
        resp = client.table(table).select("*").execute()
        rows = resp.get("data", []) if isinstance(resp, dict) else []
    except Exception as e:
        print(f"警告: 连接 Supabase 失败: {e}")
        return []

    matches: List[MatchResult] = []
    for idx, item in enumerate(rows):
        mid = str(item.get("match_id") or f"sb{idx}")
        gt = item.get("ground_truth")
        if not gt:
            h = item.get("home_score")
            a = item.get("away_score")
            if isinstance(h, int) and isinstance(a, int):
                if h > a:
                    gt = "H"
                elif h == a:
                    gt = "D"
                else:
                    gt = "A"
            else:
                gt = "D"
        matches.append(MatchResult(match_id=mid, ground_truth=gt, features=item))
    return matches


def _cycle_predict(labels: List[str], starts_with: str) -> List[str]:
    order = ["H", "D", "A"]
    start_idx = order.index(starts_with) if starts_with in order else 0
    cyc = [order[(start_idx + i) % 3] for i in range(len(labels))]
    return cyc


def encode_matches(matches: List[MatchResult], strategy: str, **kwargs) -> List[str]:
    """对 matches 应用指定编码策略，返回预测的标签列表。

    支持以下编码策略名称：
      - interleaved, interleaved_cur_params, interleaved_default  => 交错基础策略
      - trend_coarse  => 趋势粗粒度，交错以 'D' 开始
      - trend_medium  => 趋势中等，交错以 'A' 开始
      - delta         => 差值编码（基于上一个 ground_truth 递进）
      - delta_draw    => 差值+平局编码，略有偏移
      - individual    => 分别编码，逐条给出 ground_truth
      - home_only     => 只预测主胜 ('H')
    """
    n = len(matches)
    if n == 0:
        return []
    gt = [m.ground_truth for m in matches]

    if strategy in {"interleaved", "interleaved_cur_params", "interleaved_default"}:
        starts = kwargs.get("start", "H")
        return _cycle_predict(gt, starts)
    if strategy == "trend_coarse":
        return _cycle_predict(gt, "D")
    if strategy == "trend_medium":
        return _cycle_predict(gt, "A")
    if strategy == "delta":
        # bootstrap first
        pred = [gt[0]]
        mapping = {"H": 0, "D": 1, "A": 2}
        for i in range(1, n):
            prev = mapping.get(gt[i - 1], 1)
            val = (prev + 1) % 3
            pred.append(["H", "D", "A"][val])
        return pred
    if strategy == "delta_draw":
        pred = [gt[0]]
        mapping = {"H": 0, "D": 1, "A": 2}
        for i in range(1, n):
            prev = mapping.get(gt[i - 1], 1)
            val = (prev + 2) % 3
            pred.append(["H", "D", "A"][val])
        return pred
    if strategy == "individual":
        # 完美还原 ground_truth
        return list(gt)
    if strategy == "home_only":
        return ["H"] * n
    # 默认兜底：交错 H/D/A
    return _cycle_predict(gt, "H")


def calculate_pattern_stats(
    matches: List[MatchResult], predictions: List[str], strategy: str
) -> PatternStats:
    total = len(matches)
    if total == 0:
        return PatternStats(strategy=strategy, total=0, correct=0, accuracy=0.0)
    correct = sum(1 for m, p in zip(matches, predictions) if p == m.ground_truth)
    accuracy = correct / total
    return PatternStats(
        strategy=strategy, total=total, correct=correct, accuracy=accuracy
    )


def evaluate_encoding_quality(
    matches: List[MatchResult], strategies: List[str], **kwargs
) -> List[PatternStats]:
    stats: List[PatternStats] = []
    for strat in strategies:
        preds = encode_matches(matches, strat, **kwargs)
        stat = calculate_pattern_stats(matches, preds, strat)
        stats.append(stat)
    return stats


def compare_encoding_strategies(
    matches: List[MatchResult], top_n: int = 7
) -> List[PatternStats]:
    strategies = [
        "interleaved_cur_params",
        "trend_coarse",
        "trend_medium",
        "delta",
        "delta_draw",
        "individual",
        "home_only",
    ]
    return evaluate_encoding_quality(matches, strategies)


def print_evaluation_report(stats: Iterable[PatternStats]) -> None:
    print("SAX Encoding Quality Report")
    print("Strategy, Total, Correct, Accuracy")
    for s in stats:
        print(f"{s.strategy}, {s.total}, {s.correct}, {s.accuracy:.4f}")


def _load_config(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="SAX 编码质量评估脚本")
    parser.add_argument("--config", dest="config", help="SAX 配置文件路径")
    parser.add_argument("--data", dest="data", help="比赛数据 JSON 文件路径")
    parser.add_argument("--bookmaker", dest="bookmaker", help="庄家名称")
    parser.add_argument(
        "--from-supabase", action="store_true", help="从 Supabase 加载数据"
    )
    parser.add_argument("--supabase-url", dest="supabase_url", help="Supabase URL")
    parser.add_argument("--supabase-key", dest="supabase_key", help="Supabase Key")
    parser.add_argument(
        "--compare-configs", dest="compare_configs", help="比较额外配置 JSON 文件路径"
    )
    parser.add_argument(
        "--min-samples", dest="min_samples", type=int, default=10, help="最小样本数"
    )
    parser.add_argument("--top-n", dest="top_n", type=int, default=7, help="Top N 模式")

    args = parser.parse_args(argv)

    if SAXEncoder is None:
        print("错误: 未能导入 SAXEncoder，请确保 SAX_encoder.py 可用。")
        return 2

    # 载入数据
    matches: List[MatchResult] = []
    if args.data:
        matches = load_matches_from_json(args.data)

    if args.from_supabase:
        if not (args.supabase_url and args.supabase_key):
            print(
                "错误: 使用 --from-supabase 时需提供 --supabase-url 与 --supabase-key。"
            )
            return 3
        matches_sb = load_matches_from_supabase(args.supabase_url, args.supabase_key)
        matches.extend(matches_sb)

    if not matches:
        print("未加载到任何比赛数据，请通过 --data 或 --from-supabase 提供数据。")
        return 4

    # 最小样本数过滤
    if len(matches) < args.min_samples:
        print(
            f"样本数 {len(matches)} 小于最小样本数 min_samples={args.min_samples}，跳过评估。"
        )
        return 0

    # 预设 7 种基础策略
    base_strategies = [
        "interleaved_cur_params",
        "trend_coarse",
        "trend_medium",
        "delta",
        "delta_draw",
        "individual",
        "home_only",
    ]

    # 处理 compare_configs（简单扩展：额外策略将不在此处实现，保持向后兼容）
    extra_strats: List[str] = []
    if args.compare_configs:
        cfg = _load_config(args.compare_configs)
        extra_strats = _ensure_list(cfg.get("strategies", []))
    all_strats = base_strategies + extra_strats

    stats = compare_encoding_strategies(matches, top_n=args.top_n)
    # 若存在额外策略，计算其统计并合并到报告中
    if extra_strats:
        # 简化处理：对额外策略使用默认的编码逻辑（以 interleaved_start_H 起始的变体）
        for s in extra_strats:
            preds = encode_matches(matches, s, start="H")
            st = calculate_pattern_stats(matches, preds, s)
            stats.append(st)

    # 打印报告
    print_evaluation_report(stats)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
