#!/usr/bin/env python3
"""
SAX编码质量评估 - 完整版

从 bet_365_details.json 加载赔率数据，
从 Supabase 获取比赛结果，
评估不同SAX编码策略的质量。
"""

import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sax_encoder import SAXEncoder, create_joint_series, create_delta_series


@dataclass
class MatchData:
    """比赛数据"""

    match_id: int
    hometeam: str
    guestteam: str
    season: str
    home_odds: List[float]
    draw_odds: List[float]
    away_odds: List[float]
    final_score: Optional[str] = None

    def get_result(self) -> Optional[str]:
        """解析比赛结果: 'home', 'draw', 'away'"""
        if not self.final_score or "-" not in self.final_score:
            return None
        try:
            home_goals, away_goals = map(int, self.final_score.split("-"))
            if home_goals > away_goals:
                return "home"
            elif home_goals == away_goals:
                return "draw"
            else:
                return "away"
        except (ValueError, IndexError):
            return None


@dataclass
class PatternStats:
    """SAX模式统计"""

    pattern: str
    total: int
    home_wins: int = 0
    draws: int = 0
    away_wins: int = 0
    purity: float = 0.0

    def compute_purity(self):
        """计算纯度"""
        if self.total > 0:
            self.purity = max(self.home_wins, self.draws, self.away_wins) / self.total
        return self.purity

    def get_dominant_result(self) -> Tuple[str, float]:
        """获取主导结果及占比"""
        results = {"home": self.home_wins, "draw": self.draws, "away": self.away_wins}
        dominant = max(results, key=results.get)
        ratio = results[dominant] / self.total if self.total > 0 else 0
        return dominant, ratio


def load_odds_from_json(
    json_path: str, bookmaker_name: str = "Bet 365"
) -> List[MatchData]:
    """从JSON加载赔率数据"""
    print(f"加载赔率数据: {json_path}")
    print(f"目标庄家: {bookmaker_name}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    matches = []
    skipped = 0

    for item in data:
        bookmakers = item.get("bookmakers", [])

        # 查找指定庄家数据
        target_bm_data = None
        for bm in bookmakers:
            if bm.get("bookmakerName") == bookmaker_name:
                target_bm_data = bm
                break

        if not target_bm_data:
            skipped += 1
            continue

        running_odds = target_bm_data.get("runningOdds", [])
        if len(running_odds) < 2:
            skipped += 1
            continue

        # 提取赔率序列
        home_odds, draw_odds, away_odds = [], [], []
        for odds in running_odds:
            try:
                h = float(odds.get("home", 0))
                d = float(odds.get("draw", 0))
                a = float(odds.get("away", 0))
                if h > 0 and d > 0 and a > 0:
                    home_odds.append(h)
                    draw_odds.append(d)
                    away_odds.append(a)
            except (ValueError, TypeError):
                continue

        if len(home_odds) < 2:
            skipped += 1
            continue

        match = MatchData(
            match_id=item.get("scheduleId", 0),
            hometeam=item.get("hometeam", ""),
            guestteam=item.get("guestteam", ""),
            season=item.get("season", ""),
            home_odds=home_odds,
            draw_odds=draw_odds,
            away_odds=away_odds,
        )
        matches.append(match)

    print(f"  加载 {len(matches)} 场比赛, 跳过 {skipped} 场")
    return matches


def load_results_from_supabase(match_ids: List[int]) -> Dict[int, str]:
    """从Supabase加载比赛结果"""
    print("从Supabase加载比赛结果...")

    try:
        from dotenv import load_dotenv

        load_dotenv()

        from supabase import create_client

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

        if not url or not key:
            print("  警告: 未配置Supabase环境变量")
            return {}

        client = create_client(url, key)

        # 分批查询（避免请求过大）
        results = {}
        batch_size = 500

        for i in range(0, len(match_ids), batch_size):
            batch = match_ids[i : i + batch_size]
            response = (
                client.table("league_matches")
                .select("match_id, final_score")
                .in_("match_id", batch)
                .execute()
            )

            for row in response.data:
                if row.get("final_score"):
                    results[row["match_id"]] = row["final_score"]

        print(f"  获取 {len(results)} 场比赛结果")
        return results

    except Exception as e:
        print(f"  错误: {e}")
        return {}


def encode_match(
    match: MatchData, encoder: SAXEncoder, encoding_type: str
) -> Optional[str]:
    """对单场比赛进行SAX编码"""
    try:
        if encoding_type == "interleaved":
            joint = create_joint_series(
                match.home_odds, match.draw_odds, match.away_odds
            )
            return encoder.encode(joint, encoder.word_size * 3)

        elif encoding_type == "delta":
            delta = create_delta_series(match.home_odds, match.away_odds)
            return encoder.encode(delta, encoder.word_size)

        elif encoding_type == "delta_draw":
            delta = create_delta_series(match.home_odds, match.away_odds)
            avg_odds = [(h + a) / 2 for h, a in zip(match.home_odds, match.away_odds)]
            draw_dev = [d - avg for d, avg in zip(match.draw_odds, avg_odds)]
            delta_pattern = encoder.encode(delta, encoder.word_size)
            draw_pattern = encoder.encode(draw_dev, encoder.word_size)
            return delta_pattern + draw_pattern

        elif encoding_type == "individual":
            sax_home = encoder.encode(match.home_odds, encoder.word_size)
            sax_draw = encoder.encode(match.draw_odds, encoder.word_size)
            sax_away = encoder.encode(match.away_odds, encoder.word_size)
            return sax_home + sax_draw + sax_away

        elif encoding_type == "home_only":
            return encoder.encode(match.home_odds, encoder.word_size)

        elif encoding_type == "draw_only":
            return encoder.encode(match.draw_odds, encoder.word_size)

        elif encoding_type == "away_only":
            return encoder.encode(match.away_odds, encoder.word_size)

        else:
            return None

    except Exception as e:
        return None


def evaluate_strategy(
    matches: List[MatchData],
    encoder: SAXEncoder,
    encoding_type: str,
    min_samples: int = 10,
) -> Dict:
    """评估单个编码策略"""

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
        pattern_stats[pattern] = stats

    # 评估质量
    if not pattern_stats:
        return {
            "encoding_type": encoding_type,
            "total_patterns": 0,
            "total_matches": len(matches),
            "avg_purity": 0,
            "coverage": 0,
            "high_purity_count": 0,
        }

    # 按纯度排序
    sorted_patterns = sorted(
        pattern_stats.values(), key=lambda x: (-x.purity, -x.total)
    )
    top_20 = sorted_patterns[:20]

    total_matches = sum(s.total for s in pattern_stats.values())
    covered_matches = sum(s.total for s in top_20)

    avg_purity = np.mean([s.purity for s in top_20]) if top_20 else 0
    high_purity = [s for s in pattern_stats.values() if s.purity >= 0.7]

    return {
        "encoding_type": encoding_type,
        "word_size": encoder.word_size,
        "alphabet_size": encoder.alphabet_size,
        "total_patterns": len(pattern_stats),
        "total_matches": total_matches,
        "avg_purity": avg_purity,
        "coverage": covered_matches / total_matches if total_matches > 0 else 0,
        "high_purity_count": len(high_purity),
        "high_purity_coverage": sum(s.total for s in high_purity) / total_matches
        if total_matches > 0
        else 0,
        "top_patterns": top_20[:10],
    }


def print_strategy_report(result: Dict):
    """打印策略评估报告"""
    print(f"\n{'=' * 70}")
    print(
        f"策略: {result['encoding_type']} (word_size={result['word_size']}, alphabet_size={result['alphabet_size']})"
    )
    print("=" * 70)

    print(f"\n[统计摘要]")
    print(f"  总模式数: {result['total_patterns']}")
    print(f"  有效比赛数: {result['total_matches']}")
    print(f"  Top 20平均纯度: {result['avg_purity']:.2%}")
    print(f"  覆盖率: {result['coverage']:.2%}")
    print(f"  高纯模式数(≥70%): {result['high_purity_count']}")
    print(f"  高纯模式覆盖率: {result['high_purity_coverage']:.2%}")

    print(f"\n[Top 10 模式详情]")
    print(
        f"{'排名':<4} {'模式':<25} {'样本':<6} {'主胜':<6} {'平局':<6} {'客胜':<6} {'纯度':<8} {'预测'}"
    )
    print("-" * 70)

    for i, stats in enumerate(result["top_patterns"], 1):
        dominant, ratio = stats.get_dominant_result()
        print(
            f"{i:<4} {stats.pattern:<25} {stats.total:<6} {stats.home_wins:<6} "
            f"{stats.draws:<6} {stats.away_wins:<6} {stats.purity:<8.2%} {dominant}({ratio:.0%})"
        )


def compare_all_strategies(matches: List[MatchData]):
    """对比所有编码策略"""

    strategies = [
        # (word_size, alphabet_size, encoding_type)
        (8, 7, "interleaved"),  # 当前参数
        (6, 5, "interleaved"),  # 中等粒度
        (4, 3, "interleaved"),  # 趋势粗粒度
        (6, 4, "delta"),  # 差值编码
        (4, 3, "delta_draw"),  # 差值+平局
        (4, 3, "individual"),  # 分别编码
        (6, 4, "home_only"),  # 主胜专注
        (6, 4, "draw_only"),  # 平局专注
        (6, 4, "away_only"),  # 客胜专注
    ]

    print("\n" + "=" * 70)
    print("开始对比9种编码策略")
    print("=" * 70)

    results = []
    for word_size, alphabet_size, encoding_type in strategies:
        print(f"\n测试: {encoding_type} (w={word_size}, a={alphabet_size})")
        encoder = SAXEncoder(word_size=word_size, alphabet_size=alphabet_size)
        result = evaluate_strategy(matches, encoder, encoding_type, min_samples=10)
        results.append(result)
        print_strategy_report(result)

    # 汇总对比
    print("\n" + "=" * 70)
    print("策略对比汇总")
    print("=" * 70)
    print(
        f"{'策略':<20} {'w×a':<8} {'模式空间':<12} {'平均纯度':<12} {'覆盖率':<12} {'高纯模式':<10}"
    )
    print("-" * 70)

    for r in results:
        space_size = r["alphabet_size"] ** r["word_size"]
        print(
            f"{r['encoding_type']:<20} {r['word_size']}×{r['alphabet_size']:<4} {space_size:<12} "
            f"{r['avg_purity']:<12.2%} {r['coverage']:<12.2%} {r['high_purity_count']:<10}"
        )

    # 推荐最佳策略
    def score(r):
        return r["avg_purity"] * r["coverage"] * np.log1p(r["high_purity_count"])

    best = max(results, key=score)
    print(f"\n[推荐策略]")
    print(f"  策略: {best['encoding_type']}")
    print(
        f"  参数: word_size={best['word_size']}, alphabet_size={best['alphabet_size']}"
    )
    print(f"  平均纯度: {best['avg_purity']:.2%}")
    print(f"  覆盖率: {best['coverage']:.2%}")
    print(f"  高纯模式数: {best['high_purity_count']}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SAX编码质量评估")
    parser.add_argument("--data", required=True, help="赔率数据JSON文件路径")
    parser.add_argument("--bookmaker", default="Bet 365", help="庄家名称")
    parser.add_argument(
        "--no-supabase", action="store_true", help="不从Supabase加载结果"
    )
    parser.add_argument("--min-samples", type=int, default=10, help="模式最小样本数")
    args = parser.parse_args()

    # 1. 加载赔率数据
    matches = load_odds_from_json(args.data, args.bookmaker)

    if not matches:
        print("错误: 未加载到比赛数据")
        return

    # 2. 从Supabase加载比赛结果
    if not args.no_supabase:
        match_ids = [m.match_id for m in matches]
        results = load_results_from_supabase(match_ids)

        # 关联结果
        for match in matches:
            if match.match_id in results:
                match.final_score = results[match.match_id]

        # 统计
        with_result = sum(1 for m in matches if m.final_score)
        print(f"  {with_result}/{len(matches)} 场比赛有结果数据")

    # 3. 对比所有策略
    compare_all_strategies(matches)


if __name__ == "__main__":
    main()
