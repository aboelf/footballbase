#!/usr/bin/env python3
"""
多庄家赔率 SAX 编码主入口

支持多个庄家的赔率数据 SAX 编码：
- 自动发现 bookmaker_details/ 目录下所有庄家配置和数据文件
- 每个庄家使用独立的 SAX 配置参数
- 支持单独编码 (Individual) 和联合编码 (Joint)
python 3.run_sax.py --config-dir "1.generateOddsDetail/SAX encoder/bookmaker_details" --data-dir "1.generateOddsDetail/SAX encoder/bookmaker_details"
输出结构:
  - {庄家}_sax_individual.json: 各庄家分别编码结果
  - {庄家}_sax_joint.json: 各庄家联合编码结果
"""

import json
import os
import sys
import glob
from datetime import datetime
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any, cast
import argparse
import numpy as np


def find_bookmaker_configs(config_dir: str) -> Dict[str, str]:
    """自动发现所有庄家配置文件"""
    config_pattern = os.path.join(config_dir, "sax_config_*.json")
    configs = {}
    for config_path in glob.glob(config_pattern):
        filename = os.path.basename(config_path)
        # 从文件名提取庄家名: sax_config_bet_365.json -> bet_365
        bookmaker = filename.replace("sax_config_", "").replace(".json", "")
        configs[bookmaker] = config_path
    return configs


def find_data_files(data_dir: str) -> List[str]:
    """自动发现所有数据文件"""
    data_pattern = os.path.join(data_dir, "*_details.json")
    files = glob.glob(data_pattern)
    return sorted(files)


def load_bookmaker_config(config_path: str) -> dict:
    """加载庄家配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_odds_series_from_bookmaker(
    bookmaker_data: dict,
) -> Optional[Tuple[List[float], List[float], List[float]]]:
    """
    从庄家数据中提取赔率序列

    Args:
        bookmaker_data: 包含 runningOdds 的庄家数据

    Returns:
        (home_series, draw_series, away_series): 三个赔率序列，或 None
    """
    running = bookmaker_data.get("runningOdds", [])
    if not running:
        return None

    home, draw, away = [], [], []
    for o in running:
        try:
            h = float(o["home"])
            d = float(o["draw"])
            a = float(o["away"])
            # 过滤无效值（引号、0、负数等）
            if h > 0 and d > 0 and a > 0:
                home.append(h)
                draw.append(d)
                away.append(a)
        except (ValueError, TypeError, KeyError):
            continue

    if len(home) < 2:
        return None

    return home, draw, away


def extract_odds_series_from_match(
    match: dict, bookmaker_name: Optional[str] = None
) -> Optional[Tuple[List[float], List[float], List[float]]]:
    """
    从比赛数据中提取指定庄家的赔率序列

    Args:
        match: 比赛数据 dict
        bookmaker_name: 庄家名称（可选，用于多庄家数据）

    Returns:
        (home_series, draw_series, away_series): 三个赔率序列，或 None
    """
    # 尝试从 bookmakers 数组中提取（多庄家格式）
    if "bookmakers" in match and match["bookmakers"]:
        bookmakers = match["bookmakers"]

        if bookmaker_name:
            # 查找指定庄家
            for bm in bookmakers:
                if bm.get("bookmakerName") == bookmaker_name:
                    return extract_odds_series_from_bookmaker(bm)
            return None
        else:
            # 返回第一个庄家
            return extract_odds_series_from_bookmaker(bookmakers[0])

    # 兼容旧格式：runningOdds 直接在 match 层
    running = match.get("runningOdds", [])
    if not running:
        return None

    home = [float(o["home"]) for o in running]
    draw = [float(o["draw"]) for o in running]
    away = [float(o["away"]) for o in running]

    return home, draw, away


def create_joint_series(
    home: List[float], draw: List[float], away: List[float]
) -> List[float]:
    """创建联合序列（交错拼接）"""
    min_len = min(len(home), len(draw), len(away))
    joint = []
    for i in range(min_len):
        joint.append(home[i])
        joint.append(draw[i])
        joint.append(away[i])
    return joint


def create_delta_series(home: List[float], away: List[float]) -> List[float]:
    """创建差值序列（主客队赔率差）"""
    min_len = min(len(home), len(away))
    return [home[i] - away[i] for i in range(min_len)]


class MultiBookmakerSAXProcessor:
    """多庄家 SAX 编码处理器"""

    def __init__(self, config_dir: str, data_dir: str):
        self.config_dir = config_dir
        self.data_dir = data_dir
        self.encoders: Dict[str, Dict[str, Any]] = {}  # 庄家名 -> {encoder, params}
        self.configs: Dict[str, dict] = {}  # 庄家名 -> 配置

    def load_all_configs(self):
        """加载所有庄家配置文件"""
        configs = find_bookmaker_configs(self.config_dir)
        print(f"发现 {len(configs)} 个庄家配置文件:")
        for bookmaker, config_path in configs.items():
            config = load_bookmaker_config(config_path)
            self.configs[bookmaker] = config
            print(
                f"  - {bookmaker}: strategy={config.get('strategy', 'individual')}, "
                f"word_size={config.get('word_size')}, "
                f"alphabet_size={config.get('alphabet_size')}, "
                f"interpolate_len={config.get('interpolate_len')}"
            )

    @staticmethod
    def normalize_bookmaker_name(name: str) -> str:
        """标准化庄家名称：Bet 365 -> bet_365"""
        return name.lower().replace(" ", "_")

    def get_encoder(self, bookmaker: str):
        """获取或创建指定庄家的编码器"""
        if bookmaker not in self.encoders:
            # 标准化名称用于匹配配置文件
            normalized = self.normalize_bookmaker_name(bookmaker)
            if normalized in self.configs:
                config = self.configs[normalized]
                config_path = os.path.join(
                    self.config_dir, f"sax_config_{normalized}.json"
                )
                self.encoders[bookmaker] = {
                    "encoder": self._create_encoder_from_config(config, config_path),
                    "params": {
                        "word_size": config.get("word_size", 8),
                        "alphabet_size": config.get("alphabet_size", 7),
                        "interpolate_len": config.get("interpolate_len", 32),
                        "strategy": config.get("strategy", "individual"),
                        "alphabet": None,
                        "breakpoints": None,
                    },
                }
                enc = self.encoders[bookmaker]["encoder"]
                self.encoders[bookmaker]["params"]["alphabet"] = enc.alphabet
                self.encoders[bookmaker]["params"]["breakpoints"] = (
                    enc.breakpoints.round(3).tolist()
                )
            else:
                # 使用默认参数
                self.encoders[bookmaker] = {
                    "encoder": self._create_encoder_from_config({}, None),
                    "params": {
                        "word_size": 8,
                        "alphabet_size": 7,
                        "interpolate_len": 32,
                        "strategy": "individual",
                        "alphabet": None,
                        "breakpoints": None,
                    },
                }
                enc = self.encoders[bookmaker]["encoder"]
                self.encoders[bookmaker]["params"]["alphabet"] = enc.alphabet
                self.encoders[bookmaker]["params"]["breakpoints"] = (
                    enc.breakpoints.round(3).tolist()
                )

        return self.encoders[bookmaker]

    def _create_encoder_from_config(self, config: dict, config_path: Optional[str]):
        """根据配置创建 SAX 编码器"""
        from sax_encoder import SAXEncoder

        if config_path and os.path.exists(config_path):
            return SAXEncoder(config_path=config_path)
        else:
            return SAXEncoder(
                word_size=config.get("word_size", 8),
                alphabet_size=config.get("alphabet_size", 7),
            )

    def process_match_individual(self, match: dict, bookmaker: str):
        """处理比赛 - 分别编码方案"""
        result = extract_odds_series_from_match(match, bookmaker)
        if result is None:
            return None
        home, draw, away = result

        encoder_data = self.get_encoder(bookmaker)
        encoder = encoder_data["encoder"]
        params = encoder_data["params"]

        return {
            "scheduleId": match.get("scheduleId"),
            "hometeam": match.get("hometeam"),
            "guestteam": match.get("guestteam"),
            "matchTime": match.get("matchTime"),
            "season": match.get("season"),
            "bookmaker": bookmaker,
            "sax_home": encoder.encode(home, params["interpolate_len"]),
            "sax_draw": encoder.encode(draw, params["interpolate_len"]),
            "sax_away": encoder.encode(away, params["interpolate_len"]),
            "stats": {
                "home_mean": round(float(np.mean(home)), 3),  # type: ignore
                "draw_mean": round(float(np.mean(draw)), 3),  # type: ignore
                "away_mean": round(float(np.mean(away)), 3),  # type: ignore
                "home_std": round(float(np.std(home)), 3),  # type: ignore
                "draw_std": round(float(np.std(draw)), 3),  # type: ignore
                "away_std": round(float(np.std(away)), 3),  # type: ignore
                "running_odds_count": len(home),
            },
        }

    def process_match_joint(self, match: dict, bookmaker: str):
        """处理比赛 - 联合编码方案 (根据config中的strategy)"""
        result = extract_odds_series_from_match(match, bookmaker)
        if result is None:
            return None
        home, draw, away = result

        encoder_data = self.get_encoder(bookmaker)
        encoder = encoder_data["encoder"]
        params = encoder_data["params"]
        strategy = params.get("strategy", "individual")

        # 根据strategy生成不同的编码
        if strategy == "interleaved":
            # 使用交错拼接编码
            joint_series = create_joint_series(home, draw, away)
            sax_interleaved = encoder.encode(
                joint_series, params["interpolate_len"] * 3
            )

            return {
                "scheduleId": match.get("scheduleId"),
                "hometeam": match.get("hometeam"),
                "guestteam": match.get("guestteam"),
                "matchTime": match.get("matchTime"),
                "season": match.get("season"),
                "bookmaker": bookmaker,
                "sax_interleaved": sax_interleaved,
                "sax_delta": None,
                "stats": {
                    "home_mean": round(float(np.mean(home)), 3),
                    "draw_mean": round(float(np.mean(draw)), 3),
                    "away_mean": round(float(np.mean(away)), 3),
                    "running_odds_count": len(home),
                },
            }
        else:
            # 使用传统的joint编码 (interleaved + delta)
            joint_series = create_joint_series(home, draw, away)
            delta_series = create_delta_series(home, away)

            return {
                "scheduleId": match.get("scheduleId"),
                "hometeam": match.get("hometeam"),
                "guestteam": match.get("guestteam"),
                "matchTime": match.get("matchTime"),
                "season": match.get("season"),
                "bookmaker": bookmaker,
                "sax_interleaved": encoder.encode(
                    joint_series, params["interpolate_len"] * 3
                ),
                "sax_delta": encoder.encode(delta_series, params["interpolate_len"]),
                "stats": {
                    "home_mean": round(float(np.mean(home)), 3),
                    "draw_mean": round(float(np.mean(draw)), 3),
                    "away_mean": round(float(np.mean(away)), 3),
                    "running_odds_count": len(home),
                },
            }

    def process_file(self, data_file: str) -> Tuple[Dict[str, List], Dict[str, List]]:
        """
        处理单个数据文件

        Returns:
            (results_individual, results_joint): 分别编码和联合编码结果
        """
        print(f"\n处理数据文件: {data_file}")

        with open(data_file, "r", encoding="utf-8") as f:
            matches = json.load(f)
        print(f"  比赛总数: {len(matches)}")

        # 检测文件格式
        first_match = matches[0] if matches else {}
        if "bookmakers" in first_match:
            # 多庄家格式：从每场比赛提取所有庄家
            bookmakers_in_file = set()
            for match in matches:
                for bm in match.get("bookmakers", []):
                    bookmakers_in_file.add(bm.get("bookmakerName"))
            print(f"  庄家列表: {sorted(bookmakers_in_file)}")
        else:
            # 单庄家格式：从文件名提取
            filename = os.path.basename(data_file)
            if "_details.json" in filename:
                bookmaker = filename.replace("_details.json", "")
            else:
                bookmaker = "unknown"
            bookmakers_in_file = {bookmaker}
            print(f"  庄家: {bookmaker}")

        # 初始化结果字典
        results_individual: Dict[str, List] = {}
        results_joint: Dict[str, List] = {}
        for bm in bookmakers_in_file:
            results_individual[bm] = []
            results_joint[bm] = []

        # 处理每场比赛
        errors = []
        for i, match in enumerate(matches):
            for bookmaker in bookmakers_in_file:
                # 分别编码
                result_ind = self.process_match_individual(match, bookmaker)
                if result_ind:
                    results_individual[bookmaker].append(result_ind)

                # 联合编码
                result_joint = self.process_match_joint(match, bookmaker)
                if result_joint:
                    results_joint[bookmaker].append(result_joint)

            if (i + 1) % 500 == 0:
                print(f"    已处理: {i + 1}/{len(matches)}")

        # 打印统计
        for bookmaker in bookmakers_in_file:
            print(
                f"  {bookmaker}: 成功编码 {len(results_individual[bookmaker])} 场比赛"
            )

        return results_individual, results_joint

    def save_results(
        self,
        results_individual: Dict[str, List],
        results_joint: Dict[str, List],
        output_dir: str,
    ):
        """保存所有结果"""
        os.makedirs(output_dir, exist_ok=True)

        all_individual_path = os.path.join(output_dir, "all_sax_individual.json")
        all_joint_path = os.path.join(output_dir, "all_sax_joint.json")

        # 保存汇总文件
        with open(all_individual_path, "w", encoding="utf-8") as f:
            json.dump(results_individual, f, ensure_ascii=False, indent=2)
        print(f"\n分别编码汇总: {all_individual_path}")

        with open(all_joint_path, "w", encoding="utf-8") as f:
            json.dump(results_joint, f, ensure_ascii=False, indent=2)
        print(f"联合编码汇总: {all_joint_path}")

        # 保存各庄家独立文件
        for bookmaker in results_individual:
            if results_individual[bookmaker]:
                # 分别编码
                ind_path = os.path.join(output_dir, f"{bookmaker}_sax_individual.json")
                with open(ind_path, "w", encoding="utf-8") as f:
                    json.dump(
                        results_individual[bookmaker], f, ensure_ascii=False, indent=2
                    )
                print(f"  {bookmaker} 分别编码: {ind_path}")

                # 联合编码
                joint_path = os.path.join(output_dir, f"{bookmaker}_sax_joint.json")
                with open(joint_path, "w", encoding="utf-8") as f:
                    json.dump(results_joint[bookmaker], f, ensure_ascii=False, indent=2)
                print(f"  {bookmaker} 联合编码: {joint_path}")

    def print_statistics(
        self, results_individual: Dict[str, List], results_joint: Dict[str, List]
    ):
        """打印编码统计"""
        print(f"\n{'=' * 60}")
        print("编码结果统计")
        print("=" * 60)

        for bookmaker, results in results_individual.items():
            if not results:
                continue

            print(f"\n【{bookmaker}】")

            # 分别编码 - 各类型分布
            home_counts = Counter(r["sax_home"] for r in results)
            draw_counts = Counter(r["sax_draw"] for r in results)
            away_counts = Counter(r["sax_away"] for r in results)

            print(f"分别编码 - 模式分布 (Top 5):")
            print(f"  主胜 (home): {home_counts.most_common(5)}")
            print(f"  平局 (draw): {draw_counts.most_common(5)}")
            print(f"  客胜 (away): {away_counts.most_common(5)}")

            # 联合编码
            if bookmaker in results_joint and results_joint[bookmaker]:
                joint_results = results_joint[bookmaker]
                joint_counts = Counter(r["sax_interleaved"] for r in joint_results)
                delta_counts = Counter(r["sax_delta"] for r in joint_results)

                print(f"联合编码 - 模式分布 (Top 5):")
                print(f"  交错拼接: {joint_counts.most_common(5)}")
                print(f"  差值编码: {delta_counts.most_common(5)}")


def main():
    parser = argparse.ArgumentParser(description="多庄家赔率 SAX 编码")
    parser.add_argument(
        "--config-dir",
        default="./1.generateOddsDetail/SAX encoder/bookmaker_details",
        help="庄家配置文件目录",
    )
    parser.add_argument(
        "--data-dir",
        default="./1.generateOddsDetail/SAX encoder/bookmaker_details",
        help="数据文件目录",
    )
    parser.add_argument("--output-dir", default="./sax_results", help="输出目录")
    parser.add_argument("--bookmaker", help="指定只处理单个庄家 (如 bet_365, easybets)")
    args = parser.parse_args()

    print("=" * 60)
    print("多庄家赔率 SAX 编码")
    print("=" * 60)

    # 初始化处理器
    processor = MultiBookmakerSAXProcessor(args.config_dir, args.data_dir)

    # 加载所有配置
    processor.load_all_configs()

    # 发现数据文件
    data_files = find_data_files(args.data_dir)
    if not data_files:
        print(f"\n错误: 在 {args.data_dir} 目录下找不到 *_details.json 文件")
        sys.exit(1)

    print(f"\n发现 {len(data_files)} 个数据文件:")
    for f in data_files:
        print(f"  - {f}")

    # 过滤指定庄家
    if args.bookmaker:
        data_files = [f for f in data_files if args.bookmaker in f]
        if not data_files:
            print(f"\n错误: 找不到庄家 {args.bookmaker} 的数据文件")
            sys.exit(1)

    # 处理所有文件
    all_individual: Dict[str, List] = {}
    all_joint: Dict[str, List] = {}

    for data_file in data_files:
        results_ind, results_joint = processor.process_file(data_file)

        # 合并结果
        for bm, res in results_ind.items():
            if bm not in all_individual:
                all_individual[bm] = []
            all_individual[bm].extend(res)

        for bm, res in results_joint.items():
            if bm not in all_joint:
                all_joint[bm] = []
            all_joint[bm].extend(res)

    # 打印统计
    processor.print_statistics(all_individual, all_joint)

    # 保存结果
    processor.save_results(all_individual, all_joint, args.output_dir)

    print(f"\n{'=' * 60}")
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
