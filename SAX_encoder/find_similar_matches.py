#!/usr/bin/env python3
"""
根据比赛 ID 查找 SAX 编码最相似的比赛

功能:
1. 输入比赛ID，下载 titan007.com 的比赛数据
2. 提取 Bet 365 赔率数据
3. 进行 SAX 编码
4. 从 Supabase 查找 SAX 编码最相似的 top10 比赛
5. 使用浏览器打开分析页面

用法:
    python find_similar_matches.py <match_id> [options]

示例:
    # 基本用法（使用均赔筛选，默认）
    python find_similar_matches.py 2799893

    # 使用终盘赔率进行筛选
    python find_similar_matches.py 2799893 --use-final

    # 使用初盘赔率进行筛选
    python find_similar_matches.py 2799893 --use-initial

    # 自定义赔率容忍度（±10%）
    python find_similar_matches.py 2799893 --use-final --tolerance 10

选项:
    --use-final      使用终盘赔率进行筛选（running_odds 最后一条记录）
    --use-initial    使用初盘赔率进行筛选
    --use-mean      使用平均赔率进行筛选（默认）
    --tolerance N   赔率容忍百分比（默认 5.0）

输出:
    - 控制台打印最相似的比赛列表
    - 自动使用浏览器打开分析页面
"""

import json
import os
import re
import sys
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

import numpy as np
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


# ============ SAX 距离计算 ============

# SAX 断点表 (支持 alphabet_size 3-8)
SAX_BREAKPOINTS = {
    3: [-0.43, 0.43],
    4: [-0.67, 0, 0.67],
    5: [-0.84, -0.25, 0.25, 0.84],
    6: [-0.97, -0.43, 0, 0.43, 0.97],
    7: [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
    8: [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15],
}


def symbol_to_number(symbol: str) -> int:
    """符号转数值 (a=0, b=1, ...)"""
    return ord(symbol) - ord("a")


def mindist_sax(s1: str, s2: str, word_size: int, alphabet_size: int) -> float:
    """
    计算两个 SAX 字符串的 MINDIST 距离

    参考: Lin, J., Keogh, E., Lonardi, S., & Chiu, B. (2003).
    A symbolic representation of time series, with implications for streaming algorithms.
    """
    if not s1 or not s2:
        return float("inf")

    # 使用较短的长度
    min_len = min(len(s1), len(s2), word_size)

    breakpoints = SAX_BREAKPOINTS[alphabet_size]

    def char_distance(c1: str, c2: str) -> float:
        n1, n2 = symbol_to_number(c1), symbol_to_number(c2)
        if n1 == n2:
            return 0.0

        # 相邻符号之间的距离为 0
        if abs(n1 - n2) == 1:
            return 0.0

        # 获取断点值（处理最高符号 'g' 的情况）
        def get_breakpoint(n):
            if n < len(breakpoints):
                return breakpoints[n]
            # 最后一个符号 'g' 之后没有断点，使用一个大的正值
            return 3.0

        b1, b2 = get_breakpoint(min(n1, n2)), get_breakpoint(max(n1, n2))
        return (b2 - b1) ** 2

    dist = sum(char_distance(s1[i], s2[i]) for i in range(min_len))
    return np.sqrt(dist / word_size) if word_size > 0 else float("inf")


# ============ SAX 编码器 ============


class SAXEncoder:
    """SAX 编码器 - Symbolic Aggregate Approximation"""

    def __init__(self, word_size=8, alphabet_size=7, config_path=None):
        self.word_size = word_size
        self.alphabet_size = alphabet_size

        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self.breakpoints = self._compute_breakpoints_gaussian()
        self.alphabet = self._generate_alphabet()

    def load_config(self, config_path):
        """从配置文件加载参数"""
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.word_size = config.get("word_size", self.word_size)
        self.alphabet_size = config.get("alphabet_size", self.alphabet_size)

        if (
            config.get("breakpoints_type") == "empirical"
            and "empirical_breakpoints" in config
        ):
            self.breakpoints = np.array(config["empirical_breakpoints"])
        else:
            self.breakpoints = self._compute_breakpoints_gaussian()

        print(
            f"  从配置文件加载 SAX 参数: word_size={self.word_size}, alphabet_size={self.alphabet_size}"
        )

    def _compute_breakpoints_gaussian(self):
        """计算高斯断点"""
        from scipy.stats import norm

        breakpoints = [
            norm.ppf(i / self.alphabet_size) for i in range(1, self.alphabet_size)
        ]
        return np.array(breakpoints)

    def _generate_alphabet(self):
        """生成字母表"""
        return "abcdefghijklmnopqrstuvwxyz"[: self.alphabet_size]

    def normalize(self, series):
        """Z-score 归一化"""
        series = np.array(series, dtype=float)
        mean = np.mean(series)
        std = np.std(series)
        if std == 0:
            return series - mean
        return (series - mean) / std

    def _interpolate_to_length(self, series, target_length):
        """插值到目标长度"""
        if len(series) == target_length:
            return np.array(series)

        if len(series) < target_length:
            x = np.linspace(0, 1, len(series))
            x_new = np.linspace(0, 1, target_length)
            return np.interp(x_new, x, series)
        else:
            return self._paa(series, target_length)

    def _paa(self, series, target_length):
        """PAA 降维"""
        series = np.array(series, dtype=float)
        n = len(series)

        if n == target_length:
            return series

        segment_size = n / target_length
        paa_values = np.zeros(target_length)

        for i in range(target_length):
            start = int(i * segment_size)
            end = int((i + 1) * segment_size)
            if end > n:
                end = n
            if end > start:
                paa_values[i] = np.mean(series[start:end])
            else:
                paa_values[i] = series[start] if start < n else 0

        return paa_values

    def _map_to_symbol(self, value):
        """映射到字母符号"""
        for i, breakpoint in enumerate(self.breakpoints):
            if value < breakpoint:
                return self.alphabet[i]
        return self.alphabet[-1]

    def encode(self, series, interpolate_len=32):
        """SAX 编码"""
        if len(series) < 2:
            return self.alphabet[0] * self.word_size

        interpolated = self._interpolate_to_length(series, interpolate_len)
        normalized = self.normalize(interpolated)
        paa = self._paa(normalized, self.word_size)
        sax_string = "".join([self._map_to_symbol(v) for v in paa])

        return sax_string


# ============ 数据下载 ============


def download_match_data(match_id: str) -> Optional[str]:
    """
    从 titan007.com 下载比赛数据

    Args:
        match_id: 比赛 ID

    Returns:
        下载的 JS 文件内容，或 None（如果下载失败）
    """
    url = f"https://1x2d.titan007.com/{match_id}.js"

    print(f"下载比赛数据: {url}")

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "*/*",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        )

        with urllib.request.urlopen(req, timeout=30) as response:
            content = response.read().decode("utf-8")
            print(f"  下载成功: {len(content)} 字节")
            return content

    except urllib.error.HTTPError as e:
        print(f"  HTTP 错误: {e.code} - {e.reason}")
        return None
    except urllib.error.URLError as e:
        print(f"  URL 错误: {e.reason}")
        return None
    except Exception as e:
        print(f"  下载失败: {e}")
        return None


# ============ Bet 365 赔率提取 ============


@dataclass
class RunningOdds:
    home: str
    draw: str
    away: str
    time: str
    return_rate_home: str
    return_rate_draw: str
    return_rate_away: str


@dataclass
class MatchOdds:
    schedule_id: int
    hometeam: str
    guestteam: str
    match_time: str
    initial_odds_home: str
    initial_odds_draw: str
    initial_odds_away: str
    final_odds_home: str
    final_odds_draw: str
    final_odds_away: str
    running_odds: list[RunningOdds]


def extract_bet365_odds(content: str) -> Optional[MatchOdds]:
    """
    从下载的 JS 内容中提取 Bet 365 赔率数据

    参考 detail.js 的解析逻辑

    Args:
        content: JS 文件内容

    Returns:
        MatchOdds 对象，或 None（如果提取失败）
    """
    # 提取基本信息
    schedule_id_match = re.search(r"var ScheduleID=(\d+);", content)
    hometeam_match = re.search(r'var hometeam="([^"]+)";', content)
    guestteam_match = re.search(r'var guestteam="([^"]+)";', content)
    match_time_match = re.search(r'var MatchTime="([^"]+)";', content)

    if not schedule_id_match:
        print("  错误: 无法找到 ScheduleID")
        return None

    # 提取 game 数据
    game_match = re.search(r"var game=Array\(([\s\S]*?)\);", content)
    if not game_match:
        print("  错误: 无法找到 game 数据")
        return None

    game_content = game_match.group(1)
    # 去除首尾的引号
    game_content = game_content.strip('"')
    game_items = game_content.split('","')

    # 找到 Bet 365 的庄家ID
    bet365_id = None
    for item in game_items:
        parts = item.split("|")
        if len(parts) >= 3 and parts[2] == "Bet 365":
            bet365_id = parts[1]
            break

    if not bet365_id:
        print("  错误: 未找到 Bet 365 赔率数据")
        return None

    # 提取 gameDetail 数据
    game_detail_match = re.search(r"var gameDetail=Array\(([\s\S]*?)\);", content)
    if not game_detail_match:
        print("  错误: 无法找到 gameDetail 数据")
        return None

    detail_content = game_detail_match.group(1)
    # 去除首尾的引号
    detail_content = detail_content.strip('"')
    detail_items = detail_content.split('","')

    bet365_detail = None
    for item in detail_items:
        if item.startswith(bet365_id + "^"):
            bet365_detail = item
            break

    if not bet365_detail:
        print("  错误: 未找到 Bet 365 详细赔率数据")
        return None

    # 解析 Bet 365 的详细赔率
    try:
        odds_records = bet365_detail.split("^")[1].split(";")
        odds_records = [r for r in odds_records if r.strip()]

        running_odds = []
        for record in odds_records:
            parts = record.split("|")
            if len(parts) >= 7:
                running_odds.append(
                    RunningOdds(
                        home=parts[0],
                        draw=parts[1],
                        away=parts[2],
                        time=parts[3],
                        return_rate_home=parts[4],
                        return_rate_draw=parts[5],
                        return_rate_away=parts[6],
                    )
                )
    except Exception as e:
        print(f"  解析详细赔率失败: {e}")
        return None

    # 获取初始赔率和终盘赔率
    initial_odds_home = ""
    initial_odds_draw = ""
    initial_odds_away = ""
    final_odds_home = ""
    final_odds_draw = ""
    final_odds_away = ""

    for item in game_items:
        if "|Bet 365|" in item:
            clean_data = item.replace('^"|"$', "").strip('"')
            parts = clean_data.split("|")
            if len(parts) >= 14:
                # parts[3-5] 是初赔
                initial_odds_home = parts[3]
                initial_odds_draw = parts[4]
                initial_odds_away = parts[5]
                # parts[10-12] 是终赔
                final_odds_home = parts[10]
                final_odds_draw = parts[11]
                final_odds_away = parts[12]
            break

    return MatchOdds(
        schedule_id=int(schedule_id_match[1]) if schedule_id_match else 0,
        hometeam=hometeam_match[1] if hometeam_match else "",
        guestteam=guestteam_match[1] if guestteam_match else "",
        match_time=match_time_match[1] if match_time_match else "",
        initial_odds_home=initial_odds_home,
        initial_odds_draw=initial_odds_draw,
        initial_odds_away=initial_odds_away,
        final_odds_home=final_odds_home,
        final_odds_draw=final_odds_draw,
        final_odds_away=final_odds_away,
        running_odds=running_odds,
    )


def extract_odds_series(match_odds: MatchOdds):
    """
    从 MatchOdds 中提取数值序列

    Args:
        match_odds: MatchOdds 对象

    Returns:
        (home_series, draw_series, away_series): 三个赔率序列
    """
    home = [float(o.home) for o in match_odds.running_odds]
    draw = [float(o.draw) for o in match_odds.running_odds]
    away = [float(o.away) for o in match_odds.running_odds]

    return home, draw, away


# ============ Supabase 查询 ============


def get_supabase_client():
    """获取 Supabase REST 客户端"""
    from postgrest import SyncPostgrestClient

    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_KEY", "")

    if not url or not key:
        raise ValueError("请在 .env 文件中设置 SUPABASE_URL 和 SUPABASE_KEY")

    return SyncPostgrestClient(
        f"{url}/rest/v1",
        headers={
            "Authorization": f"Bearer {key}",
            "apikey": key,
            "Content-Type": "application/json",
        },
    )


def fetch_all_sax_data(client) -> list[dict]:
    """从 Supabase 获取所有 SAX 数据"""
    print("从 Supabase 获取 SAX 数据...")

    try:
        # 获取所有数据（没有过滤条件）
        result = client.table("match_odds_sax").select("*").execute()

        if not result.data:
            print("  警告: Supabase 中没有找到数据")
            return []

        print(f"  已加载 {len(result.data)} 条记录")
        return result.data

    except Exception as e:
        print(f"  获取数据失败: {e}")
        return []


def find_similar_matches(
    client,
    target_sax_interleaved: str,
    target_sax_delta: str,
    target_odds: dict,
    word_size: int = 8,
    alphabet_size: int = 7,
    top_n: int = 10,
    odds_tolerance_pct: float = 5.0,
    use_initial_odds: bool = True,
    use_final_odds: bool = False,
) -> list[dict]:
    """
    查找 SAX 编码最相似的比赛（带赔率筛选）

    Args:
        client: Supabase 客户端
        target_sax_interleaved: 目标比赛的交错 SAX 编码
        target_sax_delta: 目标比赛的差值 SAX 编码
        target_odds: 目标比赛的赔率
            - 使用初盘时: {'init_win': x, 'init_draw': x, 'init_lose': x}
            - 使用均赔时: {'home_mean': x, 'draw_mean': x, 'away_mean': x}
            - 使用终盘时: {'final_win': x, 'final_draw': x, 'final_lose': x}
        word_size: SAX 分段数
        alphabet_size: 字母表大小
        top_n: 返回前 N 个最相似的比赛
        odds_tolerance_pct: 赔率容忍百分比（默认 ±5%）
        use_initial_odds: 是否使用初盘赔率筛选（默认 False，与终盘互斥）
        use_final_odds: 是否使用终盘赔率筛选（默认 False，与初盘互斥）

    Returns:
        最相似的比赛列表
    """
    # 确定赔率筛选类型
    if use_final_odds:
        odds_type = "终盘"
    elif use_initial_odds:
        odds_type = "初盘"
    else:
        odds_type = "均赔"

    print(
        f"\n查找 SAX 编码最相似的比赛 (赔率筛选: {odds_type}, ±{odds_tolerance_pct}%)..."
    )
    print(f"  目标交错编码: {target_sax_interleaved}")
    print(f"  目标差值编码: {target_sax_delta}")

    if use_final_odds:
        print(
            f"  目标终盘赔率: 胜={target_odds.get('final_win'):.2f}, 平={target_odds.get('final_draw'):.2f}, 负={target_odds.get('final_lose'):.2f}"
        )
    elif use_initial_odds:
        print(
            f"  目标初盘赔率: 胜={target_odds.get('init_win'):.2f}, 平={target_odds.get('init_draw'):.2f}, 负={target_odds.get('init_lose'):.2f}"
        )
    else:
        print(
            f"  目标平均赔率: 胜={target_odds.get('home_mean'):.2f}, 平={target_odds.get('draw_mean'):.2f}, 负={target_odds.get('away_mean'):.2f}"
        )

    # 获取所有数据
    all_data = fetch_all_sax_data(client)

    if not all_data:
        return []

    # 计算距离（仅限赔率相近的比赛）
    similarities = []
    filtered_count = 0

    if use_final_odds:
        # 使用终盘赔率筛选
        target_home = target_odds.get("final_win", 0)
        target_draw = target_odds.get("final_draw", 0)
        target_away = target_odds.get("final_lose", 0)
    elif use_initial_odds:
        # 使用初盘赔率筛选
        target_home = target_odds.get("init_win", 0)
        target_draw = target_odds.get("init_draw", 0)
        target_away = target_odds.get("init_lose", 0)
    else:
        # 使用均赔筛选
        target_home = target_odds.get("home_mean", 0)
        target_draw = target_odds.get("draw_mean", 0)
        target_away = target_odds.get("away_mean", 0)

    # 转换为百分比阈值
    home_threshold = target_home * (odds_tolerance_pct / 100)
    draw_threshold = target_draw * (odds_tolerance_pct / 100)
    away_threshold = target_away * (odds_tolerance_pct / 100)

    for item in all_data:
        sax_interleaved = item.get("sax_interleaved", "")
        sax_delta = item.get("sax_delta", "")

        if not sax_interleaved:
            continue

        if use_final_odds:
            # 使用终盘赔率
            item_home = item.get("final_win", 0)
            item_draw = item.get("final_draw", 0)
            item_away = item.get("final_lose", 0)
        elif use_initial_odds:
            # 使用初盘赔率
            item_home = item.get("init_win", 0)
            item_draw = item.get("init_draw", 0)
            item_away = item.get("init_lose", 0)
        else:
            # 使用均赔
            item_home = item.get("home_mean", 0)
            item_draw = item.get("draw_mean", 0)
            item_away = item.get("away_mean", 0)

        # 跳过无效赔率
        if not item_home or not item_draw or not item_away:
            continue

        # 赔率筛选：三个赔率都在 ±5% 范围内
        if (
            abs(item_home - target_home) > home_threshold
            or abs(item_draw - target_draw) > draw_threshold
            or abs(item_away - target_away) > away_threshold
        ):
            continue

        filtered_count += 1

        # 计算 MINDIST 距离
        dist_interleaved = mindist_sax(
            target_sax_interleaved, sax_interleaved, word_size, alphabet_size
        )
        dist_delta = (
            mindist_sax(target_sax_delta, sax_delta, word_size, alphabet_size)
            if sax_delta
            else float("inf")
        )

        # 综合距离（加权平均）
        combined_dist = (
            (dist_interleaved + dist_delta) / 2 if sax_delta else dist_interleaved
        )

        similarities.append(
            {
                "match_id": item.get("match_id"),
                "sax_interleaved": sax_interleaved,
                "sax_delta": sax_delta,
                "dist_interleaved": dist_interleaved,
                "dist_delta": dist_delta,
                "combined_dist": combined_dist,
                "item_home": item_home,
                "item_draw": item_draw,
                "item_away": item_away,
            }
        )

    print(f"  赔率筛选后剩余: {filtered_count}/{len(all_data)} 场比赛")

    # 按综合距离排序
    similarities.sort(key=lambda x: x["combined_dist"])

    # 返回 top N（排除自身）
    result = [s for s in similarities if s["match_id"] != target_sax_interleaved][
        :top_n
    ]

    return result


def extract_odds_for_sax(home, draw, away):
    """提取赔率序列用于 SAX 编码"""
    return home, draw, away


# ============ 浏览器控制 ============


def open_analysis_pages(matches: list[dict]):
    """
    使用浏览器打开分析页面

    Args:
        matches: 最相似的比赛列表
    """
    import webbrowser
    import subprocess

    print(f"\n准备打开分析页面...")

    base_url = "https://zq.titan007.com/analysis/"

    urls = []
    for match in matches:
        match_id = match.get("match_id")
        if match_id:
            url = f"{base_url}{match_id}.htm"
            urls.append(url)

    if not urls:
        print("  没有有效的比赛 ID，无法打开页面")
        return

    print(f"  将打开 {len(urls)} 个页面:")
    for i, url in enumerate(urls, 1):
        print(f"    {i}. {url}")

    # 使用系统浏览器打开所有页面
    try:
        # macOS 使用 open 命令，其他系统使用 webbrowser
        import platform

        system = platform.system()

        if system == "Darwin":  # macOS
            for url in urls:
                subprocess.run(["open", "-g", url], check=False)
        else:
            # 其他系统使用 webbrowser
            browser = webbrowser.get()
            # 使用 new=2 参数在新窗口中打开
            for url in urls:
                webbrowser.open(url, new=2)

        print(f"\n  ✓ 已打开 {len(urls)} 个页面")
    except Exception as e:
        print(f"  打开页面失败: {e}")
        print("  请手动访问:")
        for url in urls:
            print(f"    {url}")


# ============ 主程序 ============


def main():
    print("=" * 60)
    print("根据比赛 ID 查找 SAX 编码最相似的比赛")
    print("=" * 60)

    # 加载 SAX 编码参数
    config_path = "sax_config.json"
    WORD_SIZE = 8
    ALPHABET_SIZE = 7
    INTERPOLATE_LEN = 32

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        WORD_SIZE = config.get("word_size", WORD_SIZE)
        ALPHABET_SIZE = config.get("alphabet_size", ALPHABET_SIZE)
        INTERPOLATE_LEN = config.get("interpolate_len", INTERPOLATE_LEN)
        print(
            f"  从配置文件加载参数: word_size={WORD_SIZE}, alphabet_size={ALPHABET_SIZE}, interpolate_len={INTERPOLATE_LEN}"
        )

    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python find_similar_matches.py <match_id> [options]")
        print("示例: python find_similar_matches.py 2799893")
        print("")
        print("选项:")
        print("  --use-final      使用终盘赔率进行筛选")
        print("  --use-initial    使用初盘赔率进行筛选")
        print("  --use-mean       使用平均赔率进行筛选（默认）")
        print("  --tolerance N    赔率容忍百分比（默认 5）")
        sys.exit(1)

    match_id = sys.argv[1]

    # 解析可选参数
    use_final_odds = False
    use_initial_odds = False  # 默认使用均赔
    odds_tolerance_pct = 5.0

    i = 2
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg == "--use-final":
            use_final_odds = True
            use_initial_odds = False
        elif arg == "--use-initial":
            use_final_odds = False
            use_initial_odds = True
        elif arg == "--use-mean":
            use_final_odds = False
            use_initial_odds = False
        elif arg == "--tolerance" and i + 1 < len(sys.argv):
            try:
                odds_tolerance_pct = float(sys.argv[i + 1])
                i += 1
            except ValueError:
                pass
        i += 1

    # 1. 下载比赛数据
    print(f"\n[1/5] 下载比赛数据 (ID: {match_id})")
    content = download_match_data(match_id)

    if not content:
        print("错误: 下载比赛数据失败")
        sys.exit(1)

    # 2. 提取 Bet 365 赔率
    print(f"\n[2/5] 提取 Bet 365 赔率数据")
    match_odds = extract_bet365_odds(content)

    if not match_odds:
        print("错误: 提取赔率数据失败")
        sys.exit(1)

    print(f"  主队: {match_odds.hometeam}")
    print(f"  客队: {match_odds.guestteam}")
    print(f"  比赛时间: {match_odds.match_time}")
    print(
        f"  初始赔率: 胜={match_odds.initial_odds_home}, 平={match_odds.initial_odds_draw}, 负={match_odds.initial_odds_away}"
    )
    print(f"  赔率变化记录数: {len(match_odds.running_odds)}")
    print(
        f"  终盘赔率: 胜={match_odds.final_odds_home}, 平={match_odds.final_odds_draw}, 负={match_odds.final_odds_away}"
    )

    # 打印所有赔率记录
    print(f"\n  [所有赔率记录] 共 {len(match_odds.running_odds)} 条:")
    print(f"  {'序号':<6} {'时间':<25} {'胜':<8} {'平':<8} {'负':<8}")
    print(f"  {'-' * 60}")
    for i, ro in enumerate(match_odds.running_odds, 1):
        print(f"  {i:<6} {ro.time:<25} {ro.home:<8} {ro.draw:<8} {ro.away:<8}")

    if len(match_odds.running_odds) < 2:
        print("错误: 赔率变化记录太少，无法进行 SAX 编码")
        sys.exit(1)

    # 3. SAX 编码
    print(f"\n[3/5] SAX 编码")
    encoder = SAXEncoder(
        word_size=WORD_SIZE,
        alphabet_size=ALPHABET_SIZE,
        config_path=config_path if os.path.exists(config_path) else None,
    )

    home, draw, away = extract_odds_series(match_odds)

    # 交错拼接序列
    min_len = min(len(home), len(draw), len(away))
    joint_series = []
    for i in range(min_len):
        joint_series.append(home[i])
        joint_series.append(draw[i])
        joint_series.append(away[i])

    # 差值序列
    delta_series = [home[i] - away[i] for i in range(min_len)]

    # 编码
    sax_interleaved = encoder.encode(joint_series, INTERPOLATE_LEN * 3)
    sax_delta = encoder.encode(delta_series, INTERPOLATE_LEN)

    # 计算目标比赛的平均赔率
    import numpy as np

    # 初始赔率
    init_home = (
        float(match_odds.initial_odds_home) if match_odds.initial_odds_home else 0
    )
    init_draw = (
        float(match_odds.initial_odds_draw) if match_odds.initial_odds_draw else 0
    )
    init_away = (
        float(match_odds.initial_odds_away) if match_odds.initial_odds_away else 0
    )

    # 终盘赔率（从 match_odds 字段获取）
    final_home = float(match_odds.final_odds_home) if match_odds.final_odds_home else 0
    final_draw = float(match_odds.final_odds_draw) if match_odds.final_odds_draw else 0
    final_away = float(match_odds.final_odds_away) if match_odds.final_odds_away else 0

    target_odds = {
        "init_win": init_home,
        "init_draw": init_draw,
        "init_lose": init_away,
        "home_mean": round(float(np.mean(home)), 3),
        "draw_mean": round(float(np.mean(draw)), 3),
        "away_mean": round(float(np.mean(away)), 3),
        "final_win": final_home,
        "final_draw": final_draw,
        "final_lose": final_away,
    }

    print(f"  交错编码: {sax_interleaved}")
    print(f"  差值编码: {sax_delta}")
    print(f"  初盘赔率: 胜={init_home:.2f}, 平={init_draw:.2f}, 负={init_away:.2f}")
    print(f"  终盘赔率: 胜={final_home:.2f}, 平={final_draw:.2f}, 负={final_away:.2f}")
    print(
        f"  均赔: 胜={target_odds['home_mean']:.2f}, 平={target_odds['draw_mean']:.2f}, 负={target_odds['away_mean']:.2f}"
    )
    print(f"  编码参数: word_size={WORD_SIZE}, alphabet_size={ALPHABET_SIZE}")

    # 4. 从 Supabase 查找相似比赛
    print(f"\n[4/5] 从 Supabase 查找相似比赛")

    try:
        client = get_supabase_client()
        similar_matches = find_similar_matches(
            client,
            sax_interleaved,
            sax_delta,
            target_odds=target_odds,
            word_size=WORD_SIZE,
            alphabet_size=ALPHABET_SIZE,
            top_n=10,
            odds_tolerance_pct=odds_tolerance_pct,
            use_initial_odds=use_initial_odds,
            use_final_odds=use_final_odds,
        )

        if not similar_matches:
            print("  未找到相似的比赛")
        else:
            # 根据筛选类型显示对应的赔率列名
            if use_final_odds:
                odds_label = "终盘赔率(胜/平/负)"
            elif use_initial_odds:
                odds_label = "初盘赔率(胜/平/负)"
            else:
                odds_label = "均赔(胜/平/负)"

            print(f"\n找到 {len(similar_matches)} 场最相似的比赛:")
            print("-" * 60)
            print(
                f"{'排名':<4} {'比赛ID':<12} {'交错编码':<20} {'距离':<8} {odds_label:<20}"
            )
            print("-" * 60)

            for i, match in enumerate(similar_matches, 1):
                print(
                    f"{i:<4} {match['match_id']:<12} {match['sax_interleaved']:<20} "
                    f"{match['combined_dist']:.4f} "
                    f"{match['item_home']:.2f}/{match['item_draw']:.2f}/{match['item_away']:.2f}"
                )

            print("-" * 60)

    except ValueError as e:
        print(f"\n配置错误: {e}")
        print("请确保 .env 文件中已设置 SUPABASE_URL 和 SUPABASE_KEY")
        similar_matches = []
    except Exception as e:
        print(f"\n查询失败: {e}")
        import traceback

        traceback.print_exc()
        similar_matches = []

    # 5. 打开分析页面
    print(f"\n[5/5] 打开分析页面")
    if similar_matches:
        open_analysis_pages(similar_matches)
    else:
        print("  没有相似的比赛需要打开")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
