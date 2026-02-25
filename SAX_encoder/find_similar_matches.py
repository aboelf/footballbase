#!/usr/bin/env python3
"""
根据比赛 ID 查找 SAX 编码最相似的比赛

功能:
1. 输入比赛ID，下载 titan007.com 的比赛数据
2. 提取指定庄家赔率数据（支持 Bet 365, Easybets）
3. 进行 SAX 编码
4. 从 Supabase 查找 SAX 编码最相似的比赛
5. 使用浏览器打开分析页面

用法:
    python find_similar_matches.py <match_id> [options]

示例:
    # 基本用法（Bet 365，均赔筛选）
    python find_similar_matches.py 2799893

    # 使用 Easybets 庄家
    python find_similar_matches.py 2799893 --bookmaker Easybets

    # 使用终盘赔率进行筛选
    python find_similar_matches.py 2799893 --use-final

    # 使用初盘赔率进行筛选
    python find_similar_matches.py 2799893 --use-initial

    # 自定义赔率容忍度（±10%）
    python find_similar_matches.py 2799893 --use-final --tolerance 10

选项:
    --bookmaker NAME  指定庄家 (bet_365, easybets，默认: bet_365)
    --use-final       使用终盘赔率进行筛选
    --use-initial     使用初盘赔率进行筛选
    --use-both        同时使用初盘和终盘赔率进行筛选
    --use-mean        使用平均赔率进行筛选（默认）
    --tolerance N     赔率容忍百分比（默认 5.0）
    --use-dist N       combined_dist 阈值，只保留距离小于等于该值的比赛（默认 0.5）

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

        # 相邻符号之间使用较小的距离（非零）
        if abs(n1 - n2) == 1:
            # 计算相邻符号之间的断点距离的一半作为最小距离
            min_idx = min(n1, n2)
            if min_idx < len(breakpoints):
                b = breakpoints[min_idx]
                if min_idx > 0:
                    b_prev = breakpoints[min_idx - 1]
                    return ((b - b_prev) ** 2) / 4
            return 0.01  # 默认小距离
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
        """插值到目标长度（已标准化的数据）"""
        if len(series) == target_length:
            return np.array(series)

        if len(series) < target_length:
            # 上采样：线性插值
            x = np.linspace(0, 1, len(series))
            x_new = np.linspace(0, 1, target_length)
            return np.interp(x_new, x, series)
        else:
            # 下采样：均匀采样（保持形状）
            return self._simple_downsample(series, target_length)

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
        """
        SAX 编码
        
        注意: 输入数据应该已经是 Z-Score 标准化的，且长度固定为20点
        这里只做必要的插值和符号映射，不重复标准化和PAA
        """
        if len(series) < 2:
            return self.alphabet[0] * self.word_size

        # 直接对已标准化的数据进行处理
        # 1. 插值到目标长度（如果需要）
        interpolated = self._interpolate_to_length(series, interpolate_len)
        
        # 2. 降维到 word_size（使用简单的均匀采样替代PAA，保持形状）
        # 由于数据已经是20点标准化数据，直接均匀采样到word_size
        paa = self._simple_downsample(interpolated, self.word_size)
        
        # 3. 映射到符号
        sax_string = "".join([self._map_to_symbol(v) for v in paa])

        return sax_string
    
    def _simple_downsample(self, series, target_length):
        """
        简单降维：均匀采样
        
        替代PAA，保持数据的原始形状特征
        """
        series = np.array(series, dtype=float)
        n = len(series)
        
        if n == target_length:
            return series
        
        # 均匀采样 indices
        indices = np.linspace(0, n - 1, target_length, dtype=int)
        return series[indices]


# ============ 赔率数据预处理：时间对齐与重采样 ============


from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional


def parse_datetime(time_str: str, year_hint: int = None) -> Optional[datetime]:
    """
    解析各种格式的时间字符串
    
    支持格式:
    - "2024-01-15 10:30:00"
    - "2024/01/15 10:30"
    - "01-15 10:30" (月-日 时:分)
    - "02-21 22:45" (月-日 时:分，从详细记录)
    - "2026,02-1,21,15,00,00" (年,月-日,时,分,秒) - 注意: 月-日 可能是 "月-日"
    """
    if not time_str:
        return None
    
    import re
    
    # 处理详细记录格式: "02-21 22:45" (月-日 时:分)
    match = re.match(r'(\d{1,2})-(\d{1,2})\s+(\d{1,2}):(\d{1,2})', time_str)
    if match:
        month, day, hour, minute = match.groups()
        year = year_hint if year_hint else datetime.now().year
        try:
            return datetime(year, int(month), int(day), int(hour), int(minute))
        except ValueError:
            pass
    
    # 处理特殊格式: "2026,02-1,21,15,00,00"
    # 注意: 这里 "02-1" 可能是 "02-1" 即 2月1日，但实际应该是 2月21日
    # 这个格式解析有误，暂时不使用
    # 格式: 年,月-日,时,分,秒
    # match = re.match(r'(\d{4}),(\d{1,2})-(\d{1,2}),(\d{1,2}),(\d{1,2}),(\d{1,2})', time_str)
    # 实际上这个格式 "02-1" 更像是 "月份-日期" 但日期可能是1位数
    # 从详细记录看应该是 "月-日" 格式，如 "02-21"
    
    # 尝试解析: 年,月-日,时:分:秒
    match = re.match(r'(\d{4}),(\d{1,2})-(\d{1,2}),(\d{1,2}):(\d{1,2}):(\d{1,2})', time_str)
    if match:
        try:
            year, month, day, hour, minute, second = match.groups()
            # 这里 day=1 可能是日期，需要结合实际数据修正
            dt = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second))
            # 如果解析出的日期与详细记录不符，需要修正
            # 从详细记录我们知道比赛是2月21日
            # 这里先按解析结果返回，后续可修正
            return dt
        except ValueError:
            pass
    
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(time_str.strip(), fmt)
        except ValueError:
            continue
    
    return None


def parse_match_time_from_detail(match_time_str: str, odds_records_times: list = None) -> datetime:
    """
    从比赛时间字符串和赔率记录时间推断正确的比赛时间
    
    titan007 的 MatchTime 格式: "2026,02-1,11,19,40,00" -> 2026年2月11日19:40:00
    格式: year, month-day, day, hour, minute, second
    注意: "02-1" 实际上是 "month-day" 但day可能是1位数
    
    Args:
        match_time_str: MatchTime 字段值
        odds_records_times: 详细赔率记录的时间列表（已解析的datetime）
        
    Returns:
        修正后的比赛时间 (北京时间)
    """
    import re
    
    # 尝试解析 "2026,02-1,11,19,40,00" 格式
    # 关键: 用 (\d+-\d+) 捕获完整的 "02-1" 而不是 (\d+)-(\d+)
    match = re.match(r'(\d+),(\d+-\d+),(\d+),(\d+),(\d+),(\d+)', match_time_str)
    if match:
        try:
            year = int(match.group(1))
            month_day = match.group(2)  # "02-1" or "02-21"
            day = int(match.group(3))   # "11"
            hour = int(match.group(4))
            minute = int(match.group(5))
            second = int(match.group(6))
            
            # 解析月份
            md = month_day.split('-')
            month = int(md[0])
            
            # JS数据时间是UTC，需要+8小时转北京时间
            utc_time = datetime(year, month, day, hour, minute, second)
            beijing_time = utc_time + timedelta(hours=8)
            return beijing_time
        except (ValueError, AttributeError):
            pass
    
    # 方法2: 直接解析标准格式
    parsed = parse_datetime(match_time_str)
    if parsed:
        return parsed
    
    # 方法3: 从赔率记录最后一条推断（备用）
    if odds_records_times and len(odds_records_times) > 0:
        last_time = max(odds_records_times)
        # 假设比赛在最后一次赔率变化后15分钟左右
        return last_time + timedelta(minutes=15)
    
    raise ValueError(f"无法解析比赛时间: {match_time_str}")


def odds_to_implied_probability(odds: float) -> float:
    """
    赔率转换为隐含胜率（未扣返还率）
    
    Args:
        odds: 赔率值 (如 2.10)
        
    Returns:
        隐含胜率 (如 0.476)
    """
    if not odds or odds <= 0:
        return 0.0
    return 1.0 / odds


def odds_to_real_probability(home_odds: float, draw_odds: float, away_odds: float) -> Tuple[float, float, float]:
    """
    赔率转换为真实胜率（扣除返还率）
    
    计算步骤:
    1. 先算隐含概率: p = 1/odds
    2. 计算三项总和: total = p_home + p_draw + p_away
    3. 真实概率: real_p = p / total
    
    Args:
        home_odds: 主胜赔率 (如 1.75)
        draw_odds: 平局赔率 (如 3.70)
        away_odds: 客胜赔率 (如 4.33)
        
    Returns:
        (real_home, real_draw, real_away): 真实胜率
        
    Example:
        1.75, 3.70, 4.33
        → 隐含: 0.5714, 0.2703, 0.2309
        → 总和: 1.0726
        → 真实: 0.533, 0.252, 0.215
    """
    if not home_odds or not draw_odds or not away_odds:
        return 0.0, 0.0, 0.0
    if home_odds <= 0 or draw_odds <= 0 or away_odds <= 0:
        return 0.0, 0.0, 0.0
    
    # 隐含概率
    implied_home = 1.0 / home_odds
    implied_draw = 1.0 / draw_odds
    implied_away = 1.0 / away_odds
    
    # 三项总和（返还率）
    total = implied_home + implied_draw + implied_away
    
    if total <= 0:
        return 0.0, 0.0, 0.0
    
    # 真实概率
    real_home = implied_home / total
    real_draw = implied_draw / total
    real_away = implied_away / total
    
    return real_home, real_draw, real_away


def resample_odds_to_fixed_timeline(
    running_odds: list,
    match_time: str,
    initial_odds: Tuple[float, float, float] = None,
    time_step_minutes: int = 10,
    pre_match_hours: int = 48,
) -> Dict[str, List[float]]:
    """
    赔率数据时间对齐与重采样
    
    时间轴分段:
    1. 开盘: 第一个赔率
    2. 开盘到赛前48小时: 赔率均值（时间加权）
    3. 赛前48小时到开赛: 每10分钟一个数据点
    
    Args:
        running_odds: 赔率变化记录列表 [{home, draw, away, time}, ...]
        match_time: 比赛时间 (字符串)
        initial_odds: (可选) 初始赔率 (home, draw, away)
        time_step_minutes: 时间步长（分钟），默认10
        pre_match_hours: 赛前多少小时开始详细采样，默认48
        
    Returns:
        {
            'home': [开盘胜率, 48小时均值, 47小时50分, 47小时40分, ..., 开赛胜率],
            'draw': [...],
            'away': [...]
        }
    """
    # 解析比赛时间（使用新函数修正）
    # 先收集详细记录的时间
    odds_times_for_match = []
    for record in running_odds:
        rt = parse_datetime(record.time)
        if rt:
            odds_times_for_match.append(rt)
    
    # 使用修正后的比赛时间
    kickoff = parse_match_time_from_detail(match_time, odds_times_for_match)
    if not kickoff:
        print(f"  警告: 无法解析比赛时间 {match_time}")
        return {'home': [], 'draw': [], 'away': []}
    
    print(f"  修正后的比赛时间: {kickoff}")
    
    # 解析赔率变化记录
    odds_records = []
    for record in running_odds:
        # 使用带年份提示的解析
        record_time = parse_datetime(record.time, year_hint=kickoff.year)
        if not record_time:
            continue
        
        # 处理年份：如果没有解析出年份，使用比赛年份
        if record_time.year == 1900 or record_time.year == 1:
            record_time = record_time.replace(year=kickoff.year)
        
        # 如果比比赛时间还晚，说明是去年
        if record_time > kickoff:
            record_time = record_time.replace(year=kickoff.year - 1)
        
        try:
            home = float(record.home)
            draw = float(record.draw)
            away = float(record.away)
            
            # 计算真实胜率（扣除返还率）
            real_home, real_draw, real_away = odds_to_real_probability(home, draw, away)
            
            odds_records.append({
                'time': record_time,
                'home': home,
                'draw': draw,
                'away': away,
                'home_prob': real_home,
                'draw_prob': real_draw,
                'away_prob': real_away,
            })
        except (ValueError, TypeError):
            continue
    
    if not odds_records:
        print("  警告: 无有效的赔率记录")
        return {'home': [], 'draw': [], 'away': []}
    
    # 按时间排序
    odds_records.sort(key=lambda x: x['time'])
    
    earliest = odds_records[0]['time']
    cutoff_48h = kickoff - timedelta(hours=pre_match_hours)
    
    print(f"  最早赔率时间: {earliest}")
    print(f"  赛前48小时截止: {cutoff_48h}")
    print(f"  比赛时间: {kickoff}")
    print(f"  有效记录数: {len(odds_records)}")
    
    # ========== 第一部分: 开盘赔率 ==========
    first_record = odds_records[0]
    opening_odds = {
        'home': first_record['home_prob'],
        'draw': first_record['draw_prob'],
        'away': first_record['away_prob'],
    }
    print(f"  开盘胜率: {opening_odds['home']:.3f}")
    
    # ========== 第二部分: 开盘到赛前48小时均值（时间加权）===========
    early_records = [r for r in odds_records if r['time'] <= cutoff_48h]
    
    if len(early_records) >= 2:
        # 时间加权均值
        total_weight = 0.0
        weighted_home = 0.0
        weighted_draw = 0.0
        weighted_away = 0.0
        
        for i in range(1, len(early_records)):
            # 权重 = 持续时间（小时）
            duration = (early_records[i]['time'] - early_records[i-1]['time']).total_seconds() / 3600
            if duration <= 0:
                duration = 0.1  # 最小权重
            
            weighted_home += early_records[i]['home_prob'] * duration
            weighted_draw += early_records[i]['draw_prob'] * duration
            weighted_away += early_records[i]['away_prob'] * duration
            total_weight += duration
        
        if total_weight > 0:
            early_avg_odds = {
                'home': weighted_home / total_weight,
                'draw': weighted_draw / total_weight,
                'away': weighted_away / total_weight,
            }
        else:
            # 如果无法计算加权，使用简单均值
            early_avg_odds = {
                'home': sum(r['home_prob'] for r in early_records) / len(early_records),
                'draw': sum(r['draw_prob'] for r in early_records) / len(early_records),
                'away': sum(r['away_prob'] for r in early_records) / len(early_records),
            }
    else:
        # 记录太少，使用最后一个可用值
        early_avg_odds = {
            'home': early_records[-1]['home_prob'] if early_records else opening_odds['home'],
            'draw': early_records[-1]['draw_prob'] if early_records else opening_odds['draw'],
            'away': early_records[-1]['away_prob'] if early_records else opening_odds['away'],
        }
    
    print(f"  48小时均值胜率: {early_avg_odds['home']:.3f}")
    
    # ========== 第三部分: 赛前48小时到开赛，精简采样 ==========
    # 方案A: 固定总点数，确保所有比赛时间轴一致
    TARGET_POINTS = 20  # 目标总点数
    
    cutoff_24h = kickoff - timedelta(hours=24)
    
    # 找出所有赔率变化点 (只在24h内)
    change_points = []
    prev_home = early_avg_odds['home']
    for record in odds_records:
        if cutoff_24h < record['time'] <= kickoff:
            if abs(record['home_prob'] - prev_home) > 0.001:  # 0.1%变化阈值
                change_points.append(record['time'])
                prev_home = record['home_prob']
    
    print(f"  24h内赔率变化点: {len(change_points)} 个")
    
    # ========== 构建时间轴 (固定策略) ==========
    # 策略:
    # - 开盘 + 48h均值 = 2个固定点 (在resampled中)
    # - 48h~24h: 保留3个点 (每12h一个)
    # - 24h~开赛: 保留15个点 (变化点优先，不足则均匀采样)
    # - 开赛: 1个点
    # 总计: 2 + 3 + 15 + 1 = 21 (接近目标)
    
    time_points = []
    
    # 开盘 (索引0)
    # 48h均值 (索引1) - 已包含在resampled中
    
    # 48h ~ 24h: 每12小时 (3个点: 48h, 36h, 24h)
    for hours in [48, 36, 24]:
        time_points.append(kickoff - timedelta(hours=hours))
    
    # 24h ~ 开赛: 优先保留变化点，不足则均匀采样
    # 先收集所有候选点
    candidate_points = []
    
    # 24h内均匀采样点 (每2小时) - 覆盖整个24h范围
    for hours in range(24, 0, -2):
        candidate_points.append(('fixed', kickoff - timedelta(hours=hours)))
    
    # 添加变化点作为候选
    for cp in change_points:
        candidate_points.append(('change', cp))
    
    # 按时间排序
    candidate_points.sort(key=lambda x: x[1])
    
    # 选择16个点: 优先选择变化点，不足则用均匀点
    selected_change_points = [cp for t, cp in candidate_points if t == 'change']
    selected_fixed_points = [cp for t, cp in candidate_points if t == 'fixed']
    
    # 选择变化点(优先) + 固定点(不足时补充)
    final_24h_points = []
    used_times = set()
    
    # 先选变化点
    for cp in selected_change_points:
        if len(final_24h_points) >= 12:  # 最多12个变化点
            break
        # 检查时间是否已存在
        time_key = cp.strftime('%Y%m%d%H%M')
        if time_key not in used_times:
            final_24h_points.append(cp)
            used_times.add(time_key)
    
    # 不足16个则用固定点补充
    while len(final_24h_points) < 16 and selected_fixed_points:
        fp = selected_fixed_points.pop(0)
        time_key = fp.strftime('%Y%m%d%H%M')
        if time_key not in used_times:
            final_24h_points.append(fp)
            used_times.add(time_key)
    
    # 按时间排序
    final_24h_points.sort()
    
    # 添加到时间轴
    for tp in final_24h_points:
        time_points.append(tp)
    
    # 开赛时间
    time_points.append(kickoff)
    
    # 去重并排序
    time_points = sorted(list(set(time_points)))
    
    print(f"  精简后时间节点数: {len(time_points)}")
    print(f"  - 48h~24h: {len([t for t in time_points if cutoff_48h <= t < cutoff_24h])} 个点")
    print(f"  - 24h~开赛: {len([t for t in time_points if cutoff_24h <= t <= kickoff])} 个点")
    
    # 为每个时间点分配赔率（向后填充）
    resampled = {
        'home': [opening_odds['home'], early_avg_odds['home']],  # 开盘, 48小时均值
        'draw': [opening_odds['draw'], early_avg_odds['draw']],
        'away': [opening_odds['away'], early_avg_odds['away']],
    }
    
    # 用于填充的当前赔率
    current_odds = {
        'home': early_avg_odds['home'],
        'draw': early_avg_odds['draw'],
        'away': early_avg_odds['away'],
    }
    
    # 遍历时间点（从48小时前到开赛）
    odds_idx = 0
    for tp in time_points[1:]:  # 跳过第一个（48小时截止点已在resampled中）
        # 找到该时间点之后的第一条记录
        while odds_idx < len(odds_records) and odds_records[odds_idx]['time'] < tp:
            current_odds = {
                'home': odds_records[odds_idx]['home_prob'],
                'draw': odds_records[odds_idx]['draw_prob'],
                'away': odds_records[odds_idx]['away_prob'],
            }
            odds_idx += 1
        
        resampled['home'].append(current_odds['home'])
        resampled['draw'].append(current_odds['draw'])
        resampled['away'].append(current_odds['away'])
    
    print(f"  重采样后数据点数: home={len(resampled['home'])}, draw={len(resampled['draw'])}, away={len(resampled['away'])}")
    print(f"  最终胜率: home={resampled['home'][-1]:.3f}, draw={resampled['draw'][-1]:.3f}, away={resampled['away'][-1]:.3f}")
    
    return resampled


def get_interleaved_series(resampled_odds: Dict[str, List[float]]) -> List[float]:
    """
    将重采样后的赔率转换为交错序列
    
    Args:
        resampled_odds: {'home': [...], 'draw': [...], 'away': [...]}
        
    Returns:
        交错序列: [home0, draw0, away0, home1, draw1, away1, ...]
    """
    min_len = min(len(resampled_odds['home']), len(resampled_odds['draw']), len(resampled_odds['away']))
    
    interleaved = []
    for i in range(min_len):
        interleaved.append(resampled_odds['home'][i])
        interleaved.append(resampled_odds['draw'][i])
        interleaved.append(resampled_odds['away'][i])
    
    return interleaved


def get_delta_series(resampled_odds: Dict[str, List[float]]) -> List[float]:
    """
    计算胜率差值序列 (主胜 - 客胜)
    
    Args:
        resampled_odds: {'home': [...], 'draw': [...], 'away': [...]}
        
    Returns:
        差值序列: [home0-away0, home1-away1, ...]
    """
    min_len = min(len(resampled_odds['home']), len(resampled_odds['away']))
    
    delta = []
    for i in range(min_len):
        delta.append(resampled_odds['home'][i] - resampled_odds['away'][i])
    
    return delta


def z_score_normalize(series: List[float]) -> List[float]:
    """
    Z-Score 标准化
    
    公式: z = (x - μ) / σ
    
    作用: 消除赔率绝对值的影响，只保留波动的形状
    - 主胜赔率从 1.2 变到 1.3，与从 5.0 变到 5.1，在绝对值上一样
    - 但在博弈意义上，1.2→1.3 的变化比 5.0→5.1 重要得多
    - Z-Score 标准化后，这些变化会被正确反映
    
    Args:
        series: 原始数值序列
        
    Returns:
        标准化后的序列，均值为0，标准差为1
    """
    import numpy as np
    
    arr = np.array(series, dtype=float)
    mean = arr.mean()
    std = arr.std()
    
    if std == 0:
        # 防止除零，返回零中心化的序列
        return (arr - mean).tolist()
    
    return ((arr - mean) / std).tolist()


def normalize_odds_series(resampled_odds: Dict[str, List[float]]) -> Dict[str, List[float]]:
    """
    对赔率序列进行 Z-Score 标准化
    
    Args:
        resampled_odds: {'home': [...], 'draw': [...], 'away': [...]}
        
    Returns:
        标准化后的序列
    """
    return {
        'home': z_score_normalize(resampled_odds['home']),
        'draw': z_score_normalize(resampled_odds['draw']),
        'away': z_score_normalize(resampled_odds['away']),
    }


# ============ 亚盘 SAX 编码器 ============


def parse_handicap_odds_detail(odds_detail_json) -> tuple:
    """
    解析 odds_detail JSONB 数据，提取亚盘赔率序列

    Args:
        odds_detail_json: JSONB 格式的 odds_detail 数据

    Returns:
        (home_odds_list, away_odds_list, handicap_list): 三个列表
    """
    if not odds_detail_json:
        return [], [], []

    # 解析 JSONB 数据
    if isinstance(odds_detail_json, str):
        try:
            detail_list = json.loads(odds_detail_json)
        except json.JSONDecodeError:
            return [], [], []
    elif isinstance(odds_detail_json, list):
        detail_list = odds_detail_json
    else:
        return [], [], []

    if not detail_list:
        return [], [], []

    home_odds = []
    away_odds = []
    handicap_list = []

    for item in detail_list:
        if not isinstance(item, dict):
            continue
        home = item.get("home")
        away = item.get("away")
        handicap = item.get("handicap")

        if home is not None and away is not None:
            home_odds.append(float(home))
            away_odds.append(float(away))
            handicap_list.append(handicap)

    return home_odds, away_odds, handicap_list


def encode_handicap_sax(
    encoder: SAXEncoder,
    odds_detail_json,
    interpolate_len: int = 32,
) -> Optional[dict]:
    """
    对亚盘数据进行 SAX 编码

    Args:
        encoder: SAXEncoder 实例
        odds_detail_json: JSONB 格式的 odds_detail 数据
        interpolate_len: 插值长度

    Returns:
        dict: 包含 sax_home, sax_away, sax_diff, handicap_changes
        None: 如果数据不足
    """
    home_odds, away_odds, handicap_list = parse_handicap_odds_detail(odds_detail_json)

    if len(home_odds) < 2 or len(away_odds) < 2:
        return None

    # 编码主队赔率
    sax_home = encoder.encode(home_odds, interpolate_len)

    # 编码客队赔率
    sax_away = encoder.encode(away_odds, interpolate_len)

    # 编码差值 (主队 - 客队)
    min_len = min(len(home_odds), len(away_odds))
    diff_odds = [home_odds[i] - away_odds[i] for i in range(min_len)]
    sax_diff = encoder.encode(diff_odds, interpolate_len)

    # 计算盘口变化次数
    handicap_changes = 0
    if handicap_list:
        prev = handicap_list[0]
        for h in handicap_list[1:]:
            if h != prev:
                handicap_changes += 1
                prev = h

    return {
        "sax_home": sax_home,
        "sax_away": sax_away,
        "sax_diff": sax_diff,
        "handicap_changes": handicap_changes,
        "home_count": len(home_odds),
        "away_count": len(away_odds),
    }


def calculate_handicap_distance(
    target_sax: dict,
    item_sax: dict,
    word_size: int,
    alphabet_size: int,
) -> float:
    """
    计算亚盘 SAX 距离

    Args:
        target_sax: 目标比赛的 SAX 编码
        item_sax: 数据库比赛的 SAX 编码
        word_size: SAX 分段数
        alphabet_size: 字母表大小

    Returns:
        float: 综合距离
    """
    if not item_sax:
        return float("inf")

    # 主队赔率距离
    dist_home = mindist_sax(
        target_sax.get("sax_home", ""),
        item_sax.get("sax_home", ""),
        word_size,
        alphabet_size,
    )

    # 客队赔率距离
    dist_away = mindist_sax(
        target_sax.get("sax_away", ""),
        item_sax.get("sax_away", ""),
        word_size,
        alphabet_size,
    )

    # 差值距离
    dist_diff = mindist_sax(
        target_sax.get("sax_diff", ""),
        item_sax.get("sax_diff", ""),
        word_size,
        alphabet_size,
    )

    # 综合距离 (平均)
    combined = (dist_home + dist_away + dist_diff) / 3

    return combined


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


def fetch_all_sax_data(client, bookmaker: str = "Bet 365") -> list[dict]:
    """从 Supabase 获取指定庄家的 SAX 数据 (含亚盘)"""
    print(f"从 Supabase 获取 SAX 数据 (庄家: {bookmaker})...")

    try:
        # 从视图获取指定庄家的数据 (含亚盘)
        result = (
            client.table("v_match_odds_sax_handicap")
            .select("*")
            .eq("bookmaker_name", bookmaker)
            .execute()
        )

        if not result.data:
            print(f"  警告: Supabase 中没有找到庄家 '{bookmaker}' 的数据")
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
    use_initial_odds: bool = False,
    use_final_odds: bool = False,
    use_both_odds: bool = False,
    bookmaker: str = "Bet 365",
    use_dist: float = 0.5,
    use_handicap: bool = False,
    target_handicap_sax: Optional[dict] = None,
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
        use_initial_odds: 是否使用初盘赔率筛选
        use_final_odds: 是否使用终盘赔率筛选
        use_both_odds: 是否同时使用初盘和终盘赔率筛选
        bookmaker: 庄家名称（默认 Bet 365）

    Returns:
        最相似的比赛列表
    """
    # 确定赔率筛选类型
    if use_both_odds:
        odds_type = "初盘+终盘"
    elif use_final_odds:
        odds_type = "终盘"
    elif use_initial_odds:
        odds_type = "初盘"
    else:
        odds_type = "均赔"

    print(
        f"\n查找 SAX 编码最相似的比赛 (庄家: {bookmaker}, 赔率筛选: {odds_type}, ±{odds_tolerance_pct}%, 距离阈值: ≤{use_dist})..."
    )
    print(f"  目标交错编码: {target_sax_interleaved}")
    print(f"  目标差值编码: {target_sax_delta}")

    if use_both_odds:
        print(
            f"  目标初盘赔率: 胜={target_odds.get('init_win'):.2f}, 平={target_odds.get('init_draw'):.2f}, 负={target_odds.get('init_lose'):.2f}"
        )
        print(
            f"  目标终盘赔率: 胜={target_odds.get('final_win'):.2f}, 平={target_odds.get('final_draw'):.2f}, 负={target_odds.get('final_lose'):.2f}"
        )
    elif use_final_odds:
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

    # 获取指定庄家的数据
    all_data = fetch_all_sax_data(client, bookmaker)

    if not all_data:
        return []

    # 计算距离（仅限赔率相近的比赛）
    similarities = []
    filtered_count = 0

    for item in all_data:
        sax_interleaved = item.get("sax_interleaved", "")
        sax_delta = item.get("sax_delta", "")

        if not sax_interleaved:
            continue

        if use_both_odds:
            # 同时使用初盘和终盘赔率筛选
            item_init_home = item.get("init_win", 0)
            item_init_draw = item.get("init_draw", 0)
            item_init_away = item.get("init_lose", 0)
            item_final_home = item.get("final_win", 0)
            item_final_draw = item.get("final_draw", 0)
            item_final_away = item.get("final_lose", 0)

            # 目标初盘赔率
            target_init_home = target_odds.get("init_win", 0)
            target_init_draw = target_odds.get("init_draw", 0)
            target_init_away = target_odds.get("init_lose", 0)
            # 目标终盘赔率
            target_final_home = target_odds.get("final_win", 0)
            target_final_draw = target_odds.get("final_draw", 0)
            target_final_away = target_odds.get("final_lose", 0)

            # 跳过无效赔率
            if not item_init_home or not item_init_draw or not item_init_away:
                continue
            if not item_final_home or not item_final_draw or not item_final_away:
                continue
            if not target_init_home or not target_init_draw or not target_init_away:
                continue
            if not target_final_home or not target_final_draw or not target_final_away:
                continue

            # 初盘赔率筛选：三个赔率都在 ±5% 范围内
            init_home_threshold = target_init_home * (odds_tolerance_pct / 100)
            init_draw_threshold = target_init_draw * (odds_tolerance_pct / 100)
            init_away_threshold = target_init_away * (odds_tolerance_pct / 100)

            init_pass = (
                abs(item_init_home - target_init_home) <= init_home_threshold
                and abs(item_init_draw - target_init_draw) <= init_draw_threshold
                and abs(item_init_away - target_init_away) <= init_away_threshold
            )

            # 终盘赔率筛选：三个赔率都在 ±5% 范围内
            final_home_threshold = target_final_home * (odds_tolerance_pct / 100)
            final_draw_threshold = target_final_draw * (odds_tolerance_pct / 100)
            final_away_threshold = target_final_away * (odds_tolerance_pct / 100)

            final_pass = (
                abs(item_final_home - target_final_home) <= final_home_threshold
                and abs(item_final_draw - target_final_draw) <= final_draw_threshold
                and abs(item_final_away - target_final_away) <= final_away_threshold
            )

            if not (init_pass and final_pass):
                continue

            # 使用终盘赔率作为显示
            item_home = item_final_home
            item_draw = item_final_draw
            item_away = item_final_away

        elif use_final_odds:
            # 使用终盘赔率
            target_home = target_odds.get("final_win", 0)
            target_draw = target_odds.get("final_draw", 0)
            target_away = target_odds.get("final_lose", 0)

            item_home = item.get("final_win", 0)
            item_draw = item.get("final_draw", 0)
            item_away = item.get("final_lose", 0)

            # 跳过无效赔率
            if not item_home or not item_draw or not item_away:
                continue

            # 赔率筛选
            home_threshold = target_home * (odds_tolerance_pct / 100)
            draw_threshold = target_draw * (odds_tolerance_pct / 100)
            away_threshold = target_away * (odds_tolerance_pct / 100)

            if (
                abs(item_home - target_home) > home_threshold
                or abs(item_draw - target_draw) > draw_threshold
                or abs(item_away - target_away) > away_threshold
            ):
                continue

        elif use_initial_odds:
            # 使用初盘赔率
            target_home = target_odds.get("init_win", 0)
            target_draw = target_odds.get("init_draw", 0)
            target_away = target_odds.get("init_lose", 0)

            item_home = item.get("init_win", 0)
            item_draw = item.get("init_draw", 0)
            item_away = item.get("init_lose", 0)

            # 跳过无效赔率
            if not item_home or not item_draw or not item_away:
                continue

            # 赔率筛选
            home_threshold = target_home * (odds_tolerance_pct / 100)
            draw_threshold = target_draw * (odds_tolerance_pct / 100)
            away_threshold = target_away * (odds_tolerance_pct / 100)

            if (
                abs(item_home - target_home) > home_threshold
                or abs(item_draw - target_draw) > draw_threshold
                or abs(item_away - target_away) > away_threshold
            ):
                continue

        else:
            # 使用均赔
            target_home = target_odds.get("home_mean", 0)
            target_draw = target_odds.get("draw_mean", 0)
            target_away = target_odds.get("away_mean", 0)

            item_home = item.get("home_mean", 0)
            item_draw = item.get("draw_mean", 0)
            item_away = item.get("away_mean", 0)

            # 跳过无效赔率
            if not item_home or not item_draw or not item_away:
                continue

            # 赔率筛选
            home_threshold = target_home * (odds_tolerance_pct / 100)
            draw_threshold = target_draw * (odds_tolerance_pct / 100)
            away_threshold = target_away * (odds_tolerance_pct / 100)

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

        # 计算亚盘 SAX 距离
        dist_handicap = 0.0
        item_handicap_sax = None
        if use_handicap and target_handicap_sax:
            # 从数据库获取 odds_detail 并编码
            odds_detail = item.get("odds_detail")
            if odds_detail:
                # 创建临时 encoder 用于亚盘
                handicap_encoder = SAXEncoder(word_size=word_size, alphabet_size=alphabet_size)
                item_handicap_sax = encode_handicap_sax(handicap_encoder, odds_detail)
            
            if item_handicap_sax:
                dist_handicap = calculate_handicap_distance(
                    target_handicap_sax, item_handicap_sax, word_size, alphabet_size
                )
            else:
                dist_handicap = float("inf")

        # 综合距离（加权平均）
        # 处理 dist_delta 为 infinity 的情况
        if dist_delta == float("inf"):
            eu_dist = dist_interleaved
        else:
            eu_dist = (dist_interleaved + dist_delta) / 2 if sax_delta else dist_interleaved
        
        if use_handicap and target_handicap_sax and dist_handicap != float("inf"):
            # 欧赔距离占 60%，亚盘距离占 40%
            combined_dist = eu_dist * 0.6 + dist_handicap * 0.4
        else:
            combined_dist = eu_dist

        similarities.append(
            {
                "match_id": item.get("match_id"),
                "sax_interleaved": sax_interleaved,
                "sax_delta": sax_delta,
                "dist_interleaved": dist_interleaved,
                "dist_delta": dist_delta,
                "dist_handicap": dist_handicap if use_handicap else None,
                "combined_dist": combined_dist,
                "item_home": item_home,
                "item_draw": item_draw,
                "item_away": item_away,
                "final_score": item.get("final_score", ""),
                # 亚盘数据
                "init_handicap": item.get("init_handicap", ""),
                "init_odds_home": item.get("init_odds_home"),
                "init_odds_away": item.get("init_odds_away"),
                "final_handicap": item.get("final_handicap", ""),
                "final_odds_home": item.get("final_odds_home"),
                "final_odds_away": item.get("final_odds_away"),
            }
        )

    print(f"  赔率筛选后剩余: {filtered_count}/{len(all_data)} 场比赛")

    similarities.sort(key=lambda x: x["combined_dist"])
    similarities = [s for s in similarities if s["combined_dist"] <= use_dist]

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
    import argparse

    parser = argparse.ArgumentParser(
        description="根据比赛 ID 查找 SAX 编码最相似的比赛",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法（Bet 365，均赔筛选）
  python find_similar_matches.py 2799893

  # 使用 Easybets 庄家
  python find_similar_matches.py 2799893 --bookmaker Easybets

  # 使用终盘赔率筛选
  python find_similar_matches.py 2799893 --use-final

  # 使用初盘赔率筛选
  python find_similar_matches.py 2799893 --use-initial

  # 同时使用初盘和终盘赔率筛选
  python find_similar_matches.py 2799893 --use-both

  # 自定义赔率容忍度
  python find_similar_matches.py 2799893 --use-final --tolerance 10
        """,
    )

    parser.add_argument("match_id", type=str, help="比赛 ID")
    parser.add_argument(
        "--bookmaker",
        type=str,
        default="Bet 365",
        help="指定庄家 (bet_365, easybets，默认: Bet 365)",
    )
    parser.add_argument(
        "--use-final",
        action="store_true",
        help="使用终盘赔率进行筛选",
    )
    parser.add_argument(
        "--use-initial",
        action="store_true",
        help="使用初盘赔率进行筛选",
    )
    parser.add_argument(
        "--use-both",
        action="store_true",
        help="同时使用初盘和终盘赔率进行筛选",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=5.0,
        help="赔率容忍百分比（默认 5.0）",
    )
    parser.add_argument(
        "--use-dist",
        type=float,
        default=0.5,
        help="combined_dist 阈值，只保留距离小于等于该值的比赛（默认 0.5）",
    )
    parser.add_argument(
        "--use-handicap",
        action="store_true",
        help="使用亚盘数据进行筛选",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="不自动打开分析页面",
    )

    args = parser.parse_args()

    # 标准化庄家名称
    bookmaker = args.bookmaker
    if bookmaker.lower() in ["bet365", "bet_365", "bet-365"]:
        bookmaker = "Bet 365"
    elif bookmaker.lower() in ["easybets", "easybet"]:
        bookmaker = "Easybets"

    match_id = args.match_id
    use_final_odds = args.use_final
    use_initial_odds = args.use_initial
    use_both_odds = args.use_both
    use_handicap = args.use_handicap
    no_open = args.no_open
    if not use_initial_odds and not use_final_odds and not use_both_odds:
        use_final_odds = True
    odds_tolerance_pct = args.tolerance
    use_dist = args.use_dist

    print("=" * 60)
    print("根据比赛 ID 查找 SAX 编码最相似的比赛")
    print("=" * 60)

    # 加载 SAX 编码参数（根据庄家加载对应配置文件）
    config_dir = "SAX_encoder/1.generateOddsDetail/SAX encoder/bookmaker_details"
    word_size = 8
    alphabet_size = 4
    interpolate_len = 32

    # 根据庄家名称确定配置文件路径
    if bookmaker == "Bet 365":
        config_path = os.path.join(config_dir, "sax_config_bet_365.json")
    elif bookmaker == "Easybets":
        config_path = os.path.join(config_dir, "sax_config_easybets.json")

    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        word_size = config.get("word_size", word_size)
        alphabet_size = config.get("alphabet_size", alphabet_size)
        interpolate_len = config.get("interpolate_len", interpolate_len)
        print(
            f"  从配置文件加载 SAX 参数 (庄家: {bookmaker}): word_size={word_size}, alphabet_size={alphabet_size}, interpolate_len={interpolate_len}"
        )

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
    sax_config_path = (
        config_path if config_path and os.path.exists(config_path) else None
    )
    encoder = SAXEncoder(
        word_size=word_size,
        alphabet_size=alphabet_size,
        config_path=sax_config_path,
    )

    # 使用新的重采样函数，确保固定20点时间轴
    resampled_odds = resample_odds_to_fixed_timeline(
        match_odds.running_odds, 
        match_odds.match_time
    )
    
    if not resampled_odds or len(resampled_odds['home']) < 2:
        print("错误: 赔率重采样失败，无法进行 SAX 编码")
        sys.exit(1)
    
    # Z-Score 标准化 (消除赔率绝对值影响，保留波动形状)
    normalized_odds = normalize_odds_series(resampled_odds)
    
    home = normalized_odds['home']
    draw = normalized_odds['draw']
    away = normalized_odds['away']

    print(f"  重采样后数据点数: {len(home)}")
    print(f"  Z-Score 标准化后: 均值≈0, 标准差≈1")

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
    sax_interleaved = encoder.encode(joint_series, interpolate_len * 3)
    sax_delta = encoder.encode(delta_series, interpolate_len)

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
    print(f"  编码参数: word_size={word_size}, alphabet_size={alphabet_size}")

    # 尝试获取目标比赛的亚盘数据
    target_handicap_sax = None
    if use_handicap:
        try:
            client = get_supabase_client()
            # 从数据库获取目标比赛的亚盘数据
            result = (
                client.table("v_match_odds_sax_handicap")
                .select("odds_detail")
                .eq("match_id", int(match_id))
                .eq("bookmaker_name", bookmaker)
                .execute()
            )
            has_handicap_data = False
            if result.data and len(result.data) > 0:
                first_item = result.data[0]
                if isinstance(first_item, dict) and first_item.get("odds_detail"):
                    handicap_encoder = SAXEncoder(word_size=word_size, alphabet_size=alphabet_size)
                    target_handicap_sax = encode_handicap_sax(
                        handicap_encoder, 
                        first_item["odds_detail"],
                        interpolate_len * 3
                    )
                    if target_handicap_sax:
                        print(f"  亚盘编码: {target_handicap_sax['sax_home']}, {target_handicap_sax['sax_away']}, {target_handicap_sax['sax_diff']}")
                        print(f"  亚盘变化次数: {target_handicap_sax['handicap_changes']}")
                        has_handicap_data = True
                    else:
                        print("  警告: 目标比赛亚盘数据不足")
                else:
                    print("  警告: 目标比赛无亚盘数据")
            else:
                print("  警告: 目标比赛无亚盘数据")
            
            # 如果没有亚盘数据，下载并本地解析
            if not has_handicap_data:
                print(f"\n  正在下载亚盘数据...")
                import subprocess
                # 项目根目录
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                download_script = os.path.join(project_root, "download_module", "download_total_handicap.py")
                output_dir = os.path.join(project_root, "rawdata", "odds", "total_handicap")
                try:
                    # 下载亚盘数据
                    result = subprocess.run(
                        [sys.executable, download_script, "--match-id", str(match_id), "--output-dir", output_dir],
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    if result.returncode == 0:
                        print("  亚盘数据下载成功，正在解析...")
                        # 本地解析（不上传到数据库）
                        module_path = os.path.join(project_root, "download_module")
                        sys.path.insert(0, module_path)
                        
                        try:
                            from parse_total_handicap import parse_html_file
                            
                            html_dir = os.path.join(project_root, "rawdata", "odds", "total_handicap")
                            html_file = os.path.join(html_dir, f"{match_id}_total.html")
                            
                            if os.path.exists(html_file):
                                records = parse_html_file(html_file)
                                # 找到对应庄家的数据
                                bookmaker_key = "bet365" if bookmaker == "Bet 365" else "easybets" if bookmaker == "Easybets" else bookmaker.lower()
                                for record in records:
                                    if record.get("bookmaker") == bookmaker_key and record.get("odds_detail"):
                                        handicap_encoder = SAXEncoder(word_size=word_size, alphabet_size=alphabet_size)
                                        target_handicap_sax = encode_handicap_sax(
                                            handicap_encoder, 
                                            record["odds_detail"],
                                            interpolate_len * 3
                                        )
                                        if target_handicap_sax:
                                            print(f"  亚盘编码: {target_handicap_sax['sax_home']}, {target_handicap_sax['sax_away']}, {target_handicap_sax['sax_diff']}")
                                            print(f"  亚盘变化次数: {target_handicap_sax['handicap_changes']}")
                                            has_handicap_data = True
                                            break
                                if not has_handicap_data:
                                    print("  警告: 解析后未找到对应庄家数据")
                            else:
                                print(f"  警告: HTML 文件不存在: {html_file}")
                                
                        except ImportError as e:
                            print(f"  警告: 导入解析模块失败: {e}")
                            use_handicap = False
                            has_handicap_data = False
                    else:
                        print(f"  警告: 下载失败 - {result.stderr}")
                except subprocess.TimeoutExpired:
                    print("  警告: 下载超时")
                except Exception as e:
                    print(f"  警告: 下载/解析过程出错: {e}")
                    
        except Exception as e:
            print(f"  警告: 获取亚盘数据失败: {e}")
            use_handicap = False

    # 4. 从 Supabase 查找相似比赛
    print(f"\n[4/5] 从 Supabase 查找相似比赛")

    try:
        client = get_supabase_client()

        if use_both_odds:
            similar_matches_final = find_similar_matches(
                client,
                sax_interleaved,
                sax_delta,
                target_odds=target_odds,
                word_size=word_size,
                alphabet_size=alphabet_size,
                top_n=10,
                odds_tolerance_pct=odds_tolerance_pct,
                use_initial_odds=False,
                use_final_odds=True,
                use_both_odds=False,
                bookmaker=bookmaker,
                use_dist=use_dist,
                use_handicap=use_handicap,
                target_handicap_sax=target_handicap_sax,
            )

            similar_matches_initial = find_similar_matches(
                client,
                sax_interleaved,
                sax_delta,
                target_odds=target_odds,
                word_size=word_size,
                alphabet_size=alphabet_size,
                top_n=10,
                odds_tolerance_pct=odds_tolerance_pct,
                use_initial_odds=True,
                use_final_odds=False,
                use_both_odds=False,
                bookmaker=bookmaker,
                use_dist=use_dist,
                use_handicap=use_handicap,
                target_handicap_sax=target_handicap_sax,
            )

            combined = similar_matches_final + similar_matches_initial
            combined.sort(key=lambda x: x["combined_dist"])
            similar_matches = combined[:20]

        else:
            similar_matches = find_similar_matches(
                client,
                sax_interleaved,
                sax_delta,
                target_odds=target_odds,
                word_size=word_size,
                alphabet_size=alphabet_size,
                top_n=10,
                odds_tolerance_pct=odds_tolerance_pct,
                use_initial_odds=use_initial_odds,
                use_final_odds=use_final_odds,
                use_both_odds=False,
                bookmaker=bookmaker,
                use_dist=use_dist,
                use_handicap=use_handicap,
                target_handicap_sax=target_handicap_sax,
            )

        if not similar_matches:
            print("  未找到相似的比赛")
        else:
            # 根据筛选类型显示对应的赔率列名
            if use_both_odds:
                odds_label = "初盘+终盘赔率"
            elif use_final_odds:
                odds_label = "终盘赔率(胜/平/负)"
            elif use_initial_odds:
                odds_label = "初盘赔率(胜/平/负)"
            else:
                odds_label = "均赔(胜/平/负)"

            print(f"\n找到 {len(similar_matches)} 场最相似的比赛:")
            print("-" * 100)
            # 根据是否使用亚盘显示不同表头
            if use_handicap and target_handicap_sax:
                print(
                    f"{'排名':<4} {'比赛ID':<12} {'交错编码':<16} {'欧赔距离':<10} {'亚盘距离':<10} {'综合距离':<10} {'结果':<8} {odds_label:<20}"
                )
            else:
                print(
                    f"{'排名':<4} {'比赛ID':<12} {'交错编码':<16} {'距离':<8} {'结果':<8} {odds_label:<20} {'初盘亚盘':<15} {'终盘亚盘':<15}"
                )
            print("-" * 100)

            for i, match in enumerate(similar_matches, 1):
                final_score = match.get('final_score', '')
                # 亚盘数据
                init_handicap = match.get('init_handicap', '') or ''
                final_handicap = match.get('final_handicap', '') or ''
                init_ah = f"{init_handicap} {match.get('init_odds_home', '')}/{match.get('init_odds_away', '')}" if init_handicap else '-'
                final_ah = f"{final_handicap} {match.get('final_odds_home', '')}/{match.get('final_odds_away', '')}" if final_handicap else '-'
                
                if use_handicap and target_handicap_sax:
                    # 显示欧赔距离、亚盘距离和综合距离
                    dist_inter = match.get('dist_interleaved', 0) or 0
                    dist_delta = match.get('dist_delta', 0)
                    # 处理 infinity 的情况
                    if dist_delta == float("inf"):
                        eu_dist = dist_inter
                    else:
                        eu_dist = (dist_inter + dist_delta) / 2
                    ah_dist = match.get('dist_handicap', 0) or 0
                    if ah_dist == float("inf"):
                        ah_dist = 0
                    print(
                        f"{i:<4} {match['match_id']:<12} {match['sax_interleaved']:<16} "
                        f"{eu_dist:.4f}     {ah_dist:.4f}     {match['combined_dist']:.4f} "
                        f"{final_score:<8} "
                        f"{match['item_home']:.2f}/{match['item_draw']:.2f}/{match['item_away']:.2f}"
                    )
                else:
                    print(
                        f"{i:<4} {match['match_id']:<12} {match['sax_interleaved']:<16} "
                        f"{match['combined_dist']:.4f} "
                        f"{final_score:<8} "
                        f"{match['item_home']:.2f}/{match['item_draw']:.2f}/{match['item_away']:.2f}"
                        f" {init_ah:<15} {final_ah:<15}"
                    )

            print("-" * 100)

            # 统计结果
            win_count = 0
            draw_count = 0
            loss_count = 0
            for match in similar_matches:
                final_score = match.get('final_score', '')
                if final_score and '-' in final_score:
                    try:
                        parts = final_score.split('-')
                        home_goals = int(parts[0])
                        away_goals = int(parts[1])
                        if home_goals > away_goals:
                            win_count += 1
                        elif home_goals == away_goals:
                            draw_count += 1
                        else:
                            loss_count += 1
                    except (ValueError, IndexError):
                        pass

            print(f"\n结果统计: 胜 {win_count} 场, 平 {draw_count} 场, 负 {loss_count} 场")
            if win_count + draw_count + loss_count > 0:
                total = win_count + draw_count + loss_count
                print(f"         胜率: {win_count/total*100:.1f}%, 平率: {draw_count/total*100:.1f}%, 负率: {loss_count/total*100:.1f}%")

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
    if no_open:
        print(f"\n[5/5] 跳过打开分析页面 (--no-open)")
    else:
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
