#!/usr/bin/env python3
"""
SAX 编码器 - Symbolic Aggregate Approximation

基于分析得出的参数:
  - word_size = 8 (分段数)
  - alphabet_size = 7 (字母表 a-g)
"""

import numpy as np
from scipy.stats import norm
import os
import json


class SAXEncoder:
    """
    SAX 编码器

    1. Z-score 归一化
    2. PAA (Piecewise Aggregate Approximation) 降维
    3. 映射到字母符号
    """

    def __init__(self, word_size=8, alphabet_size=7, config_path=None):
        """
        初始化 SAX 编码器

        Args:
            word_size: 分段数 (PAA 段数)
            alphabet_size: 字母表大小 (a-z 的前 n 个字母)
            config_path: 配置文件路径 (可选)
        """
        self.word_size = word_size
        self.alphabet_size = alphabet_size
        self.breakpoints = None
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        else:
            self.breakpoints = self._compute_breakpoints_gaussian()
        
        self.alphabet = self._generate_alphabet()

    def load_config(self, config_path):
        """从配置文件加载参数"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.word_size = config.get("word_size", self.word_size)
        self.alphabet_size = config.get("alphabet_size", self.alphabet_size)
        
        if config.get("breakpoints_type") == "empirical" and "empirical_breakpoints" in config:
            self.breakpoints = np.array(config["empirical_breakpoints"])
        else:
            self.breakpoints = self._compute_breakpoints_gaussian()
            
        print(f"已从 {config_path} 加载 SAX 配置: word_size={self.word_size}, alphabet_size={self.alphabet_size}")

    def _compute_breakpoints_gaussian(self):
        """计算高斯断点，将标准正态分布分成 alphabet_size 个等概率区间"""
        breakpoints = [norm.ppf(i / self.alphabet_size) for i in range(1, self.alphabet_size)]
        return np.array(breakpoints)

    def _generate_alphabet(self):
        """生成字母表"""
        return 'abcdefghijklmnopqrstuvwxyz'[:self.alphabet_size]

    def normalize(self, series):
        """
        Z-score 归一化

        Args:
            series: 数值序列 (list 或 np.array)

        Returns:
            归一化后的序列
        """
        series = np.array(series, dtype=float)
        mean = np.mean(series)
        std = np.std(series)
        if std == 0:
            return series - mean
        return (series - mean) / std

    def _interpolate_to_length(self, series, target_length):
        """
        将序列插值到目标长度

        Args:
            series: 输入序列
            target_length: 目标长度

        Returns:
            插值后的序列
        """
        if len(series) == target_length:
            return np.array(series)

        if len(series) < target_length:
            # 上采样：使用线性插值
            x = np.linspace(0, 1, len(series))
            x_new = np.linspace(0, 1, target_length)
            return np.interp(x_new, x, series)
        else:
            # 下采样：使用 PAA
            return self._paa(series, target_length)

    def _paa(self, series, target_length):
        """
        PAA (Piecewise Aggregate Approximation) 降维

        将序列分成 target_length 段，取每段的平均值

        Args:
            series: 输入序列
            target_length: 目标长度（段数）

        Returns:
            PAA 降维后的序列
        """
        series = np.array(series, dtype=float)
        n = len(series)

        if n == target_length:
            return series

        # 每段的大小
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
        """将归一化后的值映射到字母符号"""
        for i, breakpoint in enumerate(self.breakpoints):
            if value < breakpoint:
                return self.alphabet[i]
        return self.alphabet[-1]  # 最大的值

    def encode(self, series, interpolate_len=32):
        """
        SAX 编码

        Args:
            series: 输入数值序列
            interpolate_len: 插值后的统一长度（用于对齐不同长度的序列）

        Returns:
            SAX 编码字符串
        """
        if len(series) < 2:
            return self.alphabet[0] * self.word_size

        # 1. 插值到统一长度（便于比较）
        interpolated = self._interpolate_to_length(series, interpolate_len)

        # 2. Z-score 归一化
        normalized = self.normalize(interpolated)

        # 3. PAA 降维
        paa = self._paa(normalized, self.word_size)

        # 4. 映射到字母
        sax_string = ''.join([self._map_to_symbol(v) for v in paa])

        return sax_string

    def encode_with_paa(self, series):
        """
        直接对序列进行 SAX 编码（不插值到固定长度）

        适用于已经对齐的序列

        Args:
            series: 输入数值序列

        Returns:
            SAX 编码字符串
        """
        if len(series) < self.word_size:
            # 序列太短，先上采样
            series = self._interpolate_to_length(series, self.word_size * 2)

        # 归一化
        normalized = self.normalize(series)

        # PAA
        paa = self._paa(normalized, self.word_size)

        # 映射到字母
        return ''.join([self._map_to_symbol(v) for v in paa])


def extract_odds_series(match):
    """
    从比赛数据中提取赔率序列

    Args:
        match: 比赛数据 dict

    Returns:
        (home_series, draw_series, away_series): 三个赔率序列
    """
    running = match.get('runningOdds', [])
    if not running:
        return None, None, None

    home = [float(o['home']) for o in running]
    draw = [float(o['draw']) for o in running]
    away = [float(o['away']) for o in running]

    return home, draw, away


def create_joint_series(home, draw, away):
    """
    创建联合序列（交错拼接）

    Args:
        home, draw, away: 三个赔率序列

    Returns:
        交错拼接后的序列: [h1, d1, a1, h2, d2, a2, ...]
    """
    min_len = min(len(home), len(draw), len(away))
    joint = []
    for i in range(min_len):
        joint.append(home[i])
        joint.append(draw[i])
        joint.append(away[i])
    return joint


def create_delta_series(home, away):
    """
    创建差值序列（主客队赔率差）

    Args:
        home: 主胜赔率序列
        away: 客胜赔率序列

    Returns:
        差值序列
    """
    min_len = min(len(home), len(away))
    return [home[i] - away[i] for i in range(min_len)]


if __name__ == '__main__':
    # 快速测试
    encoder = SAXEncoder(word_size=8, alphabet_size=7)

    # 测试数据
    test_series = [2.1, 2.2, 2.3, 2.5, 2.4, 2.3, 2.2, 2.1, 2.0, 1.9,
                   1.8, 1.7, 1.8, 1.9, 2.0, 2.1]

    sax = encoder.encode(test_series)
    print(f"测试序列: {test_series}")
    print(f"SAX 编码: {sax}")
    print(f"参数: word_size={encoder.word_size}, alphabet_size={encoder.alphabet_size}")
    print(f"字母表: {encoder.alphabet}")
    print(f"断点: {encoder.breakpoints}")
