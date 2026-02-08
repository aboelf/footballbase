import json
import numpy as np
from collections import Counter
from scipy import stats
import math
import os


def normalize_series(series):
    """Z-score 归一化"""
    series = np.array(series, dtype=float)
    if len(series) < 2:
        return np.zeros_like(series)
    std = np.std(series)
    if std == 0:
        return series - np.mean(series)
    return (series - np.mean(series)) / std


def calculate_entropy(labels):
    """计算序列的熵"""
    if len(labels) == 0:
        return 0
    counts = Counter(labels)
    total = sum(counts.values())
    probs = [count / total for count in counts.values()]
    return -sum(p * math.log2(p) for p in probs if p > 0)


def analyze_odds_data(json_file, output_config="sax_config.json"):
    """分析赔率数据分布并生成配置建议"""

    print("=" * 60)
    print("Bet 365 赔率数据分布深度分析报告")
    print("=" * 60)

    if not os.path.exists(json_file):
        print(f"错误: 找不到数据文件 {json_file}")
        return None

    with open(json_file, 'r', encoding='utf-8') as f:
        matches = json.load(f)

    print(f"\n[1] 数据集概况")
    print(f"  - 比赛总数: {len(matches)}")

    # 收集数据
    home_series_all = []
    draw_series_all = []
    away_series_all = []
    normalized_all = []
    lengths = []

    for match in matches:
        running = match.get('runningOdds', [])
        if len(running) >= 2:
            lengths.append(len(running))
            h = [float(o['home']) for o in running]
            d = [float(o['draw']) for o in running]
            a = [float(o['away']) for o in running]
            
            home_series_all.extend(h)
            draw_series_all.extend(d)
            away_series_all.extend(a)
            
            normalized_all.extend(normalize_series(h))
            normalized_all.extend(normalize_series(d))
            normalized_all.extend(normalize_series(a))

    if not lengths:
        print("错误: 未找到有效的赔率序列数据")
        return None

    # 1. 序列长度分布
    print(f"\n[2] 序列长度特征")
    p25, p50, p75, p90 = np.percentile(lengths, [25, 50, 75, 90])
    print(f"  - 平均长度: {np.mean(lengths):.1f}")
    print(f"  - 长度中位数: {p50:.0f}")
    print(f"  - 75% 分位数: {p75:.0f}")
    print(f"  - 90% 分位数: {p90:.0f}")
    print(f"  - 最大长度: {max(lengths)}")

    # 2. 正态性分析
    print(f"\n[3] 数据分布正态性检查")
    skew = stats.skew(normalized_all)
    kurt = stats.kurtosis(normalized_all)
    _, p_val = stats.normaltest(normalized_all)
    
    print(f"  - 偏度 (Skewness): {skew:.4f} (越接近0越对称)")
    print(f"  - 峰度 (Kurtosis): {kurt:.4f} (越接近0越符合正态分布峰值)")
    print(f"  - K^2 检验 p-value: {p_val:.4e}")
    
    is_gaussian = p_val > 0.05
    if not is_gaussian:
        print("  - 结论: 数据呈现典型的'厚尾'分布 (尖峰厚尾)，建议使用经验分位数断点。")
    else:
        print("  - 结论: 数据基本符合正态分布。")

    # 3. 熵与字母表大小
    print(f"\n[4] 字母表大小 (alphabet_size) 优化分析")
    print(f"  {'Size':<6} | {'Entropy':<10} | {'Efficiency':<10}")
    print("-" * 35)
    
    # 我们希望字母表大小能区分不同波动，但又不至于太稀疏
    # 使用理论上的最大熵作为参考
    best_alphabet_size = 8
    
    for size in range(3, 11):
        # 模拟经验分位数编码
        bins = np.percentile(normalized_all, np.linspace(0, 100, size + 1))
        digitized = np.digitize(normalized_all, bins[1:-1])
        ent = calculate_entropy(digitized)
        efficiency = ent / math.log2(size)
        print(f"  {size:<6} | {ent:<10.4f} | {efficiency:.2%}")

    # 4. 参数建议逻辑
    # word_size: 降维后的长度。通常取中位长度的 1/2。
    # 如果中位数是 15，word_size 取 8 比较合适。
    suggested_word_size = 8
    if p50 > 20:
        suggested_word_size = 10
    elif p50 < 10:
        suggested_word_size = 4
        
    # alphabet_size: 经验上 6-8 对金融/赔率类数据较好
    suggested_alphabet_size = 8
    
    # interpolate_len: 统一插值长度。应该大于大部分序列的长度，以减少信息损失。
    # 建议取 P90 或更高。
    suggested_interpolate_len = int(max(32, p90))

    # 生成经验断点
    emp_breakpoints = np.percentile(normalized_all, np.linspace(0, 100, suggested_alphabet_size + 1)[1:-1])

    print(f"\n{'=' * 60}")
    print("最终系统配置建议")
    print("=" * 60)
    print(f"  - word_size (分段数): {suggested_word_size}")
    print(f"  - alphabet_size (字母表大小): {suggested_alphabet_size}")
    print(f"  - interpolate_len (统一插值长度): {suggested_interpolate_len}")
    print(f"  - breakpoints_type (断点类型): {'empirical' if not is_gaussian else 'gaussian'}")
    print("=" * 60)

    config = {
        "word_size": int(suggested_word_size),
        "alphabet_size": int(suggested_alphabet_size),
        "interpolate_len": int(suggested_interpolate_len),
        "breakpoints_type": "empirical" if not is_gaussian else "gaussian",
        "empirical_breakpoints": [float(b) for b in emp_breakpoints],
        "gaussian_breakpoints": [float(b) for b in stats.norm.ppf(np.linspace(0, 1, suggested_alphabet_size + 1)[1:-1])],
        "metadata": {
            "n_matches": len(matches),
            "median_length": float(p50),
            "skewness": float(skew),
            "kurtosis": float(kurt)
        }
    }

    with open(output_config, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n配置已保存至: {output_config}")
    return config


if __name__ == '__main__':
    import sys
    
    json_path = 'SAX encoder/SAX encoder/bet365_details/bet365_details.json'
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    
    if not os.path.exists(json_path):
        # 尝试一些常见的相对路径
        paths = [
            'SAX encoder/bet365_details/bet365_details.json',
            'bet365_details.json',
            '../SAX encoder/SAX encoder/bet365_details/bet365_details.json'
        ]
        for p in paths:
            if os.path.exists(p):
                json_path = p
                break

    analyze_odds_data(json_path)

