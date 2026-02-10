import json
import numpy as np
from collections import Counter
from scipy import stats
import math
import os
import glob


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


def extract_running_odds(match, bookmaker_name=None):
    """
    从比赛数据中提取运行盘赔率
    支持新旧两种格式:
    - 新格式: match.bookmakers[].runningOdds
    - 旧格式: match.runningOdds
    """
    # 新格式: match.bookmakers 数组
    if 'bookmakers' in match and isinstance(match['bookmakers'], list):
        if bookmaker_name:
            # 指定庄家
            for bm in match['bookmakers']:
                if bm.get('bookmakerName') == bookmaker_name:
                    return bm.get('runningOdds', [])
            return []
        else:
            # 返回第一个有数据的庄家
            for bm in match['bookmakers']:
                odds = bm.get('runningOdds', [])
                if odds and len(odds) >= 2:
                    return odds
            return []
    
    # 旧格式: 直接的 runningOdds
    return match.get('runningOdds', [])


def load_match_data(json_file):
    """
    加载比赛数据
    支持:
    - 单个 JSON 文件
    - 目录（自动处理目录下所有 JSON 文件）
    """
    data = []
    
    if os.path.isfile(json_file):
        files = [json_file]
    elif os.path.isdir(json_file):
        files = glob.glob(os.path.join(json_file, '*.json'))
        files.sort()
    else:
        return data
    
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
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
    """分析单个庄家的数据"""
    print("=" * 70)
    print(f"庄家数据分布深度分析报告: {bookmaker_name}")
    print("=" * 70)
    
    # 收集数据
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
                h = [safe_float(o.get('home')) for o in running]
                d = [safe_float(o.get('draw')) for o in running]
                a = [safe_float(o.get('away')) for o in running]
                
                # 过滤无效数据
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
    
    best_alphabet_size = 8
    
    for size in range(3, 11):
        bins = np.percentile(normalized_all, np.linspace(0, 100, size + 1))
        digitized = np.digitize(normalized_all, bins[1:-1])
        ent = calculate_entropy(digitized)
        efficiency = ent / math.log2(size)
        print(f"  {size:<6} | {ent:<10.4f} | {efficiency:.2%}")
    
    # 4. 参数建议逻辑
    suggested_word_size = 8
    if p50 > 20:
        suggested_word_size = 10
    elif p50 < 10:
        suggested_word_size = 4
        
    suggested_alphabet_size = 8
    suggested_interpolate_len = int(max(32, p90))
    
    # 生成经验断点
    emp_breakpoints = np.percentile(normalized_all, np.linspace(0, 100, suggested_alphabet_size + 1)[1:-1])
    
    print(f"\n{'=' * 70}")
    print("最终系统配置建议")
    print("=" * 70)
    print(f"  - word_size (分段数): {suggested_word_size}")
    print(f"  - alphabet_size (字母表大小): {suggested_alphabet_size}")
    print(f"  - interpolate_len (统一插值长度): {suggested_interpolate_len}")
    print(f"  - breakpoints_type: {'empirical' if not is_gaussian else 'gaussian'}")
    print("=" * 70)
    
    config = {
        "bookmaker": bookmaker_name,
        "word_size": int(suggested_word_size),
        "alphabet_size": int(suggested_alphabet_size),
        "interpolate_len": int(suggested_interpolate_len),
        "breakpoints_type": "empirical" if not is_gaussian else "gaussian",
        "empirical_breakpoints": [float(b) for b in emp_breakpoints],
        "gaussian_breakpoints": [float(b) for b in stats.norm.ppf(np.linspace(0, 1, suggested_alphabet_size + 1)[1:-1])],
        "metadata": {
            "n_matches_total": len(matches),
            "n_matches_valid": processed_count,
            "median_length": float(p50),
            "skewness": float(skew),
            "kurtosis": float(kurt)
        }
    }
    
    with open(output_config, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n配置已保存至: {output_config}")
    return config


def analyze_all_bookmakers(data_dir, output_dir=None):
    """
    分析目录下所有 JSON 文件中的所有庄家数据
    """
    if output_dir is None:
        output_dir = data_dir
    
    matches = load_match_data(data_dir)
    
    if not matches:
        print(f"错误: 未找到有效数据")
        return
    
    # 收集所有庄家名称
    bookmakers_found = set()
    for match in matches:
        if 'bookmakers' in match and isinstance(match['bookmakers'], list):
            for bm in match['bookmakers']:
                name = bm.get('bookmakerName')
                if name:
                    bookmakers_found.add(name)
    
    if not bookmakers_found:
        print("错误: 数据中未找到庄家信息 (使用旧数据格式)")
        return
    
    print(f"发现庄家: {', '.join(sorted(bookmakers_found))}")
    print(f"比赛总数: {len(matches)}")
    print()
    
    # 为每个庄家生成分析报告
    all_configs = {}
    for bm_name in sorted(bookmakers_found):
        safe_name = bm_name.lower().replace(' ', '_').replace('/', '_')
        output_config = os.path.join(output_dir, f"sax_config_{safe_name}.json")
        config = analyze_single_bookmaker(matches, bm_name, output_config)
        if config:
            all_configs[bm_name] = config
        print()
    
    return all_configs


def analyze_odds_data(json_file, output_config="sax_config.json", bookmaker=None):
    """
    分析赔率数据分布并生成配置建议
    
    参数:
        json_file: JSON 文件或目录路径
        output_config: 输出配置文件路径
        bookmaker: 指定分析的庄家名称（可选）
    """
    
    # 判断输入是文件还是目录
    if os.path.isdir(json_file):
        # 目录模式: 分析所有庄家
        base_output_dir = os.path.dirname(output_config) if output_config != "sax_config.json" else json_file
        os.makedirs(base_output_dir, exist_ok=True)
        return analyze_all_bookmakers(json_file, base_output_dir)
    else:
        # 单文件模式
        matches = load_match_data(json_file)
        
        if not matches:
            print(f"错误: 未找到有效数据")
            return None
        
        # 如果指定了庄家
        if bookmaker:
            return analyze_single_bookmaker(matches, bookmaker, output_config)
        
        # 否则分析第一个有数据的庄家
        sample_match = matches[0]
        if 'bookmakers' in sample_match and isinstance(sample_match['bookmakers'], list):
            # 使用第一个有数据的庄家
            for bm in sample_match['bookmakers']:
                if bm.get('runningOdds') and len(bm['runningOdds']) >= 2:
                    return analyze_single_bookmaker(matches, bm['bookmakerName'], output_config)
        
        # 旧格式兼容
        return analyze_single_bookmaker(matches, "Unknown", output_config)


def print_usage():
    """打印使用说明"""
    print("""
使用方式:
    python 2.analyze_odds.py [数据路径] [选项]

参数:
    数据路径: JSON 文件或目录路径
              默认: ./1.generateOddsDetail/SAX encoder/bookmaker_details

选项:
    --bookmaker <名称>    指定分析的庄家名称
    --output <文件>       输出配置文件路径
    --list                列出所有可用的庄家

示例:
    # 分析目录下所有庄家
    python 2.analyze_odds.py "./1.generateOddsDetail/SAX encoder/bookmaker_details"
    
    # 分析单个庄家
    python 2.analyze_odds.py "./1.generateOddsDetail/SAX encoder/bookmaker_details" --bookmaker "Bet 365"
    
    # 指定输出文件
    python 2.analyze_odds.py "./1.generateOddsDetail/SAX encoder/bookmaker_details" --output my_config.json
""")


if __name__ == '__main__':
    import sys
    
    # 默认路径
    data_path = './1.generateOddsDetail/SAX encoder/bookmaker_details'
    bookmaker = None
    output_config = 'sax_config.json'
    
    # 解析参数
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg in ['--help', '-h']:
            print_usage()
            sys.exit(0)
        elif arg == '--bookmaker' and i + 1 < len(sys.argv):
            bookmaker = sys.argv[i + 1]
            i += 2
        elif arg == '--output' and i + 1 < len(sys.argv):
            output_config = sys.argv[i + 1]
            i += 2
        elif arg == '--list':
            # 列出所有庄家
            matches = load_match_data(data_path)
            if matches:
                bookmakers = set()
                for m in matches:
                    if 'bookmakers' in m:
                        for bm in m['bookmakers']:
                            if bm.get('bookmakerName'):
                                bookmakers.add(bm['bookmakerName'])
                print("可用的庄家:")
                for bm in sorted(bookmakers):
                    print(f"  - {bm}")
            sys.exit(0)
        else:
            data_path = arg
            i += 1
    
    # 尝试多个可能的路径
    if not os.path.exists(data_path):
        paths_to_try = [
            data_path,
            '../' + data_path,
        ]
        for p in paths_to_try:
            if os.path.exists(p):
                data_path = p
                break
    
    if not os.path.exists(data_path):
        print(f"错误: 找不到数据路径 '{data_path}'")
        print_usage()
        sys.exit(1)
    
    analyze_odds_data(data_path, output_config, bookmaker)
