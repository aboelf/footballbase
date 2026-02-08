#!/usr/bin/env python3
"""
比赛聚类分析脚本

基于 SAX 编码进行聚类分析，找出赔率变化模式相似的比赛群组

使用方法：
1. 确保 Supabase 配置正确 (.env)
2. 运行: python cluster_matches.py
"""

import json
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
from dotenv import load_dotenv
from postgrest import SyncPostgrestClient
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# 加载环境变量
load_dotenv()


# ============ SAX 距离计算 ============

# SAX 断点表 (7个符号 a-g，对应标准正态分布的分位数)
SAX_BREAKPOINTS = {
    3: [-0.43, 0.43],
    4: [-0.67, 0, 0.67],
    5: [-0.84, -0.25, 0.25, 0.84],
    6: [-0.97, -0.43, 0, 0.43, 0.97],
    7: [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
}

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def symbol_to_number(symbol: str) -> int:
    """符号转数值 (a=0, b=1, ...)"""
    return ord(symbol) - ord('a')


def mindist_sax(s1: str, s2: str, word_size: int, alphabet_size: int) -> float:
    """
    计算两个 SAX 字符串的 MINDIST 距离

    参考: Lin, J., Keogh, E., Lonardi, S., & Chiu, B. (2003).
    A symbolic representation of time series, with implications for streaming algorithms.
    """
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

        # 非相邻符号，使用断点差值
        b1, b2 = breakpoints[min(n1, n2)], breakpoints[max(n1, n2)]
        return (b2 - b1) ** 2

    dist = sum(char_distance(s1[i], s2[i]) for i in range(min_len))
    return np.sqrt(dist / word_size)


def build_distance_matrix(sax_list: list[str], word_size: int = 8, alphabet_size: int = 7) -> np.ndarray:
    """构建 SAX 距离矩阵"""
    n = len(sax_list)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = mindist_sax(sax_list[i], sax_list[j], word_size, alphabet_size)
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    return dist_matrix


# ============ 聚类方法 ============

@dataclass
class ClusterResult:
    labels: list[int]
    n_clusters: int
    method: str


def cluster_by_sax_kmeans(
    sax_strings: list[str],
    n_clusters: int = 8,
    word_size: int = 8,
    alphabet_size: int = 7,
) -> ClusterResult:
    """基于 SAX 的 K-Means 聚类"""
    # 将 SAX 字符串转换为数值特征
    features = encode_sax_to_features(sax_strings, word_size, alphabet_size)

    # 聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    return ClusterResult(labels=labels.tolist(), n_clusters=n_clusters, method="kmeans_sax")


def cluster_by_sax_dbscan(
    sax_strings: list[str],
    eps: float = 0.5,
    min_samples: int = 5,
    word_size: int = 8,
    alphabet_size: int = 7,
) -> ClusterResult:
    """基于 SAX 的 DBSCAN 聚类"""
    # 构建距离矩阵
    dist_matrix = build_distance_matrix(sax_strings, word_size, alphabet_size)

    # DBSCAN 需要距离矩阵
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = dbscan.fit_predict(dist_matrix)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"  DBSCAN 聚类结果: {n_clusters} 个簇, {n_noise} 个噪声点")

    return ClusterResult(labels=labels.tolist(), n_clusters=n_clusters, method="dbscan_sax")


def cluster_by_features(
    features: np.ndarray,
    n_clusters: int = 8,
) -> ClusterResult:
    """基于数值特征的 K-Means 聚类"""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)

    return ClusterResult(labels=labels.tolist(), n_clusters=n_clusters, method="kmeans_features")


def encode_sax_to_features(sax_strings: list[str], word_size: int, alphabet_size: int) -> np.ndarray:
    """将 SAX 字符串转换为数值特征"""
    n = len(sax_strings)
    features = np.zeros((n, word_size * 2))  # interleaved + delta

    for i, sax in enumerate(sax_strings):
        if len(sax) >= word_size:
            # 前 word_size 个字符是 interleaved
            for j in range(word_size):
                features[i, j] = symbol_to_number(sax[j])
            # 剩余字符是 delta（如果存在）
            for j in range(word_size):
                if word_size + j < len(sax):
                    features[i, word_size + j] = symbol_to_number(sax[word_size + j])

    return features


# ============ 结果分析 ============

def analyze_clusters(
    matches: list[dict],
    labels: list[int],
    output_prefix: str = "cluster",
) -> dict:
    """分析聚类结果"""
    n = len(matches)
    cluster_ids = set(labels)

    # 1. 按簇统计
    cluster_stats = {}
    for cluster_id in sorted(cluster_ids):
        if cluster_id == -1:
            continue
        cluster_matches = [m for m, l in zip(matches, labels) if l == cluster_id]
        cluster_stats[cluster_id] = {
            "count": len(cluster_matches),
            "sax_patterns": Counter(m.get("sax_interleaved", "")[:8] for m in cluster_matches).most_common(5),
            "avg_final_win": np.mean([m.get("final_win", 0) for m in cluster_matches if m.get("final_win")]),
            "avg_final_draw": np.mean([m.get("final_draw", 0) for m in cluster_matches if m.get("final_draw")]),
            "avg_final_lose": np.mean([m.get("final_lose", 0) for m in cluster_matches if m.get("final_lose")]),
        }

    # 2. 按 SAX 模式统计胜率
    sax_win_rates = {}
    for m, l in zip(matches, labels):
        if l == -1:
            continue
        sax = m.get("sax_interleaved", "")[:8]
        if sax:
            if sax not in sax_win_rates:
                sax_win_rates[sax] = {"total": 0, "wins": 0, "draws": 0}
            sax_win_rates[sax]["total"] += 1
            # 解析比分
            score = m.get("final_score", "")
            if score and "-" in score:
                parts = score.split("-")
                if len(parts) == 2:
                    try:
                        home, away = int(parts[0]), int(parts[1])
                        if home > away:
                            sax_win_rates[sax]["wins"] += 1
                        elif home == away:
                            sax_win_rates[sax]["draws"] += 1
                    except ValueError:
                        pass

    # 计算胜率
    for sax, stats in sax_win_rates.items():
        if stats["total"] >= 10:
            stats["win_rate"] = stats["wins"] / stats["total"]
            stats["draw_rate"] = stats["draws"] / stats["total"]

    # 3. 保存结果
    result = {
        "n_matches": n,
        "n_clusters": len([c for c in cluster_ids if c != -1]),
        "cluster_stats": cluster_stats,
        "sax_win_rates": {k: v for k, v in sorted(sax_win_rates.items(), key=lambda x: -x[1].get("win_rate", 0)) if v.get("total", 0) >= 10},
    }

    # 保存 JSON
    output_file = f"{output_prefix}_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"分析结果已保存: {output_file}")

    return result


def print_cluster_summary(result: dict):
    """打印聚类摘要"""
    print("\n" + "=" * 60)
    print("聚类分析结果")
    print("=" * 60)

    print(f"\n总比赛数: {result['n_matches']}")
    print(f"聚类数: {result['n_clusters']}")

    print("\n--- 按簇统计 ---")
    for cluster_id, stats in result["cluster_stats"].items():
        print(f"\n簇 {cluster_id}: {stats['count']} 场比赛")
        print(f"  常见 SAX 模式: {stats['sax_patterns'][:3]}")
        print(f"  平均赔率: 胜={stats['avg_final_win']:.2f}, 平={stats['avg_final_draw']:.2f}, 负={stats['avg_final_lose']:.2f}")

    print("\n--- SAX 模式胜率 (样本量>=10) ---")
    for sax, stats in list(result["sax_win_rates"].items())[:20]:
        print(f"  {sax}: 胜率={stats.get('win_rate', 0)*100:.1f}%, 平率={stats.get('draw_rate', 0)*100:.1f}% (n={stats['total']})")


# ============ 主程序 ============

def get_supabase_client() -> SyncPostgrestClient:
    """获取 Supabase REST 客户端"""
    url = os.getenv('SUPABASE_URL', '').rstrip('/')
    key = os.getenv('SUPABASE_KEY', '')

    if not url or not key:
        raise ValueError("请在 .env 文件中设置 SUPABASE_URL 和 SUPABASE_KEY")

    return SyncPostgrestClient(f"{url}/rest/v1", headers={
        "Authorization": f"Bearer {key}",
        "apikey": key,
        "Content-Type": "application/json"
    })


def fetch_match_odds_sax(client: SyncPostgrestClient, limit: int = 5000) -> list[dict]:
    """从 Supabase 获取 match_odds_sax 数据"""
    print("从 Supabase 获取数据...")

    result = client.table('match_odds_sax').select("*").limit(limit).execute()

    if not result.data:
        raise ValueError("未找到数据")

    print(f"  已加载 {len(result.data)} 条记录")
    return result.data


def main():
    print("=" * 60)
    print("比赛聚类分析 - 基于 SAX 编码")
    print("=" * 60)

    WORD_SIZE = 8
    ALPHABET_SIZE = 7

    try:
        # 1. 获取数据
        client = get_supabase_client()
        matches = fetch_match_odds_sax(client)

        # 2. 提取 SAX 字符串
        sax_strings = [m.get("sax_interleaved", "") for m in matches if m.get("sax_interleaved")]
        valid_indices = [i for i, m in enumerate(matches) if m.get("sax_interleaved")]

        print(f"\n有效 SAX 记录: {len(sax_strings)}")

        if len(sax_strings) < 10:
            print("错误: 数据量太少，无法进行聚类")
            return

        # 3. 方法1: K-Means 聚类 (SAX 特征)
        print("\n--- 方法1: K-Means (SAX 特征) ---")
        n_clusters = min(10, len(sax_strings) // 50)
        if n_clusters < 4:
            n_clusters = 4

        kmeans_result = cluster_by_sax_kmeans(sax_strings, n_clusters=n_clusters)
        print(f"  聚类数: {kmeans_result.n_clusters}")

        # 只用有 SAX 的数据进行聚类分析
        valid_matches = [matches[i] for i in valid_indices]
        kmeans_analysis = analyze_clusters(valid_matches, kmeans_result.labels, output_prefix="kmeans")
        print_cluster_summary(kmeans_analysis)

        # 4. 方法2: DBSCAN 聚类（跳过，数据量大时距离矩阵计算太慢）
        print("\n--- 方法2: DBSCAN (已跳过，可手动指定较小数据集) ---")
        # dbscan_result = cluster_by_sax_dbscan(sax_strings, eps=0.3, min_samples=10)
        # dbscan_analysis = analyze_clusters(valid_matches, dbscan_result.labels, output_prefix="dbscan")
        # print_cluster_summary(dbscan_analysis)

        # 5. 方法3: K-Means (数值特征)
        print("\n--- 方法3: K-Means (数值特征) ---")
        features = []
        for m in valid_matches:
            features.append([
                m.get("init_win", 0) or 0,
                m.get("final_win", 0) or 0,
                m.get("home_mean", 0) or 0,
                m.get("away_mean", 0) or 0,
            ])
        features = np.array(features)

        feature_result = cluster_by_features(features, n_clusters=n_clusters)
        feature_analysis = analyze_clusters(valid_matches, feature_result.labels, output_prefix="features")
        print_cluster_summary(feature_analysis)

        # 6. 保存聚类标签到文件
        for i, idx in enumerate(valid_indices):
            matches[idx]["cluster_kmeans"] = kmeans_result.labels[i]
            # matches[idx]["cluster_dbscan"] = dbscan_result.labels[i]  # DBSCAN 已跳过
            matches[idx]["cluster_features"] = feature_result.labels[i]

        with open("matches_with_clusters.json", 'w', encoding='utf-8') as f:
            json.dump(matches, f, ensure_ascii=False, indent=2)
        print(f"\n带聚类标签的数据已保存: matches_with_clusters.json")

        # 7. 打印示例查询
        print("\n" + "=" * 60)
        print("示例 Supabase 查询")
        print("=" * 60)
        query = """
-- 查看各簇的比赛分布
SELECT cluster_kmeans, COUNT(*) as cnt
FROM match_odds_sax
WHERE sax_interleaved IS NOT NULL
GROUP BY cluster_kmeans ORDER BY cnt DESC;

-- 查看特定 SAX 模式的比赛
SELECT * FROM match_odds_sax
WHERE sax_interleaved = 'aaabbbcc'
LIMIT 10;
"""
        print(query)

    except ValueError as e:
        print(f"\n配置错误: {e}")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
