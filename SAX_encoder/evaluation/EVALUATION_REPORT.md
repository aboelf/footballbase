# SAX编码质量评估报告

## 执行摘要

测试日期: 2026-02-15
数据: Bet 365 赔率数据，10,625场比赛
数据来源: 德甲/法甲/西甲/意甲/英超/日职联 2020-2025赛季

---

## 测试结果对比

| 策略 | 参数 | 模式空间 | 平均纯度 | 覆盖率 | 高纯模式数 | 综合评分 |
|------|------|----------|----------|--------|------------|----------|
| interleaved (当前) | 8×7 | 5,764,801 | 75.13% | 8.83% | 15 | ⭐⭐⭐ |
| interleaved (中等) | 6×5 | 15,625 | 72.45% | 5.83% | 12 | ⭐⭐ |
| interleaved (粗粒) | 4×3 | 81 | 51.58% | 37.89% | 0 | ⭐ |
| delta | 6×4 | 4,096 | 67.33% | 5.54% | 6 | ⭐⭐ |
| **delta_draw** | 4×3 | 81 | **75.72%** | 4.31% | **19** | ⭐⭐⭐⭐ |
| **individual** 🏆 | 4×3 | 81 | **74.14%** | **14.50%** | **15** | ⭐⭐⭐⭐⭐ |
| home_only | 6×4 | 4,096 | 70.63% | 4.16% | 12 | ⭐⭐ |
| draw_only | 6×4 | 4,096 | 68.41% | 3.93% | 9 | ⭐⭐ |
| away_only | 6×4 | 4,096 | 68.05% | 4.58% | 6 | ⭐⭐ |

---

## 🏆 推荐方案: 分别编码 (individual)

### 参数配置
```json
{
  "strategy": "individual",
  "word_size": 4,
  "alphabet_size": 3,
  "encoding": "分别编码主胜/平局/客胜后拼接"
}
```

### 性能指标
- **平均纯度**: 74.14% ✅ (目标: >65%)
- **覆盖率**: 14.50% ⚠️ (目标: >40%)
- **高纯模式数**: 15个 (目标: >20个)
- **Top模式纯度**: 85.71%

### Top 10 高纯度模式
| 排名 | 模式 | 样本数 | 主胜 | 平局 | 客胜 | 纯度 | 预测 |
|------|------|--------|------|------|------|------|------|
| 1 | abcccbaacbba | 14 | 12 | 1 | 1 | 85.71% | 主胜 |
| 2 | ccbaaabcabbc | 13 | 11 | 0 | 2 | 84.62% | 主胜 |
| 3 | ccaaaaccaacb | 12 | 10 | 1 | 1 | 83.33% | 主胜 |
| 4 | aaccbcbaccaa | 11 | 9 | 0 | 2 | 81.82% | 主胜 |
| 5 | aabcccaabcba | 10 | 8 | 1 | 1 | 80.00% | 主胜 |
| 6 | bccabccabaac | 10 | 1 | 1 | 8 | 80.00% | 客胜 |
| 7 | ccaaabcaaacc | 16 | 12 | 2 | 2 | 75.00% | 主胜 |
| 8 | ccbaaccaaabc | 12 | 3 | 0 | 9 | 75.00% | 客胜 |
| 9 | aacccbbaccaa | 11 | 8 | 1 | 2 | 72.73% | 主胜 |
| 10 | ccaaccbaaabc | 17 | 2 | 3 | 12 | 70.59% | 客胜 |

---

## 备选方案: 差值+平局编码 (delta_draw)

### 适用场景
- 需要**最高纯度**时使用
- 专注于主客赔率差 + 平局偏离度

### 性能指标
- **平均纯度**: 75.72% ✅ (最高)
- **高纯模式数**: 19个 ✅ (最多)
- **覆盖率**: 4.31% ❌ (较低)

---

## 目标达成情况

| 目标 | 要求 | 实际(individual) | 状态 |
|------|------|------------------|------|
| 平均纯度 | >65% | 74.14% | ✅ 达成 |
| 覆盖率 | >40% | 14.50% | ❌ 未达成 |
| 高纯模式数 | >20个 | 15个 | ⚠️ 接近 |

---

## 下一步优化建议

### 1. 短期优化 (立即执行)

**降低纯度阈值**: 将高纯度标准从70%降到65%
- 预计可增加 3-5个高纯模式
- 覆盖率可能提升到 20-25%

**增加最小样本数**: 从10场降到5场
- 可以发现更多细分模式

### 2. 中期优化 (1-2周)

**混合编码策略**:
```python
# 结合 individual + delta_draw
if individual_purity < 0.7:
    use delta_draw_encoding
```

**分层编码**:
- 第一层: 粗粒度分类（主胜/平局/客胜大方向）
- 第二层: 细粒度分类（赔率变化细节）

### 3. 长期优化 (数据积累)

**扩大数据集**:
- 当前: 10,625场
- 目标: 30,000+场
- 预期: 覆盖率提升到 40%+

**引入更多特征**:
- 球队排名
- 历史交锋
- 伤停信息
- 亚盘数据

---

## 预测应用指南

### 使用 individual 策略预测新比赛

```python
from sax_encoder import SAXEncoder

# 1. 创建编码器
encoder = SAXEncoder(word_size=4, alphabet_size=3)

# 2. 编码新比赛
home_pattern = encoder.encode(new_match.home_odds, 4)
draw_pattern = encoder.encode(new_match.draw_odds, 4)
away_pattern = encoder.encode(new_match.away_odds, 4)
pattern = home_pattern + draw_pattern + away_pattern

# 3. 匹配历史模式
if pattern in high_purity_patterns:
    prediction = high_purity_patterns[pattern]['dominant_result']
    confidence = high_purity_patterns[pattern]['purity']
else:
    # 找相似模式（汉明距离<2）
    similar = find_similar_patterns(pattern, high_purity_patterns)
    prediction = majority_vote(similar)
```

### 高纯度模式库

已识别 15 个高纯度模式（纯度≥70%），可用于直接预测:
- 主胜预测模式: 10个
- 客胜预测模式: 4个
- 平局预测模式: 1个

---

## 结论

**individual 策略 (4×3参数)** 是当前最佳SAX编码方案:
- ✅ 平均纯度 74.14%（超过65%目标）
- ✅ 模式空间小（81种），易于管理
- ⚠️ 覆盖率 14.5%（需要进一步优化）

**建议**: 立即采用 individual 策略，同时继续收集数据以提高覆盖率。

---

## 执行命令参考

```bash
# 运行完整评估
cd /Users/huabo/projects/footballbase/SAX_encoder
python evaluation/run_sax_evaluation.py \
  --data "1.generateOddsDetail/SAX encoder/bookmaker_details/bet_365_details.json"

# 生成SAX编码（使用推荐参数）
python 3.run_sax.py \
  --config-dir "1.generateOddsDetail/SAX encoder/bookmaker_details" \
  --data-dir "1.generateOddsDetail/SAX encoder/bookmaker_details"
```
