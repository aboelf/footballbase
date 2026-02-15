# SAX编码覆盖率优化测试结果

## 测试执行摘要

- **测试时间**: 2026-02-15
- **测试数据**: Bet 365 (10,625场) + Easybets (10,405场)
- **测试目标**: 找到覆盖率>35%且准确率>65%的参数配置

---

## 测试结果对比

### Bet 365 测试结果

| 配置 | 模式数 | 覆盖率 | 准确率 | 覆盖场数 | 评价 |
|------|--------|--------|--------|----------|------|
| 原始参数 (4×3, 70%, min10) | 15 | **1.98%** | **75.24%** | 210 | ❌ 覆盖太低 |
| 降低粒度 (3×3, 70%, min10) | 13 | 1.84% | 75.38% | 195 | ❌ 覆盖更低 |
| 放宽阈值 (4×3, 65%, min5) | 71 | 6.37% | **74.74%** | 677 | ⚠️ 仍不足 |
| 双优化 (3×3, 65%, min5) ⭐ | 57 | 6.17% | 71.95% | 656 | ⚠️ 仍不足 |
| 激进参数 (3×3, 60%, min3) | 194 | **16.34%** | 67.80% | 1736 | ⚠️ 接近目标 |
| **软匹配 (3×3, 65%, min5+软)** ⭐⭐ | 57 | **47.61%** | 49.61% | 5059 | ❌ 准确太低 |

### Easybets 测试结果

| 配置 | 模式数 | 覆盖率 | 准确率 | 覆盖场数 | 评价 |
|------|--------|--------|--------|----------|------|
| 原始参数 | 16 | 2.03% | 77.73% | 211 | ❌ 覆盖太低 |
| 激进参数 | 233 | **25.08%** | 67.16% | 2610 | ⚠️ 接近目标 |
| **软匹配** | 72 | **58.90%** | 49.45% | 6129 | ❌ 准确太低 |

---

## 关键发现

### 1. 软匹配的双刃剑效应

**软匹配（汉明距离≤1）**:
- ✅ 覆盖率: 1.98% → **47.61%** (24倍提升！)
- ❌ 准确率: 75.24% → **49.61%** (接近随机猜测)

**分析**: 软匹配过于"宽容"，将不同模式强行归类，导致准确率大幅下降。

### 2. 激进参数的平衡点

**激进参数 (3×3, 60%, min3)**:
- ✅ 覆盖率: **16.34%** / **25.08%**
- ✅ 准确率: **67.80%** / **67.16%**
- ⚠️ 覆盖率仍低于35%目标

### 3. 多庄家融合效果有限

**双庄家融合**:
- 可用覆盖率: **14.00%**
- 双高准确率: **82.35%** (高置信度)
- 单高准确率: **71.32%** (中置信度)

**问题**: 两个庄家的模式重叠度低，融合后覆盖率没有显著提升。

---

## 问题根源分析

### 为什么覆盖率难以提升？

```
当前模式: individual (分别编码主/平/客)
编码长度: word_size × 3 = 4×3 = 12字符 (Bet 365)
模式空间: 3^12 = 531,441 种可能

比赛数: 10,625场
平均每个模式: 10,625 / 531,441 ≈ 0.02场/模式

结论: 模式空间过大，导致比赛极度分散
```

### 覆盖率计算公式

```
覆盖率 = 高纯模式覆盖的比赛数 / 总比赛数

高纯模式 = 纯度≥70% 且 样本数≥10的模式

问题: 10,000+场比赛分散在50万+模式中，难以形成大样本高纯模式
```

---

## 修正优化方案

### 方案A: 极端压缩模式空间（推荐尝试）

```python
# 使用 interleaved 代替 individual
# 将三个赔率序列交错编码为1个序列

参数:
  strategy: "interleaved"
  word_size: 4        # 4段
  alphabet_size: 3    # 3字母
  编码长度: 4字符
  模式空间: 3^4 = 81种

预期:
  平均每个模式: 10,625 / 81 ≈ 131场比赛
  预期覆盖率: 40-60%
  预期准确率: 60-65%
```

### 方案B: 分层分类

```python
# 第一层: 粗粒度（主胜/平局/客胜大方向）
# 使用赔率平均值判断大方向

参数:
  level1_threshold: 0.6  # 60%纯度即可
  
# 第二层: 细粒度（在大方向内细分）
  strategy: "individual"
  word_size: 3
  alphabet_size: 3

预期:
  第一层覆盖: 100%
  第二层高纯覆盖: 30-40%
```

### 方案C: 特征工程

```python
# 引入更多特征，不仅依赖SAX编码

特征:
  1. SAX编码 (individual 3×3)
  2. 赔率平均值 (home_mean, draw_mean, away_mean)
  3. 赔率标准差 (波动性)
  4. 球队排名差
  5. 历史交锋记录

方法: 用这些特征训练简单分类器（如决策树）

预期:
  覆盖率: >50%
  准确率: 65-70%
```

---

## 立即可执行的优化

### 步骤1: 测试 interleaved 策略

```bash
cd /Users/huabo/projects/footballbase/SAX_encoder
python evaluation/run_sax_evaluation.py \
  --data "1.generateOddsDetail/SAX encoder/bookmaker_details/bet_365_details.json" \
  --bookmaker "Bet 365" \
  --word-size 4 \
  --alphabet-size 3 \
  --strategy interleaved
```

### 步骤2: 放宽软匹配条件

```python
# 当前: 汉明距离≤1 (太宽松)
# 优化: 只匹配前N个字符

def partial_match(pattern, library, n=6):
    """只匹配前6个字符"""
    prefix = pattern[:n]
    for lib_pattern in library:
        if lib_pattern[:n] == prefix:
            return library[lib_pattern]
    return None
```

### 步骤3: 动态阈值

```python
def get_threshold(sample_count):
    """根据样本数动态调整阈值"""
    if sample_count >= 30:
        return 0.60  # 大样本，要求可放宽
    elif sample_count >= 15:
        return 0.65
    elif sample_count >= 5:
        return 0.70
    else:
        return 0.75  # 小样本，要求更高
```

---

## 结论与建议

### 当前状况

| 指标 | 原始 | 优化后 | 目标 |
|------|------|--------|------|
| 覆盖率 | 1.98% | 16-25% | 35% ❌ |
| 准确率 | 75% | 67-75% | 65% ✅ |

### 推荐下一步

1. **立即测试**: interleaved (4×3) 策略
   - 模式空间从 531,441 降到 81
   - 预期覆盖率: 40%+

2. **数据扩充**: 收集更多比赛数据
   - 从 10,000场 扩展到 30,000+场
   - 每个模式的平均样本数增加3倍

3. **特征融合**: 引入更多特征训练分类器
   - 不仅依赖SAX编码
   - 使用赔率统计特征 + SAX编码

### 现实预期

- **短期** (1周内): 覆盖率 25-30%，准确率 65-70%
- **中期** (1月内): 覆盖率 35-40%，准确率 65-70%
- **长期** (3月内): 覆盖率 50%+，准确率 70%+

---

## 执行命令

```bash
# 1. 测试 interleaved 策略
cd /Users/huabo/projects/footballbase/SAX_encoder
python -c "
from evaluation.run_sax_evaluation import *
from sax_encoder import SAXEncoder

matches = load_odds_from_json('1.generateOddsDetail/SAX encoder/bookmaker_details/bet_365_details.json', 'Bet 365')
# 加载结果...

encoder = SAXEncoder(word_size=4, alphabet_size=3)
library = build_pattern_library(matches, encoder, 'interleaved', 5, 0.65)
print(f'interleaved模式库: {len(library)}')
"

# 2. 对比所有策略
python evaluation/run_sax_evaluation.py \
  --data "1.generateOddsDetail/SAX encoder/bookmaker_details/bet_365_details.json" \
  --bookmaker "Bet 365"
```
