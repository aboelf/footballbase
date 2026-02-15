# SAX编码质量评估方案

## 目标
找到一种SAX编码参数，使得同一模式下的比赛：
1. 赔率序列相似
2. 比赛结果一致（主胜/平局/客胜）

## 验证指标

### 1. 模式纯度 (Purity)
```
对于某个SAX模式P：
- n_total = 模式P的比赛总数
- n_home = 主胜场数
- n_draw = 平局场数  
- n_away = 客胜场数

纯度 = max(n_home, n_draw, n_away) / n_total
```
**目标**: Top 20模式的平均纯度 > 65%

### 2. 模式覆盖率 (Coverage)
```
覆盖率 = (被Top 20模式覆盖的比赛数) / (总比赛数)
```
**目标**: 覆盖率 > 40%

### 3. 模式稳定性 (Stability)
```
对同一比赛的不同时间子序列编码，
结果应该属于同一模式或相邻模式
```
**目标**: 稳定性 > 80%

## 评估步骤

### Step 1: 准备数据
```sql
-- 从Supabase获取带结果的比赛数据
SELECT 
  match_id,
  final_score,
  sax_interleaved,
  sax_delta,
  home_mean,
  draw_mean,
  away_mean
FROM v_match_odds_sax
WHERE bookmaker_name = 'Bet 365'
  AND final_score IS NOT NULL
```

### Step 2: 计算模式纯度
```python
from collections import Counter
import numpy as np

def calculate_purity(matches_with_sax):
    """
    计算所有SAX模式的纯度
    
    Args:
        matches_with_sax: [{match_id, sax_pattern, final_score, ...}, ...]
    
    Returns:
        pattern_stats: {sax_pattern: {total, home, draw, away, purity}}
    """
    pattern_stats = {}
    
    for match in matches_with_sax:
        pattern = match['sax_interleaved']  # 或 sax_delta
        score = match['final_score']
        
        if pattern not in pattern_stats:
            pattern_stats[pattern] = {
                'total': 0, 'home': 0, 'draw': 0, 'away': 0
            }
        
        pattern_stats[pattern]['total'] += 1
        
        # 解析比分
        if score and '-' in score:
            try:
                home_goals, away_goals = map(int, score.split('-'))
                if home_goals > away_goals:
                    pattern_stats[pattern]['home'] += 1
                elif home_goals == away_goals:
                    pattern_stats[pattern]['draw'] += 1
                else:
                    pattern_stats[pattern]['away'] += 1
            except:
                pass
    
    # 计算纯度
    for pattern, stats in pattern_stats.items():
        if stats['total'] > 0:
            stats['purity'] = max(
                stats['home'], stats['draw'], stats['away']
            ) / stats['total']
    
    return pattern_stats

def evaluate_encoding_quality(pattern_stats, top_n=20):
    """
    评估编码质量
    
    Returns:
        {
            'top_patterns': [...],  # Top N模式
            'avg_purity': float,     # 平均纯度
            'coverage': float,       # 覆盖率
            'total_matches': int,
            'covered_matches': int
        }
    """
    # 只保留样本数>=10的模式
    valid_patterns = {
        k: v for k, v in pattern_stats.items() 
        if v['total'] >= 10
    }
    
    # 按纯度排序
    sorted_patterns = sorted(
        valid_patterns.items(),
        key=lambda x: (-x[1]['purity'], -x[1]['total'])
    )
    
    top_patterns = sorted_patterns[:top_n]
    
    # 计算平均纯度
    avg_purity = np.mean([p['purity'] for _, p in top_patterns])
    
    # 计算覆盖率
    total_matches = sum(s['total'] for s in pattern_stats.values())
    covered_matches = sum(p['total'] for _, p in top_patterns)
    coverage = covered_matches / total_matches if total_matches > 0 else 0
    
    return {
        'top_patterns': top_patterns,
        'avg_purity': avg_purity,
        'coverage': coverage,
        'total_matches': total_matches,
        'covered_matches': covered_matches
    }
```

### Step 3: 对比不同编码方案
```python
# 测试多种编码参数组合
test_configs = [
    {'word_size': 4, 'alphabet_size': 3, 'name': '粗粒度'},
    {'word_size': 6, 'alphabet_size': 4, 'name': '中等粒度'},
    {'word_size': 8, 'alphabet_size': 5, 'name': '细粒度'},
    {'word_size': 4, 'alphabet_size': 5, 'name': '趋势+细节'},
]

results = []
for config in test_configs:
    # 用该参数重新编码所有比赛
    encoded = reencode_matches(matches, config)
    
    # 计算模式纯度
    pattern_stats = calculate_purity(encoded)
    
    # 评估质量
    quality = evaluate_encoding_quality(pattern_stats)
    
    results.append({
        'config': config,
        'quality': quality
    })

# 打印对比结果
for r in results:
    print(f"\n{r['config']['name']}:")
    print(f"  平均纯度: {r['quality']['avg_purity']:.2%}")
    print(f"  覆盖率: {r['quality']['coverage']:.2%}")
    print(f"  Top模式数: {len(r['quality']['top_patterns'])}")
```

## 预测应用

一旦找到好的编码参数，预测流程：

```python
def predict_match(match_odds, sax_encoder, pattern_stats):
    """
    预测单场比赛结果
    
    Args:
        match_odds: 新比赛的赔率序列
        sax_encoder: 训练好的SAX编码器
        pattern_stats: 历史模式统计
    
    Returns:
        {
            'predicted_result': 'home'/'draw'/'away',
            'confidence': float,
            'similar_matches': [...]
        }
    """
    # 1. 编码新比赛
    sax_pattern = sax_encoder.encode(match_odds)
    
    # 2. 查找历史相似模式
    if sax_pattern in pattern_stats:
        stats = pattern_stats[sax_pattern]
        
        # 找到占比最高的结果
        results = {
            'home': stats['home'] / stats['total'],
            'draw': stats['draw'] / stats['total'],
            'away': stats['away'] / stats['total']
        }
        predicted = max(results, key=results.get)
        confidence = results[predicted]
        
        return {
            'predicted_result': predicted,
            'confidence': confidence,
            'pattern_stats': stats
        }
    else:
        # 没有找到完全匹配的模式
        # 可以找相似模式（汉明距离<2）
        return find_similar_patterns(sax_pattern, pattern_stats)
```

## 下一步行动

1. **立即执行**: 用当前数据运行评估脚本，建立baseline
2. **对比实验**: 测试3-5种不同的SAX参数组合
3. **选择最优**: 根据纯度和覆盖率选择最佳参数
4. **集成预测**: 将最优编码应用到预测流程

需要我帮你实现这个评估脚本吗？
