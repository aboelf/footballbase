# SAX编码质量评估与优化计划

## 问题背景

当前SAX编码存在区分度不足的问题：
- 模式分布过于均匀（8段×7字母 = 5,764,801种可能）
- 同一模式下比赛结果一致性差
- 缺乏系统性的编码质量评估方法

## 验证标准

**好编码的定义：**
同一SAX模式下的比赛应该满足：
1. 赔率序列相似（变化趋势相近）
2. 比赛结果一致（主胜/平局/客胜占比>60%）

**量化指标：**
- 模式纯度（Purity）> 65%：同一模式下主导结果的占比
- 模式覆盖率（Coverage）> 40%：Top 20模式覆盖的比赛比例
- 高纯度模式数 > 20个：纯度>70%且有10+样本的模式

## 工作计划

### Phase 1: 实现评估脚本

**任务1.1: 创建评估框架**
- 文件: `SAX_encoder/evaluation/evaluate_sax_quality.py`
- 功能:
  - 从JSON或Supabase加载带结果的比赛数据
  - 使用不同SAX参数编码比赛
  - 计算模式纯度和覆盖率
  - 生成对比报告

**任务1.2: 数据加载器**
- 支持从 `bet_365_details.json` 加载原始赔率数据
- 支持从Supabase `v_match_odds_sax` 视图查询（带final_score）
- 解析比赛结果（主胜/平局/客胜）

**任务1.3: 评估指标计算**
- 模式纯度计算：`max(home, draw, away) / total`
- 覆盖率计算：`sum(top_20_matches) / total_matches`
- 生成Top模式列表及其胜率分布

### Phase 2: 对比测试7种编码策略

**策略1: 当前参数（Baseline）**
- word_size=8, alphabet_size=7, interleaved
- 目的：建立评估基准

**策略2: 趋势粗粒度**
- word_size=4, alphabet_size=3, interleaved
- 目的：大幅减少模式空间，提高集中度

**策略3: 趋势中等粒度**
- word_size=6, alphabet_size=4, interleaved
- 目的：平衡区分度和集中度

**策略4: 差值编码**
- word_size=6, alphabet_size=4, delta (home-away)
- 目的：专注主客实力对比变化

**策略5: 差值+平局偏离**
- word_size=4, alphabet_size=3, delta_draw
- 目的：同时编码主客差和平局偏离度

**策略6: 分别编码**
- word_size=4, alphabet_size=3, individual (h+d+a)
- 目的：分别编码三个赔率后拼接

**策略7: 主胜专注**
- word_size=6, alphabet_size=4, home_only
- 目的：只关注主胜赔率变化趋势

### Phase 3: 分析结果并推荐

**输出:**
1. 7种策略的对比表格
2. 每种策略的Top 10模式详情
3. 推荐的最佳编码参数及理由
4. 预测准确率预估

## 执行命令

```bash
cd /Users/huabo/projects/footballbase/SAX_encoder

# 1. 对比所有编码策略
python evaluation/evaluate_sax_quality.py --compare-configs --data "1.generateOddsDetail/SAX encoder/bookmaker_details/bet_365_details.json"

# 2. 或从Supabase获取最新数据（包含比赛结果）
python evaluation/evaluate_sax_quality.py --compare-configs --from-supabase --bookmaker "Bet 365"
```

## 预期成果

- 找到一种编码策略，使得：
  - Top 20模式平均纯度 > 65%
  - 覆盖率 > 40%
  - 高纯度模式（>70%）数量 >= 20个

- 输出可用于预测的模式库：
  - 每个高纯度模式 → 对应预测结果（主胜/平局/客胜）
  - 新比赛编码后匹配模式 → 输出预测

## 下一步

运行 `/start-work` 开始执行此计划。
