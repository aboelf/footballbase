# Work Plan: 亚盘 SAX 编码相似度匹配

## TL;DR

> 在 `find_similar_matches.py` 中添加亚盘数据的 SAX 编码和相似度计算，提高比赛匹配准确率。

> **Deliverables**:
> - 亚盘 SAX 编码函数 (parse, encode, distance)
> - 集成到相似度搜索流程
> - 结果输出包含亚盘 SAX 距离

> **Estimated Effort**: Short
> **Parallel Execution**: NO (sequential task)
> **Critical Path**: 添加函数 → 集成搜索 → 输出结果

---

## Context

### Original Request
用户希望利用 `v_match_odds_sax_handicap` 视图中的 `odds_detail` (JSONB) 数据，通过 SAX 编码提高相似度匹配准确率。

### Data Structure
- `odds_detail` 是 JSONB 格式，包含亚盘赔率随时间变化的详细记录
- 格式: `[{"handicap": "半球/一球", "home": 0.85, "away": 1.00, "time": "2024-01-15 10:00"}, ...]`

---

## Work Objectives

### Core Objective
在现有欧赔 SAX 相似度匹配基础上，增加亚盘数据的 SAX 编码和距离计算。

### Concrete Deliverables
1. 添加 `parse_handicap_odds_detail()` - 解析 odds_detail JSONB
2. 添加 `encode_handicap_sax()` - 对亚盘数据进行 SAX 编码
3. 添加 `calculate_handicap_distance()` - 计算亚盘 SAX 距离
4. 修改 `find_similar_matches()` - 集成亚盘距离计算
5. 修改输出 - 显示亚盘 SAX 距离

### Definition of Done
- [ ] 运行 `python find_similar_matches.py 2799893` 成功
- [ ] 输出包含亚盘 SAX 编码信息
- [ ] 相似度结果同时考虑欧赔和亚盘

---

## Execution Strategy

### Task Breakdown

#### Task 1: 添加亚盘 SAX 编码函数
**What to do**:
1. 在 `SAXEncoder` 类后添加 `parse_handicap_odds_detail()` 函数
2. 添加 `encode_handicap_sax()` 函数
3. 添加 `calculate_handicap_distance()` 函数

**Acceptance Criteria**:
- [ ] 函数可以正确解析 odds_detail JSONB
- [ ] 函数可以生成 SAX 编码字符串

#### Task 2: 集成到相似度搜索
**What to do**:
1. 修改 `find_similar_matches()` 函数签名，添加亚盘相关参数
2. 在主循环中计算亚盘 SAX 距离
3. 综合欧赔距离和亚盘距离

**Acceptance Criteria**:
- [ ] 搜索时同时考虑欧赔和亚盘
- [ ] 距离计算正确

#### Task 3: 更新输出和参数
**What to do**:
1. 修改命令行参数，添加 `--use-handicap` (已有占位)
2. 修改输出表格，添加亚盘 SAX 距离列

**Acceptance Criteria**:
- [ ] 输出显示亚盘 SAX 距离
- [ ] `--use-handicap` 参数生效

---

## QA Scenarios

### Scenario: 正常运行测试
  Tool: Bash
  Preconditions: 数据库有数据
  Steps:
    1. cd SAX_encoder
    2. python find_similar_matches.py 2799893 --use-handicap
  Expected Result: 输出包含亚盘信息，无报错

### Scenario: 无亚盘数据
  Tool: Bash
  Preconditions: 某比赛无 odds_detail
  Steps:
    1. python find_similar_matches.py <无亚盘的ID> --use-handicap
  Expected Result: 优雅降级，只使用欧赔距离

---

## Success Criteria

### Verification Commands
```bash
cd SAX_encoder
python find_similar_matches.py 2799893 --use-handicap
# 期望: 输出包含 "亚盘" 相关列
```
