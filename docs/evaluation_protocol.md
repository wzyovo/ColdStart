# 冷启动评估协议

## 适用范围

本文档用于定义严格冷启动用户在三个阶段下的推荐效果评估方式。

## 用户划分

- `warm_train`
- `warm_valid`
- `cold_zero_shot`
- `cold_one_shot`
- `cold_three_shot`

在进入指定阶段之前，冷启动用户的历史交互不能泄露给模型输入。

## 阶段定义

### Zero-shot

可用信号：

- device
- location
- query
- circle
- budget
- declared tags

不可用信号：

- browsing history
- clicks
- collections
- orders

### One-shot

可用信号：

- 所有 zero-shot 可用信号
- 第 1 次交互的商品和行为类型

### Three-shot

可用信号：

- 所有 zero-shot 可用信号
- 前 3 次交互的商品、时间和行为类型

## 候选集生成

同一轮比较中，所有模型必须使用相同的候选集规模。

推荐默认值：

- 候选集大小：200
- 重排 Top-K：10

## 对比基线

1. Random
2. Popularity
3. 仅内容标签匹配
4. Structured XGBoost
5. Multimodal CNN + LSTM + attention

## 提升标准

主要目标：

- 严格冷启动用户上的 `Precision@10` 相比最强基线至少提升 5%。

次要目标：

- 用户在 3 次交互后的 `Precision@10` 明显高于随机策略。

## 延迟要求

需要测量：

- 单用户推荐延迟
- 100 用户批量延迟
- p50、p95、p99

目标：

- `p95_latency_ms < 100`
