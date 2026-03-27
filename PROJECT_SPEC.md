# 冷启动项目说明

## 项目目标对齐

本仓库正在从“基于人工特征的排序原型”升级为“面向冷启动推荐的完整项目”，目标对齐以下四类要求：

1. 数据维度
   - 构建带有多标签属性的外卖仿真数据集。
   - 至少覆盖 500 个商家、2000 个商品，以及 100 个严格意义上的冷启动新用户。
   - 商品和标签分布需要体现长尾特征。
2. 技术维度
   - 使用 `CNN` 与 `LSTM` 组成多模态基座。
   - 引入多标签学习，预测用户口味、场景、价格偏好。
   - 通过注意力机制融合文本、图像、标签和上下文特征。
3. 性能维度
   - 新用户 `Precision@10` 相比选定基线提升至少 5%。
   - 新用户在 3 次交互后，推荐效果显著优于随机策略。
   - 在线推理目标耗时低于 100 ms。
4. 可视化维度
   - 提供可解释推荐的 Web 演示界面。
   - 展示推荐逻辑从静态冷启动匹配逐步演化为动态个性化排序的过程。

## 当前项目状态

现有仓库已经具备以下基础：

- 来自 Tianchi 衍生数据的用户 ID 和商品 ID。
- `real_enhanced_data/` 下的增强结构化数据表。
- 热门推荐基线。
- XGBoost 的 warm-user 排序模型。

目前仍然存在的缺口：

- 还没有严格的冷启动评估协议。
- 还没有真正的多模态字段。
- 还没有 `CNN/LSTM/Attention` 深度模型训练主线。
- 还没有正式的 Web 界面。
- 还没有延迟测试和在线性能验证。

## 目标目录结构

```text
docs/
  evaluation_protocol.md
src/
  coldstart/
    __init__.py
    config.py
    data_pipeline.py
    datasets.py
    model.py
    trainer.py
    evaluate.py
    inference.py
demo/
  index.html
  styles.css
  app.js
requirements.txt
```

## 数据协议

目标数据集会在现有结构化数据基础上补充多模态字段和冷启动协议文件。

### 商品字段

- `item_id`
- `merchant_id`
- `title_text`
- `description_text`
- `primary_cuisine`
- `price_band`
- `scene_tags`
- `taste_tags`
- `image_token`
- `image_vector_path`
- `popularity_score`
- `long_tail_bucket`

### 用户字段

- `user_id`
- `user_type`
- `timestamp`
- `device_type`
- `location_id`
- `query_id`
- `circle_id`
- `budget_level`
- `declared_tags`

### 协议文件

- `cold_start_eval_users.csv`
  - 严格冷启动用户，只允许使用画像和上下文特征。
- `cold_interactions_step1.csv`
  - 冷启动用户的第 1 次交互记录。
- `cold_interactions_step3.csv`
  - 冷启动用户前 3 次交互记录。
- `item_multimodal_features.csv`
  - 商品的轻量级图文特征或预计算向量。

## 建模方案

### 基座结构

- 文本编码器：`Embedding -> TextCNN -> BiLSTM`
- 图像分支：对预计算图像向量做线性投影
- 用户上下文分支：对圈层、设备、价格、位置、查询等离散特征做嵌入
- 用户交互分支：对前 `k` 次交互序列使用 `LSTM`
- 标签分支：对口味、场景、价格标签做嵌入表示

### 融合方式

- 使用跨模态注意力机制，以用户上下文/查询为 Query，以商品文本、图像、标签表示为 Key/Value。
- 输出融合后的用户-商品联合表示，用于排序与解释。

### 学习目标

- 主任务：排序分数学习
- 辅助任务：用户多标签偏好预测
- 可选任务：冷启动阶段分类（`0-shot`、`1-shot`、`3-shot`）

## 评估协议

冷启动效果必须分别在三个阶段下评估：

1. `zero_shot`
   - 用户没有任何可用交互。
2. `one_shot`
   - 用户有 1 次交互。
3. `three_shot`
   - 用户有 3 次交互。

必须包含的对比基线：

- Random
- Popularity
- Content matching
- Structured XGBoost ranker
- Multimodal deep model

必须输出的指标：

- `Precision@10`
- `Recall@10`
- `NDCG@10`
- `MRR@10`
- `HitRate@10`
- `p95_latency_ms`

## 演示要求

演示页面至少需要展示：

- 推荐候选列表及得分
- 标签贡献度条形图
- 不同交互阶段下的解释结果
- “规则驱动”向“行为驱动”演化的过程说明

## 交付路径

### 第一阶段

- 固定数据协议
- 补齐代码骨架
- 提供无依赖静态演示
### 第二阶段

- 安装运行依赖
- 生成更完整的多模态字段
- 训练基线模型和深度模型

### 第三阶段

- 做推理延迟基准测试
- 完成最终可视化解释和汇报材料
