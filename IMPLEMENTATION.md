# 实现说明

本文档说明CI-SDR系统的实现细节，基于论文《Eliminating Sentiment Bias in Recommender Systems by Counterfactual Inference》和需求文档。

## 实现概览

本项目完整实现了基于反事实推理的情感偏差消除推荐系统，主要包括以下组件：

### 1. 数据处理模块 (`data_loader/dataset.py`)

- **ReviewDataset类**：处理Amazon/Yelp格式的数据集
  - 支持JSON和CSV格式
  - 自动聚合用户和物品的评论
  - 使用Word2Vec生成评论嵌入向量
  - 计算用户和物品的情感极性（用于评估）

### 2. 文本嵌入模块 (`utils/embeddings.py`)

- **Word2Vec模型训练**：从评论文本中学习词向量
- **文本到向量转换**：使用平均池化将评论转换为固定长度向量
- **情感极性计算**：使用TextBlob计算情感得分（用于验证和评估）

### 3. 模型架构 (`models/ci_sdr.py`)

实现了完整的CI-SDR模型，包含：

#### 3.1 NCF主干网络
- 用户和物品的嵌入层
- MLP网络处理用户-物品交互特征
- 输出：`q_m = f_m(h_u, h_i)`

#### 3.2 用户情感分支
- MLP网络处理用户评论嵌入 `z_u`
- 输出：`y_u = f_u(z_u)`

#### 3.3 物品情感分支
- MLP网络处理物品评论嵌入 `z_i`
- 输出：`y_i = f_i(z_i)`

#### 3.4 情感控制因子
- 计算 `s_{u,i} = f_s(z_u * z_i)`
- 使用逐元素乘积作为交互特征

#### 3.5 训练阶段预测
- `y_{u,i} = q_m + y_u + y_i`
- `y_{u,i,s} = y_{u,i} * σ(s_{u,i})`

#### 3.6 去偏推理
- `y_debiased = (y_{u,i} - β) * σ(s_{u,i})`
- `β` 是可调超参数，范围 [1e-5, 0.5]

### 4. 训练模块 (`train.py`)

- **多任务损失函数**：
  - `L = L_RC + α_u * L_U + α_i * L_I`
  - `L_RC`：主任务损失（评分预测）
  - `L_U`：用户情感分支损失
  - `L_I`：物品情感分支损失
  
- **优化器配置**：
  - Adam优化器
  - 学习率：0.002
  - 权重衰减：1e-6

- **模型保存**：自动保存验证集上表现最好的模型

### 5. 评估模块 (`evaluate.py` 和 `utils/metrics.py`)

实现了三个关键评估指标：

- **MSE**：均方误差，衡量整体预测准确度
- **BU (User Sentiment Bias)**：用户情感偏差
  - 计算积极用户（前10%）与消极用户（后10%）的MSE之差
- **BI (Item Sentiment Bias)**：物品情感偏差
  - 计算热门物品（前10%）与冷门物品（后10%）的MSE之差

## 关键设计决策

### 1. 因果图建模

根据论文，我们将情感作为中介变量（Mediator），而非混淆变量（Confounder）。这意味着：
- 用户和物品 → 情感 → 评分
- 需要消除的路径：`U → S → Y` 和 `I → S → Y`
- 使用自然直接效应（NDE）进行去偏

### 2. 情感表示

- 不使用外部情感分析工具进行标注
- 通过模型内部学习 `σ(s_{u,i})` 来捕获情感
- 使用TextBlob仅用于验证和评估阶段

### 3. 去偏公式

推理阶段的去偏公式：
```
y_debiased = (y_{u,i} - β) * σ(s_{u,i})
```

这等价于：
```
y_debiased = y_{u,i} * σ(s_{u,i}) - β * σ(s_{u,i})
```

即从总预测中扣除 `β * σ(s_{u,i})` 部分，这是情感偏差的估计。

### 4. 数据预处理

- 训练阶段：从所有评论中训练Word2Vec模型
- 测试阶段：复用训练集的Word2Vec模型和评论聚合
- 对于测试集中出现的新用户/物品，使用零向量作为嵌入

## 与论文的对应关系

| 论文概念 | 代码实现 |
|---------|---------|
| 用户U | `user_idx`, `user_embedding` |
| 物品I | `item_idx`, `item_embedding` |
| 情感S | `s_{u,i}`, `σ(s_{u,i})` |
| 评分Y | `rating`, `y_pred` |
| NCF主干 | `CISDR.ncf` |
| 用户情感分支 | `CISDR.user_sentiment_mlp` |
| 物品情感分支 | `CISDR.item_sentiment_mlp` |
| 情感控制因子 | `CISDR.sentiment_mlp` |
| 去偏推理 | `CISDR.predict_debiased()` |

## 超参数说明

### 模型架构参数

- `review_embedding_dim`: 300（Word2Vec向量维度）
- `user_embedding_dim`: 64（用户嵌入维度）
- `item_embedding_dim`: 64（物品嵌入维度）
- `ncf_hidden_dims`: [128, 64, 32]（NCF隐藏层）
- `user_mlp_hidden_dims`: [256, 128, 64]（用户情感分支）
- `item_mlp_hidden_dims`: [256, 128, 64]（物品情感分支）
- `sentiment_mlp_hidden_dims`: [128, 64]（情感因子MLP）

### 训练参数

- `batch_size`: 256
- `learning_rate`: 0.002
- `weight_decay`: 1e-6
- `alpha_u`: 0.001（用户情感分支权重）
- `alpha_i`: 0.001（物品情感分支权重）

### 去偏参数

- `beta`: 0.1（可调范围 [1e-5, 0.5]）
  - 需要通过验证集上的BU和BI指标进行调优

## 扩展建议

1. **使用预训练词向量**：可以使用GloVe或Word2Vec预训练模型替代从头训练
2. **更复杂的文本编码器**：可以使用BERT等Transformer模型替代Word2Vec
3. **超参数搜索**：实现自动超参数搜索（如Grid Search或Bayesian Optimization）
4. **多GPU训练**：支持分布式训练以加速大规模数据集训练
5. **可视化工具**：添加TensorBoard支持，可视化训练过程和指标

## 注意事项

1. **数据格式**：确保输入数据包含必需字段（user_id, item_id, rating, review_text）
2. **内存使用**：大规模数据集可能需要较大的内存用于存储嵌入向量
3. **训练时间**：Word2Vec模型训练和嵌入向量生成需要一定时间
4. **beta参数调优**：需要根据具体数据集和任务调整beta值以获得最佳去偏效果

