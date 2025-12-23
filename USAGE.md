# 使用说明

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

数据文件应为JSON或CSV格式，包含以下字段：
- `user_id`: 用户ID（字符串或整数）
- `item_id`: 物品ID（字符串或整数）
- `rating`: 评分（1-5之间的浮点数或整数）
- `review_text`: 评论文本（字符串）

**示例数据格式（JSON）：**
```json
[
  {
    "user_id": "user_1",
    "item_id": "item_1",
    "rating": 5,
    "review_text": "Great product! Highly recommend."
  },
  ...
]
```

**使用示例数据：**
```bash
python generate_sample_data.py
```

这将生成一个示例数据集到 `data/sample_reviews.json`。

### 3. 训练模型

```bash
python train.py --data_path data/sample_reviews.json --epochs 50
```

**主要参数：**
- `--data_path`: 数据文件路径
- `--epochs`: 训练轮数（默认50）
- `--batch_size`: 批次大小（默认256）
- `--lr`: 学习率（默认0.002）
- `--alpha_u`: 用户情感分支损失权重（默认0.001）
- `--alpha_i`: 物品情感分支损失权重（默认0.001）

训练完成后，最佳模型会保存到 `checkpoints/best_model.pth`。

### 4. 评估模型

```bash
python evaluate.py --model_path checkpoints/best_model.pth --beta 0.1
```

**主要参数：**
- `--model_path`: 模型文件路径
- `--data_path`: 测试数据文件路径（默认与训练数据相同）
- `--beta`: 去偏参数beta（默认0.1，可调范围[1e-5, 0.5]）
- `--use_debiased`: 使用去偏推理（默认True）

评估结果会保存到 `checkpoints/best_model_results.json`。

## 模型架构说明

### 训练阶段

模型包含三个并行分支：

1. **NCF主干网络**：处理用户-物品交互特征 `q_m = f_m(h_u, h_i)`
2. **用户情感分支**：处理用户评论嵌入 `y_u = f_u(z_u)`
3. **物品情感分支**：处理物品评论嵌入 `y_i = f_i(z_i)`

融合预测：
- `y_{u,i} = q_m + y_u + y_i`
- `y_{u,i,s} = y_{u,i} * σ(s_{u,i})`，其中 `s_{u,i} = f_s(z_u * z_i)`

损失函数：
- `L = L_RC + α_u * L_U + α_i * L_I`

### 推理阶段（去偏）

使用反事实推理消除情感偏差：
- `y_debiased = (y_{u,i} - β) * σ(s_{u,i})`

其中 `β` 是可调超参数，用于控制去偏强度。

## 评估指标

- **MSE**: 均方误差，衡量整体评分预测准确度
- **BU (User Sentiment Bias)**: 用户情感偏差，积极用户与消极用户的MSE之差
- **BI (Item Sentiment Bias)**: 物品情感偏差，热门物品与冷门物品的MSE之差

BU和BI越小，说明模型对不同类型的用户和物品的推荐越公平。

## 调参建议

1. **beta参数**：用于控制去偏强度
   - 范围：[1e-5, 0.5]
   - 较小值：去偏效果较弱，但可能保持更好的预测性能
   - 较大值：去偏效果较强，但可能影响预测性能
   - 建议：从0.1开始，根据验证集上的BU和BI指标调整

2. **alpha_u和alpha_i**：控制情感分支的权重
   - 默认值：0.001
   - 增大：更强调情感分支的学习
   - 减小：更强调主交互分支的学习

3. **学习率**：默认0.002
   - 如果损失不收敛，可以尝试降低学习率
   - 如果训练太慢，可以尝试提高学习率

## 注意事项

1. 数据预处理需要一定时间，特别是Word2Vec模型训练和嵌入向量生成
2. 确保数据集足够大（建议至少5000条记录）以获得稳定的Word2Vec模型
3. 对于真实数据集，建议使用独立的训练集和测试集
4. GPU加速：如果有CUDA设备，代码会自动使用GPU训练

