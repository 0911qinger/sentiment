# 个人尝试


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org/)

## 特性

- ✅ 基于反事实推理的情感偏差消除
- ✅ 自动GPU/CPU检测和使用
- ✅ 完整的多任务学习框架
- ✅ 支持大规模数据集
- ✅ 详细的评估指标（MSE, BU, BI）

## 项目结构

```
sentiment_basis/
├── data/                    # 数据目录（需要用户放置数据集）
├── models/                  # 模型定义
│   └── ci_sdr.py
├── data_loader/             # 数据处理
│   └── dataset.py
├── utils/                   # 工具函数
│   ├── embeddings.py
│   └── metrics.py
├── train.py                 # 训练脚本
├── evaluate.py              # 评估脚本
├── config.py                # 配置文件
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

生成示例数据（用于快速测试）：
```bash
python generate_sample_data.py
```

或使用您自己的数据集（JSON/CSV格式，包含 user_id, item_id, rating, review_text 字段）

### 3. 训练模型

```bash
python train.py --data_path data/sample_reviews.json --epochs 50
```

### 4. 评估模型

```bash
python evaluate.py --model_path checkpoints/best_model.pth --beta 0.1
```

## 文档说明

- **[操作文档.md](操作文档.md)**: 详细的操作指南，包含完整步骤、参数说明、常见问题等
- **[快速参考.md](快速参考.md)**: 常用命令速查表
- **[USAGE.md](USAGE.md)**: 使用说明和架构介绍
- **[IMPLEMENTATION.md](IMPLEMENTATION.md)**: 实现细节和技术说明

## 配置参数

在`config.py`中可以调整超参数：
- 学习率：0.002
- 权重衰减：1e-6
- 批次大小：256
- beta范围：[1e-5, 0.5]
- 情感分支权重：alpha_u=0.001, alpha_i=0.001

更多参数说明请参考 `操作文档.md`

## GPU支持

本项目支持GPU加速训练，会自动检测并使用可用的GPU设备。

### 安装GPU版本的PyTorch

**使用Conda（推荐）：**
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

**使用pip：**
```bash
# 访问 https://pytorch.org/ 获取适合您CUDA版本的命令
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

验证GPU可用性：
```bash
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
```

详细说明请参考 [GITHUB_SETUP.md](GITHUB_SETUP.md)

## GitHub部署

要将项目上传到GitHub或在GPU环境中使用，请参考：
- **[QUICK_START_GITHUB.md](QUICK_START_GITHUB.md)**: 快速部署指南（推荐新手）
- **[GITHUB_SETUP.md](GITHUB_SETUP.md)**: 详细的GitHub上传和GPU环境配置指南

### 快速上 GitHub

```bash
# 使用部署脚本（最简单）
./deploy_to_github.sh "初始提交"
```

### 从GitHub克隆到GPU服务器

```bash
git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
cd REPO_NAME
conda create -n ci-sdr python=3.8 -y
conda activate ci-sdr
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
python train.py --data_path data/sample_reviews.json
```

