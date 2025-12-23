# GitHub 部署指南

本文档说明如何将CI-SDR项目上传到GitHub，以及如何在有GPU的环境中使用。

## 上传到GitHub

### 步骤1：初始化Git仓库

```bash
# 在项目根目录下
cd /Users/qing/PycharmProjects/sentiment_basis

# 初始化Git仓库
git init

# 添加所有文件
git add .

# 提交
git commit -m "Initial commit: CI-SDR sentiment bias elimination recommender system"
```

### 步骤2：创建GitHub仓库

1. 登录GitHub
2. 点击右上角的 "+" 按钮，选择 "New repository"
3. 填写仓库信息：
   - Repository name: `ci-sdr-recommender` (或您喜欢的名称)
   - Description: `Eliminating Sentiment Bias in Recommender Systems by Counterfactual Inference`
   - 选择 Public 或 Private
   - **不要**勾选 "Initialize this repository with a README"（因为我们已经有文件了）
4. 点击 "Create repository"

### 步骤3：连接本地仓库到GitHub

```bash
# 添加远程仓库（将YOUR_USERNAME和REPO_NAME替换为您的实际信息）
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# 或者使用SSH（如果您配置了SSH密钥）
# git remote add origin git@github.com:YOUR_USERNAME/REPO_NAME.git

# 查看远程仓库
git remote -v
```

### 步骤4：推送代码

```bash
# 推送到GitHub（首次推送）
git branch -M main
git push -u origin main
```

### 步骤5：后续更新

```bash
# 添加更改的文件
git add .

# 提交更改
git commit -m "描述您的更改"

# 推送到GitHub
git push
```

## 从GitHub克隆到GPU环境

### 步骤1：克隆仓库

```bash
# 使用HTTPS
git clone https://github.com/YOUR_USERNAME/REPO_NAME.git

# 或使用SSH
# git clone git@github.com:YOUR_USERNAME/REPO_NAME.git

# 进入项目目录
cd REPO_NAME
```

### 步骤2：设置GPU环境

#### 选项A：使用Conda（推荐）

```bash
# 创建Conda环境
conda create -n ci-sdr python=3.8 -y
conda activate ci-sdr

# 安装PyTorch（GPU版本）
# CUDA 11.8版本
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 或CUDA 12.1版本
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 安装其他依赖
pip install -r requirements.txt
```

#### 选项B：使用pip

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate   # Windows

# 安装PyTorch GPU版本
# 访问 https://pytorch.org/ 获取适合您CUDA版本的安装命令
# 例如 CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt
```

### 步骤3：验证GPU可用性

```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU数量: {torch.cuda.device_count()}'); print(f'GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

如果输出显示 `CUDA可用: True`，说明GPU环境配置成功！

### 步骤4：准备数据

```bash
# 使用示例数据
python generate_sample_data.py

# 或上传您自己的数据到 data/ 目录
```

### 步骤5：训练模型（自动使用GPU）

```bash
# 代码会自动检测并使用GPU（如果可用）
python train.py --data_path data/sample_reviews.json --epochs 50
```

## GPU使用说明

### 自动GPU检测

代码已经配置为自动检测和使用GPU。在 `config.py` 中：

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

这意味着：
- 如果有GPU，代码会自动使用GPU
- 如果没有GPU，代码会自动回退到CPU

### 手动指定设备

如果需要手动指定设备，可以修改 `config.py`：

```python
# 强制使用GPU
DEVICE = torch.device("cuda")

# 强制使用CPU
DEVICE = torch.device("cpu")

# 使用特定GPU（多GPU环境）
DEVICE = torch.device("cuda:0")  # 第一个GPU
DEVICE = torch.device("cuda:1")  # 第二个GPU
```

### GPU内存优化

如果遇到GPU内存不足的问题：

1. **减小批次大小**：
```bash
python train.py --data_path data/your_data.json --batch_size 128
```

2. **使用梯度累积**（需要修改代码）

3. **使用混合精度训练**（需要修改代码）

### 监控GPU使用情况

```bash
# 实时监控GPU使用情况
nvidia-smi -l 1

# 或在训练时监控
watch -n 1 nvidia-smi
```

## 常用Git命令

```bash
# 查看状态
git status

# 查看更改
git diff

# 查看提交历史
git log

# 创建新分支
git checkout -b feature/new-feature

# 切换分支
git checkout main

# 合并分支
git merge feature/new-feature

# 拉取最新代码
git pull

# 查看远程仓库
git remote -v
```

## 贡献指南

如果您想贡献代码：

1. Fork 仓库
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -m "Add your feature"`
4. 推送到分支：`git push origin feature/your-feature`
5. 创建 Pull Request

## 许可证

请根据您的需求添加许可证文件（如 MIT License, Apache License 2.0 等）。

## 问题反馈

如果遇到问题，请：
1. 检查本文档的常见问题部分
2. 查看 Issues 页面是否有类似问题
3. 创建新的 Issue 描述您的问题

