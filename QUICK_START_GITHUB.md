# GitHub 快速部署指南

## 一键部署到GitHub

### 方法1：使用部署脚本（推荐）

```bash
# 给脚本添加执行权限（首次使用）
chmod +x deploy_to_github.sh

# 运行部署脚本
./deploy_to_github.sh "初始提交：CI-SDR项目"
```

### 方法2：手动部署

```bash
# 1. 初始化Git仓库（如果还没有）
git init

# 2. 添加所有文件
git add .

# 3. 提交
git commit -m "初始提交：CI-SDR情感偏差消除推荐系统"

# 4. 在GitHub上创建新仓库，然后添加远程仓库
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# 5. 推送到GitHub
git branch -M main
git push -u origin main
```

## 从GitHub克隆到GPU服务器

### 完整流程

```bash
# 1. 克隆仓库
git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
cd REPO_NAME

# 2. 创建Conda环境（推荐用于GPU环境）
conda create -n ci-sdr python=3.8 -y
conda activate ci-sdr

# 3. 安装PyTorch GPU版本（CUDA 11.8）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. 安装其他依赖
pip install -r requirements.txt

# 5. 验证GPU
python -c "import torch; print('GPU可用:', torch.cuda.is_available()); print('GPU名称:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# 6. 准备数据
python generate_sample_data.py

# 7. 训练模型（自动使用GPU）
python train.py --data_path data/sample_reviews.json --epochs 50
```

## GPU环境安装命令参考

### CUDA 11.8（推荐）

```bash
# Conda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### CUDA 12.1

```bash
# Conda
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### CPU版本（如果没有GPU）

```bash
pip install torch torchvision torchaudio
```

## 注意事项

1. **数据文件不会被上传**：`.gitignore` 已配置忽略 `data/` 目录
2. **模型文件不会被上传**：`checkpoints/` 目录也会被忽略
3. **自动GPU检测**：代码会自动检测并使用GPU（如果可用）
4. **查看GPU使用情况**：运行 `nvidia-smi` 监控GPU使用

## 更新代码

如果从GitHub克隆后需要更新：

```bash
git pull origin main
```

## 常见问题

### 问题：GitHub上缺少某些文件？

检查 `.gitignore` 文件，某些文件可能被设置为不跟踪。

### 问题：GPU不可用？

1. 确认安装了GPU版本的PyTorch
2. 检查CUDA版本：`nvcc --version`
3. 验证GPU：`nvidia-smi`

### 问题：内存不足？

减小批次大小：
```bash
python train.py --data_path data/your_data.json --batch_size 128
```

更多详细信息请参考：
- [GITHUB_SETUP.md](GITHUB_SETUP.md) - 完整部署指南
- [操作文档.md](操作文档.md) - 详细操作说明

