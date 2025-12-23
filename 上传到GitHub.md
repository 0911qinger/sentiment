# 直接上传到GitHub - 步骤指南

## 方式一：使用自动化脚本（最简单）

```bash
# 1. 运行部署脚本
./deploy_to_github.sh "初始提交：CI-SDR情感偏差消除推荐系统"

# 脚本会自动引导您完成所有步骤
```

## 方式二：手动上传（分步执行）

### 步骤1：检查当前状态

```bash
cd /Users/qing/PycharmProjects/sentiment_basis
git status
```

### 步骤2：初始化Git仓库（如果还没有）

```bash
git init
```

### 步骤3：添加所有文件

```bash
git add .
```

### 步骤4：提交代码

```bash
git commit -m "初始提交：CI-SDR基于反事实推理的情感偏差消除推荐系统"
```

### 步骤5：在GitHub上创建新仓库

1. 打开浏览器，访问 https://github.com
2. 点击右上角的 "+" 号，选择 "New repository"
3. 填写信息：
   - **Repository name**: `ci-sdr-recommender`（或您喜欢的名称）
   - **Description**: `Eliminating Sentiment Bias in Recommender Systems by Counterfactual Inference`
   - 选择 **Public** 或 **Private**
   - ⚠️ **不要**勾选 "Add a README file"（因为我们已经有了）
   - ⚠️ **不要**勾选 "Add .gitignore"（因为我们已经有了）
   - ⚠️ **不要**选择许可证（可稍后添加）
4. 点击绿色的 "Create repository" 按钮

### 步骤6：连接本地仓库到GitHub

复制GitHub提供的命令，通常是这样的格式：

```bash
# 替换YOUR_USERNAME和REPO_NAME为您的实际信息
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
```

例如：
```bash
git remote add origin https://github.com/yourusername/ci-sdr-recommender.git
```

### 步骤7：推送到GitHub

```bash
# 设置主分支为main（如果没有的话）
git branch -M main

# 推送到GitHub（首次推送）
git push -u origin main
```

### 步骤8：验证上传成功

1. 刷新GitHub仓库页面
2. 您应该能看到所有文件已经上传

## 完整命令序列（复制粘贴）

**⚠️ 请先替换 YOUR_USERNAME 和 REPO_NAME**

```bash
cd /Users/qing/PycharmProjects/sentiment_basis

# 初始化（如果还没有）
git init

# 添加文件
git add .

# 提交
git commit -m "初始提交：CI-SDR情感偏差消除推荐系统"

# 添加远程仓库（替换为您的GitHub仓库URL）
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git

# 推送到GitHub
git branch -M main
git push -u origin main
```

## 如果遇到问题

### 问题1：远程仓库已存在

如果提示 `remote origin already exists`，执行：

```bash
# 查看现有远程仓库
git remote -v

# 如果需要更换远程仓库
git remote set-url origin https://github.com/YOUR_USERNAME/REPO_NAME.git
```

### 问题2：需要身份验证

GitHub现在要求使用Personal Access Token：

1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token" → "Generate new token (classic)"
3. 设置权限：至少勾选 `repo`
4. 生成token并复制
5. 推送时，密码处输入token（不是GitHub密码）

### 问题3：推送被拒绝

如果提示 `rejected`，可能是因为远程有内容：

```bash
# 强制推送（谨慎使用，会覆盖远程内容）
git push -u origin main --force
```

### 问题4：分支名称不同

如果您的默认分支是 `master` 而不是 `main`：

```bash
git branch -M master
git push -u origin master
```

或者：

```bash
git branch -M main
git push -u origin main
```

## 验证上传

上传成功后，访问您的GitHub仓库页面，应该能看到：

- ✅ README.md
- ✅ 所有Python源代码文件
- ✅ requirements.txt
- ✅ 配置文件
- ✅ 文档文件

**注意**：以下内容不会被上传（已在.gitignore中）：
- data/ 目录中的实际数据文件
- checkpoints/ 目录中的模型文件
- __pycache__/ 缓存文件
- venv/ 虚拟环境

## 后续更新

以后如果修改了代码，只需要：

```bash
git add .
git commit -m "描述您的更改"
git push
```

## 需要帮助？

- 查看 [GITHUB_SETUP.md](GITHUB_SETUP.md) 获取详细说明
- 查看 [QUICK_START_GITHUB.md](QUICK_START_GITHUB.md) 获取快速指南

