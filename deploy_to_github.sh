#!/bin/bash

# CI-SDR GitHub部署脚本
# 使用方法: ./deploy_to_github.sh "您的提交信息"

echo "================================"
echo "CI-SDR GitHub 部署脚本"
echo "================================"
echo ""

# 检查是否在Git仓库中
if [ ! -d ".git" ]; then
    echo "初始化Git仓库..."
    git init
    echo "✓ Git仓库初始化完成"
fi

# 检查是否有未提交的更改
if [ -n "$(git status --porcelain)" ]; then
    echo "发现未提交的更改"
    git status
    
    # 获取提交信息
    COMMIT_MSG=${1:-"Update CI-SDR project"}
    
    echo ""
    echo "准备提交更改: $COMMIT_MSG"
    read -p "继续? (y/n): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git add .
        git commit -m "$COMMIT_MSG"
        echo "✓ 更改已提交"
    else
        echo "取消提交"
        exit 1
    fi
else
    echo "✓ 没有未提交的更改"
fi

# 检查是否设置了远程仓库
if git remote get-url origin &> /dev/null; then
    REMOTE_URL=$(git remote get-url origin)
    echo "当前远程仓库: $REMOTE_URL"
else
    echo ""
    echo "未设置远程仓库。请先运行以下命令设置远程仓库："
    echo "  git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git"
    echo ""
    read -p "是否现在设置远程仓库? (y/n): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "请输入GitHub仓库URL: " REPO_URL
        git remote add origin "$REPO_URL"
        echo "✓ 远程仓库已设置"
    else
        echo "请稍后手动设置远程仓库"
        exit 1
    fi
fi

# 推送到GitHub
echo ""
echo "推送到GitHub..."
BRANCH=$(git branch --show-current)
if [ -z "$BRANCH" ]; then
    BRANCH="main"
    git branch -M main
fi

echo "推送到分支: $BRANCH"
read -p "继续? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    git push -u origin "$BRANCH"
    echo ""
    echo "================================"
    echo "✓ 部署完成！"
    echo "================================"
else
    echo "取消推送"
fi

