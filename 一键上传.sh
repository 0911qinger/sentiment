#!/bin/bash

# CI-SDR 一键上传到GitHub脚本
# 使用方法: ./一键上传.sh

echo "=========================================="
echo "  CI-SDR 项目 - GitHub 上传工具"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否在项目目录
if [ ! -f "config.py" ]; then
    echo -e "${RED}错误：请在项目根目录运行此脚本${NC}"
    exit 1
fi

# 步骤1：初始化Git仓库
echo -e "${YELLOW}[1/6]${NC} 初始化Git仓库..."
if [ ! -d ".git" ]; then
    git init
    echo -e "${GREEN}✓ Git仓库初始化完成${NC}"
else
    echo -e "${GREEN}✓ Git仓库已存在${NC}"
fi
echo ""

# 步骤2：添加所有文件
echo -e "${YELLOW}[2/6]${NC} 添加文件到Git..."
git add .
echo -e "${GREEN}✓ 文件已添加${NC}"
echo ""

# 步骤3：检查是否有更改
if [ -z "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}没有需要提交的更改${NC}"
    echo ""
else
    # 步骤4：提交
    echo -e "${YELLOW}[3/6]${NC} 提交更改..."
    COMMIT_MSG="初始提交：CI-SDR基于反事实推理的情感偏差消除推荐系统"
    git commit -m "$COMMIT_MSG"
    echo -e "${GREEN}✓ 更改已提交${NC}"
    echo ""
fi

# 步骤5：检查远程仓库
echo -e "${YELLOW}[4/6]${NC} 检查远程仓库配置..."
if git remote get-url origin &> /dev/null; then
    REMOTE_URL=$(git remote get-url origin)
    echo -e "${GREEN}✓ 远程仓库已配置: ${REMOTE_URL}${NC}"
    echo ""
    read -p "是否使用现有远程仓库? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        git remote remove origin
        REMOTE_URL=""
    fi
else
    REMOTE_URL=""
fi

# 步骤6：配置远程仓库
if [ -z "$REMOTE_URL" ]; then
    echo -e "${YELLOW}[5/6]${NC} 配置远程仓库..."
    echo ""
    echo "请提供您的GitHub仓库URL"
    echo "格式示例: https://github.com/yourusername/repo-name.git"
    echo ""
    read -p "请输入GitHub仓库URL: " REPO_URL
    
    if [ -z "$REPO_URL" ]; then
        echo -e "${RED}错误：未提供仓库URL${NC}"
        exit 1
    fi
    
    git remote add origin "$REPO_URL"
    echo -e "${GREEN}✓ 远程仓库已配置${NC}"
    echo ""
else
    echo -e "${YELLOW}[5/6]${NC} 使用现有远程仓库"
    echo ""
fi

# 步骤7：推送到GitHub
echo -e "${YELLOW}[6/6]${NC} 推送到GitHub..."
echo ""

# 确定分支名称
CURRENT_BRANCH=$(git branch --show-current 2>/dev/null || echo "main")
if [ -z "$CURRENT_BRANCH" ]; then
    git branch -M main
    CURRENT_BRANCH="main"
fi

echo "准备推送到分支: $CURRENT_BRANCH"
echo ""
read -p "继续推送? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "正在推送到GitHub..."
    git push -u origin "$CURRENT_BRANCH"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}=========================================="
        echo -e "  ✓ 上传成功！"
        echo -e "==========================================${NC}"
        echo ""
        echo "您的代码已成功上传到GitHub"
        echo "访问仓库查看: $REPO_URL"
    else
        echo ""
        echo -e "${RED}推送失败，请检查：${NC}"
        echo "1. GitHub仓库URL是否正确"
        echo "2. 是否已配置身份验证（Personal Access Token）"
        echo "3. 是否有推送权限"
        echo ""
        echo "如需帮助，请查看 上传到GitHub.md 文档"
    fi
else
    echo "已取消推送"
fi

