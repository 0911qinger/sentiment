"""
训练脚本：训练CI-SDR模型
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from tqdm import tqdm
import os
import argparse
from pathlib import Path
from typing import Dict, Tuple

from config import Config
from data_loader.dataset import ReviewDataset
from models.ci_sdr import CISDR
from utils.metrics import evaluate_model


class MultiTaskLoss(nn.Module):
    """
    多任务损失函数
    
    L = L_RC + α_u * L_U + α_i * L_I
    
    其中：
    - L_RC: 主任务损失（评分预测）
    - L_U: 用户情感分支损失
    - L_I: 物品情感分支损失
    """
    
    def __init__(self, alpha_u: float = 0.001, alpha_i: float = 0.001):
        """
        Args:
            alpha_u: 用户情感分支损失权重
            alpha_i: 物品情感分支损失权重
        """
        super(MultiTaskLoss, self).__init__()
        self.alpha_u = alpha_u
        self.alpha_i = alpha_i
        self.mse_loss = nn.MSELoss()
    
    def forward(self, 
                y_pred: torch.Tensor,
                y_u: torch.Tensor,
                y_i: torch.Tensor,
                y_true: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算多任务损失
        
        Args:
            y_pred: 主预测评分 [batch_size]
            y_u: 用户情感分支输出 [batch_size]
            y_i: 物品情感分支输出 [batch_size]
            y_true: 真实评分 [batch_size]
            
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        # 主任务损失：L_RC
        L_RC = self.mse_loss(y_pred, y_true)
        
        # 用户情感分支损失：L_U（辅助任务，也用真实评分作为目标）
        L_U = self.mse_loss(y_u, y_true)
        
        # 物品情感分支损失：L_I（辅助任务，也用真实评分作为目标）
        L_I = self.mse_loss(y_i, y_true)
        
        # 总损失
        total_loss = L_RC + self.alpha_u * L_U + self.alpha_i * L_I
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'L_RC': L_RC.item(),
            'L_U': L_U.item(),
            'L_I': L_I.item()
        }
        
        return total_loss, loss_dict


def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device) -> Dict[str, float]:
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    loss_history = {'total_loss': [], 'L_RC': [], 'L_U': [], 'L_I': []}
    
    for batch in tqdm(dataloader, desc="Training"):
        user_idx = batch['user_idx'].to(device)
        item_idx = batch['item_idx'].to(device)
        rating = batch['rating'].to(device)
        user_embedding = batch['user_embedding'].to(device)
        item_embedding = batch['item_embedding'].to(device)
        
        # 前向传播
        y_pred, y_u, y_i, _ = model(user_idx, item_idx, user_embedding, item_embedding)
        
        # 计算损失
        loss, loss_dict = criterion(y_pred, y_u, y_i, rating)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录损失
        total_loss += loss.item()
        for key in loss_history:
            if key in loss_dict:
                loss_history[key].append(loss_dict[key])
    
    # 计算平均损失
    avg_losses = {key: np.mean(values) for key, values in loss_history.items()}
    avg_losses['total_loss'] = total_loss / len(dataloader)
    
    return avg_losses


def main():
    parser = argparse.ArgumentParser(description='训练CI-SDR模型')
    parser.add_argument('--data_path', type=str, default=Config.DATA_PATH,
                        help='数据文件路径')
    parser.add_argument('--epochs', type=int, default=Config.NUM_EPOCHS,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=Config.WEIGHT_DECAY,
                        help='权重衰减')
    parser.add_argument('--alpha_u', type=float, default=Config.ALPHA_U,
                        help='用户情感分支损失权重')
    parser.add_argument('--alpha_i', type=float, default=Config.ALPHA_I,
                        help='物品情感分支损失权重')
    parser.add_argument('--save_dir', type=str, default=Config.MODEL_SAVE_DIR,
                        help='模型保存目录')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = Config.DEVICE
    print(f"使用设备: {device}")
    
    # 加载数据集
    print("加载数据集...")
    dataset = ReviewDataset(
        data_path=args.data_path,
        embedding_dim=Config.REVIEW_EMBEDDING_DIM,
        mode='train'
    )
    
    # 划分训练集和验证集（80-20）
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 创建模型
    print("初始化模型...")
    model = CISDR(
        num_users=dataset.get_num_users(),
        num_items=dataset.get_num_items(),
        review_embedding_dim=Config.REVIEW_EMBEDDING_DIM,
        user_embedding_dim=Config.USER_EMBEDDING_DIM,
        item_embedding_dim=Config.ITEM_EMBEDDING_DIM,
        ncf_hidden_dims=Config.NCF_HIDDEN_DIMS,
        user_mlp_hidden_dims=Config.USER_MLP_HIDDEN_DIMS,
        item_mlp_hidden_dims=Config.ITEM_MLP_HIDDEN_DIMS,
        sentiment_mlp_hidden_dims=Config.SENTIMENT_MLP_HIDDEN_DIMS
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 损失函数和优化器
    criterion = MultiTaskLoss(alpha_u=args.alpha_u, alpha_i=args.alpha_i)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 获取情感信息（用于评估）
    user_sentiment_dict, item_sentiment_dict = dataset.get_sentiment_info()
    
    # 训练循环
    best_val_mse = float('inf')
    best_model_path = None
    
    print("\n开始训练...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # 训练
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"训练损失 - 总计: {train_losses['total_loss']:.4f}, "
              f"L_RC: {train_losses['L_RC']:.4f}, "
              f"L_U: {train_losses['L_U']:.4f}, "
              f"L_I: {train_losses['L_I']:.4f}")
        
        # 验证
        val_metrics = evaluate_model(
            model, val_loader, device,
            user_sentiment_dict, item_sentiment_dict,
            beta=Config.BETA, use_debiased=True
        )
        print(f"验证指标 - MSE: {val_metrics['MSE']:.4f}, "
              f"BU: {val_metrics['BU']:.4f}, "
              f"BI: {val_metrics['BI']:.4f}")
        
        # 保存最佳模型
        if val_metrics['MSE'] < best_val_mse:
            best_val_mse = val_metrics['MSE']
            best_model_path = os.path.join(args.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mse': best_val_mse,
                'config': Config.__dict__,
                'num_users': dataset.get_num_users(),
                'num_items': dataset.get_num_items(),
            }, best_model_path)
            print(f"保存最佳模型 (MSE: {best_val_mse:.4f})")
    
    print(f"\n训练完成！最佳模型保存在: {best_model_path}")


if __name__ == '__main__':
    main()

