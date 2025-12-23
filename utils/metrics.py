"""
评估指标：MSE, BU (User Sentiment Bias), BI (Item Sentiment Bias)
"""
import numpy as np
import torch
from typing import Dict, List, Tuple


def compute_mse(predictions: np.ndarray, targets: np.ndarray) -> float:
    """
    计算均方误差 (MSE)
    
    Args:
        predictions: 预测值
        targets: 真实值
        
    Returns:
        MSE值
    """
    return np.mean((predictions - targets) ** 2)


def compute_bu_bi(predictions: np.ndarray,
                  targets: np.ndarray,
                  user_sentiment_dict: Dict[int, float],
                  item_sentiment_dict: Dict[int, float],
                  user_indices: np.ndarray,
                  item_indices: np.ndarray) -> Tuple[float, float]:
    """
    计算用户情感偏差 (BU) 和物品情感偏差 (BI)
    
    BU: 积极用户与消极用户的MSE之差
    BI: 热门物品与冷门物品的MSE之差
    
    Args:
        predictions: 预测值 [n_samples]
        targets: 真实值 [n_samples]
        user_sentiment_dict: {user_idx: sentiment_score}
        item_sentiment_dict: {item_idx: sentiment_score}
        user_indices: 用户索引 [n_samples]
        item_indices: 物品索引 [n_samples]
        
    Returns:
        (BU, BI): 用户情感偏差和物品情感偏差
    """
    # 获取每个样本对应的用户和物品情感
    user_sentiments = np.array([user_sentiment_dict.get(uid, 0.0) for uid in user_indices])
    item_sentiments = np.array([item_sentiment_dict.get(iid, 0.0) for iid in item_indices])
    
    # 计算BU：按用户情感分组
    # 前10%为积极用户，后10%为消极用户
    user_sentiment_threshold_high = np.percentile(user_sentiments, 90)
    user_sentiment_threshold_low = np.percentile(user_sentiments, 10)
    
    positive_user_mask = user_sentiments >= user_sentiment_threshold_high
    negative_user_mask = user_sentiments <= user_sentiment_threshold_low
    
    if np.sum(positive_user_mask) > 0 and np.sum(negative_user_mask) > 0:
        positive_user_mse = compute_mse(predictions[positive_user_mask], targets[positive_user_mask])
        negative_user_mse = compute_mse(predictions[negative_user_mask], targets[negative_user_mask])
        BU = abs(positive_user_mse - negative_user_mse)
    else:
        BU = 0.0
    
    # 计算BI：按物品情感分组
    # 前10%为热门物品，后10%为冷门物品
    item_sentiment_threshold_high = np.percentile(item_sentiments, 90)
    item_sentiment_threshold_low = np.percentile(item_sentiments, 10)
    
    positive_item_mask = item_sentiments >= item_sentiment_threshold_high
    negative_item_mask = item_sentiments <= item_sentiment_threshold_low
    
    if np.sum(positive_item_mask) > 0 and np.sum(negative_item_mask) > 0:
        positive_item_mse = compute_mse(predictions[positive_item_mask], targets[positive_item_mask])
        negative_item_mse = compute_mse(predictions[negative_item_mask], targets[negative_item_mask])
        BI = abs(positive_item_mse - negative_item_mse)
    else:
        BI = 0.0
    
    return BU, BI


def evaluate_model(model: torch.nn.Module,
                  dataloader: torch.utils.data.DataLoader,
                  device: torch.device,
                  user_sentiment_dict: Dict,
                  item_sentiment_dict: Dict,
                  beta: float = 0.1,
                  use_debiased: bool = True) -> Dict[str, float]:
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 设备
        user_sentiment_dict: 用户情感字典
        item_sentiment_dict: 物品情感字典
        beta: 去偏参数（仅在use_debiased=True时使用）
        use_debiased: 是否使用去偏推理
        
    Returns:
        评估指标字典：{'MSE', 'BU', 'BI'}
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_user_indices = []
    all_item_indices = []
    
    with torch.no_grad():
        for batch in dataloader:
            user_idx = batch['user_idx'].to(device)
            item_idx = batch['item_idx'].to(device)
            rating = batch['rating'].to(device)
            user_embedding = batch['user_embedding'].to(device)
            item_embedding = batch['item_embedding'].to(device)
            
            if use_debiased:
                # 使用去偏推理
                predictions = model.predict_debiased(
                    user_idx, item_idx, user_embedding, item_embedding, beta
                )
            else:
                # 使用标准推理
                predictions, _, _, _ = model(
                    user_idx, item_idx, user_embedding, item_embedding
                )
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(rating.cpu().numpy())
            all_user_indices.append(user_idx.cpu().numpy())
            all_item_indices.append(item_idx.cpu().numpy())
    
    # 合并所有批次的结果
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    all_user_indices = np.concatenate(all_user_indices)
    all_item_indices = np.concatenate(all_item_indices)
    
    # 计算指标
    mse = compute_mse(all_predictions, all_targets)
    BU, BI = compute_bu_bi(
        all_predictions, all_targets,
        user_sentiment_dict, item_sentiment_dict,
        all_user_indices, all_item_indices
    )
    
    return {
        'MSE': mse,
        'BU': BU,
        'BI': BI
    }

