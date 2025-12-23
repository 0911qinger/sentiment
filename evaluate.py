"""
评估脚本：评估训练好的CI-SDR模型
"""
import torch
from torch.utils.data import DataLoader
import argparse
import os
import json

from config import Config
from data_loader.dataset import ReviewDataset
from models.ci_sdr import CISDR
from utils.metrics import evaluate_model


def main():
    parser = argparse.ArgumentParser(description='评估CI-SDR模型')
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型文件路径')
    parser.add_argument('--data_path', type=str, default=Config.DATA_PATH,
                        help='测试数据文件路径')
    parser.add_argument('--beta', type=float, default=Config.BETA,
                        help='去偏参数beta')
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE,
                        help='批次大小')
    parser.add_argument('--use_debiased', action='store_true', default=True,
                        help='是否使用去偏推理')
    
    args = parser.parse_args()
    
    device = Config.DEVICE
    print(f"使用设备: {device}")
    
    # 加载模型checkpoint
    print(f"加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # 加载训练数据集（用于获取Word2Vec模型和ID映射）
    print("加载训练数据集...")
    train_dataset = ReviewDataset(
        data_path=args.data_path,
        embedding_dim=Config.REVIEW_EMBEDDING_DIM,
        mode='train'
    )
    
    # 加载测试数据集
    print("加载测试数据集...")
    # 注意：实际应用中，测试集应该单独的文件
    # 这里为了演示，使用训练数据集进行评估（实际应该使用独立的测试集）
    # 测试集应该复用训练集的Word2Vec模型和ID映射
    test_dataset = train_dataset  # 实际使用时应该加载独立的测试集文件
    
    # 创建测试数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # 创建模型
    print("初始化模型...")
    num_users = checkpoint.get('num_users', train_dataset.get_num_users())
    num_items = checkpoint.get('num_items', train_dataset.get_num_items())
    
    model = CISDR(
        num_users=num_users,
        num_items=num_items,
        review_embedding_dim=Config.REVIEW_EMBEDDING_DIM,
        user_embedding_dim=Config.USER_EMBEDDING_DIM,
        item_embedding_dim=Config.ITEM_EMBEDDING_DIM,
        ncf_hidden_dims=Config.NCF_HIDDEN_DIMS,
        user_mlp_hidden_dims=Config.USER_MLP_HIDDEN_DIMS,
        item_mlp_hidden_dims=Config.ITEM_MLP_HIDDEN_DIMS,
        sentiment_mlp_hidden_dims=Config.SENTIMENT_MLP_HIDDEN_DIMS
    ).to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型已加载，epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"验证MSE: {checkpoint.get('val_mse', 'N/A'):.4f}")
    
    # 获取情感信息
    user_sentiment_dict, item_sentiment_dict = test_dataset.get_sentiment_info()
    
    # 评估模型
    print("\n开始评估...")
    if args.use_debiased:
        print(f"使用去偏推理，beta = {args.beta}")
    else:
        print("使用标准推理（不去偏）")
    
    metrics = evaluate_model(
        model, test_loader, device,
        user_sentiment_dict, item_sentiment_dict,
        beta=args.beta,
        use_debiased=args.use_debiased
    )
    
    # 打印结果
    print("\n评估结果:")
    print(f"MSE: {metrics['MSE']:.4f}")
    print(f"BU (User Sentiment Bias): {metrics['BU']:.4f}")
    print(f"BI (Item Sentiment Bias): {metrics['BI']:.4f}")
    
    # 保存结果
    results = {
        'model_path': args.model_path,
        'beta': args.beta,
        'use_debiased': args.use_debiased,
        'metrics': metrics
    }
    
    results_path = args.model_path.replace('.pth', '_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {results_path}")


if __name__ == '__main__':
    main()

