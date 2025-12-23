"""
配置文件：定义模型和训练的超参数
"""
import torch

class Config:
    """配置类"""
    
    # 数据相关
    DATA_PATH = "data/reviews.json"
    MIN_REVIEWS_PER_USER = 5
    MIN_REVIEWS_PER_ITEM = 5
    
    # 文本嵌入相关
    WORD2VEC_DIM = 300
    REVIEW_EMBEDDING_DIM = 300
    MAX_REVIEW_LENGTH = 500
    
    # 模型架构相关
    USER_EMBEDDING_DIM = 64
    ITEM_EMBEDDING_DIM = 64
    NCF_HIDDEN_DIMS = [128, 64, 32]  # NCF的隐藏层维度
    USER_MLP_HIDDEN_DIMS = [256, 128, 64]  # 用户情感分支MLP
    ITEM_MLP_HIDDEN_DIMS = [256, 128, 64]  # 物品情感分支MLP
    SENTIMENT_MLP_HIDDEN_DIMS = [128, 64]  # 情感因子MLP
    
    # 训练相关
    BATCH_SIZE = 256
    LEARNING_RATE = 0.002
    WEIGHT_DECAY = 1e-6
    NUM_EPOCHS = 50
    ALPHA_U = 0.001  # 用户情感分支损失权重
    ALPHA_I = 0.001  # 物品情感分支损失权重
    
    # 去偏推理相关
    BETA = 0.1  # 反事实推理的beta参数，可调范围[1e-5, 0.5]
    
    # 设备
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 路径
    MODEL_SAVE_DIR = "checkpoints"
    LOG_DIR = "logs"

