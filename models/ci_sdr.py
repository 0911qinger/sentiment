"""
CI-SDR模型：基于反事实推理的情感偏差消除推荐系统
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class NCF(nn.Module):
    """
    Neural Collaborative Filtering (NCF) 主干网络
    用于处理用户-物品交互特征
    """
    
    def __init__(self, 
                 num_users: int,
                 num_items: int,
                 user_embedding_dim: int = 64,
                 item_embedding_dim: int = 64,
                 hidden_dims: list = [128, 64, 32]):
        """
        Args:
            num_users: 用户数量
            num_items: 物品数量
            user_embedding_dim: 用户嵌入维度
            item_embedding_dim: 物品嵌入维度
            hidden_dims: 隐藏层维度列表
        """
        super(NCF, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, user_embedding_dim)
        self.item_embedding = nn.Embedding(num_items, item_embedding_dim)
        
        # 构建MLP层
        input_dim = user_embedding_dim + item_embedding_dim
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        # 输出层（输出标量）
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)
        
        # 初始化嵌入层
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
    
    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            user_idx: 用户索引 [batch_size]
            item_idx: 物品索引 [batch_size]
            
        Returns:
            交互特征 q_m [batch_size, 1]
        """
        user_emb = self.user_embedding(user_idx)  # [batch_size, user_embedding_dim]
        item_emb = self.item_embedding(item_idx)  # [batch_size, item_embedding_dim]
        
        # 拼接用户和物品嵌入
        interaction = torch.cat([user_emb, item_emb], dim=1)  # [batch_size, user_emb_dim + item_emb_dim]
        
        # 通过MLP
        q_m = self.mlp(interaction)  # [batch_size, 1]
        
        return q_m


class SentimentMLP(nn.Module):
    """
    情感分支MLP：处理评论嵌入向量，输出辅助评分
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dims: list = [256, 128, 64],
                 output_dim: int = 1):
        """
        Args:
            input_dim: 输入维度（评论嵌入维度）
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度（标量评分）
        """
        super(SentimentMLP, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            current_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(current_dim, output_dim))
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入向量 [batch_size, input_dim]
            
        Returns:
            预测评分 [batch_size, 1]
        """
        return self.mlp(x)


class CISDR(nn.Module):
    """
    CI-SDR模型：基于反事实推理的情感偏差消除推荐系统
    
    架构：
    1. NCF主干网络：提取用户-物品交互特征 q_m
    2. 用户情感分支：处理用户评论嵌入 z_u，输出 y_u
    3. 物品情感分支：处理物品评论嵌入 z_i，输出 y_i
    4. 情感控制因子：s_{u,i} = f_s(z_u * z_i)
    5. 融合预测：y_{u,i,s} = (q_m + y_u + y_i) * σ(s_{u,i})
    """
    
    def __init__(self,
                 num_users: int,
                 num_items: int,
                 review_embedding_dim: int = 300,
                 user_embedding_dim: int = 64,
                 item_embedding_dim: int = 64,
                 ncf_hidden_dims: list = [128, 64, 32],
                 user_mlp_hidden_dims: list = [256, 128, 64],
                 item_mlp_hidden_dims: list = [256, 128, 64],
                 sentiment_mlp_hidden_dims: list = [128, 64]):
        """
        Args:
            num_users: 用户数量
            num_items: 物品数量
            review_embedding_dim: 评论嵌入维度
            user_embedding_dim: 用户嵌入维度
            item_embedding_dim: 物品嵌入维度
            ncf_hidden_dims: NCF隐藏层维度
            user_mlp_hidden_dims: 用户情感分支MLP隐藏层维度
            item_mlp_hidden_dims: 物品情感分支MLP隐藏层维度
            sentiment_mlp_hidden_dims: 情感因子MLP隐藏层维度
        """
        super(CISDR, self).__init__()
        
        # NCF主干网络
        self.ncf = NCF(
            num_users=num_users,
            num_items=num_items,
            user_embedding_dim=user_embedding_dim,
            item_embedding_dim=item_embedding_dim,
            hidden_dims=ncf_hidden_dims
        )
        
        # 用户情感分支
        self.user_sentiment_mlp = SentimentMLP(
            input_dim=review_embedding_dim,
            hidden_dims=user_mlp_hidden_dims,
            output_dim=1
        )
        
        # 物品情感分支
        self.item_sentiment_mlp = SentimentMLP(
            input_dim=review_embedding_dim,
            hidden_dims=item_mlp_hidden_dims,
            output_dim=1
        )
        
        # 情感控制因子MLP：处理 z_u * z_i
        sentiment_input_dim = review_embedding_dim  # z_u * z_i 的维度
        sentiment_layers = []
        current_dim = sentiment_input_dim
        
        for hidden_dim in sentiment_mlp_hidden_dims:
            sentiment_layers.append(nn.Linear(current_dim, hidden_dim))
            sentiment_layers.append(nn.ReLU())
            sentiment_layers.append(nn.Dropout(0.2))
            current_dim = hidden_dim
        
        sentiment_layers.append(nn.Linear(current_dim, 1))
        self.sentiment_mlp = nn.Sequential(*sentiment_layers)
    
    def forward(self, 
                user_idx: torch.Tensor,
                item_idx: torch.Tensor,
                user_embedding: torch.Tensor,
                item_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            user_idx: 用户索引 [batch_size]
            item_idx: 物品索引 [batch_size]
            user_embedding: 用户评论嵌入 z_u [batch_size, review_embedding_dim]
            item_embedding: 物品评论嵌入 z_i [batch_size, review_embedding_dim]
            
        Returns:
            y_pred: 预测评分 [batch_size]
            y_u: 用户情感分支输出 [batch_size]
            y_i: 物品情感分支输出 [batch_size]
            s_ui: 情感控制因子（未经过sigmoid）[batch_size]
        """
        batch_size = user_idx.size(0)
        
        # 1. NCF主干网络：q_m
        q_m = self.ncf(user_idx, item_idx).squeeze(-1)  # [batch_size]
        
        # 2. 用户情感分支：y_u
        y_u = self.user_sentiment_mlp(user_embedding).squeeze(-1)  # [batch_size]
        
        # 3. 物品情感分支：y_i
        y_i = self.item_sentiment_mlp(item_embedding).squeeze(-1)  # [batch_size]
        
        # 4. 融合：y_{u,i} = q_m + y_u + y_i
        y_ui = q_m + y_u + y_i  # [batch_size]
        
        # 5. 情感控制因子：s_{u,i} = f_s(z_u * z_i)
        # 使用逐元素乘积作为交互特征
        interaction_feature = user_embedding * item_embedding  # [batch_size, review_embedding_dim]
        s_ui = self.sentiment_mlp(interaction_feature).squeeze(-1)  # [batch_size]
        
        # 6. 最终预测：y_{u,i,s} = y_{u,i} * σ(s_{u,i})
        s_ui_sigmoid = torch.sigmoid(s_ui)  # [batch_size]
        y_pred = y_ui * s_ui_sigmoid  # [batch_size]
        
        return y_pred, y_u, y_i, s_ui
    
    def predict_debiased(self,
                        user_idx: torch.Tensor,
                        item_idx: torch.Tensor,
                        user_embedding: torch.Tensor,
                        item_embedding: torch.Tensor,
                        beta: float = 0.1) -> torch.Tensor:
        """
        去偏推理：应用反事实推理消除情感偏差
        
        公式：y_debiased = (y_{u,i} - β) * σ(s_{u,i})
        
        Args:
            user_idx: 用户索引 [batch_size]
            item_idx: 物品索引 [batch_size]
            user_embedding: 用户评论嵌入 [batch_size, review_embedding_dim]
            item_embedding: 物品评论嵌入 [batch_size, review_embedding_dim]
            beta: 反事实推理参数
            
        Returns:
            去偏后的预测评分 [batch_size]
        """
        batch_size = user_idx.size(0)
        
        # 获取各个组件
        q_m = self.ncf(user_idx, item_idx).squeeze(-1)  # [batch_size]
        y_u = self.user_sentiment_mlp(user_embedding).squeeze(-1)  # [batch_size]
        y_i = self.item_sentiment_mlp(item_embedding).squeeze(-1)  # [batch_size]
        
        # y_{u,i} = q_m + y_u + y_i
        y_ui = q_m + y_u + y_i  # [batch_size]
        
        # 计算情感控制因子
        interaction_feature = user_embedding * item_embedding
        s_ui = self.sentiment_mlp(interaction_feature).squeeze(-1)
        s_ui_sigmoid = torch.sigmoid(s_ui)  # [batch_size]
        
        # 去偏公式：y_debiased = (y_{u,i} - β) * σ(s_{u,i})
        y_debiased = (y_ui - beta) * s_ui_sigmoid
        
        return y_debiased

