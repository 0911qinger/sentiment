"""
数据集加载模块：处理Amazon/Yelp数据集
"""
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from collections import defaultdict
from utils.embeddings import build_word2vec_model, text_to_embedding, compute_sentiment_polarity
import os


class ReviewDataset(Dataset):
    """
    评论文本推荐系统数据集
    
    数据格式：包含user_id, item_id, rating, review_text字段的JSON或CSV
    """
    
    def __init__(self, 
                 data_path: str,
                 word2vec_model=None,
                 embedding_dim: int = 300,
                 mode: str = 'train',
                 user_reviews_dict: Dict = None,
                 item_reviews_dict: Dict = None,
                 user_sentiment_dict: Dict = None,
                 item_sentiment_dict: Dict = None):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            word2vec_model: 预训练的Word2Vec模型
            embedding_dim: 嵌入向量维度
            mode: 'train' 或 'test'
            user_reviews_dict: 用户评论字典（用于测试集）
            item_reviews_dict: 物品评论字典（用于测试集）
            user_sentiment_dict: 用户情感字典（用于测试集）
            item_sentiment_dict: 物品情感字典（用于测试集）
        """
        self.embedding_dim = embedding_dim
        self.word2vec_model = word2vec_model
        self.mode = mode
        
        # 加载数据
        self.data = self._load_data(data_path)
        
        # 构建用户和物品的ID映射
        self.user_to_idx, self.item_to_idx = self._build_id_mapping()
        self.idx_to_user = {v: k for k, v in self.user_to_idx.items()}
        self.idx_to_item = {v: k for k, v in self.item_to_idx.items()}
        
        # 构建用户和物品的评论聚合（训练阶段）
        if mode == 'train':
            self.user_reviews_dict, self.item_reviews_dict = self._aggregate_reviews()
            self.user_sentiment_dict, self.item_sentiment_dict = self._compute_sentiments()
        else:
            # 测试集使用传入的字典
            self.user_reviews_dict = user_reviews_dict
            self.item_reviews_dict = item_reviews_dict
            self.user_sentiment_dict = user_sentiment_dict
            self.item_sentiment_dict = item_sentiment_dict
        
        # 构建评论嵌入向量
        if mode == 'train' and word2vec_model is None:
            # 训练阶段：需要先训练Word2Vec模型
            all_reviews = list(self.user_reviews_dict.values()) + list(self.item_reviews_dict.values())
            print("正在训练Word2Vec模型...")
            self.word2vec_model = build_word2vec_model(all_reviews, embedding_dim)
            print("Word2Vec模型训练完成")
        else:
            self.word2vec_model = word2vec_model
        
        # 预计算嵌入向量
        print("正在生成评论嵌入向量...")
        self.user_embeddings = self._compute_embeddings(self.user_reviews_dict)
        self.item_embeddings = self._compute_embeddings(self.item_reviews_dict)
        print("嵌入向量生成完成")
        
        # 转换数据为索引格式
        self.samples = self._convert_to_indices()
        
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """加载数据文件"""
        if data_path.endswith('.json'):
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"不支持的文件格式: {data_path}")
        
        # 确保必要字段存在
        required_fields = ['user_id', 'item_id', 'rating', 'review_text']
        for field in required_fields:
            if field not in df.columns:
                raise ValueError(f"数据文件缺少必要字段: {field}")
        
        return df
    
    def _build_id_mapping(self) -> Tuple[Dict, Dict]:
        """构建用户和物品的ID到索引的映射"""
        unique_users = sorted(self.data['user_id'].unique())
        unique_items = sorted(self.data['item_id'].unique())
        
        user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
        item_to_idx = {item_id: idx for idx, item_id in enumerate(unique_items)}
        
        return user_to_idx, item_to_idx
    
    def _aggregate_reviews(self) -> Tuple[Dict[str, str], Dict[str, str]]:
        """
        聚合用户和物品的评论
        
        Returns:
            user_reviews_dict: {user_id: aggregated_review_text}
            item_reviews_dict: {item_id: aggregated_review_text}
        """
        user_reviews = defaultdict(list)
        item_reviews = defaultdict(list)
        
        for _, row in self.data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            review_text = str(row['review_text']) if pd.notna(row['review_text']) else ""
            
            user_reviews[user_id].append(review_text)
            item_reviews[item_id].append(review_text)
        
        # 拼接评论
        user_reviews_dict = {
            user_id: " ".join(reviews) 
            for user_id, reviews in user_reviews.items()
        }
        item_reviews_dict = {
            item_id: " ".join(reviews) 
            for item_id, reviews in item_reviews.items()
        }
        
        return user_reviews_dict, item_reviews_dict
    
    def _compute_sentiments(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        计算用户和物品的平均情感极性
        
        Returns:
            user_sentiment_dict: {user_id: average_sentiment}
            item_sentiment_dict: {item_id: average_sentiment}
        """
        user_sentiments = defaultdict(list)
        item_sentiments = defaultdict(list)
        
        for _, row in self.data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            review_text = str(row['review_text']) if pd.notna(row['review_text']) else ""
            
            sentiment = compute_sentiment_polarity(review_text)
            user_sentiments[user_id].append(sentiment)
            item_sentiments[item_id].append(sentiment)
        
        user_sentiment_dict = {
            user_id: np.mean(sentiments)
            for user_id, sentiments in user_sentiments.items()
        }
        item_sentiment_dict = {
            item_id: np.mean(sentiments)
            for item_id, sentiments in item_sentiments.items()
        }
        
        return user_sentiment_dict, item_sentiment_dict
    
    def _compute_embeddings(self, reviews_dict: Dict[str, str]) -> Dict[int, np.ndarray]:
        """
        计算评论嵌入向量
        
        Args:
            reviews_dict: {id: review_text}
            
        Returns:
            {idx: embedding_vector}
        """
        embeddings = {}
        # 先处理所有在数据集中出现的用户/物品
        for entity_id, review_text in reviews_dict.items():
            if entity_id in self.user_to_idx:
                idx = self.user_to_idx[entity_id]
            elif entity_id in self.item_to_idx:
                idx = self.item_to_idx[entity_id]
            else:
                continue
            
            embedding = text_to_embedding(
                review_text, 
                self.word2vec_model, 
                self.embedding_dim
            )
            embeddings[idx] = embedding
        
        # 对于在数据集中出现但没有评论的实体，使用零向量
        # 这主要处理测试集中可能出现的新用户/物品
        if self.mode == 'train':
            # 训练集：所有用户和物品都应该有评论
            pass
        else:
            # 测试集：可能存在没有评论的新用户/物品，使用零向量
            for user_id, user_idx in self.user_to_idx.items():
                if user_idx not in embeddings:
                    embeddings[user_idx] = np.zeros(self.embedding_dim, dtype=np.float32)
            for item_id, item_idx in self.item_to_idx.items():
                if item_idx not in embeddings:
                    embeddings[item_idx] = np.zeros(self.embedding_dim, dtype=np.float32)
        
        return embeddings
    
    def _convert_to_indices(self) -> List[Tuple[int, int, float]]:
        """将数据转换为索引格式"""
        samples = []
        for _, row in self.data.iterrows():
            user_id = row['user_id']
            item_id = row['item_id']
            rating = float(row['rating'])
            
            user_idx = self.user_to_idx[user_id]
            item_idx = self.item_to_idx[item_id]
            
            samples.append((user_idx, item_idx, rating))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个数据样本
        
        Returns:
            {
                'user_idx': 用户索引,
                'item_idx': 物品索引,
                'rating': 真实评分,
                'user_embedding': 用户评论嵌入向量,
                'item_embedding': 物品评论嵌入向量
            }
        """
        user_idx, item_idx, rating = self.samples[idx]
        
        # 获取嵌入向量
        user_embedding = torch.FloatTensor(self.user_embeddings[user_idx])
        item_embedding = torch.FloatTensor(self.item_embeddings[item_idx])
        
        return {
            'user_idx': torch.LongTensor([user_idx])[0],
            'item_idx': torch.LongTensor([item_idx])[0],
            'rating': torch.FloatTensor([rating])[0],
            'user_embedding': user_embedding,
            'item_embedding': item_embedding
        }
    
    def get_num_users(self) -> int:
        """获取用户数量"""
        return len(self.user_to_idx)
    
    def get_num_items(self) -> int:
        """获取物品数量"""
        return len(self.item_to_idx)
    
    def get_sentiment_info(self) -> Tuple[Dict, Dict]:
        """获取情感信息（用于评估）"""
        return self.user_sentiment_dict, self.item_sentiment_dict

