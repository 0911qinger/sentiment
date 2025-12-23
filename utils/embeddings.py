"""
文本嵌入工具：使用Word2Vec生成评论向量
"""
import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
import re
from typing import List, Dict
import torch
from textblob import TextBlob


def preprocess_text(text: str) -> List[str]:
    """
    预处理文本：分词并转换为小写
    
    Args:
        text: 原始文本
        
    Returns:
        分词后的词列表
    """
    if not isinstance(text, str):
        text = str(text)
    # 转换为小写并移除特殊字符
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # 分词
    words = text.split()
    return words


def build_word2vec_model(reviews: List[str], 
                         embedding_dim: int = 300,
                         min_count: int = 2) -> Word2Vec:
    """
    训练Word2Vec模型
    
    Args:
        reviews: 评论文本列表
        embedding_dim: 词向量维度
        min_count: 最小词频
        
    Returns:
        训练好的Word2Vec模型
    """
    # 预处理所有评论
    processed_reviews = [preprocess_text(review) for review in reviews]
    
    # 过滤空评论
    processed_reviews = [review for review in processed_reviews if len(review) > 0]
    
    # 训练Word2Vec模型
    model = Word2Vec(
        sentences=processed_reviews,
        vector_size=embedding_dim,
        window=5,
        min_count=min_count,
        workers=4,
        sg=1,  # skip-gram
        epochs=10
    )
    
    return model


def text_to_embedding(text: str, 
                     word2vec_model: Word2Vec,
                     embedding_dim: int = 300) -> np.ndarray:
    """
    将文本转换为固定长度的向量（使用平均池化）
    
    Args:
        text: 输入文本
        word2vec_model: Word2Vec模型
        embedding_dim: 向量维度
        
    Returns:
        固定长度的向量
    """
    words = preprocess_text(text)
    
    # 获取每个词的向量并求平均
    vectors = []
    for word in words:
        if word in word2vec_model.wv:
            vectors.append(word2vec_model.wv[word])
    
    if len(vectors) == 0:
        # 如果没有找到任何词，返回零向量
        return np.zeros(embedding_dim, dtype=np.float32)
    
    # 平均池化
    embedding = np.mean(vectors, axis=0)
    return embedding.astype(np.float32)


def compute_sentiment_polarity(text: str) -> float:
    """
    使用TextBlob计算情感极性
    
    Args:
        text: 输入文本
        
    Returns:
        情感极性值（-1到1之间）
    """
    if not isinstance(text, str):
        text = str(text)
    blob = TextBlob(text)
    return blob.sentiment.polarity

