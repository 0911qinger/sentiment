"""
生成示例数据集（用于测试）
实际使用时，请替换为真实的Amazon或Yelp数据集
"""
import json
import random
import os

def generate_sample_data(num_samples=1000, output_path="data/sample_reviews.json"):
    """
    生成示例数据集
    
    Args:
        num_samples: 生成的样本数量
        output_path: 输出文件路径
    """
    # 生成用户和物品ID
    num_users = num_samples // 10
    num_items = num_samples // 20
    
    users = [f"user_{i}" for i in range(num_users)]
    items = [f"item_{i}" for i in range(num_items)]
    
    # 示例评论模板
    positive_reviews = [
        "Great product! Highly recommend.",
        "Excellent quality and fast shipping.",
        "Love it! Works perfectly as described.",
        "Amazing value for money.",
        "Very satisfied with this purchase.",
    ]
    
    negative_reviews = [
        "Not worth the price.",
        "Poor quality, broke after a few uses.",
        "Disappointed with this product.",
        "Does not meet expectations.",
        "Would not recommend.",
    ]
    
    neutral_reviews = [
        "It's okay, nothing special.",
        "Average product, does the job.",
        "Decent quality for the price.",
    ]
    
    # 生成数据
    data = []
    for i in range(num_samples):
        user_id = random.choice(users)
        item_id = random.choice(items)
        
        # 随机生成评分和评论（评分越高，使用正面评论的概率越大）
        rating = random.choices(
            [1, 2, 3, 4, 5],
            weights=[0.1, 0.15, 0.2, 0.3, 0.25]
        )[0]
        
        if rating >= 4:
            review_text = random.choice(positive_reviews)
        elif rating <= 2:
            review_text = random.choice(negative_reviews)
        else:
            review_text = random.choice(neutral_reviews)
        
        # 添加一些随机性
        review_text = review_text + " " + f"Sample review text {i}."
        
        data.append({
            "user_id": user_id,
            "item_id": item_id,
            "rating": rating,
            "review_text": review_text
        })
    
    # 保存数据
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"已生成 {num_samples} 条示例数据，保存到: {output_path}")
    print(f"用户数: {num_users}, 物品数: {num_items}")

if __name__ == '__main__':
    generate_sample_data(num_samples=5000, output_path="data/sample_reviews.json")

