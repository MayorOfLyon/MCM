import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 假设有一个用户-物品矩阵（用户对物品的评分）
user_item_matrix = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [0, 0, 4, 4],
])

# 计算用户之间的余弦相似度
user_similarity = cosine_similarity(user_item_matrix)

# 定义目标用户和其对物品的评分
target_user_index = 0
target_user_ratings = user_item_matrix[target_user_index]

# 找到与目标用户相似度最高的用户
similar_users = np.argsort(user_similarity[target_user_index])[::-1][1:]

# 生成推荐列表
recommendations = np.sum(user_item_matrix[similar_users] * user_similarity[target_user_index, similar_users, np.newaxis], axis=0)
recommendations[target_user_ratings.nonzero()] = 0  # 将目标用户已有的评分置为0

# 获取前N个推荐物品
top_N = 2
top_N_recommendations = np.argsort(recommendations)[::-1][:top_N]

print(f"推荐物品索引：{top_N_recommendations}")
