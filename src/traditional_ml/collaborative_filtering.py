"""
协同过滤（Collaborative Filtering）推荐算法实现
基于用户或物品相似度的推荐系统算法
"""

import numpy as np


class CollaborativeFiltering:
    def __init__(self, method='user_based', k=5):
        """
        协同过滤初始化

        参数:
            method: 推荐方法，'user_based'（基于用户）或'item_based'（基于物品）
            k: 邻居数量
        """
        self.method = method
        self.k = k
        self.ratings = None
        self.similarity_matrix = None

    def fit(self, ratings_matrix):
        """
        训练协同过滤模型

        参数:
            ratings_matrix: 评分矩阵，形状为(n_users, n_items)，0表示未评分
        """
        self.ratings = np.array(ratings_matrix)

        if self.method == 'user_based':
            self.similarity_matrix = self._compute_user_similarity()
        else:
            self.similarity_matrix = self._compute_item_similarity()

        print(f"协同过滤模型训练完成，方法: {self.method}")

    def _compute_user_similarity(self):
        """计算用户相似度（余弦相似度）"""
        n_users = self.ratings.shape[0]
        similarity = np.zeros((n_users, n_users))

        for i in range(n_users):
            for j in range(i+1, n_users):
                # 获取共同评分的物品
                common = (self.ratings[i] > 0) & (self.ratings[j] > 0)

                if np.sum(common) == 0:
                    continue

                # 计算余弦相似度
                ratings_i = self.ratings[i][common]
                ratings_j = self.ratings[j][common]

                similarity[i, j] = np.dot(ratings_i, ratings_j) / \
                                  (np.linalg.norm(ratings_i) * np.linalg.norm(ratings_j) + 1e-10)
                similarity[j, i] = similarity[i, j]

        return similarity

    def _compute_item_similarity(self):
        """计算物品相似度（余弦相似度）"""
        n_items = self.ratings.shape[1]
        similarity = np.zeros((n_items, n_items))

        for i in range(n_items):
            for j in range(i+1, n_items):
                # 获取共同评分的用户
                common = (self.ratings[:, i] > 0) & (self.ratings[:, j] > 0)

                if np.sum(common) == 0:
                    continue

                # 计算余弦相似度
                ratings_i = self.ratings[common, i]
                ratings_j = self.ratings[common, j]

                similarity[i, j] = np.dot(ratings_i, ratings_j) / \
                                  (np.linalg.norm(ratings_i) * np.linalg.norm(ratings_j) + 1e-10)
                similarity[j, i] = similarity[i, j]

        return similarity

    def predict(self, user_id, item_id):
        """
        预测用户对物品的评分

        参数:
            user_id: 用户ID
            item_id: 物品ID

        返回:
            预测评分
        """
        if self.method == 'user_based':
            return self._predict_user_based(user_id, item_id)
        else:
            return self._predict_item_based(user_id, item_id)

    def _predict_user_based(self, user_id, item_id):
        """基于用户的预测"""
        # 找到对目标物品评分的用户
        rated_users = np.where(self.ratings[:, item_id] > 0)[0]
        rated_users = rated_users[rated_users != user_id]

        if len(rated_users) == 0:
            return np.mean(self.ratings[user_id][self.ratings[user_id] > 0])

        # 获取相似度
        similarities = self.similarity_matrix[user_id, rated_users]

        # 选择最相似的k个用户
        k_nearest = np.argsort(similarities)[-self.k:]

        # 加权平均预测
        numerator = np.sum(similarities[k_nearest] * self.ratings[rated_users[k_nearest], item_id])
        denominator = np.sum(np.abs(similarities[k_nearest]))

        if denominator == 0:
            return np.mean(self.ratings[user_id][self.ratings[user_id] > 0])

        return numerator / denominator

    def _predict_item_based(self, user_id, item_id):
        """基于物品的预测"""
        # 找到用户评分的物品
        rated_items = np.where(self.ratings[user_id] > 0)[0]

        if len(rated_items) == 0:
            return 0

        # 获取相似度
        similarities = self.similarity_matrix[item_id, rated_items]

        # 选择最相似的k个物品
        k_nearest = np.argsort(similarities)[-self.k:]

        # 加权平均预测
        numerator = np.sum(similarities[k_nearest] * self.ratings[user_id, rated_items[k_nearest]])
        denominator = np.sum(np.abs(similarities[k_nearest]))

        if denominator == 0:
            return np.mean(self.ratings[user_id][rated_items])

        return numerator / denominator

    def recommend(self, user_id, n_recommendations=5):
        """
        为用户推荐物品

        参数:
            user_id: 用户ID
            n_recommendations: 推荐数量

        返回:
            推荐物品列表
        """
        n_items = self.ratings.shape[1]
        unrated_items = np.where(self.ratings[user_id] == 0)[0]

        # 预测所有未评分物品的评分
        predictions = []
        for item_id in unrated_items:
            predicted_rating = self.predict(user_id, item_id)
            predictions.append((item_id, predicted_rating))

        # 按预测评分排序
        predictions.sort(key=lambda x: -x[1])

        # 返回前n个推荐
        return predictions[:n_recommendations]


if __name__ == '__main__':
    # 示例：使用协同过滤进行推荐
    print("=== 协同过滤推荐算法示例 ===")

    # 创建示例评分矩阵（5个用户，6个物品）
    ratings_matrix = np.array([
        [5, 3, 0, 0, 4, 0],
        [4, 0, 0, 5, 3, 0],
        [0, 5, 4, 0, 0, 3],
        [3, 0, 5, 4, 0, 4],
        [0, 4, 0, 0, 5, 0]
    ])

    print("评分矩阵:")
    print(ratings_matrix)

    # 基于用户的协同过滤
    print("\n基于用户的协同过滤:")
    cf_user = CollaborativeFiltering(method='user_based', k=3)
    cf_user.fit(ratings_matrix)

    # 预测用户0对物品2的评分
    predicted_rating = cf_user.predict(0, 2)
    print(f"预测用户0对物品2的评分: {predicted_rating:.2f}")

    # 为用户0推荐物品
    recommendations = cf_user.recommend(0, n_recommendations=3)
    print(f"为用户0推荐的物品: {recommendations}")

    # 基于物品的协同过滤
    print("\n基于物品的协同过滤:")
    cf_item = CollaborativeFiltering(method='item_based', k=3)
    cf_item.fit(ratings_matrix)

    # 预测用户1对物品2的评分
    predicted_rating = cf_item.predict(1, 2)
    print(f"预测用户1对物品2的评分: {predicted_rating:.2f}")

    # 为用户1推荐物品
    recommendations = cf_item.recommend(1, n_recommendations=3)
    print(f"为用户1推荐的物品: {recommendations}")
