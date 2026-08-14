"""
隐马尔可夫模型（Hidden Markov Model）算法实现
用于处理序列数据，包含隐藏状态和观测状态
"""

import numpy as np


class HMM:
    def __init__(self, n_states, n_observations):
        """
        HMM初始化

        参数:
            n_states: 隐藏状态数量
            n_observations: 观测状态数量
        """
        self.n_states = n_states
        self.n_observations = n_observations
        self.initial_prob = None
        self.transition_prob = None
        self.emission_prob = None

    def fit(self, observations, n_iterations=100):
        """
        使用Baum-Welch算法训练HMM

        参数:
            observations: 观测序列
            n_iterations: 迭代次数
        """
        observations = np.array(observations)

        # 随机初始化参数
        self.initial_prob = np.random.dirichlet(np.ones(self.n_states))
        self.transition_prob = np.random.dirichlet(np.ones(self.n_states), size=self.n_states)
        self.emission_prob = np.random.dirichlet(np.ones(self.n_observations), size=self.n_states)

        # 迭代训练
        for iteration in range(n_iterations):
            # E步：前向-后向算法
            alpha = self._forward(observations)
            beta = self._backward(observations)

            # 计算gamma和xi
            gamma, xi = self._compute_gamma_xi(observations, alpha, beta)

            # M步：更新参数
            self._update_parameters(observations, gamma, xi)

            if iteration % 20 == 0:
                log_likelihood = self._compute_log_likelihood(observations)
                print(f"迭代 {iteration}, 对数似然: {log_likelihood:.4f}")

        print(f"HMM训练完成，隐藏状态数: {self.n_states}")

    def _forward(self, observations):
        """前向算法"""
        T = len(observations)
        alpha = np.zeros((T, self.n_states))

        # 初始化
        alpha[0] = self.initial_prob * self.emission_prob[:, observations[0]]

        # 递推
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = self.emission_prob[j, observations[t]] * \
                             np.sum(alpha[t-1] * self.transition_prob[:, j])

        return alpha

    def _backward(self, observations):
        """后向算法"""
        T = len(observations)
        beta = np.zeros((T, self.n_states))

        # 初始化
        beta[T-1] = 1

        # 递推
        for t in range(T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(
                    self.transition_prob[i] * self.emission_prob[:, observations[t+1]] * beta[t+1]
                )

        return beta

    def _compute_gamma_xi(self, observations, alpha, beta):
        """计算gamma和xi"""
        T = len(observations)
        gamma = np.zeros((T, self.n_states))
        xi = np.zeros((T-1, self.n_states, self.n_states))

        # 计算gamma
        for t in range(T):
            gamma[t] = alpha[t] * beta[t]
            gamma[t] /= np.sum(gamma[t])

        # 计算xi
        for t in range(T-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = alpha[t, i] * self.transition_prob[i, j] * \
                                 self.emission_prob[j, observations[t+1]] * beta[t+1, j]
            xi[t] /= np.sum(xi[t])

        return gamma, xi

    def _update_parameters(self, observations, gamma, xi):
        """更新参数"""
        T = len(observations)

        # 更新初始概率
        self.initial_prob = gamma[0]

        # 更新转移概率
        for i in range(self.n_states):
            denom = np.sum(gamma[:-1, i])
            for j in range(self.n_states):
                self.transition_prob[i, j] = np.sum(xi[:, i, j]) / (denom + 1e-10)

        # 更新发射概率
        for j in range(self.n_states):
            denom = np.sum(gamma[:, j])
            for k in range(self.n_observations):
                self.emission_prob[j, k] = np.sum(gamma[observations == k, j]) / (denom + 1e-10)

    def _compute_log_likelihood(self, observations):
        """计算对数似然"""
        alpha = self._forward(observations)
        return np.log(np.sum(alpha[-1]) + 1e-10)

    def predict(self, observations):
        """
        使用Viterbi算法预测最可能的隐藏状态序列

        参数:
            observations: 观测序列

        返回:
            最可能的隐藏状态序列
        """
        observations = np.array(observations)
        T = len(observations)

        # Viterbi算法
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)

        # 初始化
        delta[0] = self.initial_prob * self.emission_prob[:, observations[0]]

        # 递推
        for t in range(1, T):
            for j in range(self.n_states):
                temp = delta[t-1] * self.transition_prob[:, j]
                delta[t, j] = np.max(temp) * self.emission_prob[j, observations[t]]
                psi[t, j] = np.argmax(temp)

        # 回溯
        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(delta[T-1])

        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]

        return states


if __name__ == '__main__':
    # 示例：使用HMM进行序列分析
    print("=== 隐马尔可夫模型算法示例 ===")

    # 生成观测序列
    np.random.seed(42)
    observations = np.random.randint(0, 3, size=100)

    # 创建并训练模型
    hmm = HMM(n_states=2, n_observations=3)
    hmm.fit(observations, n_iterations=50)

    # 预测最可能的隐藏状态
    hidden_states = hmm.predict(observations)

    print(f"\n观测序列: {observations[:20]}")
    print(f"隐藏状态: {hidden_states[:20]}")
    print(f"\n初始概率: {hmm.initial_prob}")
    print(f"转移概率矩阵:\n{hmm.transition_prob}")
    print(f"发射概率矩阵:\n{hmm.emission_prob}")
