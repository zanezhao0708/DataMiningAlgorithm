"""
Apriori关联规则挖掘算法实现
通过频繁项集挖掘发现数据项之间的关联关系
"""

import numpy as np
from itertools import combinations


class Apriori:
    def __init__(self, min_support=0.5, min_confidence=0.7):
        """
        Apriori算法初始化

        参数:
            min_support: 最小支持度阈值
            min_confidence: 最小置信度阈值
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = {}
        self.rules = []

    def fit(self, transactions):
        """
        训练Apriori模型，挖掘频繁项集和关联规则

        参数:
            transactions: 交易数据列表，每个元素是一个集合或列表
        """
        # 转换为集合列表
        transactions = [set(t) for t in transactions]
        n_transactions = len(transactions)

        # 获取所有唯一项
        all_items = set()
        for transaction in transactions:
            all_items.update(transaction)

        # 生成频繁1-项集
        L1 = self._generate_L1(transactions, all_items)
        self.frequent_itemsets[1] = L1

        print(f"频繁1-项集: {len(L1)}个")

        # 迭代生成频繁k-项集
        k = 2
        while True:
            # 生成候选k-项集
            candidates = self._generate_candidates(self.frequent_itemsets[k-1], k)

            if not candidates:
                break

            # 计算支持度并筛选
            Lk = {}
            for candidate in candidates:
                support = self._calculate_support(candidate, transactions)
                if support >= self.min_support:
                    Lk[candidate] = support

            if not Lk:
                break

            self.frequent_itemsets[k] = Lk
            print(f"频繁{k}-项集: {len(Lk)}个")
            k += 1

        # 生成关联规则
        self._generate_rules(transactions)

        print(f"\nApriori算法完成")
        print(f"频繁项集总数: {sum(len(v) for v in self.frequent_itemsets.values())}个")
        print(f"关联规则数量: {len(self.rules)}条")

    def _generate_L1(self, transactions, all_items):
        """生成频繁1-项集"""
        L1 = {}
        for item in all_items:
            support = self._calculate_support({item}, transactions)
            if support >= self.min_support:
                L1[frozenset({item})] = support
        return L1

    def _generate_candidates(self, prev_itemset, k):
        """生成候选k-项集"""
        candidates = set()
        items = list(prev_itemset.keys())

        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                # 合并两个(k-1)-项集
                union = items[i] | items[j]
                if len(union) == k:
                    # 检查所有(k-1)-子集是否频繁（Apriori剪枝）
                    all_subsets_frequent = True
                    for subset in combinations(union, k-1):
                        if frozenset(subset) not in prev_itemset:
                            all_subsets_frequent = False
                            break
                    if all_subsets_frequent:
                        candidates.add(union)

        return candidates

    def _calculate_support(self, itemset, transactions):
        """计算支持度"""
        count = sum(1 for transaction in transactions if itemset.issubset(transaction))
        return count / len(transactions)

    def _generate_rules(self, transactions):
        """生成关联规则"""
        self.rules = []

        # 对每个频繁项集生成规则
        for k in range(2, len(self.frequent_itemsets) + 1):
            if k not in self.frequent_itemsets:
                continue

            for itemset in self.frequent_itemsets[k]:
                support = self.frequent_itemsets[k][itemset]

                # 生成所有可能的规则
                items = list(itemset)
                for i in range(1, len(items)):
                    for antecedent in combinations(items, i):
                        antecedent = set(antecedent)
                        consequent = itemset - antecedent

                        # 计算置信度
                        antecedent_support = self._calculate_support(antecedent, transactions)
                        confidence = support / antecedent_support if antecedent_support > 0 else 0

                        if confidence >= self.min_confidence:
                            self.rules.append({
                                'antecedent': antecedent,
                                'consequent': consequent,
                                'support': support,
                                'confidence': confidence
                            })

        # 按置信度排序
        self.rules.sort(key=lambda x: x['confidence'], reverse=True)

    def get_top_rules(self, n=10):
        """
        获取前n条关联规则

        参数:
            n: 规则数量

        返回:
            关联规则列表
        """
        return self.rules[:n]


if __name__ == '__main__':
    # 示例：使用Apriori算法挖掘关联规则
    print("=== Apriori关联规则挖掘示例 ===")

    # 示例交易数据（购物篮数据）
    transactions = [
        {'牛奶', '面包', '黄油'},
        {'面包', '黄油'},
        {'牛奶', '面包', '鸡蛋'},
        {'牛奶', '面包', '黄油', '鸡蛋'},
        {'面包', '黄油', '鸡蛋'},
        {'牛奶', '鸡蛋'},
        {'牛奶', '面包', '黄油', '鸡蛋'},
        {'面包', '黄油'}
    ]

    print(f"交易数据数量: {len(transactions)}")
    print(f"交易数据:")
    for i, t in enumerate(transactions, 1):
        print(f"  {i}. {t}")

    # 创建并训练模型
    apriori = Apriori(min_support=0.3, min_confidence=0.6)
    apriori.fit(transactions)

    # 显示频繁项集
    print(f"\n频繁项集:")
    for k in sorted(apriori.frequent_itemsets.keys()):
        print(f"\n频繁{k}-项集:")
        for itemset, support in apriori.frequent_itemsets[k].items():
            print(f"  {set(itemset)}: 支持度={support:.3f}")

    # 显示关联规则
    print(f"\n关联规则（按置信度排序）:")
    top_rules = apriori.get_top_rules(10)
    for i, rule in enumerate(top_rules, 1):
        print(f"  {i}. {rule['antecedent']} -> {rule['consequent']}")
        print(f"     支持度={rule['support']:.3f}, 置信度={rule['confidence']:.3f}")