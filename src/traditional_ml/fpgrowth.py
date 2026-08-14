"""
FP-Growth（Frequent Pattern Growth）关联规则算法实现
比Apriori更高效的频繁项集挖掘算法，使用FP树结构
"""

from collections import defaultdict


class FPGrowth:
    def __init__(self, min_support=0.5, min_confidence=0.7):
        """
        FP-Growth初始化

        参数:
            min_support: 最小支持度阈值（比例，0到1之间）
            min_confidence: 最小置信度阈值（比例，0到1之间）
        """
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.frequent_itemsets = {}
        self.rules = []
        self.n_transactions = 0

    def fit(self, transactions):
        """
        训练FP-Growth模型

        参数:
            transactions: 交易数据列表，每个元素是一个可迭代对象
        """
        transactions = [list(t) for t in transactions]
        self.n_transactions = len(transactions)
        min_support_count = self.min_support * self.n_transactions

        # 1. 计算单项支持度
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1

        # 过滤频繁项并按支持度降序排序
        frequent_items = {item: count for item, count in item_counts.items()
                         if count >= min_support_count}
        self.frequent_items = dict(sorted(frequent_items.items(), key=lambda x: -x[1]))

        print(f"频繁单项: {len(self.frequent_items)}个")

        # 2. 构建FP树
        self.tree, self.header_table = self._build_tree(transactions)

        # 3. 递归挖掘频繁项集（存储为计数）
        self.frequent_itemsets = {}
        self._mine_tree(self.tree, self.header_table, set(), min_support_count)

        # 将计数转换为支持度比例
        self.frequent_itemsets = {
            itemset: count / self.n_transactions
            for itemset, count in self.frequent_itemsets.items()
        }

        # 4. 生成关联规则
        self._generate_rules()

        print(f"\nFP-Growth算法完成")
        print(f"频繁项集总数: {len(self.frequent_itemsets)}个")
        print(f"关联规则数量: {len(self.rules)}条")

    class _Node:
        def __init__(self, item, count, parent):
            self.item = item
            self.count = count
            self.parent = parent
            self.children = {}
            self.link = None

    def _build_tree(self, transactions):
        """构建FP树"""
        root = self._Node(None, 1, None)
        header_table = {}

        for transaction in transactions:
            # 过滤并排序事务中的项
            items = [item for item in transaction if item in self.frequent_items]
            items.sort(key=lambda x: -self.frequent_items[x])

            current = root
            for item in items:
                if item in current.children:
                    current.children[item].count += 1
                else:
                    new_node = self._Node(item, 1, current)
                    current.children[item] = new_node

                    # 维护header table的节点链表
                    if item in header_table:
                        header_table[item].link = new_node
                    else:
                        header_table[item] = new_node

                current = current.children[item]

        return root, header_table

    def _mine_tree(self, tree, header_table, prefix, min_support_count):
        """
        递归挖掘频繁项集

        参数:
            tree: 当前FP树根节点
            header_table: header表（item -> 节点链表头）
            prefix: 当前前缀项集
            min_support_count: 最小支持度计数
        """
        # 按支持度升序处理（从底部开始）
        items = sorted(header_table.keys(), key=lambda x: self.frequent_items[x])

        for item in items:
            # 当前项的完整项集 = 前缀 + 当前项
            new_itemset = prefix | {item}

            # 计算当前项的总支持度
            total_count = 0
            node = header_table[item]
            while node is not None:
                total_count += node.count
                node = node.link

            # 记录频繁项集
            self.frequent_itemsets[frozenset(new_itemset)] = total_count

            # 构建条件模式基
            conditional_base = []
            node = header_table[item]
            while node is not None:
                # 回溯到根，构建前缀路径
                path = []
                parent = node.parent
                while parent is not None and parent.item is not None:
                    path.append((parent.item, node.count))
                    parent = parent.parent
                if path:
                    conditional_base.append(path)
                node = node.link

            if conditional_base:
                # 构建条件FP树
                cond_header = {}
                for path in conditional_base:
                    for path_item, count in path:
                        cond_header[path_item] = cond_header.get(path_item, 0) + count

                # 过滤不频繁项
                cond_header = {k: v for k, v in cond_header.items()
                              if v >= min_support_count}

                if cond_header:
                    # 构建条件FP树
                    cond_tree, cond_header_table = self._build_conditional_tree(
                        conditional_base, cond_header
                    )
                    # 递归挖掘
                    self._mine_tree(cond_tree, cond_header_table, new_itemset, min_support_count)

    def _build_conditional_tree(self, conditional_base, cond_header):
        """构建条件FP树"""
        root = self._Node(None, 1, None)
        header_table = {}

        for path in conditional_base:
            # path中的项按全局支持度降序排序
            path_sorted = sorted(path, key=lambda x: -self.frequent_items[x[0]])

            current = root
            for item, count in path_sorted:
                if item in current.children:
                    current.children[item].count += count
                else:
                    new_node = self._Node(item, count, current)
                    current.children[item] = new_node

                    if item in header_table:
                        header_table[item].link = new_node
                    else:
                        header_table[item] = new_node

                current = current.children[item]

        return root, header_table

    def _generate_rules(self):
        """生成关联规则"""
        self.rules = []

        for itemset, support in self.frequent_itemsets.items():
            if len(itemset) < 2:
                continue

            items = list(itemset)
            n = len(items)

            from itertools import combinations
            for i in range(1, n):
                for antecedent in combinations(items, i):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent

                    antecedent_support = self.frequent_itemsets.get(antecedent, 0)

                    if antecedent_support > 0:
                        confidence = support / antecedent_support

                        if confidence >= self.min_confidence:
                            self.rules.append({
                                'antecedent': set(antecedent),
                                'consequent': set(consequent),
                                'support': support,
                                'confidence': confidence
                            })

        # 按置信度排序
        self.rules.sort(key=lambda x: -x['confidence'])

    def get_top_rules(self, n=10):
        """获取前n条关联规则"""
        return self.rules[:n]


if __name__ == '__main__':
    # 示例：使用FP-Growth挖掘关联规则
    print("=== FP-Growth关联规则示例 ===")

    transactions = [
        ['牛奶', '面包', '黄油'],
        ['面包', '黄油'],
        ['牛奶', '面包', '鸡蛋'],
        ['牛奶', '面包', '黄油', '鸡蛋'],
        ['面包', '黄油', '鸡蛋'],
        ['牛奶', '鸡蛋'],
        ['牛奶', '面包', '黄油', '鸡蛋'],
        ['面包', '黄油']
    ]

    # 创建并训练模型
    fpgrowth = FPGrowth(min_support=0.3, min_confidence=0.6)
    fpgrowth.fit(transactions)

    # 显示频繁项集
    print(f"\n频繁项集:")
    for itemset, support in sorted(fpgrowth.frequent_itemsets.items(), key=lambda x: -x[1]):
        print(f"  {set(itemset)}: 支持度={support:.3f}")

    # 显示关联规则
    print(f"\n关联规则（按置信度排序）:")
    top_rules = fpgrowth.get_top_rules(10)
    for i, rule in enumerate(top_rules, 1):
        print(f"  {i}. {rule['antecedent']} -> {rule['consequent']}")
        print(f"     支持度={rule['support']:.3f}, 置信度={rule['confidence']:.3f}")
