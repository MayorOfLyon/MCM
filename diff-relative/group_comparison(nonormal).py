import numpy as np
from scipy.stats import mannwhitneyu, kruskal
import matplotlib.pyplot as plt

# 设置随机种子以确保可重复性
np.random.seed(42)

# 生成三组非正态分布数据
data1 = np.random.exponential(scale=2, size=100)
data2 = np.random.normal(loc=5, scale=1, size=100)
data3 = np.random.gamma(shape=2, scale=2, size=100)

# 对前两组进行曼惠特尼U检验
statistic, p_value_mannwhitney = mannwhitneyu(data1, data2)
print(f"Mann-Whitney U检验结果：统计量 = {statistic}, p值 = {p_value_mannwhitney}")

# 对全部组进行KW检验
statistic, p_value_kruskal = kruskal(data1, data2, data3)
print(f"KW检验结果：统计量 = {statistic}, p值 = {p_value_kruskal}")

# # 可视化数据分布
# plt.figure(figsize=(12, 6))
#
# plt.subplot(2, 2, 1)
# plt.hist(data1, bins=20, color='blue', alpha=0.7)
# plt.title('Group 1')
#
# plt.subplot(2, 2, 2)
# plt.hist(data2, bins=20, color='green', alpha=0.7)
# plt.title('Group 2')
#
# plt.subplot(2, 2, 3)
# plt.hist(data3, bins=20, color='orange', alpha=0.7)
# plt.title('Group 3')
#
# plt.subplot(2, 2, 4)
# plt.boxplot([data1, data2, data3], labels=['Group 1', 'Group 2', 'Group 3'])
# plt.title('Boxplot of All Groups')
#
# plt.tight_layout()
# plt.show()
