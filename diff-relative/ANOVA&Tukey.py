import numpy as np
from scipy.stats import f_oneway, ttest_ind
import matplotlib.pyplot as plt

# 生成示例数据
np.random.seed(42)

# 单因素方差分析示例数据
group1 = np.random.normal(20, 5, 100)
group2 = np.random.normal(25, 5, 100)
group3 = np.random.normal(30, 5, 100)

# 执行单因素方差分析
f_statistic, p_value = f_oneway(group1, group2, group3)
print("One-way ANOVA:")
print("F-statistic:", f_statistic)
print("P-value:", p_value)

# 多重比较（Tukey事后检验）
from statsmodels.stats.multicomp import pairwise_tukeyhsd

data = np.concatenate([group1, group2, group3])
labels = ['Group 1'] * 100 + ['Group 2'] * 100 + ['Group 3'] * 100

tukey_result = pairwise_tukeyhsd(data, labels)
print("\nTukey's HSD:")
print(tukey_result)

# 绘制数据分布图
plt.figure(figsize=(10, 6))
plt.boxplot([group1, group2, group3], labels=['Group 1', 'Group 2', 'Group 3'])
plt.title("Boxplot of Data Distribution")
plt.xlabel("Groups")
plt.ylabel("Values")
plt.show()
