import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau, linregress, chi2_contingency
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.inter_rater import cohens_kappa, to_table
import pingouin as pg


# 1. Pearson相关性分析
# 衡量两个连续变量之间的线性关系(要求 正态分布)
x = np.random.normal(loc=0, scale=1, size=100)
y = 2 * x + np.random.normal(loc=0, scale=1, size=100)
correlation_coefficient, p_value_pearson = pearsonr(x, y)
print(f"Pearson相关性分析结果：相关系数 = {correlation_coefficient}, p值 = {p_value_pearson}")


# 2. Spearman秩相关性分析
# 衡量两个变量之间的单调关系(不一定线性 (要求 变量连续或有序)
x = np.random.normal(loc=0, scale=1, size=100)
y = np.sin(x) + np.random.normal(loc=0, scale=0.1, size=100)
correlation_coefficient, p_value_spearman = spearmanr(x, y)
print(f"Spearman秩相关性分析结果：相关系数 = {correlation_coefficient}, p值 = {p_value_spearman}")


# 3. Kendall's tau-b相关性分析
# 衡量两个变量之间的非线性关系(对异常不敏感 (要求 变量连续或有序)
x = np.random.normal(loc=0, scale=1, size=100)
y = np.cos(x) + np.random.normal(loc=0, scale=0.1, size=100)
correlation_coefficient, p_value_kendall = kendalltau(x, y)
print(f"Kendall's tau-b相关性分析结果：相关系数 = {correlation_coefficient}, p值 = {p_value_kendall}")


# 4. Cochran's Q检验 用于比较三个或更多相关样本的二分类变量
# 生成三个相关样本的二分类变量数据
group1 = np.random.choice([0, 1], size=50)
group2 = np.random.choice([0, 1], size=50)
group3 = np.random.choice([0, 1], size=50)

# 构建列联表
table = np.array([group1, group2, group3])

# 转置数组，使得每一列代表一个条件下的观测结果
table = table.T

# Cochran's Q Test
result = mcnemar(table)
print(f"Cochran's Q Test结果：统计量 = {result.statistic}, p值 = {result.pvalue}")


# 5. Kappa一致性检验
# 衡量分类或测量的一致性
category1 = np.random.choice(['A', 'B', 'C'], size=100)
category2 = np.random.choice(['A', 'B', 'C'], size=100)
kappa_score = cohen_kappa_score(category1, category2)
print(f"Kappa一致性检验结果：Kappa系数 = {kappa_score}")

# 构建混淆矩阵
confusion_matrix = pd.crosstab(category1, category2, rownames=['Category1'], colnames=['Category2'])
# 可视化Kappa一致性检验结果
sns.heatmap(confusion_matrix, annot=True, cmap='Blues', fmt='g')
plt.title("Kappa-confusion result")
plt.show()


# 6. Kendall一致性检验 (类似Kappa)
# 生成两个观察者的分类数据
rater1 = np.random.choice(['A', 'B', 'C'], size=100)
rater2 = np.random.choice(['A', 'B', 'C'], size=100)

# 计算Kendall's Tau相关系数
correlation_coefficient, p_value_kendall = kendalltau(rater1, rater2)
print(f"Kendall一致性检验结果：相关系数 = {correlation_coefficient}, p值 = {p_value_kendall}")


