import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, f_oneway, levene

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 生成三组正态分布的数据
group1 = np.random.normal(20, 5, 100)
group2 = np.random.normal(25, 5, 100)
group3 = np.random.normal(30, 5, 100)

# 独立样本 t 检验
t_statistic, p_value_t = ttest_ind(group1, group2)
print("Independent Samples t-test (Group1 vs. Group2):")
print("t-statistic:", t_statistic)
print("P-value:", p_value_t)

# 单因素方差分析（方差齐性检验）
levene_statistic, p_value_levene = levene(group1, group2, group3)
print("\nLevene's Test for Homogeneity of Variance:")
print("Levene Statistic:", levene_statistic)
print("P-value:", p_value_levene)

# 根据方差齐性结果选择使用方差齐性或方差不齐性的单因素方差分析
if p_value_levene < 0.05:
    print("\nSince p-value < 0.05 (rejecting homogeneity of variance), using Welch's ANOVA.")
    f_statistic, p_value_anova = f_oneway(group1, group2, group3)
else:
    print("\nUsing ordinary one-way ANOVA (homogeneity of variance assumed).")
    f_statistic, p_value_anova = f_oneway(group1, group2, group3)

print("\nOne-way ANOVA:")
print("F-statistic:", f_statistic)
print("P-value:", p_value_anova)

