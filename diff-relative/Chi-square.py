import numpy as np
from scipy.stats import fisher_exact, chi2_contingency

# 模拟分类数据
observed_data = np.array([[30, 20], [15, 35]])

"""各类卡方检验"""

# 执行皮尔逊卡方检验
chi2_stat, p_value, dof, expected = chi2_contingency(observed_data)

# Fisher's 精确卡方检验
odds_ratio, p_fisher = fisher_exact(observed_data)

# Yates 修正
chi2_stat_yates, p_value_yates, dof_yates, expected_yates = chi2_contingency(observed_data, correction='yates')

# 连续校正（Continuity Correction）
chi2_stat_continuity, p_value_continuity, dof_continuity, expected_continuity = chi2_contingency(observed_data, correction=True)

# Pearson
print("Chi-square Statistic:", chi2_stat)
print("P-value:", p_value)
print("Degrees of Freedom:", dof)
print("\nExpected Frequencies:")
print(expected)

alpha = 0.05
if p_value < alpha:
    print("\n拒绝原假设：变量之间存在关系")
else:
    print("\n未拒绝原假设：未发现变量之间的关系")

# 精确卡方
print("Fisher's Exact Test:")
print("Odds Ratio:", odds_ratio)
print("P-value (Fisher):", p_fisher)

# Yates 修正
print("\nYates Correction:")
print("Chi-square Statistic (Yates):", chi2_stat_yates)
print("P-value (Yates):", p_value_yates)
print("Degrees of Freedom (Yates):", dof_yates)
print("\nExpected Frequencies (Yates):")
print(expected_yates)

# 连续校正
print("\nContinuity Correction:")
print("Chi-square Statistic (Continuity):", chi2_stat_continuity)
print("P-value (Continuity):", p_value_continuity)
print("Degrees of Freedom (Continuity):", dof_continuity)
print("\nExpected Frequencies (Continuity):")
print(expected_continuity)
