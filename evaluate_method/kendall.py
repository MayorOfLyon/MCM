from scipy.stats import kendalltau

# 示例数据
x = [1, 2, 3, 4, 5]
y = [5, 4, 3, 2, 1]

# 计算Kendall Rank
kendall_corr, p_value = kendalltau(x, y)

# 打印结果
print(f"Kendall Rank correlation coefficient: {kendall_corr}")
print(f"P-value: {p_value}")
