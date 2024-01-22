import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare

# 设置随机种子以确保可重复性
np.random.seed(42)

# 生成一组非正态数据进行四次测量
data = np.random.normal(loc=5, scale=2, size=(4, 100))

'''
注：威尔科克森检验和弗莱德曼检验的输入数据应该是一维数组
不是二维数组或DataFrame 不用和正态组一样转DataFrame
'''
# 假设:无差异 p小于显著性水平（通常0.05） 拒绝原假设 -> 存在显著性差异 反之没有

# 对前两次测量结果进行威尔科克森检验
statistic_wilcoxon, p_value_wilcoxon = wilcoxon(data[0], data[1])
print(f"Wilcoxon检验结果：统计量 = {statistic_wilcoxon}, p值 = {p_value_wilcoxon}")

# 对全部测量结果进行弗莱德曼检验
statistic_friedman, p_value_friedman = friedmanchisquare(data[0], data[1], data[2], data[3])
print(f"Friedman检验结果：统计量 = {statistic_friedman}, p值 = {p_value_friedman}")
