import numpy as np
from scipy.stats import ttest_rel, f_oneway
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 生成三次重复测量的正态分布数据
subjects = 50
time_points = 3

data = np.random.normal(loc=25, scale=5, size=(subjects, time_points))

# 转换成 DataFrame 方便处理
df = pd.DataFrame(data, columns=['Time1', 'Time2', 'Time3'])

# 配对样本 t 检验（前两次测量）
t_statistic_paired, p_value_paired = ttest_rel(df['Time1'], df['Time2'])
print("Paired Samples t-test (Time1 vs. Time2):")
print("t-statistic:", t_statistic_paired)
print("P-value:", p_value_paired)

# 重复测量方差分析
f_statistic_repeated, p_value_repeated = f_oneway(df['Time1'], df['Time2'], df['Time3'])
print("\nRepeated Measures ANOVA:")
print("F-statistic:", f_statistic_repeated)
print("P-value:", p_value_repeated)

'''
# 数据可视化 设置 Seaborn 风格
sns.set(style="whitegrid")

# 绘制直方图
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df, bins=20, kde=True)
plt.title('Histogram of Normal Distribution Data')

# 绘制箱线图
plt.subplot(1, 2, 2)
sns.boxplot(data=df)
plt.title('Boxplot of Normal Distribution Data')
'''
# 设置图形大小
plt.figure(figsize=(10, 6))

# 条形图展示独立样本 t 检验结果
plt.subplot(1, 2, 1)
plt.bar(['Time1 vs. Time2'], [p_value_paired], color='blue', alpha=0.7)
plt.axhline(y=0.05, color='red', linestyle='--', label='Significance Level (α=0.05)')
plt.title('Paired Samples t-test')
plt.ylabel('P-value')
plt.legend()

# 条形图展示重复测量方差分析结果
# 条形高度超过虚线 拒绝假设
plt.subplot(1, 2, 2)
plt.bar(['ANOVA'], [p_value_repeated], color='green', alpha=0.7)
plt.axhline(y=0.05, color='red', linestyle='--', label='Significance Level (α=0.05)')
plt.title('Repeated Measures ANOVA')
plt.ylabel('P-value')
plt.legend()

# 调整布局
plt.tight_layout()
plt.show()


plt.show()
