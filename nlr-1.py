import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import f

# 生成一些示例数据
x1_data = np.array([1, 2, 3, 4, 5])
x2_data = np.array([2, 3, 4, 5, 6])
y_data = np.array([2.3, 1.8, 1.2, 0.9, 0.5])

# 将数据转换为矩阵形式
X = np.column_stack((x1_data, x2_data))

# 定义多项式回归的阶数
degree = 3

# 创建多项式回归模型
model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

# 拟合数据
model.fit(X, y_data)

# 计算预测值
y_pred = model.predict(X)

# 计算均方误差和决定系数
mse = mean_squared_error(y_data, y_pred)
r2 = r2_score(y_data, y_pred)

# 计算F统计量和概率
n = len(y_data)  # 样本数量
k = X.shape[1] - 1  # 模型参数个数
df_reg = k
df_resid = n - k - 1

SSR = np.sum((y_pred - np.mean(y_data))**2)
SSE = np.sum((y_data - y_pred)**2)

MSR = SSR / df_reg
MSE = SSE / df_resid

F_statistic = MSR / MSE
p_value = 1 - f.cdf(F_statistic, df_reg, df_resid)

# 输出模型参数和性能指标
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')
print(f'F-statistic: {F_statistic}')
print(f'Probability (p-value): {p_value}')
