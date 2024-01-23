import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 生成一些示例数据
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 添加截距项
X_with_intercept = sm.add_constant(X)

# 创建OLS模型（Ordinary Least Squares）
model = sm.OLS(y, X_with_intercept).fit()

# 打印回归结果与统计量:F statistic, R-squared, p-value, standard error
print(model.summary())

# 打印回归方程
print("回归方程: y = {:.5} + {:.5}x".format(model.params[0], model.params[1]))

# 获取回归系数和截距的置信区间
conf_int = model.conf_int(alpha=0.05)  # 默认alpha=0.05表示95%置信区间
print("回归系数和截距的置信区间:")
print(conf_int)

# # 绘制原始数据和回归线
# plt.scatter(X, y, alpha=0.6, label='原始数据')
# plt.plot(X, model.predict(X_with_intercept), color='red', label='线性回归')
# plt.xlabel('X')
# plt.ylabel('y')
# plt.legend()
# plt.show()

# 获取残差和置信区间
residuals = model.resid
predictions = model.get_prediction()
conf_int = predictions.conf_int()

# 计算触须的长度
whisker_length = np.abs(conf_int[:, 1] - residuals)

# 绘制残差图
plt.scatter(range(len(residuals)), residuals, alpha=0.6)

# 在每个点上添加触须
for i in range(len(residuals)):
    x_val = i
    y_val = residuals[i]
    lower_ci, upper_ci = conf_int[i]
    length = whisker_length[i]

    plt.plot([x_val, x_val], [y_val - length, y_val + length], color='black', linewidth=1)

plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title('Residual Plot with Whiskers')
plt.xlabel('Data Points')
plt.ylabel('Residuals')
plt.show()
