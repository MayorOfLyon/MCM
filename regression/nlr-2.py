import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 生成一些示例数据
x1_data = np.array([1, 2, 3, 4, 5])
x2_data = np.array([2, 3, 4, 5, 6])
y_data = np.array([2.3, 1.8, 1.2, 0.9, 0.5])

# 创建DataFrame
df = pd.DataFrame({'X1': x1_data, 'X2': x2_data, 'Y': y_data})

# 添加非线性特征，例如 X1^2 和 X2^2
df['X1_squared'] = df['X1']**2
df['X2_squared'] = df['X2']**2

# 构建模型
model = sm.OLS(df['Y'], sm.add_constant(df[['X1', 'X2', 'X1_squared', 'X2_squared']])).fit()

# 打印模型摘要
print(model.summary())

# 绘制原始数据和拟合曲线
x1_fit = np.linspace(1, 5, 100)
x2_fit = np.linspace(2, 6, 100)
x1_fit, x2_fit = np.meshgrid(x1_fit, x2_fit)
x1_fit = x1_fit.flatten()
x2_fit = x2_fit.flatten()

y_fit = model.predict(sm.add_constant(pd.DataFrame({
    'X1': x1_fit, 
    'X2': x2_fit, 
    'X1_squared': x1_fit**2, 
    'X2_squared': x2_fit**2
})))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['X1'], df['X2'], df['Y'], label='Original Data')
ax.plot_trisurf(x1_fit, x2_fit, y_fit, color='red', alpha=0.5, label='Fitted Surface')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
ax.set_title('Multivariate Nonlinear Regression with ols')
plt.legend()
plt.show()
