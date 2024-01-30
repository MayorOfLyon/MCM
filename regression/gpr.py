import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 生成一些示例数据
rng = np.random.default_rng(seed=42)
X = np.sort(5 * rng.random((50, 1)), axis=0)
y = np.sin(X).ravel() + 0.1 * rng.normal(size=X.shape[0])

# 定义高斯过程的核函数
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))

# 创建高斯过程回归模型
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# 训练模型
gp.fit(X, y)

# 生成预测
x_pred = np.atleast_2d(np.linspace(0, 5, 1000)).T
y_pred, sigma = gp.predict(x_pred, return_std=True)

# 绘制结果
plt.figure(figsize=(8, 4))
plt.scatter(X, y, c='r', s=20, zorder=10, edgecolors=(0, 0, 0))
plt.plot(x_pred, y_pred, 'k', lw=2)
plt.fill_between(x_pred.ravel(), y_pred - sigma, y_pred + sigma, alpha=0.2, color='k')
plt.title("Gaussian Process Regression")
plt.show()
