from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# 生成一些示例数据
np.random.seed(42)
X = np.random.rand(100, 10)  # 100个样本，每个样本有10个特征
y = np.random.rand(100, 2)   # 对应的两个目标

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用线性回归模型，并使用 MultiOutputRegressor 包装
model = MultiOutputRegressor(LinearRegression())

# 训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 评估模型性能
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# 获取系数和截距
coefficients = [estimator.coef_ for estimator in model.estimators_]
intercepts = [estimator.intercept_ for estimator in model.estimators_]

print("Coefficients for Target 1:", coefficients[0])
print("Intercept for Target 1:", intercepts[0])
print("Coefficients for Target 2:", coefficients[1])
print("Intercept for Target 2:", intercepts[1])
