import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

# 生成示例时间序列数据
np.random.seed(42)
time_series = np.concatenate([np.random.normal(loc=i, scale=1, size=50) for i in range(3)])

# 将时间序列转换为二维数组，以符合模型输入格式
X = time_series.reshape(-1, 1)
print(X.shape)
# 定义 HMM 模型
model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)

# 用时间序列拟合 HMM 模型
model.fit(X)

# 预测未来的时间序列
future_steps = 10
predicted_states, _ = model.sample(future_steps)

# 获取状态转移序列
state_sequence = model.predict(X)
print(state_sequence)

# 绘制原始时间序列及预测的未来时间序列
plt.figure(figsize=(10, 6))
plt.plot(time_series, label='Observed Time Series', linestyle='-', marker='o')
plt.plot(range(len(time_series), len(time_series) + future_steps), predicted_states, label='Predicted Future States', linestyle='--', marker='x')
plt.title('HMM Model Prediction for Time Series')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
