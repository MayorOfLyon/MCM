import numpy as np
import matplotlib.pyplot as plt

def logistic_growth(t, K, r, t0):
    """
    Logistic Growth Model 的实现
    :param t: 时间
    :param K: 环境容量
    :param r: 增长速率参数
    :param t0: 开始增长的时间
    :return: 种群大小
    """
    return K / (1 + np.exp(-r * (t - t0)))

# 设置模型参数
K = 1000  # 环境容量
r = 0.1   # 增长速率
t0 = 0    # 开始增长的时间

# 生成时间序列
time_steps = np.arange(0, 500, 1)

# 计算种群大小
population_size = logistic_growth(time_steps, K, r, t0)

# 绘制 S 形曲线
plt.plot(time_steps, population_size, label='Logistic Growth Model')
plt.xlabel('Time')
plt.ylabel('Population Size')
plt.title('Logistic Growth Model')
plt.legend()
plt.show()
