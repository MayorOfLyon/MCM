import numpy as np
import matplotlib.pyplot as plt

def discrete_system(a, b, x):
    """
    一阶离散时间系统的差分方程模型
    :param a: 系统参数
    :param b: 系统参数
    :param x: 输入信号的时间序列
    :return: 输出信号的时间序列
    """
    n = len(x)
    y = np.zeros(n)

    for i in range(1, n):
        y[i] = a * y[i-1] + b * x[i]

    return y

# 生成输入信号
time_steps = np.arange(0, 10, 1)
input_signal = np.sin(time_steps)

# 设置系统参数
a = 0.5
b = 1.0

# 模拟系统响应
output_signal = discrete_system(a, b, input_signal)

# 绘制输入和输出信号
plt.plot(time_steps, input_signal, label='Input Signal')
plt.plot(time_steps, output_signal, label='Output Signal')
plt.xlabel('Time Steps')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
