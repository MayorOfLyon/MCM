import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 定义 SIR 模型的微分方程
def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

# 定义模型参数
beta = 0.3  # 传播率
gamma = 0.1  # 恢复率

# 初始条件：初始易感者、感染者、康复者的人数
initial_conditions = [0.99, 0.01, 0.0]

# 时间点
t = np.linspace(0, 200, 1000)

# 求解微分方程
solution = odeint(sir_model, initial_conditions, t, args=(beta, gamma))

# 绘制结果
plt.plot(t, solution[:, 0], label='Susceptible')
plt.plot(t, solution[:, 1], label='Infectious')
plt.plot(t, solution[:, 2], label='Recovered')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('SIR Model')
plt.legend()
plt.show()
