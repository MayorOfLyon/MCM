from scipy.optimize import minimize
from numpy import ones
import numpy as np

# 目标函数: minimize
def objective(x):
    x1, x2,x3,x4,x5 = x
    return - (x1**2 + x2**2 + 3* x3**2 + 4*x4**2 + 2*x5**2 - 8*x1 - 2*x2 - 3*x3 - x4 - 2*x5)
# 变量限界
LB = [0]*5
UB = [99]*5
bound = tuple(zip(LB,UB))

# 限界条件: constraints>=0
def constraint1(x):
    x1, x2,x3,x4,x5 = x
    return -(x1 + x2 + x3 + x4 + x5 - 400)

def constraint2(x):
    x1, x2,x3,x4,x5 = x
    return -(x1 + 2*x2 + 2*x3 + x4 + 6*x5 - 800)

def constraint3(x):
    x1, x2,x3,x4,x5 = x
    return -(2*x1 + x2 + 6*x3 - 200)

def constraint4(x):
    x1, x2,x3,x4,x5 = x
    return -(x3 + x4 + 5*x5 - 200)
# 等式约束用eq
constraints = [ {'type': 'ineq', 'fun': constraint1},
                {'type': 'ineq', 'fun': constraint2},
                {'type': 'ineq', 'fun': constraint3},
                {'type': 'ineq', 'fun': constraint4}]

# 定义蒙特卡洛抽样函数
def monte_carlo_sampling(num_samples, variable_bounds):
    samples = np.random.uniform(variable_bounds[:, 0], variable_bounds[:, 1], size=(num_samples, len(variable_bounds)))
    return samples

# 蒙特卡洛法遍历决策空间
num_samples = 10000
samples = monte_carlo_sampling(num_samples, np.array(bound))

# 记录优化结果
optimal_results = []

for sample in samples:
    result = minimize(objective, sample, bounds=bound, constraints=constraints)
    optimal_results.append(result.fun)

# 选择最佳结果
best_result_index = np.argmin(optimal_results)
best_result = optimal_results[best_result_index]
best_sample = samples[best_result_index]
print("最优结果：", -best_result)
print("最优解：", best_sample)