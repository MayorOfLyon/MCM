import numpy as np

def objective_function(x):
    return 40 * x[0] + 90 * x[1]

def constraints(x):
    constraint1 = 9*x[0] + 7*x[1] <= 56
    constraint2 = 7*x[0] + 20*x[1] <= 70
    return constraint1 and constraint2

def monte_carlo_integer_programming(iterations, bounds):
    best_solution = None
    best_value = float('-inf')

    for _ in range(iterations):
        # 随机生成整数解
        random_solution = np.random.randint(bounds[:,0], bounds[:, 1], size=(len(bounds)))  # 生成两个整数

        # 检查是否满足约束条件
        if not constraints(random_solution):
            continue

        # 评估目标函数
        value = objective_function(random_solution)

        # 如果找到更好的解，则更新最佳解
        if value > best_value:
            best_solution = random_solution
            best_value = value

    return best_solution, best_value

# bounds
LB = [0]*2
UB = [100]*2
bounds = tuple(zip(LB,UB))

iterations = 10000

# 使用蒙特卡洛方法求解整数规划问题
best_solution, best_value = monte_carlo_integer_programming(iterations, np.array(bounds))

# 输出结果
print("Best Solution:", best_solution)
print("Best Value:", best_value)
