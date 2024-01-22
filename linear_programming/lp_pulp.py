from pulp import LpVariable, lpSum, LpMaximize, LpProblem, LpStatus, LpMinimize

# 创建线性规划问题
prob = LpProblem("Minimize_Profit", LpMaximize)

# 创建变量
x1 = LpVariable("x1")
x2 = LpVariable("x2", lowBound=0)

# 定义目标函数
prob += -2 * x1 - x2

# 添加约束
prob += -1 * x1 + x2 <= 1
prob += x1 + x2 >= 2
prob += x1 - 2 * x2 <= 4
prob += x1 + 2* x2 == 3.5

# 求解问题
prob.solve()
optimal_x1 = x1.value()
optimal_x2 = x2.value()
print("Optimal x1:", optimal_x1)
print("Optimal x2:", optimal_x2)
print("Status:", prob.status)
print("Objective Value:", prob.objective.value())
