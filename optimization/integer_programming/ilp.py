from pulp import LpProblem, LpVariable, lpSum, LpMinimize, LpInteger, value, LpMaximize

# 创建线性整数规划问题
prob = LpProblem("Integer_Programming_Example", LpMaximize)

# 创建变量
x1 = LpVariable("x1", lowBound=0, cat=LpInteger)
x2 = LpVariable("x2", lowBound=0, cat=LpInteger)

# 定义目标函数
prob += 40 * x1 + 90 * x2

# 添加约束
prob += 9 * x1 + 7 * x2 <= 56
prob += 7 * x1 + 20 * x2 <= 70

# 求解问题
prob.solve()

# 输出结果
print("Status:", prob.status)
print("Objective Value:", value(prob.objective))
print("Optimal Values:")
print("x1 =", value(x1))
print("x2 =", value(x2))
