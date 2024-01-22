import cvxpy as cp

# 创建变量
x1 = cp.Variable()
x2 = cp.Variable()

# 定义目标函数和约束
objective = cp.Maximize(-2 * x1 - x2)
constraints = [ -x1 + x2 <= 1, x1 + x2 >= 2, x1 - 2 * x2 <= 4, x1 + 2 * x2 == 3.5, x2 >=0 ]

# 创建问题并求解
prob = cp.Problem(objective, constraints)
result = prob.solve()
print("Optimal solution:")
print("result =", result)
print("x1 =", x1.value)
print("x2 =", x2.value)
