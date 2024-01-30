from scipy.optimize import linprog

# 化成标准型
# 目标函数系数
c = [2, 1]
# 不等式约束，系数
A = [[-1, 1],
     [-1, -1],
     [1, -2],
     [0, -1]]
b = [[1], [-2], [4],[0]]
# 等式约束，系数
Aeq = [[1, 2]]
beq = [3.5]

res = linprog(c, A, b, Aeq, beq)
print(res.fun, res.x)
