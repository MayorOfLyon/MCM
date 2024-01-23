import numpy as np


def min2max(column):
    max_value = column.max()
    ans = max_value - column
    return np.array(ans)

def middle2max(column, best_value):
    abs_value = abs(column - best_value)
    max_value = abs_value.max()
    if max_value == 0:
        max_value = 1
    ans = 1 - abs_value / max_value
    return np.array(ans)

def interval2max(column, best_value, worst_value):
    max_value = max(worst_value - column.min(), column.max() - best_value)
    if max_value == 0:
        max_value = 1
    ans = []
    for i in range(len(column)):
        if column[i] < worst_value:
            value = 1-(worst_value - column[i]) / max_value
            ans.append(value)
        elif column[i] > best_value:
            value = 1-(column[i] - best_value) / max_value
            ans.append(value)
        else:
            ans.append(1)
    return np.array(ans)

def vec2log(vector):
    ans = []
    for i in range(len(vector)):
        if vector[i] == 0:
            ans.append(0)
        else:
            ans.append(np.log(vector[i]))
    return np.array(ans)


print("please input the number of samples:")
number = int(input())

print("please input the number of features:")
features = int(input())

print("please input the type of the features:1. max 2. min 3. middle 4. interval") 
kind = input().split(" ")

print("please input the matrix :")
matrix = np.zeros((number, features))
for i in range(number):
    matrix[i] = input().split(" ")
    matrix[i] = list(map(float, matrix[i]))

print("matrix:", matrix)

# 正向化
for i in range(features):
    # max
    if kind[i] == "2":
        matrix[:, i] = min2max(matrix[:, i])
    elif kind[i] == "3":
        print("please input the best value of feature", i+1)
        best_value = float(input())
        matrix[:, i] = middle2max(matrix[:, i], best_value)
    elif kind[i] == "4":
        print("please input the best value of feature", i+1)
        best_value = float(input())
        print("please input the worst value of feature", i+1)
        worst_value = float(input())
        matrix[:, i] = interval2max(matrix[:, i], best_value, worst_value)
print("正向化后的矩阵：", matrix)

# 标准化
for i in range(features):
    matrix[:, i] = matrix[:, i] / np.sqrt((matrix[:, i] ** 2).sum())
    
# 熵权法
weights = np.zeros(features)
for i in range(features):
    vector = matrix[:, i]
    norm_vector = vector / vector.sum()
    # 信息熵
    entropy = - np.sum((norm_vector * vec2log(norm_vector))) / np.log(number)
    weights[i] = 1- entropy
weights = weights / weights.sum()
print("熵权法权重：", weights)

# 计算最优解和最劣解
max_vector = matrix.max(axis=0)
min_vector = matrix.min(axis=0)

# 求解矩阵与最优解和最劣解的距离并使用熵权法加权
d_z = np.sqrt((weights*(matrix - max_vector) ** 2).sum(axis=1)) 
d_f = np.sqrt((weights*(matrix - min_vector) ** 2).sum(axis=1))

s = d_f / (d_f + d_z)
score = 100*s/s.sum()

for i in range(number):
    print("第", i+1, "个样本的得分为：", score[i])
