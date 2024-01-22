import numpy as np


def icc_calculate(Y, icc_type):
    [n, k] = Y.shape

    # 自由度
    dfall = n * k - 1  # 所有自由度
    dfe = (n - 1) * (k - 1)  # 剩余自由度
    dfc = k - 1  # 列自由度
    dfr = n - 1  # 行自由度

    # 所有的误差
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions
    x0 = np.tile(np.eye(n), (k, 1))  # subjects
    X = np.hstack([x, x0])

    # 误差均方
    predicted_Y = np.dot(
        np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T), Y.flatten("F")
    )
    residuals = Y.flatten("F") - predicted_Y
    SSE = (residuals ** 2).sum()

    MSE = SSE / dfe

    # 列均方
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n
    MSC = SSC / dfc

    # 行均方
    SSR = ((np.mean(Y, 1) - mean_Y) ** 2).sum() * k
    MSR = SSR / dfr

    # 单项随机
    if icc_type == "icc(1)":
        SSW = SST - SSR  # 剩余均方
        MSW = SSW / (dfall - dfr)

        ICC1 = (MSR - MSW) / (MSR + (k - 1) * MSW)
        ICC2 = (MSR - MSW) / MSR

    # 双向随机
    elif icc_type == "icc(2)":

        ICC1 = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)
        ICC2 = (MSR - MSE) / (MSR + (MSC - MSE) / n)

    # 双向混合
    elif icc_type == "icc(3)":

        ICC1 = (MSR - MSE) / (MSR + (k - 1) * MSE)
        ICC2 = (MSR - MSE) / MSR

    return ICC1, ICC2


'_____________示例______________'

a = [[90,95,89,92,89,80,91,94,84,95],
     [89,80,89,93,91,80,94,92,82,90],
     [100,100,91,91,94,81,93,92,84,96]]
b = np.array(a)
b = b.T
icc_type = "icc(1)"
icc1, icc2 = icc_calculate(b, icc_type)
print('模型{}:\t'.format(icc_type))
'''
ICC的值介于0~1之间：
    小于0.5表示一致性较差；
    0.5~0.75一致性中等；
    0.75~0.9一致性较好；
    大于0.9一致性极好
'''
print('单个测量:', icc1)
print('平均测量:', icc2)

# ICC详细讲解
# https://blog.csdn.net/qq_43426908/article/details/124365536
