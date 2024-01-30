import numpy as np
import libpysal
import pysal.model

# 生成一个随机的空间权重矩阵
np.random.seed(123)
w = libpysal.weights.lat2W(4, 4)

# 生成随机的自变量和因变量
X = np.random.rand(16, 3)
y = np.random.rand(16)

# 创建空间杜宾模型
model = pysal.model.spreg.ML_Lag(y, X, w, name_y='y', name_x=['x1', 'x2', 'x3'], name_w='w', name_ds='data')

# 输出模型摘要
print(model.summary)