import numpy as np
import pandas as pd
import statsmodels.api as sm

# 生成随机时间序列数据作为示例
np.random.seed(0)
data = np.random.randn(100, 2)  # 生成一个包含两个变量的时间序列数据

# 将数据转换为DataFrame
df = pd.DataFrame(data, columns=['Variable1', 'Variable2'])

# 拟合VAR 模型
model = sm.tsa.VAR(df)
results = model.fit()

# 打印模型的汇总信息
print(results.summary())
