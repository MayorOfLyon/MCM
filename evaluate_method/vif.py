import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 检查多重共线性

# 示例数据框
data = {
    'X1': [1, 2, 3, 4, 5],
    'X2': [5, 4, 2, 7, 10],
    'X3': [5, 6, 7, 8, 9]
}

df = pd.DataFrame(data)

# 添加截距项
df['Intercept'] = 1

# 计算VIF
def calculate_vif(data_frame, dependent_variable):
    features = data_frame.drop(dependent_variable, axis=1)
    vif_data = pd.DataFrame()
    vif_data["Variable"] = features.columns
    vif_data["VIF"] = [variance_inflation_factor(features.values, i) for i in range(features.shape[1])]
    return vif_data

# 调用函数计算VIF
vif_result = calculate_vif(df, 'X1')

# 输出结果
print(vif_result)
