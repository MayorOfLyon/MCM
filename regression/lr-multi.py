import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd

# 生成一些示例数据
np.random.seed(0)
X = np.random.rand(100, 3)
y = 2 * X[:, 0] + 3 * X[:, 1] + 4 * X[:, 2] + np.random.randn(100)

# 添加常数列
X_with_intercept = sm.add_constant(X)

# 创建多元线性回归模型
model = sm.OLS(y, X_with_intercept).fit()

# 打印统计量
print(model.summary())

# 回归系数以及置信区间
conf_int = model.conf_int(alpha=0.05)  # 默认alpha=0.05表示95%置信区间
print("回归系数和截距的置信区间:")
print(conf_int)

# 获取残差和置信区间
residuals = model.resid
predictions = model.get_prediction()
conf_int = predictions.conf_int()
# print(conf_int)

# 残差的无偏估计
residual_std_dev = np.std(residuals)
print(f"残差的标准差: {residual_std_dev}")

# 计算触须的长度
whisker_length = np.abs(conf_int[:, 1] - residuals)

# 绘制残差图
plt.scatter(range(len(residuals)), residuals, alpha=0.6)

# 在每个点上添加触须
for i in range(len(residuals)):
    x_val = i
    y_val = residuals[i]
    lower_ci, upper_ci = conf_int[i]
    length = whisker_length[i]

    plt.plot([x_val, x_val], [y_val - length, y_val + length], color='black', linewidth=1)

plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title('Residual Plot with Whiskers')
plt.xlabel('Data Points')
plt.ylabel('Residuals')
plt.show()


# step-wise regression
df = pd.DataFrame(X, columns=['X1', 'X2', 'X3'])
df['y'] = y

def forward_selected(data, response):
    """逐步回归选择特征"""
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')
    while remaining:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {}".format(response, ' + '.join(selected + [candidate]))
            model = sm.OLS.from_formula(formula, data)
            results = model.fit()
            score = results.aic
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop(0)
        if current_score > best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {}".format(response, ' + '.join(selected))
    model = sm.OLS.from_formula(formula, data)
    results = model.fit()
    return results

# 执行逐步回归
results = forward_selected(df, 'y')

# 打印结果摘要
print(results.summary())