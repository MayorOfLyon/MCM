import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from hmmlearn import hmm
import seaborn as sns

def vec2log(vector):
    ans = []
    for i in range(len(vector)):
        if vector[i] == 0:
            ans.append(0)
        else:
            ans.append(np.log(vector[i]))
    return np.array(ans)

# load data
Train_data = pd.read_csv('./preprocessed_data/merged_data.csv')
date = Train_data['review_date']
Train_data = Train_data[['votes_ratio', 'star_rating', 'text_rating', 'Repeated_Purchase']]
is_negative = Train_data.lt(0).any()
print(is_negative)
# 熵权法
# 1.数据标准化（正数）
for index, column in enumerate(Train_data.columns):
    if is_negative[index]:
        # minmax
        min_val = Train_data[column].min()
        max_val = Train_data[column].max()
        Train_data[column] = Train_data[column].apply(lambda x: (x - min_val) / (max_val - min_val))
    else:
        Train_data[column] = Train_data[column] / np.sqrt((Train_data[column] ** 2).sum())
        
# 2.计算每个指标的熵值
# 熵权法
features = len(Train_data.columns)
weights = np.zeros(features)
entropys = np.zeros(features)
for index, column in enumerate(Train_data.columns):
    vector = Train_data[column].values
    norm_vector = vector / vector.sum()
    # 信息熵
    entropy = - np.sum((norm_vector * vec2log(norm_vector))) / np.log(len(Train_data))
    weights[index] = 1 - entropy
    entropys[index] = entropy
weights = weights / weights.sum()
print("熵权法权重：", weights)
print("熵权法熵值：", entropys)

# load data
data_dryer = pd.read_csv('./postprocess_data/merged_data_dryer.csv')
data_microwave = pd.read_csv('./postprocess_data/merged_data_microwave.csv')
data_pacifier = pd.read_csv('./postprocess_data/merged_data_pacifer.csv')

data_dryer = data_dryer[data_dryer['product_parent'] == 732252283]

data_dryer = data_dryer[['review_date', 'votes_ratio', 'star_rating', 'text_rating', 'Repeated_Purchase']]
data_microwave = data_microwave[['review_date', 'votes_ratio', 'star_rating', 'text_rating', 'Repeated_Purchase']]
data_pacifier = data_pacifier[['review_date', 'votes_ratio', 'star_rating', 'text_rating', 'Repeated_Purchase']]

data_dryer['final_metric'] = data_dryer[data_dryer.columns[1:]].dot(weights)
data_microwave['final_metric'] = data_microwave[data_microwave.columns[1:]].dot(weights)
data_pacifier['final_metric'] = data_pacifier[data_pacifier.columns[1:]].dot(weights)

data_dryer['review_date'] = pd.to_datetime(data_dryer['review_date'])
data_microwave['review_date'] = pd.to_datetime(data_microwave['review_date'])
data_pacifier['review_date'] = pd.to_datetime(data_pacifier['review_date'])

data_dryer = data_dryer.sort_values(by='review_date')
data_microwave = data_microwave.sort_values(by='review_date')
data_pacifier = data_pacifier.sort_values(by='review_date')

# # 对每一天的数据求平均
# data_dryer = data_dryer.groupby('review_date').mean().reset_index()
# data_microwave = data_microwave.groupby('review_date').mean().reset_index()
# data_pacifier = data_pacifier.groupby('review_date').mean().reset_index()

import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
np.random.seed(42)

# 训练GMM-HMM模型
data = data_dryer['final_metric'].values.reshape(-1, 1)
train_data = data[:int(0.8 * len(data))]
test_data = data[int(0.8 * len(data)):]
print("len data is ",len(data))
print("len train_data is ",len(train_data))

num_states = 8
num_components = 1  # 每个状态对应的高斯分量数量
hmm_dryer = hmm.GMMHMM(n_components=num_states, n_mix=num_components, covariance_type="full", n_iter=5000)

hmm_dryer.fit(train_data)

# 预测
hidden_state = hmm_dryer.predict(data)
predict = np.dot(hmm_dryer.means_.T, hmm_dryer.predict_proba(data).T)
predict = predict.mean(axis=1)
predict = predict.T

print("RMSE is ", np.sqrt(np.mean((predict - data) ** 2)))
# plot
hidden = hidden_state - 3
hidden = hidden * 0.05
plt.figure(figsize=(15, 8))
# plt.plot(hidden, label='Hidden State')
plt.plot(data, label='True')
plt.plot(predict, label='Predicted')
plt.title('GMM-HMM')
plt.legend()
plt.show()

# 状态转移矩阵
transmat = hmm_dryer.transmat_
max_prob_transition = np.unravel_index(np.argmax(transmat, axis=None), transmat.shape)
print(f"Most probable transition: State {max_prob_transition[0]} to State {max_prob_transition[1]} with probability {transmat[max_prob_transition]}")

sns.heatmap(transmat, cmap='Blues', annot=True, fmt=".2f", cbar=False)
plt.title('State Transition Matrix')
plt.show()

startprob = hmm_dryer.startprob_
print(f"Initial state matrix: {startprob}")
print(f"Most probable initial state: {np.argmax(startprob)} with probability {np.max(startprob)}")
# print("hidden staet is ", hidden_state)
from scipy.stats import pearsonr, spearmanr
# 计算Pearson相关系数
data = data.flatten()
corr_pearson, _ = pearsonr(data, hidden_state)
print(f'Pearson correlation: {corr_pearson}')
# 计算Spearman秩相关系数
corr_spearman, _ = spearmanr(data, hidden_state)
print(f'Spearman correlation: {corr_spearman}')

means = hmm_dryer.means_
print(f"Gaussian means: {means}")
covars = hmm_dryer.covars_
print(f"Gaussian covariances: {covars}")

# count_state3 = np.count_nonzero(hidden_state == 3)
# ratio_state3 = count_state3 / len(hidden_state)
# print(f"The ratio of state 3: {ratio_state3}")
# print("hidden state is", hidden_state)