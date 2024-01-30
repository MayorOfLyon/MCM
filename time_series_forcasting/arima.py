import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller as ADF
import itertools
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# read data
ChinaBank = pd.read_csv('../data/ChinaBank.csv',index_col = 'Date',parse_dates=['Date'])
# print(ChinaBank.head())
ChinaBank.index = pd.to_datetime(ChinaBank.index)
sub = ChinaBank.loc['2014-01':'2014-06','Close']
# print(sub.head())

# data split
train = sub.loc['2014-01':'2014-03']
test = sub.loc['2014-04':'2014-06']

# #查看训练集的时间序列与数据(只包含训练集)
# plt.figure(figsize=(12,6))
# plt.plot(train)
# plt.xticks(rotation=45) #旋转45度
# plt.show()

# # 差分
ChinaBank['diff_1'] = ChinaBank['Close'].diff(1) #1阶差分
ChinaBank['diff_2'] = ChinaBank['diff_1'].diff(1) #2阶差分
# fig = plt.figure(figsize=(12,10))
# #原数据
# ax1 = fig.add_subplot(311)
# ax1.plot(ChinaBank['Close'])
# #1阶差分
# ax2 = fig.add_subplot(312)
# ax2.plot(ChinaBank['diff_1'])
# #2阶差分
# ax3 = fig.add_subplot(313)
# ax3.plot(ChinaBank['diff_2'])
# plt.show()

# ADF test 查看P值是否很接近0（第二列）
# 计算原始序列、一阶差分序列、二阶差分序列的单位根检验结果
ChinaBank['diff_1'] = ChinaBank['diff_1'].fillna(0)
ChinaBank['diff_2'] = ChinaBank['diff_2'].fillna(0)

timeseries_adf = ADF(ChinaBank['Close'].tolist())
timeseries_diff1_adf = ADF(ChinaBank['diff_1'].tolist())
timeseries_diff2_adf = ADF(ChinaBank['diff_2'].tolist())
# 打印单位根检验结果
print('timeseries_adf : ', timeseries_adf)
print('timeseries_diff1_adf : ', timeseries_diff1_adf)
print('timeseries_diff2_adf : ', timeseries_diff2_adf)

# # ACF and PACF
# #绘制
# fig = plt.figure(figsize=(12,7))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(train, lags=20,ax=ax1)
# ax1.xaxis.set_ticks_position('bottom') # 设置坐标轴上的数字显示的位置，top:显示在顶部  bottom:显示在底部
# #fig.tight_layout()
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(train, lags=20, ax=ax2)
# ax2.xaxis.set_ticks_position('bottom')
# #fig.tight_layout()
# plt.show()

# 寻找p,q
p_min = 0
q_min = 0
p_max = 5
q_max = 5   
d = 1
#以BIC准则为依据
results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min,p_max+1)],
                           columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])

for p,q in itertools.product(range(p_min,p_max+1),range(q_min,q_max+1)):
    if p==0 and q==0:
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
        continue
    try:
        model = sm.tsa.ARIMA(train, order=(p, d, q),
                               #enforce_stationarity=False,
                               #enforce_invertibility=False,
                              )
        results = model.fit()
        results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
    except:
        continue
    
print(results_bic)
bic_matrix = pd.DataFrame(results_bic) 
# p,q = bic_matrix.stack().idxmin() 
# print(u'BIC最小的p值和q值为：%s、%s' %(p,q))

# # 热力图
# results_bic = results_bic[results_bic.columns].astype(float)
# fig, ax = plt.subplots(figsize=(10, 8))
# ax = sns.heatmap(results_bic,
#                  mask=results_bic.isnull(),
#                  ax=ax,
#                  annot=True,
#                  fmt='.2f',
#                  cmap="Purples"
#                  )

# ax.set_title('BIC')
# plt.show()
# print("based on BIC: ",results_bic.stack().idxmin())

# # based on BIC and AIC
# train_results = sm.tsa.arma_order_select_ic(train, ic=['aic', 'bic'], trend='n', max_ar=8, max_ma=8)
# print('AIC', train_results.aic_min_order)
# print('BIC', train_results.bic_min_order)

# arima model
p = 1
q = 0
d = 1
model = sm.tsa.ARIMA(train, order=(p,d,q))
results = model.fit()

#残差检验
resid = results.resid 
fig, ax = plt.subplots(figsize=(12, 5))
ax = sm.graphics.tsa.plot_acf(resid, lags=40,ax=ax)
plt.show()

predict_sunspots = results.predict(dynamic=False)
# print(predict_sunspots)

# 测试集
plt.figure(figsize=(12,6))
plt.plot(train[1:])
plt.xticks(rotation=45) #旋转45度
plt.plot(predict_sunspots[1:])
plt.title("test")
plt.show()

# 预测
fig, ax = plt.subplots(figsize=(12, 6))
ax = sub[1:].plot(ax=ax)
predict_sunspots[1:].plot(ax=ax)
plt.title("forecast")
plt.show()