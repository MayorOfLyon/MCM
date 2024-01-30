import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.impute import KNNImputer
import scipy.stats as st
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, make_scorer
from lightgbm import LGBMRegressor
sns.set_theme(style="darkgrid")

# load data
path = './data/'
Train_data = pd.read_csv(path+'used_car_train_20200313.csv', sep=' ')

# # observe the data
# print("shape is ",Train_data.shape)
# print(pd.concat([Train_data.head(), Train_data.tail()]))
# print(Train_data.info())
# print(Train_data.describe())

# # observe the missing data
# print(Train_data.isnull().sum())
# missing = Train_data.isnull().sum()
# missing = missing[missing > 0]
# missing.sort_values(inplace=True)
# missing.plot.bar()
# plt.show()

# # deal with the missing data
# # 1. interpoaltion
# degree = 2
# for column in Train_data.columns:
#     missing = Train_data[column].isnull()
#     if missing.sum() > 0:
#         index = Train_data.index[~missing]
#         missing_index = Train_data.index[missing]
#         poly_coefficients = np.polyfit(index, Train_data.loc[~missing, column], degree)
#         poly_interpolation = np.poly1d(poly_coefficients)
#         Train_data.loc[missing_index, column] = poly_interpolation(missing_index)
# # 2. delete the data
# Train_data = Train_data.dropna()
# # 3. replace the data
# Train_data = Train_data.fillna(Train_data.mean())
# Train_data= Train_data.fillna(0)
# # 4. do nothing
# # 5. machine learning interpolation
# imputer = KNNImputer(n_neighbors=5)
# Train_data = imputer.fit_transform(Train_data)

# # observe the abnormal data. The code below is just an example.
# # - is also the null data
# print(Train_data['notRepairedDamage'].value_counts())
# # The data distribution is severely skewed
# print(Train_data["seller"].value_counts())
# del Train_data["seller"]

# # deal with the abnormal data
# clf = IsolationForest(contamination=0.05)  # contamination 表示异常值的比例
# clf.fit(Train_data[['feature_of_interest']])
# Train_data['is_outlier'] = clf.predict(Train_data[['feature_of_interest']])
# # 将异常值标记为 -1，正常值标记为 1
# Train_data['is_outlier'] = np.where(Train_data['is_outlier'] == -1, 'Outlier', 'Inlier')
# plt.scatter(Train_data.index, Train_data['feature_of_interest'], c=Train_data['is_outlier'], cmap='viridis')
# plt.title('Isolation Forest - Outlier Detection')
# plt.xlabel('Index')
# plt.ylabel('Feature of Interest')
# plt.show()

# deal with the duplicate data
duplicate_rows = Train_data[Train_data.duplicated(keep=False)]
Train_data_no_duplicates = Train_data.drop_duplicates(keep='first')
# Train_data_no_duplicates = Train_data.drop_duplicates(subset=['column1', 'column2'])

# # Understand the distribution of predicted values
# print("Skewness: %f" % Train_data['price'].skew())
# print("Kurtosis: %f" % Train_data['price'].kurt())
# y = Train_data['price']
# plt.figure(1); plt.title('Johnson SU')
# sns.distplot(y, kde=False, fit=st.johnsonsu)
# plt.figure(2); plt.title('Normal')
# sns.distplot(y, kde=False, fit=st.norm)
# plt.figure(3); plt.title('Log Normal')
# sns.distplot(y, kde=False, fit=st.lognorm)
# plt.show()

# # long tail distribution, log transformation
# Train_data['price'] = np.log1p(Train_data['price'])


# # categorical features and numeric features analysis
# numeric_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13','v_14' ]
# categorical_features = ['name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode',]
# for cat_fea in categorical_features:
#     print(cat_fea + "的特征分布如下：")
#     print("{}特征有个{}不同的值".format(cat_fea, Train_data[cat_fea].nunique()))
#     print(Train_data[cat_fea].value_counts())

# # correlation analysis
# numeric_features.append('price')
# price_numeric = Train_data[numeric_features]
# correlation = price_numeric.corr()
# print(correlation['price'].sort_values(ascending = False),'\n')
# f , ax = plt.subplots(figsize = (7, 7))
# plt.title('Correlation of Numeric Features with Price',y=1,size=16)
# sns.heatmap(correlation,square = True,  vmax=0.8)
# plt.show()

# # pairplot
# sns.set()
# columns = ['price', 'v_12', 'v_8' , 'v_0', 'power', 'v_5',  'v_2', 'v_6', 'v_1', 'v_14']
# sns.pairplot(Train_data[columns],size = 2 ,kind ='scatter',diag_kind='kde')
# plt.show()

# # Regression relationship between target variable and other numberical variables.
# # Polynomial functions can be used to fit.
# sns.set_theme(style="darkgrid")
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(24, 20))
# Y_train = Train_data['price']
# for i, feature in enumerate(['v_12', 'v_8', 'v_0', 'power']):
#     scatter_plot = pd.concat([Y_train, Train_data[feature]], axis=1)
#     color = sns.color_palette("husl")[i]  # 使用Seaborn的husl调色板获取不同颜色
#     sns.regplot(x=feature, y='price', data=scatter_plot, scatter=True, fit_reg=True, ax=eval(f'ax{i + 1}'), scatter_kws={'color': color})
# plt.show()

# # categorical features: boxplot
# categorical_features = ['model',
#  'brand',
#  'bodyType',
#  'fuelType',
#  'gearbox',
#  'notRepairedDamage']
# for c in categorical_features:
#     Train_data[c] = Train_data[c].astype('category')
#     if Train_data[c].isnull().any():
#         Train_data[c] = Train_data[c].cat.add_categories(['MISSING'])
#         Train_data[c] = Train_data[c].fillna('MISSING')
# def boxplot(x, y, **kwargs):
#     sns.boxplot(x=x, y=y)
#     x=plt.xticks(rotation=90)
# f = pd.melt(Train_data, id_vars=['price'], value_vars=categorical_features)
# g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, height=5)
# g = g.map(boxplot, "value", "price")
# plt.show()

# # categorical features: violinplot
# categorical_features = ['model',
#  'brand',
#  'bodyType',
#  'fuelType',
#  'gearbox',
#  'notRepairedDamage']
# catg_list = categorical_features
# target = 'price'
# for catg in catg_list :
#     sns.violinplot(x=catg, y=target, data=Train_data)
#     plt.show()

# # categorical features: barplot
# categorical_features = ['model',
#  'brand',
#  'bodyType',
#  'fuelType',
#  'gearbox',
#  'notRepairedDamage']
# def bar_plot(x, y, **kwargs):
#     sns.barplot(x=x, y=y)
#     x=plt.xticks(rotation=90)
# f = pd.melt(Train_data, id_vars=['price'], value_vars=categorical_features)
# g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, height=5)
# g = g.map(bar_plot, "value", "price")
# plt.show()

# # categorical features: countplot
# categorical_features = ['model',
#  'brand',
#  'bodyType',
#  'fuelType',
#  'gearbox',
#  'notRepairedDamage']
# def count_plot(x,  **kwargs):
#     sns.countplot(x=x)
#     x=plt.xticks(rotation=90)
# f = pd.melt(Train_data,  value_vars=categorical_features)
# g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False, height=5)
# g = g.map(count_plot, "value")
# plt.show()

# # binning
# bin = [i*10 for i in range(31)]
# Train_data['power_bin'] = pd.cut(Train_data['power'], bin, labels=False)
# print(Train_data[['power_bin','power']].head())

# # standardization and normalization
# scaler = StandardScaler()
# Train_data['power'] = scaler.fit_transform(Train_data['power'].values.reshape(-1, 1))
# scaler = MinMaxScaler()
# Train_data['power'] = scaler.fit_transform(Train_data['power'].values.reshape(-1, 1))
# print(Train_data['power'].head())

# # onehotencoder for categorical data
# Train_data = pd.get_dummies(Train_data, columns=['model', 'brand', 'bodyType', 'fuelType',
#                                      'gearbox', 'notRepairedDamage'])

# # labelencoder for label data
# label_encoder = LabelEncoder()
# Y_train = Train_data['price']
# Y_train= label_encoder.fit_transform(Y_train)

# feature engineering
# hahahahah

# # Data dimensionality reduction
# lgb = LGBMRegressor()
# Y_train = Train_data['price']
# Train_data = Train_data.drop(['price'], axis=1)
# selector = SelectFromModel(lgb)
# X_sfm_gbdt = selector.fit_transform(Train_data, Y_train)
# scores = cross_val_score(lgb, X=X_sfm_gbdt, y=Y_train, verbose=0, cv=5,
#                          scoring=make_scorer(mean_absolute_error))
# print(np.mean(scores))