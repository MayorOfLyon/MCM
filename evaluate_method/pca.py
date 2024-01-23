import numpy as np
import pandas as pd
from scipy import linalg

# read data
data = pd.read_csv('./data/pv.csv', encoding='gbk')
data = data.drop(['time'], axis=1)
data = data.dropna(axis=0, how='any')
data = data.to_numpy()

# data normalization
norm_data = (data-np.mean(data, axis=0))/np.std(data, axis=0, ddof=1)

# covariance matrix
cov_data = np.cov(norm_data.T)

cov_data = np.array([[1,0.79,0.36,0.76,0.25,0.51],
                      [0.79,1,0.31,0.55,0.17,0.35],
                      [0.36,0.31,1,0.35,0.64,0.58],
                      [0.76,0.55,0.35,1,0.16,0.38],
                      [0.25,0.17,0.64,0.16,1,0.63],
                      [0.51,0.35,0.58,0.38,0.63,1]])

# eigenvalue and eigenvector
eigenvalue, eigenvector = linalg.eig(cov_data)

# sort eigenvalues and eigenvectors
sorted_indices = np.argsort(eigenvalue)[::-1]
sorted_eigenvalues = eigenvalue[sorted_indices]
sorted_eigenvectors = eigenvector[:, sorted_indices]

# cumulative contribution rate
contribution_rate = sorted_eigenvalues/sorted_eigenvalues.sum()
contribution_rate = np.real(contribution_rate)
print(contribution_rate)
cum_contribution_rate = np.cumsum(contribution_rate)
print(cum_contribution_rate)

# select the first k eigenvectors
k = 2
eigenvector = sorted_eigenvectors[:, :k]
print(eigenvector[:,0])