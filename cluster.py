import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

# 生成模拟数据
n_samples = 300
random_state = 42
X, y = make_blobs(n_samples=n_samples, random_state=random_state)

# 聚类模型
kmeans = KMeans(n_clusters=3, random_state=random_state)
y_kmeans = kmeans.fit_predict(X)

# rand score
ari = adjusted_rand_score(y, y_kmeans)
print("Adjusted Rand Index (ARI):", ari)

# 可视化聚类结果
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, alpha=0.75)
plt.title('KMeans Clustering')
plt.show()

print("hello world")