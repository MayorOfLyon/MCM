import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# 生成一些示例数据
X, y = make_blobs(n_samples=300, centers=4, random_state=42)

# 使用 Agglomerative Clustering 进行聚类
agglomerative_clustering = AgglomerativeClustering(n_clusters=4)
labels = agglomerative_clustering.fit_predict(X)

# 绘制聚类结果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', s=50, edgecolors='w')
plt.title('Agglomerative Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# 绘制树状图
linked = linkage(X, 'ward')  # 'ward' 是一种聚类算法
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Cluster Distance')
plt.show()
