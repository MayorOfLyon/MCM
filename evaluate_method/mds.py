import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt

# 创建一个高维空间的距离矩阵
distance_matrix = np.array([[0, 3, 4, 2],
                            [3, 0, 1, 5],
                            [4, 1, 0, 6],
                            [2, 5, 6, 0]])

# 使用MDS进行降维
mds = MDS(n_components=2, dissimilarity='precomputed')
low_dimensional_representation = mds.fit_transform(distance_matrix)

# 绘制结果
plt.scatter(low_dimensional_representation[:, 0], low_dimensional_representation[:, 1])

# 添加标签
for i, txt in enumerate(["A", "B", "C", "D"]):
    plt.annotate(txt, (low_dimensional_representation[i, 0], low_dimensional_representation[i, 1]))

plt.title('MDS Visualization')
plt.show()
