import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# 创建一些示例二维数据
X = np.array([[1, 1], [1, 3], [2, 2], [2.5, 5], [3, 1], 
        [4, 2], [2, 3.5], [3, 3], [3.5, 4]])

# 目标：对于任意给定点找到其3个最近邻
# 寻找最近邻的数量
num_neighbors = 3
# 输入数据点
input_point = np.array([[2.6, 1.7]])
# 画出数据分布
plt.figure()
plt.scatter(X[:,0], X[:,1], marker='o', s=25, color='k')

# 建立最近邻模型
knn = NearestNeighbors(n_neighbors=num_neighbors, algorithm='ball_tree').fit(X)
# 计算输入点与输入数据中所有点的距离
distances, indices = knn.kneighbors(input_point)

# 打印k个最近邻
print("\nk nearest neighbors")
for rank, index in enumerate(indices[0][:num_neighbors]):
    print(str(rank+1) + " -->", X[index])
# indices数组是一个已排序的数组


# 画出输入数据点，并突出显示k个最近邻
plt.figure()
plt.scatter(X[:,0], X[:,1], marker='o', s=25, color='k')
plt.scatter(X[indices][0][:][:,0], X[indices][0][:][:,1], 
        marker='o', s=150, color='k', facecolors='none')
plt.scatter(input_point[:,0], input_point[:,1],
        marker='x', s=150, color='k')

plt.show()