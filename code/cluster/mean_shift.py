import numpy as np
from sklearn.cluster import MeanShift,estimate_bandwidth


def txt_read(filename,delim):
    x_data = []
    with open(filename,'r') as f:
        for line in f:
#             data = line.strip().split(delim)
            data = [float(x) for x in line.strip().split(delim)]
            x_data.append(data)

    x_data = np.array(x_data)
    return x_data

data = txt_read('data_multivar.txt',',')


def meanshift(data):
	from sklearn.cluster import MeanShift, estimate_bandwidth
	# 通过指定输入参数创建一个均值漂移模型
	# 设置带宽参数bandwidth
	bandwidth = estimate_bandwidth(data,quantile=0.1,n_samples=len(data))
	# 这一方法会根据quantile比例的最大近邻距离，估算出整个数据集的平均最大近邻距离。默认这里的quantile是0.3,取值范围在[0,1]

	# MeanShift计算聚类
	meanshift_estimator = MeanShift(bandwidth=bandwidth,bin_seeding=True)

	# 训练模型
	meanshift_estimator.fit(data)

	# 提取标记
	labels = meanshift_estimator.labels_

	# 从模型中提取集群的中心点，然后打印集群数量
	centroids = meanshift_estimator.cluster_centers_
	num_clusters = len(np.unique(labels))
	print("Number of clusters in input data = ",num_clusters)

# 把集群数据可视化
# 画出数据点和聚类中心
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure()
# 为每种集群设置不同的标记
markers = '.*xv'

# 迭代数据点并画出它们：
for i,marker in zip(range(num_clusters),markers):
    # 画出属于某个集群中心的数据点
    plt.scatter(data[labels==i,0],data[labels==i,1],marker=marker,color='k')
    # 画出集群中心
    centroid = centroids[i]
    plt.plot(centroid[0],centroid[1],marker='o',markerfacecolor='b',markeredgecolor='k',markersize=15)

plt.title('Clusters and their centroids')
plt.show()