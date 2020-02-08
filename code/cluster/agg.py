import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

# 定义一个实现凝聚层次聚类的函数
def perform_clustering(data,connectivity,title,num_clusters=3,linkage='ward'):
    plt.figure()
    model = AgglomerativeClustering(linkage=linkage,connectivity=connectivity,n_clusters=num_clusters)
    # linkage指定链接算法，取值为['ward','complete','average']
    # connectivity：一个数组或者可调用对象或者None，用于指定连接矩阵
    # n_clusters：一个整数，指定分类簇的数量
    model.fit(data)
    # 提取标记
    labels = model.labels_
    # 为每种集群设置不同的标记
    markers = '.vx'
    # 迭代数据，用不同的标记聚类的点画在图形中
    for i,marker in zip(range(num_clusters),markers):
        # 画出属于某个集群中心的数据点
        plt.scatter(data[labels==i,0],data[labels==i,1],marker=marker,color='k')

# 噪声函数
def add_noise(x,y,amplitude):
    X = np.concatenate((x,y),axis=0) # 数组拼接(默认axis=0按行拼接，即列数不变行数相加)
    X += amplitude * np.random.randn(2,X.shape[1])
    return X.T

# 定义一个函数来获取一组呈螺旋状额数据点
def get_spiral(t,noise_amplitude=0.5):
    r = t
    x = r * np.cos(t)
    y = r * np.sin(t)
    return add_noise(x,y,noise_amplitude)

# 定义一个函数来获取位于玫瑰曲线上的数据点
def get_rose(t,noise_amplitude=0.02):
    # 设置玫瑰的曲线方程；如果变量k是奇数，那么曲线有k朵花瓣；如果k是偶数，那么有2k朵花瓣
    k = 5
    r = np.cos(k*t) + 0.25
    x = r * np.cos(t)
    y = r * np.sin(t)
    return add_noise(x,y,noise_amplitude)

# 为了增加多样性，我们再定义一个hypotrochoid函数
def get_hypotrochoid(t,noise_amplitude=0):
    a,b,h = 10.0,2.0,4.0
    x = (a-b)*np.cos(t) + h*np.cos((a-b)/b*t)
    y = (a-b)*np.sin(t) - h*np.sin((a-b)/b*t)
    return add_noise(x,y,0)

# 生成样本数据
n_samples = 500
np.random.seed(2)
t = 2.5 * np.pi * (1 + 2 * np.random.rand(1,n_samples))
X = get_spiral(t)

# 不考虑螺旋形的数据连接性
connectivity = None
perform_clustering(X,connectivity,'No connectivity')

# 根据数据连接线创建K和临近点的图形
connectivity = kneighbors_graph(X,n_neighbors=10,include_self=False)
# kneighbors_graph用于计算X中k个临近点（列表）对应的权重
# n_neighbors：整数，可选（默认值为5）,用kneighbors_graph查找的近邻数
perform_clustering(X,connectivity,'K-Neighbors connectivity')

plt.show()