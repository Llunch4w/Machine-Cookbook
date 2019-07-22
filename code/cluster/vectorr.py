import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy import misc

def compress_image(img,num_clusters):
    # 将输入 的图片转换成（样本量，特征量）数组，以运行k-means聚类算法
    X = img.reshape((-1,1))
    
    # 对输入数据运行k-means聚类
    kmeans = KMeans(n_clusters=num_clusters,n_init=4,random_state=5)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_.squeeze() # 从数组的形状中删除单维条目，即把shape中为1的维度去掉
    labels = kmeans.labels_
    
    # 为每个数据配置离它最近的中心点，并转变为图片的形状
    input_image_compressed = np.choose(labels,centroids).reshape(img.shape) # 按照序号label对centroids中的数进行选择
    
    return input_image_compressed


def plot_image(img,title):
    vmin = img.min()
    vmax = img.max()
    plt.figure()
    plt.title(title)
    plt.imshow(img,cmap=plt.cm.gray,vmin=vmin,vmax=vmax)


