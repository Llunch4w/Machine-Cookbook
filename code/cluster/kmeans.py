import matplotlib.pyplot as plt
import numpy as np


def kmeans_func(num_clusters,data):
    from sklearn.cluster import KMeans
    # num_clusters = 4
    kmeans = KMeans(init='k-means++',n_clusters=num_clusters,n_init=10)
    kmeans.fit(data)
    return kmeans


def plot(kmeans,data):
    x_min,x_max = min(data[:,0])-1.0,max(data[:,0])+1.0
    y_min,y_max = min(data[:,1])-1.0,max(data[:,1])+1.0

    # 画出图像
    step_size = 0.01
    x_values,y_values = np.meshgrid(np.arange(x_min,x_max,step_size),np.arange(y_min,y_max,step_size)) # 定义网格
    # 预测网格中所有数据点的标记
    predicted_labels = kmeans.predict(np.c_[x_values.ravel(),y_values.ravel()])
    predicted_labels = predicted_labels.reshape(x_values.shape)
    plt.figure()
    plt.pcolormesh(x_values,y_values,predicted_labels,cmap=plt.cm.Paired)
    plt.scatter(data[:,0],data[:,1],c='none',s=30,edgecolors='b',linewidth=1,cmap=plt.cm.Paired)

    # 把中心点画在图形上
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:,0],centroids[:,1],marker='s',s=200,linewidths=3,
            color='green',zorder=10,facecolors='pink')


    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    plt.xticks(np.arange(int(x_min),int(x_max),1.0))
    plt.yticks(np.arange(int(y_min),int(y_max),1.0))

    plt.show()
