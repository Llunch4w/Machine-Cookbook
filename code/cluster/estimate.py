import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

def estimate(data):
    # 为了确定集群的最佳数量，我们迭代一系列的值，找出其中的峰值
    scores = []
    range_values = np.arange(2,10) #[2,10),步长为1

    for i in range_values:
        kmeans = KMeans(init='k-means++',n_clusters=i,n_init=10)
        kmeans.fit(data)
        score = metrics.silhouette_score(data, kmeans.labels_, 
                    metric='euclidean', sample_size=len(data))
        print('Number of clusters = ',i)
        print('Silhouette score = ',score)
        scores.append(score)
        

    # 画出图形并找出峰值
    plt.figure()
    plt.bar(range_values,scores,width=0.6,color='b',align='center')
    plt.title('Silhouetee score vs number of clusters')
    plt.show()

    # 画出数据
    plt.figure()
    plt.scatter(data[:,0],data[:,1],color='b',s=30)
    x_min,x_max = min(data[:,0])-1,max(data[:,0])+1
    y_min,y_max = min(data[:,1])-1,max(data[:,1])+1
    plt.title('Input data')
    plt.xlim(x_min,x_max)
    plt.ylim(y_min,y_max)
    plt.xticks(np.arange(x_min,x_max,1))
    plt.yticks(np.arange(y_min,y_max,1))
    # plt.xticks(())
    # plt.yticks(())
    plt.show()