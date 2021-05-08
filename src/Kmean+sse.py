import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

center=[[0,1],[-2,-1],[0,-2], [2, 2], [-2, 2]] #centers表示数据点中心，可以输入int数字，代表有多少个中心，也可以输入几个坐标
cluster_std=0.5 #表示分布的标准差
X,labels=make_blobs(n_samples=1000,centers=center,n_features=2,
                    cluster_std=cluster_std,random_state=0)

unique_lables=set(labels)#即unique_lables={0, 1, 2}
colors=plt.cm.Spectral(np.linspace(0,1,len(unique_lables)))#即colors=三种颜色
print(list(zip(unique_lables,colors)))
for k,col in zip(unique_lables,colors):
    x_k=X[labels==k]
    plt.plot(x_k[:,0],x_k[:,1],'o',markerfacecolor=col,markeredgecolor="k",
             markersize=14)

plt.title('data by make_blob()')
plt.savefig("blobs.png")



from sklearn.cluster import KMeans

plt.cla()
kset = []
SSE = []
plt.title('SSE _ K')
plt.xlabel('k')
plt.ylabel('SSE')
for k in range(1, 10):
    print(k)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit_predict(X)
    kset.append(k)
    SSE.append(kmeans.inertia_)

plt.plot(kset, SSE, '^-')
plt.vlines(5, 0, 5000, colors = "r", linestyles = "dashed")
plt.savefig("see.png")