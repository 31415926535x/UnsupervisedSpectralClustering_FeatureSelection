# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import seaborn as sns
# import numpy as np

# x = [i / 10 for i in range(10)]
# print(x)


# # TODO 将打分后的特征进行热力图绘制



# import numpy as np
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt
# # import matplotlib

# # matplotlib.use('TKAgg')
# sns.set()
# # np.random.seed(0)
# score = [0.0006685458443962955, 0.0009644144946927556, 0.0013858767078933965]
# data = pd.DataFrame({"LS_score": score})
# label = ["Y", "X", "Z"]
# data.index = label
# print(data)
# ax=sns.heatmap(data)
# plt.title("233333")
# plt.savefig("./test.png")
# # plt.show()

# import pandas as pd
# import random
# random.seed(0)
# rnd_1 = [random.randrange(1,20) for x in range(10)]
# rnd_2 = [random.randrange(1,20) for x in range(10)]
# rnd_3 = [random.randrange(1,20) for x in range(10)]
# rnd_4 = [random.randrange(1,20) for x in range(10)]
# rnd_5 = [random.randrange(1,20) for x in range(10)]
# fecha = pd.date_range('2012-4-10', '2012-4-19')
# data = pd.DataFrame({'fecha':fecha, 'sse': rnd_1, 'rnd_2': rnd_2, 'rnd_3': rnd_3, 'rnd_4': rnd_4, 'rnd_5': rnd_5})
# print(data)
# # print(data.describe())
# data_fecha = data.set_index('fecha')
# # # print(data_fecha.iloc[:,[0,-2]].head())
# # print(data_fecha.iloc[:, 0: -2])
# # col = [i for i in data_fecha.columns if i not in ['sse']]
# # print(data_fecha['sse'])
# # print(data_fecha[col])





# from utils.PlotUtils import generateKmeansEvaluationPlot
# k = [i for i in range(10)]
# generateKmeansEvaluationPlot(k, data_fecha)

# print(data_fecha.values)

# ddata = pd.DataFrame({"score...": data_fecha['rnd_1']})
# ddata.index = data['rnd_2']
# print(ddata)

# sns.set()
# plt.subplots(figsize=(10, 12))
# plt.subplots_adjust(left=0.2)
# # ax = sns.displot(data) # 没有这个我这里会报错
# ax = sns.heatmap(ddata, annot=True)
# ax.set_xticklabels(ddata,rotation='horizontal')
# plt.setp(ax.get_yticklabels() , rotation = 360)
# plt.savefig("test.png")


# m = 1206
# k_set = [i for i in range(10, m, 10)] # 聚类的个数集，10个以上
# k_set.append(int(pow(m, 0.5)))
# k_set = np.sort(k_set)
# print(k_set)

# import os, shutil
# PLOT_PATH = '../pic/Kmeans'
# os.makedirs(PLOT_PATH)

# from sklearn import datasets 
# from sklearn.cluster import KMeans 
# from sklearn.metrics import davies_bouldin_score 
# from sklearn.datasets import make_blobs
# # loading the dataset 
# X, y_true = make_blobs(n_samples=300, centers=4,cluster_std=0.50, random_state=0)
# # K-Means 
# print(X.head())
# kmeans = KMeans(n_clusters=4, random_state=1).fit(X)
# # we store the cluster labels 
# labels = kmeans.labels_
# # print(X)
# print(X.head())
# print(davies_bouldin_score(X, labels))
# # print(X)
# print(X.head())


# from sklearn import datasets
# iris = datasets.load_iris()
# X = iris.data
# print(X[22])
# from sklearn.cluster import KMeans
# from sklearn.metrics import davies_bouldin_score
# kmeans = KMeans(n_clusters=3, random_state=1).fit(X)
# print(X[22])
# labels = kmeans.labels_
# davies_bouldin_score(X, labels)
# print(X[22])


# a = np.array(np.arange(1, 5, 1)).reshape(2, 2)
# print(a)
# d = np.array(a.sum(axis=1))
# d2 = np.power(np.array(a.sum(axis=1)), 0.5)
# print(d)
# print(d2)
# b = np.array([[1, 2, 3],[4, 5, 6]])
# c = np.array([[1, 2, 3]])
# print(b[:, 0])
# print(c[:, 0])
# v = np.dot(np.diag(d2[:, 0]), np.ones(2))
# print(v)
# print(d[:, 0])
# print(np.diag(d[:, 0]))
# v = np.dot(np.diag(d[:, 0]), np.ones(2))
# print(v)
# a = np.array([[5, 6],[7, 8]])
# from sklearn.metrics.pairwise import rbf_kernel
# w = rbf_kernel(a)
# print(w)
# print(w.shape)
# d = np.sum(w, axis=1)
# print(d)
# print(d.shape)
# print(d[0])
# def diag(x, nn):
#     tmp = np.zeros((nn,nn))
#     for i in range(nn):
#         print(i)
#         tmp[i][i] = x[i]
#     return tmp

# l = diag(d, 2) - w
# print(l)
# sd = np.power(d, 0.5)
# print(sd)
# rsd = diag(np.power(d, -1/2), 2)
# print(rsd)
# ln = np.dot(np.dot(rsd, l), rsd)
# print(ln)

# print("-----------")

# import numpy.matlib
# d1 = np.power(np.array(w.sum(axis=1)), -0.5)
# print(d1)
# d1 = d1.reshape(2, 1)
# print(d1)
# print(d1.shape)
# print(np.diag(d1))
# print("....")
# print(np.matlib.repmat(d1, 1, 2))
# ln = (np.matlib.repmat(d1, 1, 2)) * np.array(l) * np.matlib.repmat(np.transpose(d1), 2, 1)
# print(ln)

# c = np.array([1, 2]).reshape(2, 1)
# b = np.array([[1, 2],[3, 4]])
# a = diag(c, 2)
# print(np.dot(a, b))
# print(np.dot(np.dot(a, b), a))
# print("=========")
# print(np.matlib.repmat(c, 1, 2))
# print(np.matlib.repmat(c, 1, 2) * b)
# print((np.matlib.repmat(c, 1, 2) * b * np.matlib.repmat(np.transpose(c), 2, 1)))




# import scipy
# import scipy.sparse
# dd = w.sum(axis=1).flatten()
# print(dd)
# d = scipy.sparse.spdiags(dd, [0], 2, 2, format="csr")
# print("d")
# print(d)
# l = d - w
# print(l)
# with scipy.errstate(divide="ignore"):
#     print(np.sqrt(dd))
#     diag_sqrt = 1.0 / np.sqrt(dd)
# diag_sqrt[np.isinf(diag_sqrt)] = 0
# print("diag_sqrt")
# print(diag_sqrt)
# dh = scipy.sparse.spdiags(diag_sqrt, [0], 2, 2, format="csr")
# print(dh)
# print("l.dot(dh)")
# print(l.dot(dh))
# k = l.dot(dh)
# print(k)
# ln = dh.dot(k)
# # ln = dh.dot(l.dot(dh))
# print(ln)


# print(np.diag(np.ones(10) * 100))

# a = np.arange(10)
# b = np.arange(10)
# print(a.shape)
# print(a + b)
# print(np.sum(a * b))


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