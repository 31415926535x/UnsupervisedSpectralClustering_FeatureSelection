from sklearn.preprocessing import StandardScaler
import numpy as np


class LS:
    
    def __init__(self, data, features, cluster_number=2, affinity="nearest_neighbors", n_neighbors=10, IsDebug=False):
        self.data = data
        self.features = features
        self.cluster_number = cluster_number
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.IsDebug = IsDebug
        self.name = "Laplican Score Algorithm"

    def getName(self):
        return self.name

    def featuresSelect(self):
        self.LSInit0()
        # self.LSInit1()

        # 特征选择，按得分由大到小排序
        sortFeatures, sortLSscorce = self.LSFeaturesSelect()

        # 生成LS score 得分热力图，取最多 MAX_FEATURES 个
        print("\n\n\nthe sorted features lapcian score is")
        for i in range(self.m):
            print(i, sortFeatures[i], sortLSscorce[i])
        
        MAX_FEATURES = int(0.3 * len(sortFeatures))

        k = len(sortLSscorce) if(len(sortLSscorce) < 10) else MAX_FEATURES
        
        import pandas as pd
        print(k)
        print(len(sortLSscorce))
        print(list(sortLSscorce)[:k])
        data = pd.DataFrame({"LS_score": list(sortLSscorce)[:k]})
        data.index = list(sortFeatures)[:k]
        print(data)
        
        # 绘制得分热力图
        from utils.PlotUtils import generateScoreHeatmap
        generateScoreHeatmap(data=data, k=k, n=len(sortFeatures), Name=self.name)
        


        # 使用特征指分解方式聚类
        self.LSCluster()



    def LSInit0(self):


        point_data = self.data
        self.n, self.m = self.data.shape
        self.X = point_data.copy()
        self.X = StandardScaler().fit_transform(self.X)

        if(self.IsDebug):
            print("\n\n Data is ")
            print(self.data, self.data.shape)
            print("\n\nX is ")
            print(self.X, self.X.shape)
        # exit(-1)
        self.LSInit1()
    
    def LSInit1(self):

        if(self.affinity == "nearest_neighbors"):
            print("affinity use nearest_neighbors...")
            from sklearn.neighbors import kneighbors_graph
            self.W = kneighbors_graph(self.X, n_neighbors=self.n_neighbors, include_self=True)

            if(self.IsDebug):
                print("\n\n W is ")
                print(self.W, self.W.shape)
            
            self.W = 0.5 * (self.W + self.W.T)

            if(self.IsDebug):
                print("\n\n W is ")
                print(self.W, self.W.shape)

        elif(affinity == "rbf"):
            self.W = np.empty([n, n], dtype=float, order='C')
            for i in range(0, n):
                for j in range(0, n):
                    self.W[i][j] = np.sum((self.X[i] - self.X[j]) ** 2)
            sigma = 1.0 / X.shape[1]
            W *= -sigma
            self.W = np.exp(slef.W)

        else:
            raise Exception('Error affinity, check it is ``nearest_neighbors`` or ``rbf``.....')

        
        self.D = np.sum(self.W, axis=1)
        
        if(self.IsDebug):
            print("\n\nD is")
            print(self.D, self.D.shape)
        
        def diag(x, nn):
            tmp = np.zeros((nn,nn))
            for i in range(nn):
                tmp[i][i] = x[i][0]
            return tmp

        self.L = diag(self.D, self.n) - self.W

        if(self.IsDebug):
            print("\n\n L is")
            print(self.L, self.L.shape)
        
        self.sqrtD = np.power(self.D, 0.5)
        self.sqrtD = diag(self.sqrtD, self.n)
        if(self.IsDebug):
            print("\n\n sqrtD is")
            print(self.sqrtD, self.sqrtD.shape)
        
        self.recSqrtD = diag(np.power(self.D, -1/2), self.n)

        self.LN = np.dot(np.dot(self.recSqrtD, self.L), self.recSqrtD)

        self.F = self.X
        self.D = diag(self.D, self.n)
        self.identity = np.ones([self.n, 1], dtype=float, order='C')


        if(self.IsDebug):
            print("\n\n D is")
            print(self.D, self.D.shape)
            print("\n\n sqrtD is")
            print(self.sqrtD, self.sqrtD.shape)
            print("\n\n recsqrtD is")
            print(self.recSqrtD, self.recSqrtD.shape)

            print("\n\n\n")
            print("L is ")
            print(self.L, self.L.shape)
            print("\n\nLN is")
            print(self.LN, self.LN.shape)

        
    def LSFeaturesSelect(self):

        LSScore = [0 for i in range(self.m)]

        tmp = np.dot(np.dot(self.identity.T, self.D), self.identity)

        if(self.IsDebug):
            print("\n\n tmp is", tmp)
        
        self.ftilde = np.empty([self.n, 0])
        for r in range(self.m):

            s = np.divide(np.dot(np.dot(self.F[:,r].T, self.D), self.identity), tmp)
            if(self.IsDebug):
                print("fr shape is ", self.F[:,r].T.shape) # F[:,r]应该是列向量n*1
                print("s is:", s)

            fTilde = (self.F[:, r].reshape(self.n, 1) - s * self.identity)
            # LSScore[r] = np.divide(np.dot(np.dot(fTilde.T, self.L), fTilde), np.dot(np.dot(fTilde.T, self.D), fTilde)).getA()[0][0]
            LSScore[r] = np.divide(np.dot(np.dot(fTilde.T, self.LN), fTilde), np.dot(np.dot(fTilde.T, self.D), fTilde)).getA()[0][0]
            if(self.IsDebug):
                print("LS_r is:", LSScore[r])
                print("fTilde_r is:", fTilde)
                print("shape is ", fTilde.shape)
            print(self.ftilde.shape)
            self.ftilde = np.append(self.ftilde, fTilde, axis=1)

        if(self.IsDebug):
            print("\n\n\nLS score is")
            print(LSScore)

        labels = self.features

        # sort LS to get featrues
        dictLS = dict(zip(LSScore, range(self.m)))
        sortLS = np.sort(LSScore)
        print("sortLS: ", sortLS)
        # sortLS = sortLS[::-1]
        self.sortID = [dictLS[k] for k in sortLS]
        self.anslabel = [] # 排序后的标签
        self.score = []  # 对应的LS分值
        for i in range(self.m):
            print(i, labels[self.sortID[i]], sortLS[i])
            self.anslabel.append(labels[self.sortID[i]])
            self.score.append(sortLS[i])

        return self.anslabel, self.score

    def LSCluster(self):

        k = self.cluster_number # 聚类个数，根据一些理论，可能最好的聚类个数为 \sqrt{M} 但是可以通过多次的聚类，然后根据其SSE等指标找到拐点
        # k_set = [i for i in range(100, self.m, 100)] # 聚类的个数集，10个以上
        # k_set.extend([i for i in range(2, 100, 10)])
        print(self.n)
        print(self.m)
        # k_set = [i for i in range(2, self.n if(self.n < 20) else 50, 5)]
        k_set = [i for i in range(2, 11, 1)] # 这里理论上是要选择所有可能的簇数，但是，我这里的数可能簇数较小
        # if(self.n > 100):
        #     k_set.extend([i for i in range(100, self.n, 300)])
        # k_set.append(int(pow(self.n, 0.5)))
        # if 1 in k_set:
        #     k_set.remove(1) # 聚类要求大于2
        k_set = np.sort(k_set)
        
        if(self.IsDebug):
            print(k_set)

        # https://blog.csdn.net/zhangbaoanhadoop/article/details/82721323
        # https://blog.csdn.net/weixin_36486455/article/details/112379886
        # https://zhuanlan.zhihu.com/p/115752696
        SSEs = [] # 误差平方和，越小越好
        Silhouette_score = [] # 轮廓系数法 越接近1越好
        Calinski_harabaz_score = [] # Calinski-Harabaz 越大越好
        # Compactness = [] #紧密性(CP)
        Davies_Bouldin = [] # 戴维森堡丁指数 分类适确性指标 越小越好


        for k in k_set:
            print("kmeans data by cluster_number = " + str(k))
            y_pred, sse = self.OneKmeans(k)


            print("============== Evaluation ================")
            from sklearn import metrics
            # 对于聚类效果的一个打分评估
            SSEs.append(sse)
            sils = metrics.silhouette_score(self.X, y_pred)
            Silhouette_score.append(sils)
            chs = metrics.calinski_harabasz_score(self.X, y_pred)
            Calinski_harabaz_score.append(chs)
            db = metrics.davies_bouldin_score(self.X, y_pred)
            Davies_Bouldin.append(db)
            print("\n\n\nthe sse at cluster_number=" + str(k) + " is: ", sse)
            print("the metrics.calinski_harabaz_score at cluster_number=" + str(k) + " is: ", chs)
            print("the metrics.silhouette_score at cluster_number=" + str(k) + " is: ", sils)
            print("the metrics.davies_bouldin_score at cluster_number=" + str(k) + " is: ", db)


            print("============== plt ==================")
            # 通过前面特征选择降维以及得分获取的特征和聚类后的编码y_pred来绘图
            from utils.PlotUtils import generateSubFeaturesPlot
            # generateSubFeaturesPlot(self.data, y_pred, "LS.png")
            X = self.data.copy()
            X = self.X.take(self.sortID[0: 3], axis=1)
            generateSubFeaturesPlot(X, self.anslabel[: 3],  y_pred, "LS_cluster_number_" + str(k), 'LS_score')
        
        def min_max_range(x, range_values):
            return [round( ((xx - min(x)) / (1.0*(max(x) - min(x)))) * (range_values[1] - range_values[0]) + range_values[0], 2) for xx in x]
        SSEs = min_max_range(SSEs, (0, 1))
        Silhouette_score = min_max_range(Silhouette_score, (0, 1))
        Calinski_harabaz_score = min_max_range(Calinski_harabaz_score, (0, 1))
        Davies_Bouldin = min_max_range(Davies_Bouldin, (0, 1))


        import pandas as pd
        EvaluationData = pd.DataFrame({"SSE": SSEs, "calinski_harabaz_score": Calinski_harabaz_score, "silhouette_score": Silhouette_score, "davies_bouldin_score": Davies_Bouldin})
        EvaluationData.index = k_set
        from utils.PlotUtils import generateKmeansEvaluationPlot
        generateKmeansEvaluationPlot(k_set, EvaluationData, 'LS_score')


    def OneKmeans(self, k):

        import time
        start=time.time()
        from scipy.sparse.linalg import eigsh
        from sklearn.utils import check_random_state
        random_state = check_random_state(None)

        # # TODO 尝试使用特征选择后的，这里可能是理解有误，也有可能是逻辑问题，还有可能是新数据集进行聚类方法选择有误，导致使用前置得分来特征选择k1个特征以达到降维后再聚类的时间较长
        # self.X = self.data.copy()
        # self.X = self.X.take(self.sortID[0: k], axis=1)
        # self.X = StandardScaler().fit_transform(self.X)
        # self.n, self.m = self.X.shape
        # if(self.IsDebug):
        #     print("\n\nX is ")
        #     print(self.X, self.X.shape)
        # print("\n\n\n\n\n===================================\n\n\n\n\n")
        # self.LSInit1()
        # print("\n\n\n\n\n===================================\n\n\n\n\n")

        # evc = self.LN

        # print(self.ftilde.shape)
        # print(self.ftilde)
        # evc = self.ftilde.take(self.sortID[0: k], axis=1)


        # 使用sklearn包中的特征值分解方式，其可以快速的获取到前k个最小特征值及对应的特征值向量，使用的应该是 arkpark方法
        # 最后对特征向量进行kmeans聚类即可
        eva, evc = eigsh(self.LN * -1, k=k, sigma=1.0, which='LM', tol=0.0)
        evc = evc.T[k::-1]
        # evc = evc / np.diag(D)
        evc = evc.T
        end=time.time()
        if(self.IsDebug):
            print('Running time: %s Seconds'%(end-start))
            print(evc)
            print(evc.shape)

        print("============== kmean ==================")
        from sklearn.cluster import KMeans
        from sklearn.utils import check_random_state
        random_state = check_random_state(None)
        random_state = check_random_state(random_state)
        print(random_state)
        # y_pred = KMeans(n_clusters=k, random_state=random_state).fit_predict(evc)
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=random_state)
        y_pred = kmeans.fit_predict(evc)
        sse = kmeans.inertia_
        if(self.IsDebug):
            print("sse: " + str(sse))
        # from sklearn.cluster import SpectralClustering
        # y_pred = SpectralClustering(n_clusters=2, eigen_solver='arpack', affinity="precomputed").fit(W).labels_
        if(self.IsDebug):
            print(y_pred)

        return y_pred, sse        


