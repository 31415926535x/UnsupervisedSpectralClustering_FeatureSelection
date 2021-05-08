from sklearn.preprocessing import StandardScaler
import numpy as np


class SPEC:
    
    def __init__(self, data, features, cluster_number=2, gamma=-1, style=0, IsDebug=False):
        self.data = data
        self.features = features
        self.cluster_number = cluster_number
        self.style = style
        self.gamma = gamma
        self.IsDebug = IsDebug
        self.name = "SPEC Score Algorithm"

    def getName(self):
        return self.name

    def featuresSelect(self):
        self.SPECInit0()
        self.SPECInit1()

        # 特征选择，按得分由大到小排序
        sortFeatures, sortSPECscorce = self.SPECFeaturesSelect()

        # 生成SPEC score 得分热力图，取最多 MAX_FEATURES 个
        print("\n\n\nthe sorted features SPEC score is")
        for i in range(self.m):
            print(i, sortFeatures[i], sortSPECscorce[i])
        
        MAX_FEATURES = int(0.3 * len(sortFeatures))

        k = len(sortSPECscorce) if(len(sortSPECscorce) < 10) else MAX_FEATURES
        
        import pandas as pd
        print(k)
        print(len(sortSPECscorce))
        print(list(sortSPECscorce)[:k])
        data = pd.DataFrame({"SPEC_score": list(sortSPECscorce)[:k]})
        data.index = list(sortFeatures)[:k]
        print(data)
        
        # 绘制得分热力图
        from utils.PlotUtils import generateScoreHeatmap
        generateScoreHeatmap(data=data, k=k, n=len(sortFeatures), Name=self.name)
        


        # 使用特征值分解方式聚类
        self.SPECCluster()



    def SPECInit0(self):


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
        self.SPECInit1()
    
    def SPECInit1(self):

        self.W = np.empty([self.n, self.n], dtype=float, order='C')
        for i in range(0, self.n):
            for j in range(0, self.n):
                self.W[i][j] = np.sum((self.X[i] - self.X[j]) ** 2)
        if(self.gamma == -1):
            sigma = 1.0 / self.X.shape[1]
        self.W *= -sigma
        self.W = np.exp(self.W)


        if(self.IsDebug):
            # TODO: not use self.gamma; gamma = 1/self.m
            print("rbf_kernel by numpy:")
            print(self.W)
            print("rbf_kernel by sklearn:")
            from sklearn.metrics.pairwise import rbf_kernel
            print(rbf_kernel(self.X))

        # 对W进行压缩
        if type(self.W) is np.ndarray:
            from scipy.sparse import csc_matrix
            self.W = csc_matrix(self.W)

        self.D = np.sum(self.W, axis=1)
        
        if(self.IsDebug):
            print("\n\nD is")
            print(self.D, self.D.shape)
        
        # 这里因为W的原因，D可能出现(n,)性状，一种解决方式是reshape，然后用repmat操作；或者像我这样手写将行向量转为对角矩阵，，方便后续矩阵运算的理解
        def diag(x, nn):
            tmp = np.zeros((nn,nn))
            for i in range(nn):
                tmp[i][i] = x[i][0] # 可能 [0] 或报错，，看输入的数据
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
            print("another sqrtD: ")
            print(np.power(np.array(self.W.sum(axis=1)), 0.5))
            print(np.diag(np.power(np.array(self.W.sum(axis=1)), 0.5)[:, 0]))
        
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

        
    def SPECFeaturesSelect(self):


        # 对标准化后的L进行特征值分解
        s, U = np.linalg.eigh(self.LN)
        s = np.flipud(s)
        U = np.fliplr(U)

        from numpy import linalg as LA
        v = np.dot(self.sqrtD, np.ones(self.n))
        v = v/LA.norm(v)

        if(self.IsDebug):

            
            print("s")
            print(s)

            print("v")
            print(v)


        SPECScore = [10000 for i in range(self.m)]

        from numpy import linalg as LA
        for i in range(self.m):
            f = self.F[:, i]
            F_hat = np.dot(self.sqrtD, f)

            l = LA.norm(F_hat)
            if(l < 1000 * np.spacing(1)):
                SPECScore[i] = 10000
                continue
            else:
                F_hat = F_hat / l
            
            a = np.array(np.dot(F_hat.T, U))
            # print(a)
            # print(a.shape)
            a = np.multiply(a, a)
            # print(a)
            # print(a.shape)
            # a = a.T
            # a = np.transpose(a)
            
            if(self.IsDebug):
                print("f: ")
                # print(f)
                print(f[0])
                print("F_hat: ")
                # print(F_hat)
                print(F_hat[0])
                print("l: ")
                print(l)
                print("a: ")
                print(a[0])
                a = a[0]
                # print(a.shape)
                # print(s.shape)
                # print(np.sum(a * s))


            # use f'Lf 
            if(self.style == -1):
                SPECScore[i] = np.sum(a * s)
            # use all eigenpairs expect 1st
            elif(self.style == 0):
                a1 = a[0: self.n - 1]
                SPECScore[i] = np.sum(a1 * s[0: self.n - 1]) / (1 - np.power(np.dot(F_hat.T, v), 2))
            # use first k expect the 1st

            else:
                a1 = a[self.n - self.style: self.n - 1]
                SPECScore[i] = np.sum(a1 * (2 - s[self.n - self.style: self.n - 1]))

            print("SPECScore[" + str(i) + "]: ", SPECScore[i])

        if(self.style != -1 and self.style != 0):
            SPECScore[SPECScore == 10000] = -10000
        
        if(self.IsDebug):
            print("\n\n\nSPECScore: ")
            print(SPECScore)

        labels = self.features
        if(self.IsDebug):
            print("labels")
            print(labels)


        # sort spec to get featrues
        dictSPEC = dict(zip(SPECScore, range(self.m)))
        sortSPEC = np.sort(SPECScore)
        if(self.IsDebug):
            print("sortSPEC")
            print(sortSPEC)
        # 前两种方式降序
        if(self.style == -1 or self.style == 0):
            sortSPEC = sortSPEC[::-1]
        print("sortSPEC: ", sortSPEC)
        self.sortID = [dictSPEC[k] for k in sortSPEC]
        self.anslabel = [] # 排序后的标签
        self.score = []  # 对应的SPEC分值
        for i in range(self.m):
            print(i, labels[self.sortID[i]], sortSPEC[i])
            self.anslabel.append(labels[self.sortID[i]])
            self.score.append(sortSPEC[i])

        return self.anslabel, self.score

        









    def SPECCluster(self):

        """
            null
        """
        pass




