import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

import myutils

class LS:

    Platform = ''
    data = {}
    X = None
    trait_id = None
    trait_name = None
    simple_name = None
    IsDebug = False

    def __init__(self, platform, IsDebug, IsTest=False):
        print('Use LS method....')
        self.Platform = platform
        self.data = myutils.Myutils.DatasProcess().getSimpleDataOfPlatform(platform)
        print('\n\n\n\n\n')
        self.X = self.data['Simple_datas']
        self.trait_id = self.data['Traits_labels_id']
        self.trait_name = self.data['Traits_labels']
        self.simple_name = self.data['Simple_labels']
        self.n, self.m = np.shape(self.X)

        self.IsDebug = IsDebug
        if(self.IsDebug):
            print('\n\n\n')
            print(self.X)
            print(self.X.shape)
            print('<' + str(self.n) + ', ' + str(self.m) + '>')
            print(self.trait_id)
            print(self.trait_name)
            print(self.simple_name)

    def test(self):
        import pandas as pd
        dataPath = '../res/Datas/Test/examination.csv'
        
        pass

    def Construct_W(self):
        '''
            使用高斯核函数构造的相似矩阵
            X: [n, m] n个样本m个特征的数据矩阵
        '''
        
        self.W = np.empty([self.n, self.n], dtype=float, order='C')
        for i in range(0, self.n):
            for j in range(0, self.n):
                self.W[i][j] = np.sum((self.X[i] - self.X[j]) ** 2)
        
        self.sigma = 1.0 / self.X.shape[1]

        self.W *= -self.sigma

        self.W = np.exp(self.W)

        if(self.IsDebug):
            print('\n\n\n')
            print('W is: ')
            print(self.W)
            print('sigma is ' + str(self.sigma))
            
            from sklearn.metrics.pairwise import rbf_kernel
            print('rbf_kernel processed is: ')
            print(rbf_kernel(self.X))
        
        return self.W

    def Construct_D_L_LN_F(self):
        '''
            其他基础矩阵，如相似矩阵的度矩阵、单位矩阵、拉普拉斯矩阵、标准化后的拉普拉斯矩阵、特征矩阵等
        '''
        self.D = np.sum(self.W, axis=1)

        self.L = np.diag(self.D) - self.W
        
        # get D's $D^{1/2}$
        self.sqrtD = np.diag(self.D ** 0.5)
        # recSqrtD = 1.0 / sqrtD
        self.recSqrtD = np.diag(1.0 / (self.D ** 0.5))

        # get normalized Lapician matrix $LN = sqrtD L sqrtD$
        self.LN = np.dot(np.dot(self.recSqrtD, self.L), self.recSqrtD)
        
        self.F = self.X.T
        self.D = np.diag(self.D)

        # get identity matrix
        self.identity = np.ones([self.n, 1], dtype=float, order='C')

        if(self.IsDebug):
            print('D: ' + str(np.shape(self.D)))
            print(self.D)
            print("sqrtD and recsqrtD are:", self.sqrtD, self.recSqrtD)
            print('L = D - W (laplician matrix) is: ' + str(np.shape(self.L)))
            print(self.L)
            print("normalized lapician matrix is: ", self.LN)
            print("F matrix is:", self.F)
            print("D matrix is:", self.D)
            print("identity matrix is:", self.identity)

    def FeaturesSelect(self):
        ''' 
            基于拉普拉斯分值的无特征选择算法（LS）
        '''
        print("---------------------------------------------------------")
        print("this is LS: ")
        fTilde = np.empty([self.m, self.n], dtype=float, order='C')
        LS = np.empty([1, self.m], dtype=float, order='C')    
        # get tmp 1D1^T
        tmp = np.dot(np.dot(self.identity.T, self.D), self.identity)
        if(self.IsDebug):
            print("tmp is:", tmp)

        # get $\tilde{ff_r} = f_r - \frac{{f_r}^TD{\bf 1}}{{\bf 1 ^T}D{\bf 1}}{\bf 1}$
        for r in range(self.m):
            
            s = np.divide(np.dot(np.dot(self.F[r], self.D), self.identity), tmp)
            if(self.IsDebug):
                print("s is:", s)

            fTilde[r] = (self.F[r] - s * self.identity.T)
            if(self.IsDebug):
                print("fTilde_r is:", fTilde[r])
            # get $LS_r = \frac{{\tilde{f_r}}^TL{\tilde{f_r}}}{{\tilde{f_r}^T}D\tilde{f_r}}$
            LS[0][r] = np.divide(np.dot(np.dot(fTilde[r], self.L), fTilde[r].T), np.dot(np.dot(fTilde[r], self.D), fTilde[r].T))
            if(self.IsDebug):
                print("LS_r is:", LS[0][r])
        
        print(LS)

        # # sort LS to get featrues
        # dictLS = dict(zip(LS[0], range(self.m)))
        # sortLS = np.sort(LS[0])
        # print("sortLS: ", sortLS)
        # sortLS = sortLS[::-1]
        # sortID = [dictLS[k] for k in sortLS]
        # ans = []
        # score = []
        # for i in range(self.m):
        #     print(i, self.labels[sortID[i]], sortLS[i])
        #     ans.append(self.labels[sortID[i]])
        #     score.append(sortLS[i])
        print("---------------------------------------------------------")
        # return ans, score


ls = LS('IN_RGB', True, True)
ls.Construct_W()
ls.Construct_D_L_LN_F()
ls.FeaturesSelect()