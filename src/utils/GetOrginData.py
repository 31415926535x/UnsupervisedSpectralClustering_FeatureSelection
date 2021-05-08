import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from numpy import random

class GetOrginData:
    ''' 
        对指定数据进行预处理：读取、分段、缺失值、预分析等等
        对于不同来源的数据，应该针对其数据的特点进行不同的处理
        这里将多文件描述的xlsx数据进行划分，按照数据自身 观测方式PLATFORMS 的不同来处理得到 DataFrame 格式的数据集
        处理后的数据组成为 n*m 的 n个样本以及 m个特征标签， 特征标签可以通过 df.index 获取
    '''
    # 缩水后的样本数，建议预处理以及预分析是使用，
    N = 15
    # 最大样本天数
    # DATE = {}
    # TraitID, TraitName, TraitDate, way映射表文件地址
    # TRAIT_MAP_PATH = '../TraitData_old/Trait_selected/1_IN_RGB.xlsx'
    TRAIT_MAP_PATH = '../res/Datas/Phenomics_Data/Phenomics_Data/Pheno_Information.xlsx'
    # 实际数据地址
    TRAIT_DATA_PATH = '../res/Datas/Phenomics_Data/Phenomics_Data/Phenomics_Data.xlsx'
    # 图片保存地址
    PLOT_PATH = '../pic/'

    PLATFORMS = []
    DATES = {}
    trait_map_selecteds = {}
    trait_ids = {}
    trait_datas = {}
    simple_datas = {}
    
    def __init__(self, n):
        '''
            n 样本数
            platform 测量方式 
            plot_path 保存为值
        '''
        sns.set()
        self.N = n
        import os, shutil
        if(os.path.exists(self.PLOT_PATH)):
            shutil.rmtree(self.PLOT_PATH)
        os.mkdir(self.PLOT_PATH)
    
    def read_Trait_map_selected(self):
        '''
            获取各种方式方式下的映射表数据：
            TraitID, TraitName, TraitDate, Platform
        '''
        print('读取映射表中...')
        # 原数据 4e5*15，分块读取，需要遍历
        # 性状ID与测量方法映射表
        # xlrd不支持读取xlsx，openpyxl不支持读取大excel文件，，所以这里使用处理好后的文件，待优化
        trait_map = pd.read_excel(self.TRAIT_MAP_PATH)
        # 获取所有的Platform
        self.PLATFORMS = list(set(list(trait_map['Trait_Platform'])))
        self.PLATFORMS.sort()
        print(self.PLATFORMS)
        # 测量方式对应的TraitID的数据
        for i in self.PLATFORMS:
            self.trait_map_selecteds[i] = trait_map[trait_map['Trait_Platform'].str.contains(i)]
        return self.trait_map_selecteds

    def read_Trait_data(self):
        '''
            获取实际的该测量方式下的所有数据
            TraitID, x1, x2, x3.... Date, TraitName 获取具体数据可以通过 data[PLATFORMS].iloc[:, 1: -2]
            共N个样本列，行数为映射表指定的ID数量
            trait_data 为 DataFrame格式
        '''
        if(len(self.trait_map_selecteds) == 0):
            self.read_Trait_map_selected()
        # 实际的数据ID
        for i in self.PLATFORMS:
            self.trait_ids[i] = self.trait_map_selecteds[i]['Trait_ID']
        # print(self.trait_map_selecteds)
        # print(self.trait_ids)
                
        # 原数据 4e5*15，分块读取，需要遍历
        self.trait_datas = {i: pd.DataFrame() for i in self.PLATFORMS}
        # with pd.read_table(self.TRAIT_DATA_PATH, header=0, index_col=0, chunksize=100000, iterator=True) as alldatas:
        #     for data in alldatas:
        #         for i in self.PLATFORMS:
        #             self.trait_datas[i] = self.trait_datas[i].append(data[data.index.isin(self.trait_ids[i])])
                # self.trait_datas = self.trait_data.append(data[data.index.isin(self.trait_ids)])
        data = pd.read_excel(self.TRAIT_DATA_PATH, index_col=0)
        # 读入的数据为680各样本的数据
        # 为了数据处理的方便和有效
        # 需要取较少组的数据，即N组
        # 同时要对NaN数据进行处理
        # 这里原始数据出现nan的样本较少，故采用直接丢弃法
        print(data)
        data = data.dropna(axis=1)
        print(data)
        if(self.N != -1):
            data = data.sample(n=self.N, replace=False, random_state=None, axis=1)
        if(self.N == -1):
            self.N = len(data.columns)
        print(data)
        for i in self.PLATFORMS:
            self.trait_datas[i] = self.trait_datas[i].append(data[data.index.isin(self.trait_ids[i])])

        # 各测量方式下的日期
        for i in self.PLATFORMS:
            self.DATES[i] = self.trait_map_selecteds[i]['Trait_Time']
            self.DATES[i] = self.DATES[i].replace('none', 'NaN')
            self.DATES[i] = self.DATES[i].astype('float64')
            if(self.DATES[i].isnull().values.any() == False):
                self.DATES[i] = list(self.DATES[i])
                self.DATES[i] = [int(i) for i in self.DATES[i]]
                self.trait_datas[i]['Trait_Time'] = self.DATES[i]

                # 获取该测量方式下的所有日期
                self.DATES[i] = list(set(self.DATES[i]))
                self.DATES[i].sort()
                # print(self.DATES[i])
            else:
                self.DATES[i] = []
            self.trait_datas[i]['Trait_Name'] = list(self.trait_map_selecteds[i]['Trait_Name'])

        print(self.trait_datas)
        # print(self.DATES)
        return self.trait_datas

    def read_Simple_data(self):
        '''
            获取不同样本，且以所有测量的天数为列、TraitName为行的数据
            simple_datas 中每一个为N个样本的 DataFrame()列表数据
            为之后的聚类热力图的绘制做准备
        '''
        if(len(self.trait_datas) == 0):
            self.read_Trait_data()

        day_datas = {i: {} for i in self.PLATFORMS}
        for i in self.PLATFORMS:
            if(len(self.DATES[i]) != 0):
                day_datas[i] = {j: pd.DataFrame() for j in self.DATES[i]}
                for j in self.DATES[i]:
                    day_datas[i][j] = day_datas[i][j].append(self.trait_datas[i][self.trait_datas[i]['Trait_Time'] == j])
            else:
                day_datas[i][-1] = self.trait_datas[i]

        self.simple_datas = {i: {} for i in self.PLATFORMS}
        for i in self.PLATFORMS:
            if(len(self.DATES[i]) == 0):
                # 按样本ID为列名，一张图
                self.simple_datas[i] = {str(i) + '_all_simple': pd.DataFrame()}
                for si in range(0, self.N):
                    self.simple_datas[i][str(i) +'_all_simple']['simple_' + str(si)] = list(day_datas[i][-1].iloc[:, si])
                self.simple_datas[i][str(i) + '_all_simple'].index = day_datas[i][-1]['Trait_Name']
            else:
                # 按样本的测量的天数为列名，N张图
                self.simple_datas[i] = {str(i) + '_simple_' + str(j): pd.DataFrame() for j in range(0, self.N)}
                for si in range(0, self.N):
                    for j in self.DATES[i]:
                        self.simple_datas[i][str(i) + '_simple_' + str(si)]['day_' + str(j)] = list(day_datas[i][j].iloc[:, si])
                    self.simple_datas[i][str(i) + '_simple_' + str(si)].index = day_datas[i][self.DATES[i][0]]['Trait_Name']

        print(self.simple_datas)
        return self.simple_datas
    
    def get_clustermap(self):
        
        if(len(self.simple_datas) == 0):
            self.read_Simple_data()
            
        # 聚类热力图
        # method 聚类的方式，默认 average
        # col_cluster 是否进行列聚类
        # N = 1
        
        import os
        for i in self.PLATFORMS:
            print('正在处理 %s 测量方式的热力图...' %str(i))
            path = self.PLOT_PATH + '/' + str(i) + '/'
            os.mkdir(path)
            for si in range(0, self.N):
                print('正在生成第 %d 个样本的聚类热力图...' %si)
                sn = ''
                if(len(self.DATES[i]) == 0):
                    sn = str(i) + '_all_simple'
                else:
                    sn = str(i) + '_simple_' + str(si)
                
                # self.simple_datas[i][sn] = self.simple_datas[i][sn].replace('NaN', 0)
                print(self.simple_datas[i][sn])
                try:
                    plt.ion()
                    cols = len(self.simple_datas[i][sn].columns)
                    rows = len(self.simple_datas[i][sn].index)
                    dpi = 300
                    if(rows > 100):
                        rows = rows * 6 // 10
                        cols = cols * 6 // 10
                        dpi = 80
                    # plt.figure()
                    plt.xticks(rotation=300, fontsize=10)
                    plt.yticks(fontsize=10)
                    fig = sns.clustermap(self.simple_datas[i][sn], 
                                z_score=0, method='complete', col_cluster=False, 
                                cmap='RdBu', 
                                figsize=(cols, rows),
                                )
                    fig.fig.suptitle(sn)
                    plt.pause(10)
                    if(rows > 100):
                        fig.savefig(path + sn + '.svg', dpi=dpi, format="svg")
                    else:
                        # plt.show()
                        print(path + sn)
                        fig.savefig(path + sn, dpi=dpi)
                    # plt.pause(5) # 5s后关闭
                    plt.clf()   # 释放资源
                    plt.cla()
                    # plt.close(fig)
                    plt.close()
                    plt.close('all')
                    pass
                except Exception as identifier:
                    print('数据中可能含有nan，跳过该图的绘制...')
                    print(identifier)
                    pass
                if(len(self.DATES[i]) == 0):
                    break
                    # 对于没有日期的数据仅绘制一幅
        print('Done!!!')

        # for i in range(0, self.N):
        #     print('正在生成第 %d 个样本的聚类热力图...' %i)
        #     # sns_plot.append(sns.clustermap(simple_data[i], z_score=0, method='complete', col_cluster=False, cmap=sns.diverging_palette(10, 220, sep=80, n=7)))
        #     fig = sns.clustermap(self.simple_datas[i], z_score=0, method='complete', col_cluster=False, cmap='RdBu', figsize=(len(self.simple_data[i].columns), len(self.simple_data[i].index)))
        #     plt.show()
        #     fig.savefig(self.PLOT_PATH + '/simple_%d' %i, dpi = 300)
        #     # plt.close(fig)
        #     # plt.cla()
        #     # plt.clf()
        #     plt.close("all")
        # print("Done!!!")

# GetOrginData(15).get_clustermap()


    def read_Simple_data_TraitANDDate(self):
        '''
            获取不同测量方式下的样本，以特征性状+日期组合方式表示特征标签
            返回的是以测量方式为键的字典数据
            simple_datas 中每一个为N个样本的 DataFrame()列表数据
            为之后的聚类热力图的绘制做准备
        '''
        if(len(self.trait_datas) == 0):
            self.read_Trait_data()

        day_datas = {i: {} for i in self.PLATFORMS}
        for i in self.PLATFORMS:
            if(len(self.DATES[i]) != 0):
                day_datas[i] = {j: pd.DataFrame() for j in self.DATES[i]}
                for j in self.DATES[i]:
                    day_datas[i][j] = day_datas[i][j].append(self.trait_datas[i][self.trait_datas[i]['Trait_Time'] == j])
            else:
                day_datas[i][-1] = self.trait_datas[i]

        self.simple_datas = {i: {} for i in self.PLATFORMS}
        for i in self.PLATFORMS:
            if(len(self.DATES[i]) == 0):
                # 这里应该要处理一下，不用 all_simple 
                # 按样本ID为列名，一张图
                self.simple_datas[i] = {str(i) + '_all_simple': pd.DataFrame()}
                for si in range(0, self.N):
                    self.simple_datas[i][str(i) +'_all_simple']['simple_' + str(si)] = list(day_datas[i][-1].iloc[:, si])
                self.simple_datas[i][str(i) + '_all_simple'].index = day_datas[i][-1]['Trait_Name']
            else:
                index = []
                
                self.simple_datas[i] = self.trait_datas[i].iloc[:, 1: -2]
                index = self.trait_datas[i]['Trait_Name'].map(str) + "_day_" + self.trait_datas[i]['Trait_Time'].map(str)

                self.simple_datas[i].index = index 
                print("\n\n\n------- PLATFORMS ---------")
                print(i)
                print(self.simple_datas[i])
                print("--------------\n")

        print(self.simple_datas)
        return self.simple_datas