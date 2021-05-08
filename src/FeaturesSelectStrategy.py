from LS import LS
from SPEC import SPEC
class FeaturesSelectStrategy:
    def __init__(self, getData=False, Data=None, featuresSelectAlgorithm="LS", clusterAlgorithm=None, IsDebug=False):
        self.getData = getData
        self.Data = Data
        self.featuresSelectAlgorithm = featuresSelectAlgorithm
        self.clusterAlgorithm = clusterAlgorithm
        self.IsDebug = IsDebug
        # print("\n\n\n init FeaturesSelectAlgorithm: %s", featuresSelectAlgorithm.getName())

    def featuresSelect(self):

        # getdata...
        if(self.IsDebug == True):
            print("get data....")
        # if(self.getData == None):
        if(self.getData == False):
            # 使用小数据进行演示
            from utils.GetSmallData import getSmallData
            self.data, self.features = getSmallData(self.IsDebug)
            print("Got small data....")
        else:
            # 指定获取数据的函数，数据为无监督标签的纯数据，默认获取到的数据为dataframe格式数据
            self.data = self.Data
            print(self.data)
            self.features = self.data.index
            self.data = self.data.values.T
            print("Got orgin data....")

        if(self.IsDebug == True):
            print(self.data)
            print(self.features)

        # FeaturesSelection
        if(self.IsDebug == True):
            print("featrue selecting...")

        if(self.featuresSelectAlgorithm == "SPEC"): 
            self.featuresSelectAlgorithm = SPEC(data=self.data, features=self.features, IsDebug=self.IsDebug, style=0)
        else: # 默认为LS
            self.featuresSelectAlgorithm = LS(data=self.data, features=self.features, IsDebug=self.IsDebug)
        self.featuresSelectAlgorithm.featuresSelect()


from utils.GetOrginData import GetOrginData

# data = GetOrginData(10).read_Simple_data_TraitANDDate()['IN_RGB']
# data = GetOrginData(-1).read_Simple_data_TraitANDDate()['IN_RGB']
# print("data....................................")
# print(data)
# FeaturesSelectStrategy(getData=True, Data=GetOrginData(-1).read_Simple_data_TraitANDDate()['IN_RGB']).featuresSelect()
# FeaturesSelectStrategy(getData=True, Data=GetOrginData(-1).read_Simple_data_TraitANDDate()['UAV_MUL']).featuresSelect()
FeaturesSelectStrategy(getData=True, Data=GetOrginData(-1).read_Simple_data_TraitANDDate()['UAV_MUL'], featuresSelectAlgorithm="SPEC", IsDebug=True).featuresSelect()
# FeaturesSelectStrategy(featuresSelectAlgorithm="SPEC", IsDebug=True).featuresSelect()
# FeaturesSelectStrategy(getData=True, Data=GetOrginData(-1).read_Simple_data_TraitANDDate()['UAV_RGB']).featuresSelect()
# FeaturesSelectStrategy(IsDebug=True).featuresSelect()