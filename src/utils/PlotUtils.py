

def generateSubFeaturesPlot(data, label, y_pred, filename='generateSubFeaturesPlot', Name='LS_score'):
    # 对m个特征值中，得分最小的三个进行聚类后的绘图，
    # 即使用前三个最优特征表示整个数据空间（选择三是因为生成的图像最大为三维，并不说明后续的特征无用）
    # 故结果为4张表示三维各面的子图
    import os, shutil
    PLOT_PATH = '../pic/' + Name + '/'
    # if(os.path.exists(PLOT_PATH)):
    #     shutil.rmtree(PLOT_PATH)
    if(os.path.exists(PLOT_PATH) == False):
        os.makedirs(PLOT_PATH)
    
    print(data)
    print(label)
    # n = 3
    # plt_num = (n - 1) * n / 2 + 1
    plt_num = 4
    num = 3
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    #定义坐标轴
    # fig = plt.figure()
    plt.cla()
    fig = plt.figure(dpi=80,figsize=(16,16))
    plt.rcParams.update({"font.size":40})
    # plt.title(filename, fontsize=40)
    plt.title(filename, fontweight='bold') #  这里可能有bug，无法显示标题，其次，所有的文字大小没有改变，猜测是图片太大
    plt.xlabel("step",fontsize=40)
    plt.ylabel("rate",fontsize=40)
    # ax1 = plt.axes(projection='3d')
    # ax2 = plt.axes(projection='3d')
    # ax3 = plt.axes(projection='3d')
    # fig = plt.figure(dpi=80,figsize=(32,32))
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9)
    # plt.subplots_adjust(left=0.5, right=0.55, bottom=0.05, top=0.1)
    id = 1
    for i in range(num):
        for j in range(i + 1, num):
            plt.subplot(2, int(plt_num / 2), id)
            id += 1
            print([label[i], label[j]])
            plt.xlabel(label[i])
            plt.ylabel(label[j])
            plt.title(label[i] + "----" + label[j])
            plt.scatter(data[:, i], data[:, j], c=y_pred, cmap='rainbow')
    
    # 三维立体图
    ax = fig.add_subplot(2, int(plt_num / 2), id, projection='3d')
    ax.scatter3D(data[:, 0], data[:, 1], data[:, 2], c=y_pred, cmap='rainbow')
    ax.set_xlabel(label[0])
    ax.set_ylabel(label[1])
    ax.set_zlabel(label[2])
    # plt.show()
    plt.savefig(PLOT_PATH + filename + ".png")
    print(PLOT_PATH + filename + ".png has saved.....")

def generateScoreHeatmap(data, k, n, Name="LS_score"):
    from matplotlib import pyplot as plt
    import seaborn as sns
    plt.cla()
    sns.set()
    plt.subplots(figsize=(10, 12))
    plt.subplots_adjust(left=0.25)
    plt.title(Name + " (" + str(int(k / n * 100)) + ")")
    if(k < 30):
        ax = sns.heatmap(data, annot=True)
    else:
        ax = sns.heatmap(data)
    ax.set_xticklabels(data, rotation='horizontal')
    plt.setp(ax.get_yticklabels() , rotation = 360)
    # ax=sns.heatmap(data)
    import os, shutil
    PLOT_PATH = '../pic/' + Name + "/"
    # if(os.path.exists(PLOT_PATH)):
    #     shutil.rmtree(PLOT_PATH)
    if(os.path.exists(PLOT_PATH) == False):
        os.makedirs(PLOT_PATH)
    plt.savefig(PLOT_PATH + "/heatmap.png")

def generateKmeansEvaluationPlot(n_clusters, data, Name='LS_score'):
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    print(n_clusters)
    print(data)
    col = [i for i in data.columns if i not in ['calinski_harabaz_score']]
    label = data.columns
    # data = data.values
    import os, shutil
    PLOT_PATH = '../pic/' + Name + '/'
    # if(os.path.exists(PLOT_PATH)):
    #     shutil.rmtree(PLOT_PATH)
    # os.makedirs(PLOT_PATH)
    if(os.path.exists(PLOT_PATH) == False):
        os.makedirs(PLOT_PATH)

    plt.cla()
    fig = plt.figure(dpi=100,figsize=(12,9))

    # plt.rcParams.update({"font.size":40})
    plt.title("kmeans_plot k", fontweight='bold')
    plt.xlabel("n_clusters")
    # plt.ylabel("SSEs")
    plt.ylabel("Evaluations")
    # plt.ylabel(label)
    plt.plot(n_clusters, data, 'o-')
    plt.legend(labels=label)
    plt.savefig(PLOT_PATH + "kmeans_plot.png")