import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#定义坐标轴
fig = plt.figure()
plt.figure(dpi=80,figsize=(8,8))
ax1 = plt.axes(projection='3d')
ax2 = plt.axes(projection='3d')
ax3 = plt.axes(projection='3d')


def getSmallData(IsDebug=False):
    #定义坐标轴
    fig = plt.figure()
    plt.figure(dpi=80,figsize=(8,8))
    ax1 = plt.axes(projection='3d')
    ax2 = plt.axes(projection='3d')
    ax3 = plt.axes(projection='3d')
    #ax = fig.add_subplot(111,projection='3d')  #这种方法也可以画多个子图


    d1 = generator_lines(3, 700, IsDebug=IsDebug)
    d2 = generator_lines(7, 1400, IsDebug=IsDebug)
    point_data = np.hstack((d1, d2)).T
    np.random.shuffle(point_data)
    featrues = ['X', 'Y', 'Z']
    if(IsDebug == False):
        return point_data, featrues
    # print(X)
    ax3.scatter3D(point_data[:,0], point_data[:,1], point_data[:,2])
    ax3.set_xlabel('-- X --')
    ax3.set_ylabel('-- Y --')
    ax3.set_zlabel('-- Z --')

    # plt.show()
    plt.savefig("getSmallData.png")
    # print(x, y, z)
    return point_data, featrues




def generator_lines(multi=3, dot_num=200, IsDebug=False):
    z = np.linspace(0, 13, 1000)
    x = np.random.uniform(-multi, multi, 1000)
    y = np.sqrt(np.full(z.shape, multi * multi) - x ** 2)
    import random
    for i in np.nditer(y, op_flags=['readwrite']):
        if(random.randint(1, 10) <= 5):
            i *= -1
    zd = np.random.uniform(0, 13, dot_num)
    xd = np.hstack((np.random.uniform(-multi, multi, int(dot_num / 5 * 3)), np.hstack((np.random.uniform(-0.5, 0.5, int(dot_num / 5 * 1)), np.random.uniform(multi - 1, multi, int(dot_num / 5 * 1))))))
    if(IsDebug):
        print(xd.shape)
    for i in np.nditer(xd, op_flags=['readwrite']):
        if(random.randint(1, 10) <= 6):
            i *= -1
    yd = np.sqrt(np.full(zd.shape, multi * multi) - xd ** 2)
    for i in np.nditer(yd, op_flags=['readwrite']):
        if(random.randint(1, 10) <= 5):
            i *= -1


    ax1.scatter3D(xd,yd,zd, cmap='Blues')  #绘制散点图
    ax1.plot3D(x,y,z,'gray')    #绘制空间曲线

    ax2.set_alpha(0.2)
    p = np.full(x.shape, z.min())
    ax2.plot3D(x, y, p)
    p = np.full(x.shape, y.max())
    ax2.plot3D(x, p, z)
    p = np.full(x.shape, x.min())
    ax2.plot3D(p, y, z)
    

    ax1.set_xlabel('-- X --')
    ax1.set_ylabel('-- Y --')
    ax1.set_zlabel('-- Z --')
    return np.vstack((np.vstack((xd, yd)), zd))
