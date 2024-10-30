# -*- coding: UTF-8 -*-
import numpy as np
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['axes.unicode_minus']=False
rcParams['font.family'] = 'simhei'
 
 
def dmove2( x_input, t, control):
    # 微分方程
    x, y, z, velocity, gamma, fai = x_input
    # nx, nz, gunzhuan = control
    velocity_, gamma_, fai_ = control

    x_ = velocity * np.cos(gamma) * np.cos(fai)
    y_ = velocity * np.cos(gamma) * np.sin(fai)
    z_ = velocity * np.sin(gamma)
    return np.array([x_, y_, z_, velocity_, gamma_, fai_])


def Aircraft_math_model(x, y, z, velocity, gamma, fai, miu, control1 = 0., control2 = 0., control3 = 0., time = 1,n = 2, skip=1):
    """
    control1:加速度,control2:倾角角速度,control3:航向角角速度,time:计算时间,n:时间内的点数(n>2)
    """
    if skip != 1:
        time = skip*time
        velocity = velocity/skip
        control1 = control1/(skip**2)
        control2 = control2/skip
        control3 = control3/skip

    t = np.linspace(0, time, n)
    po = odeint(dmove2, (x, y, z, velocity, gamma, fai), t, args=([control1,control2,control3],))
    if skip != 1:
        po[:,3] = po[:,3]*skip
    miu_list = np.zeros((n,1))
    miu_list[0,0] =  miu                                    
    for k in range(1, n):
        gamma_ =control2 
        fai_ = control3
        velocity = po[k,3]
        gamma = po[k,4]
        g = 9.81  # 重力加速度
        miu = np.arctan(fai_*velocity*np.cos(gamma)/(g*np.cos(gamma)+gamma_*velocity))
        miu_list[k,0] = miu

    return np.append(po,miu_list,axis=1)

if __name__=='__main__':
    # 初始值 
    velocity  =50 
    gamma, fai, miu = 0., 0., 0.            #航迹倾角（俯仰角）#航向角（偏航角）#滚转角
    x,y,z = 0,0,20    # 初始位置
    
    #todo 三个控制量，加速度、倾角角速度、航向角角速度
    control1 = 0.0
    control2 = 0.0
    control3 = 0.5

    fig = plt.figure('trajectory')
    ax = Axes3D(fig,auto_add_to_figure=False)
    fig.add_axes(ax)
    plt.title('trajectory')
    position = Aircraft_math_model(x, y, z, velocity, gamma, fai, miu, control1, control2, control3, time = 1,n = 20, skip=1)
    print("po: ",position[-1],type(position),position.shape)
    ax.plot(position[:, 0], position[:, 1], position[: ,2])
    position1 = Aircraft_math_model(x, y, z, velocity, gamma, fai, miu, control1, control2, control3, time = 1,n = 20, skip=10)
    print("po1: ",position1[-1],type(position1),position1.shape)
    ax.plot(position1[:, 0], position1[:, 1], position1[: ,2]+1)
    # ax.scatter(position[:, 0], position[:, 1], position[: ,2])
    plt.xlabel("X")
    plt.ylabel("Y")

    # plt.figure(2)
    # plt.title('miu 滚转角')
    # plt.plot(position[:,5]*180/np.pi)
    plt.show()