
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Envplot():
    def __init__(self,mode='2D'):
        # fig = plt.figure('env')
        fig0 = plt.figure('env')     #,figsize=(4, 4)
        self.mode = mode
        if self.mode == '3D':
            self.ax = Axes3D(fig0,auto_add_to_figure=False)
            fig0.add_axes(self.ax)

        elif self.mode == '2D':    
            self.ax = fig0.subplots()

        # mngr = plt.get_current_fig_manager()
        # mngr.window.wm_geometry("+0+0")  # 调整窗口在屏幕上弹出的位置

        # plt.axis([0, 30, 0, 30])
        plt.show(block=False)

    def render(self,path_x,path_y,z,goal,axis=[0, 20, 0, 20]):
        # plt.cla()
        ax = self.ax
        ax.axis(axis[0:4])
        self.ax.set_aspect('equal')
        ax.clear()
        if self.mode == '3D':    
            ax.set_zlim(z-10,z+10)
            for i in range(path_x.shape[0]):
                ax.scatter3D(goal[i,0], goal[i,1],z,marker='*') #画出目标点
                ax.plot3D(path_x[i],path_y[i],z,label=str(i))   #画出轨迹
        elif self.mode == '2D':
            for i in range(path_x.shape[0]):
                ax.scatter(goal[i,0], goal[i,1],marker='*') #画出目标点
                ax.plot(path_x[i],path_y[i],label=str(i))   #画出轨迹
        # plt.legend()
        # plt.legend(bbox_to_anchor=(1, 0), loc=3, borderaxespad=0)
        plt.pause(0.001)    
            

    def close(self):
        # plt.show(block=True)            #画面阻塞
        plt.close()                   #画面关闭


if __name__=='__main__':
    xs = np.array([[0]])
    ys = np.array([[0]])
    goal = np.array([[85,0.5]])
    env = Envplot(mode = '2D')
    # print(xs.shape,xs)
    for i in range(80):
        y = np.random.random()
        xs = np.append(xs,[[i]],axis=1)
        ys = np.append(ys,[[y]],axis=1)
        
        # print(xs.shape,xs)
        env.render(xs,ys,20,goal,[0, 90, 0, 1])
    env.close()