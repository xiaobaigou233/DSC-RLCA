import datetime
import random
from copy import copy
import math

import numpy as np
from scipy.spatial import distance
from Path_planning import Plan
np.set_printoptions(precision=5,suppress=True)
class plot2DEnv():

    def __init__(self,N=5):
        self.N = N                             # 无人机数目
        
        self.min_distance = 0.5                 # 最小距离
        self.goal_distance = 0.5

        # boundary limit set
        self.boundary = [-75, 75, -75, 75]
        # self.boundary = [-(4*N+5), (4*N+5), -(4*N+5), (4*N+5)]
        self.z = 20                         #飞行高度
        # action limit set 
        # self.action_v_threshold = [-0.2, 0.2]  # -0.1<a<0.1 #单次加速限制
        # self.action_angle_threshold = [-50 * math.pi / 180, 50 * math.pi / 180]  # 0<theta<90deg #单次转角限制

        # v limit set
        self.v_threshold = [0.7,1.0]
        # self.v_threshold = [2.1,2.4]
        self.contrl_flag = np.zeros((self.N,1))
        # self.contrl_flag[0] = 1
        # self.contrl_flag = np.ones((self.N,1))      #为0是角度控制，为1是速度控制
        #性能指标参数变量 
        self.t = None               #时间步
        self.success_done = None    #成功数目
        self.distance_sum = None    #路程
        self.min_distance_agent = None  #无人机间最小距离
        self.theta_v_agent = None             #无人机速度,角度


        self.viewer = None
        self.state = None

        self.path_x = None              #plot画图
        self.path_y = None
        self.trajectory = None          #ROS轨迹
        self.goal = None                #目标点坐标

        self.distance_goal = None       #到目标点距离    

        self.is_boundary_done = None    #出界判断
        self.is_goal_done = None        #到达终点判断
        self.is_collide_done = None     #碰撞判断
        self.reward = None
        self.done = None
        self.steps_beyond_done = None


    
    #位置更新
    def _run(self,state,action, record =False):
        from Aircraft_math_model import Aircraft_math_model
        # print(action,type(action),action.shape)
        for i in range(len(action)):
            if action[i,-1] == 1:
                position = Aircraft_math_model(state[i,0], state[i,1], 0, state[i,3], 0, state[i,2], 0, control1=action[i,0], control2=0.0, control3=0.0)    #速度控制
            elif action[i,-1] == 0:
                position = Aircraft_math_model(state[i,0], state[i,1], 0, state[i,3], 0, state[i,2], 0, control1=0.0, control2=0.0, control3=action[i,0])  #方向控制
    
            if record == True:
                if self.trajectory.shape[1] != position.shape[0]-1:
                    self.trajectory = np.zeros((len(action),position.shape[0]-1,position.shape[1]))
                self.trajectory[i] = position[1:]               #ROS所需的轨迹点
            
            self.distance_sum[i] = self.distance_sum[i] + distance.euclidean(state[i,0:2],position[-1,0:2])     
                
            state[i,0]= position[-1,0]
            state[i,1]= position[-1,1]
            # theta = self.tf_theta(position[-1,5])
            # state[i,2]= self.tf_theta(position[-1,5])
            state[i,2]= position[-1,5]                                          #当前角度
            if state[i,3]!=0:
                state[i,3]= position[-1,3].clip(self.v_threshold[0], self.v_threshold[1])                            #当前速度
            # print(position[-1,3])
            
        # print(self.distance_sum)
        return state
    #判断是否走出边界
    def _is_boundary(self):
        boundary_x = np.logical_or((self.state[:,0].reshape(self.N)>self.boundary[1]),self.state[:,0].reshape(self.N)<self.boundary[0])
        boundary_y = np.logical_or((self.state[:,1].reshape(self.N)>self.boundary[3]),(self.state[:,1].reshape(self.N)<self.boundary[2]))
        boundary_done = np.logical_or(boundary_x,boundary_y)
        # boundary_done = boundary_done.reshape((self.N,1))
        if self.done.all():
            pass
        elif not (boundary_done == self.is_boundary_done).all():
            print('is boundary')
            self.done = np.logical_or(self.done,boundary_done)
            # print(self.done)
            self.is_boundary_done = copy(boundary_done)
            print(self.is_boundary_done,'is_boundary')
        else:
            pass            

    #判断是否到达目标
    def _is_goal(self):
        # done = False
        goal_done = np.full((self.N, ), False, dtype=bool)
        goal_done = copy(self.is_goal_done)
        last_distance = copy(self.distance_goal) # 记录上一次距离

        run_reward = np.zeros(self.N)

        for i in range(self.N):
            self.distance_goal[i] = np.sqrt(np.sum(np.square(self.point[i]-self.goal[i]))) # 更新目标距离

            if self.distance_goal[i] <=  self.goal_distance:
                run_reward[i] = 0
                goal_done[i] = True
            elif (last_distance[i] <= self.state[i,3]-self.goal_distance+0.1) and (last_distance[i] >= self.goal_distance):
                
                run_reward[i] = 0
                goal_done[i] = True      
            else:
                run_reward[i] = 0.0

            if self.done[i]==False:
                self.reward[i] = self.reward[i] + run_reward[i]
                
        # print(self.distance_goal)

        goal_reward = 0*goal_done.astype(int)           #到达目标点奖励

        if self.done.all():
            pass
        elif not (goal_done == self.is_goal_done).all():
            self.reward = self.reward + goal_reward*(~(goal_done == self.is_goal_done)).astype(int)
            self.done = np.logical_or(self.done,goal_done)
            # print(self.reward)
            self.is_goal_done = copy(goal_done)
            self.success_done = copy(self.is_goal_done)     #计算成功率
            print(self.is_goal_done,'is goal')
        else:
            pass
    
    #判断是否有碰撞
    def _is_collide(self):

        distance_agent = distance.cdist(self.point,self.point)
        np.fill_diagonal(distance_agent,100)    #此处100为一个大数用来使对角线不为零，不小于安全距离
        self.min_distance_agent = np.append(self.min_distance_agent,np.min(distance_agent))     #记录性能数据        
        # print(self.min_distance_agent.shape)
        #接近惩罚
        # approach_agent = np.clip(distance_agent,a_min=0,a_max=2)
        # approach_reward = -1 * np.sum(2-approach_agent,axis=0)              
        # self.reward = self.reward + approach_reward*(~self.done).astype(int)
        
        collide_true = distance_agent <= self.min_distance    #距离小于R
        # print('collide_true',distance_agent,collide_true)
        collide_int = copy(collide_true.astype(int))    
        collide_done = np.matmul(np.ones(self.N),collide_int)
        collide_done = collide_done.astype(bool)
        collide_reward = -20*collide_done.astype(int)

        # collide_agent = collide_true[np.triu_indices(self.N, k = 1)]

        if self.done.all():
            pass
        elif not (collide_done == self.is_collide_done).all():
            self.reward = self.reward + collide_reward*(~(collide_done == self.is_collide_done)).astype(int)
            # self.done = np.logical_or(self.done,collide_done)     #单独停止
            # self.done = np.full((self.N, ), True, dtype=bool)       #全部停止
            self.done = np.full((self.N, ), False, dtype=bool)      #不停止
            # print(self.done)
            self.is_collide_done = copy(collide_done)
            print(self.is_collide_done,'is collide')

        else:
            pass
    # 规则奖励函数设计
    def _rule_reward(self,state,action):
        # print(action)
        action_reward = np.zeros(len(action))
        for i in range(len(action_reward)):
            if action[i,1] == 0:
                if action[i,2] == 0:
                    action_reward[i] = -action[i,0]**2     #角度
                if action[i,2] == 1:
                    action_reward[i] = -50*action[i,0]**2  #速度
        
        self.reward = self.reward + action_reward
    

    def step(self, action):
        #根据动作更新状态
        self.reward = np.zeros(self.N)
        
        for i in range(self.N):
            if self.done[i]:
                action[i,0] = 0
                self.state[i,3] = 0 #到达目标点则设置速度为零
                #若想让其盘旋则不修改速度为0，修改角度为固定值

        # self._rule_reward(self.state,action)
        # print(action)
        action = np.append(action,self.contrl_flag,axis=1)
        self.state = self._run(self.state,action)  #更新状态
        self.point = copy(self.state[:,:2])

        self.path_x = np.append(self.path_x,self.state[:,0].reshape((self.N,1)),axis=1)
        self.path_y = np.append(self.path_y,self.state[:,1].reshape((self.N,1)),axis=1) #各个智能体轨迹
        self.theta_v_agent = np.append(self.theta_v_agent,self.state[:,2:4].reshape((1,self.N,2)),axis=0)

        #判断片段是否结束
        # self._is_boundary()
        self._is_goal()
        self._is_collide()

        self.t = self.t + (~self.done).astype(int)           #总时间

        done = self.done.all()
        if not done:
            pass
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                print(
                    "You are calling 'step()' even though this "+
                    "environment has already returned done = True. You "+
                    "should always call 'reset()' once you receive 'done = "+
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            self.reward = np.zeros(self.N)

        return self.state, self.reward, self.done, {}

    def reset(self,env_start,env_goal,env_v,env_angle,boundary):
        # env_start,env_goal,env_v,env_angle,self.plan,self.min_distance_plan = self.env_unchanged(env_modes = 1)
        # env_start,env_goal,env_v,env_angle,self.plan,self.min_distance_plan = self.env_random_change()
        self.N = env_start.shape[0]
        # self.boundary = [-(4*self.N+5), (4*self.N+5), -(4*self.N+5), (4*self.N+5)]
        self.boundary = boundary
        self.contrl_flag = np.zeros((self.N,1))
        self.state = np.zeros((self.N,4))

        for i in range(self.N):
            self.state[i,0:2] = env_start[i]
            self.state[i,2] = env_angle[i]
            self.state[i,3] = env_v[i]
        self.point = copy(self.state[:,:2]) # 记录起点，point变量存储当前位置
        self.goal = env_goal #生成目标点
        
        # self.state = np.append(self.state,self.goal,axis=1) #目标点加入状态
        #环境state总状态包括所有无人机x,y位置，朝向角度，速度，目标点x,y
        self.path_x = copy(self.state[:,0].reshape((self.N,1)))
        self.path_y = copy(self.state[:,1].reshape((self.N,1))) # 起点写入轨迹数组
        self.trajectory = np.zeros((self.N,0,0))
        #性能参数
        self.t = np.ones(self.N)        
        self.distance_sum = np.zeros(self.N)            #总路程
        self.success_done = np.full((self.N, ), False, dtype=bool)    #成功数目
        self.min_distance_agent = np.array([min(distance.pdist(self.point[:]))])           #最小间距
        # self.theta_v_agent = np.array([self.state[:,2:4].reshape(self.N,2)])
        self.theta_v_agent = np.zeros((0,self.N,2))
        
        self.distance_goal = np.zeros(self.N) # 记录到目标点距离
        for i in range(self.N):
            self.distance_goal[i] = np.sqrt(np.sum(np.square(self.point[i]-self.goal[i]))) # 更新目标距离

        self.reward = np.zeros(self.N)
        self.is_boundary_done = np.full((self.N, ), False, dtype=bool)
        self.is_goal_done = np.full((self.N, ), False, dtype=bool)
        self.is_collide_done = np.full((self.N, ), False, dtype=bool)
        self.done = np.full((self.N, ), False, dtype=bool)
        self.steps_beyond_done = None
        return self.state

    def index(self):
        Success_rate = np.sum(self.success_done== True) / self.N
        T = self.t
        Distance_sum = self.distance_sum
        min_distance_agent = self.min_distance_agent
        theta_v_agent = self.theta_v_agent
        # actual_path = np.append(self.path_x,self.path_y,axis=0)
        return Success_rate,T,Distance_sum,min_distance_agent,theta_v_agent

    def data_save(self,file_path):
        import csv

        # 打开或创建 CSV 文件
        with open(file_path, 'w', newline='') as csvfile:
        # 创建 CSV writer 对象
            csvwriter = csv.writer(csvfile)
            # 逐行写入数据
            # print(self.theta_v_agent[:][0][1].shape)
            for i in range(self.N):
                # print([self.theta_v_agent[x][i][0] for x in range(self.theta_v_agent.shape[0])])
                x = [str(i)]+["x"]+self.path_x[i].tolist()
                y = [str(i)]+["y"]+self.path_y[i].tolist()
                theta = [str(i)]+["theta"]+[self.theta_v_agent[x][i][0] for x in range(self.theta_v_agent.shape[0])]
                # theta = ["theta"]+[x for x in agent.theta]
                v = [str(i)]+["v"]+[self.theta_v_agent[x][i][1] for x in range(self.theta_v_agent.shape[0])]
                goalx = [str(i)]+["goalx"]+[self.goal[i][0]]
                goaly = [str(i)]+["goaly"]+[self.goal[i][1]]
                step = [str(i)]+["done_step"]+[self.t[i]]
                # print(x)
                csvwriter.writerow(x)
                csvwriter.writerow(y)
                csvwriter.writerow(theta)
                csvwriter.writerow(v)
                csvwriter.writerow(goalx)
                csvwriter.writerow(goaly)
                csvwriter.writerow(step)
        print("Data saved to data.csv")
            

    def render(self,mode='plot'):

        if mode == 'plot':
            if self.viewer is None:
                from envplot import Envplot
                self.viewer = Envplot(mode='2D')
            self.viewer.render(self.path_x,self.path_y,self.z,self.goal,self.boundary)   #画出轨迹
        
        else:
            pass
        if self.state is None:
            return None

    def close(self):


        if self.viewer:
            self.viewer.close()
            self.viewer = None




# N = 1

def _transform(action,contrl_flag):
    if contrl_flag == 0:
        action_threshold =50 * math.pi / 180  # 0<theta<90deg #单次转角限制
    elif contrl_flag == 1:
        action_threshold = 0.1                  # -0.1<a<0.1 #单次加速限制
    action_low  = np.array([-action_threshold,0],dtype=np.float32)
    action_high = np.array([ action_threshold,1],dtype=np.float32)
    action = action*0.5*(action_high-action_low)+0.5*(action_high+action_low)  ########此处有问题#########
    # print(action)
    
    return action

def warmup(act_dim,contrl_flag):
    # action = np.random.uniform(-1, 1, act_dim)
    action = np.array([0,0])
    action = _transform(action,contrl_flag)
    return action


if __name__=='__main__':
    env = plot2DEnv()
    obs_dim = env.obs_space
    action_dim = env.act_space
    N = env.N

    for i in range(1):
        obs,_ = env.reset() #环境复位
        action = np.zeros((env.N,env.act_space))
        done = np.full((N, ), False, dtype=bool)
        # print(obs.shape)
        step = 0
        contrl_flag=env.contrl_flag
        episode_reward = 0
        while not done.all():
            env.render()    #渲染画面
            step += 1
            for i in range(env.N):
                action[i] = warmup(action_dim,contrl_flag[i])
            # print(action)
            # print(obs[:,3])
            next_obs, reward, done, _ = env.step(action) # 传入动作获取状态
            # print(reward)
            # print(collision)
            if step >= 200:
                break
            episode_reward += reward
            # print(reward)
        print("episode"+str(i)+str(episode_reward))
        Success_rate,T,Distance_sum,min_distance_agent, _,_= env.index()
        print(Success_rate,T,Distance_sum,min_distance_agent.shape)
        env.close(save=False)
