import random
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance
from copy import copy
class Plan():
    def __init__(self,N,center = [0,0],boundary_size = 25):
        self.N = N
        # boundary_size = 4*N
        boundary_size = 36
        self.step = 2*boundary_size
        # self.step = 90
        self.control_t =10          #每step有t个路径点
        
        boundary = [-boundary_size, boundary_size, -boundary_size, boundary_size]
        self.v_threshold = [0.7,1.0]
        # self.v_threshold = [(2*boundary_size)/self.step-0.3,(2*boundary_size)/self.step]
        
        self.center = center
        self.boundary = boundary
    def tf_theta(self,theta):
        theta = theta - 2*math.pi*np.floor((theta)/(2*math.pi)) #角度区间转换进0_2pi
        return theta
    def circle(self,same_speed,layer = 1):
        #根据速度与步长和冲突中心输出起点，终点，直线路径
        step = self.control_t*self.step
        # layer_num = np.zeros(layer)
        # for x in range(layer-1):
        #     layer_num[x] = int(self.N/layer)
        # layer_num[int(layer-1)] = self.N-(layer-1)*int(self.N/layer)
        start = np.zeros((self.N,2))
        goal = np.zeros((self.N,2))
        angle = np.zeros(self.N)
        plan = np.zeros((self.N,step,2))
        if same_speed:
            v = random.uniform(self.v_threshold[0], self.v_threshold[1])*np.ones(self.N)
            # print(v)
            v=0.7*np.ones(self.N)
        else:
            v = np.array([random.uniform(self.v_threshold[0], self.v_threshold[1]) for i in range(self.N)])
        R = random.randint(step/2, step/2)
        random_theta = random.uniform(-np.pi, np.pi)
        # random_theta = 0
        random_angle = random.uniform(2*np.pi, 2*np.pi)        
        for n in range(self.N):
            for j in range(-R,step-R):
                plan[n,j+R,0] = self.center[0] + j*v[n]/self.control_t*np.cos(n*random_angle/self.N+random_theta)
                plan[n,j+R,1] = self.center[1] + j*v[n]/self.control_t*np.sin(n*random_angle/self.N+random_theta)
            angle[n] = self.tf_theta(n*random_angle/self.N+random_theta)
        start = copy(plan[:,0,:])
        goal = copy(plan[:,-1,:])
        # min_distance_agent = np.zeros(self.step)
        # for j in range(self.step):
        #     # print(j)
        #     min_distance_agent[j] = min(distance.pdist(plan[:,int((j+1)*self.control_t-1)]))
        # min_distance_agent = np.append(min(distance.pdist(plan[:,0])),min_distance_agent)
        print(step,v,plan.shape)
        return start,goal,v,angle,plan
    
    def circle_aross(self,same_speed):
        #根据速度与步长和冲突中心输出起点，终点，直线路径
        step = self.control_t*self.step
        start = np.zeros((self.N,2))
        goal = np.zeros((self.N,2))
        angle = np.zeros(self.N)
        plan = np.zeros((self.N,step,2))
        if same_speed:
            v = random.uniform(self.v_threshold[0], self.v_threshold[1])*np.ones(self.N)
            # print(v)
            v=0.7*np.ones(self.N)
        else:
            v = np.array([random.uniform(self.v_threshold[0], self.v_threshold[1]) for i in range(self.N)])
        R = random.randint(step/2, step/2)
        random_theta = random.uniform(-np.pi, np.pi)
        # random_theta = 0
        random_angle = random.uniform(2*np.pi, 2*np.pi)        
        for n in range(self.N):
            start[n] = self.center -R*v[n]/self.control_t*np.array([np.cos(n*random_angle/self.N+random_theta),np.sin(n*random_angle/self.N+random_theta)])
            angle[n] = self.tf_theta(n*random_angle/self.N+random_theta+0.05)
            for j in range(step):
                plan[n,j] = start[n] + j*v[n]/self.control_t*np.array([np.cos(angle[n]),np.sin(angle[n])])

        goal = copy(plan[:,-1,:])
        print(step,v,plan.shape)
        return start,goal,v,angle,plan
   
    def preset_plan(self):
        step = self.control_t*self.step
        preset_start = np.array([[-29.51765273,3.36086429],
                                 [1.703126373,-26.2887085],
                                 [-34.34940498,27.16483281],
                                 [-20.04160768,28.51016228],
                                 [-18.75088862,-20.11563911],
                                 [34.36761068,-0.854167499],
                                 [-9.462017807,-34.14072726],
                                 [-34.58465304,-31.89191878],
                                 [29.59122739,32.5606045]])
        preset_goal = np.array([[30.24315501,-28.35069477],
                                [-26.97550907,34.98537239],
                                [-8.279592687,-35.26386257],
                                [7.636650189,-33.22225352],
                                [35.31129485,20.55692459],
                                [-28.22690048,-26.52329863],
                                [-6.408374381,33.44368204],
                                [30.08106812,-12.00922214],
                                [-17.85430009,-15.66697102]])
        preset_v = np.array([0.94,0.9,0.72,0.73,0.93,0.85,0.84,0.93,0.94])
        preset_angle = np.array([-0.487859193,2.008549487,-1.175215809,-1.149308251,0.644990184,-2.752421822,1.525644375,0.298294811,-2.348020504])
        plan = np.zeros((len(preset_start),step,2))
        print(plan.shape)
        for i in range(len(preset_start)):
            for j in range(step):
                plan[i,j] = preset_start[i] + j*preset_v[i]/self.control_t*np.array([np.cos(preset_angle[i]),np.sin(preset_angle[i])])

        # print(start,goal,v,angle,plan)
        return preset_start,preset_goal,preset_v,preset_angle,plan
    def random_plan(self,v,start_list,goal_list):
        step = self.control_t*self.step
        #速度，起点，方向随机，计算得到终点和路径
        while(1):
            plan = np.zeros((step,2))
            random_v = v
            random_angle = random.uniform(-np.pi, np.pi)
            random_start_x = random.uniform(self.boundary[0], self.boundary[1])
            random_start_y = random.uniform(self.boundary[2], self.boundary[3])
            random_goal_x = random_start_x + random_v/self.control_t*math.cos(random_angle)*step
            random_goal_y = random_start_y + random_v/self.control_t*math.sin(random_angle)*step
            if (random_goal_x > self.boundary[0]) and (random_goal_x < self.boundary[1]) and\
            (random_goal_y > self.boundary[2]) and (random_goal_y < self.boundary[3]):
                start = np.array([random_start_x,random_start_y])
                goal = np.array([random_goal_x,random_goal_y])
                start_dis,goal_dis = np.zeros(len(start_list)+1),np.zeros(len(start_list)+1)
                start_dis[0],goal_dis[0] = 11,11
                for i in range(len(start_list)):
                    start_dis[i+1] = distance.euclidean(start,start_list[i])
                    goal_dis[i+1] = distance.euclidean(goal,goal_list[i])
                # print(start_dis,goal_dis)
                if min(start_dis)>10 and min(goal_dis)>10 :
                    # print(min(start_dis),min(goal_dis))
                    v = np.array(random_v)
                    angle = np.array(random_angle)
                    for j in range(step):
                        plan[j,0] = random_start_x + random_v/self.control_t*math.cos(random_angle)*j
                        plan[j,1] = random_start_y + random_v/self.control_t*math.sin(random_angle)*j
                    break
        # print(start,goal,v,angle,plan)
        return start,goal,v,angle,plan
    def random_env(self,same_speed):
        step = self.control_t*self.step
        start = np.zeros((self.N,2))
        goal = np.zeros((self.N,2))
        angle = np.zeros(self.N)
        plan = np.zeros((self.N,step,2))
        start_list,goal_list = [],[]
        if same_speed:
            # v = random.uniform(self.v_threshold[0], self.v_threshold[1])*np.ones(self.N)
            v=0.93963*np.ones(self.N)   
        else:
            v = np.array([random.uniform(self.v_threshold[0], self.v_threshold[1]) for i in range(self.N)])
        for i in range(self.N):
            start[i],goal[i],v[i],angle[i],plan[i] = self.random_plan(v[i],start_list,goal_list)
            start_list.append(start[i])
            goal_list.append(goal[i])
        # min_distance_agent = np.zeros(self.step)
        # for j in range(self.step):
        #     min_distance_agent[j] = min(distance.pdist(plan[:,int((j+1)*self.control_t-1)]))
        # min_distance_agent = np.append(min(distance.pdist(plan[:,0])),min_distance_agent)
        return start,goal,v,angle,plan

    def pass_through(self):
        step = self.control_t*self.step
        start = np.zeros((self.N,2))
        goal = np.zeros((self.N,2))
        angle = np.zeros(self.N)
        plan = np.zeros((self.N,step,2))
        v = random.uniform(self.v_threshold[0], self.v_threshold[1])*np.ones(self.N)
        R = int(step/2)
        random_theta = random.uniform(-np.pi, np.pi)
        random_theta =np.pi/2
        for n in range(2):
            for j in range(-R,step-R):
                plan[n,j+R,0] = self.center[0] + j*v[n]/self.control_t*np.cos(n*np.pi+random_theta)
                plan[n,j+R,1] = self.center[1] + j*v[n]/self.control_t*np.sin(n*np.pi+random_theta)
            angle[n] = self.tf_theta(n*np.pi+random_theta)
        for m in range(self.N-2):
            for j in range(-R,step-R):
                plan[m+2,j+R,0] = plan[0,j+R,0]+ 2*int(m/2+1)*(-1)**m
                plan[m+2,j+R,1] = plan[0,j+R,1]
            angle[m+2] = angle[0]
        start = copy(plan[:,0,:])
        goal = copy(plan[:,-1,:])
        min_distance_agent = np.zeros(self.step)
        for j in range(self.step):
            # print(j)
            min_distance_agent[j] = min(distance.pdist(plan[:,int((j+1)*self.control_t-1)]))
            # min_distance_agent[j] = min(distance.pdist(plan[:,j]))
        min_distance_agent = np.append(min(distance.pdist(plan[:,0])),min_distance_agent)
        return start,goal,v,angle,plan,min_distance_agent
        pass
    def cross(self):
        step = self.control_t*self.step
        start = np.zeros((self.N,2))
        goal = np.zeros((self.N,2))
        angle = np.zeros(self.N)
        v = random.uniform(self.v_threshold[0], self.v_threshold[1])*np.ones(self.N)
        plan = np.zeros((self.N,step,2))
        R = random.randint(step/2, step/2)
        for i in range(self.N):
            angle[i] = self.tf_theta(i*np.pi/2)*((-1)**int(i/4))
        # for t in range(0,2):    
        #     for j in range(-R,step-R):
        #             plan[t,j+R,0] = self.center[0] + j*v[t]/self.control_t*np.cos(angle[t])
        #             plan[t,j+R,1] = self.center[1] + j*v[t]/self.control_t*np.sin(angle[t])
        for n in range(self.N):    
            for j in range(-R,step-R):
                    plan[n,j+R,0] = self.center[0] + j*v[n]/self.control_t*np.cos(angle[n]) + int((n+2)/4)*7*np.sin(abs(angle[n]))
                    plan[n,j+R,1] = self.center[1] + j*v[n]/self.control_t*np.sin(angle[n]) + int((n+2)/4)*7*np.cos(abs(angle[n]))
        start = copy(plan[:,0,:])
        goal = copy(plan[:,-1,:])
        return start,goal,v,angle,plan
    def index(self,plan):
        #输出参数用于性能指标，包括时间步，总路程，机间最小距离
        T = np.zeros(self.N)
        distance_sum = np.zeros(self.N)
        # distance_sum2 = np.zeros(self.N)
        min_distance_agent = np.zeros(self.step)
        for i in range(plan.shape[0]):
            T[i] = self.step
            # distance_sum2[i] = distance.euclidean(plan[i,0],plan[i,-1])
            for t in range(plan.shape[1]-1):
                distance_sum[i] = distance.euclidean(plan[i,t,0:2],plan[i,t+1,0:2]) + distance_sum[i]
        for j in range(self.step):
            min_distance_agent[j] = min(distance.pdist(plan[:,int((j+1)*self.control_t-1)]))
        min_distance_agent = np.append(min(distance.pdist(plan[:,0])),min_distance_agent)
        # print(plan.shape)
        # print(distance_sum)
        return T,distance_sum,min_distance_agent
    #####选择场景
    def env_random_change(self):
        self.N = random.randrange(self.N-2, self.N+4, 2)
        # self.N = random.randrange(self.N-1, self.N+2, 1)
        # self.boundary = [-(4*self.N+5), (4*self.N+5), -(4*self.N+5), (4*self.N+5)]

        # env_plan = Plan(self.N)
        env_modes = [1,2,3]
        random_env= random.choices(env_modes,weights=[0.5,0.3,0.2])
        if random_env[0] == 1:
            env_start,env_goal,env_v,env_angle,plan = self.circle(same_speed=True)
        elif random_env[0] == 2:
            env_start,env_goal,env_v,env_angle,plan = self.circle(same_speed=False)
        elif random_env[0] == 3:
            env_start,env_goal,env_v,env_angle,plan = self.random_env(same_speed=False)
        env_message = [env_start,env_goal,env_v,env_angle,self.boundary]
        return env_message,plan
    def env_unchanged(self,env_modes = None):
        if env_modes == 1:
            env_start,env_goal,env_v,env_angle,plan = self.circle(same_speed=True)
        elif env_modes == 2:
            env_start,env_goal,env_v,env_angle,plan = self.circle(same_speed=False)
        elif env_modes == 3:
            env_start,env_goal,env_v,env_angle,plan = self.random_env(same_speed=True)
        elif env_modes == 4:
            env_start,env_goal,env_v,env_angle,plan = self.random_env(same_speed=False)
        elif env_modes == 5:
            env_start,env_goal,env_v,env_angle,plan = self.pass_through()
        elif env_modes == 6:
            env_start,env_goal,env_v,env_angle,plan = self.cross()
        elif env_modes == 7:
            env_start,env_goal,env_v,env_angle,plan = self.circle_aross(same_speed=True)
        elif env_modes == 8:
            env_start,env_goal,env_v,env_angle,plan = self.circle_aross(same_speed=False)
        elif env_modes == 9:
            env_start,env_goal,env_v,env_angle,plan = self.preset_plan()    
        env_message = [env_start,env_goal,env_v,env_angle,self.boundary]
        return env_message,plan



if __name__ == "__main__":
    N = 10
    plan = Plan(N)
    # env_info,path = plan.env_unchanged(1)
    start,goal,v,angle,path = plan.cross()
    for n in range(N):
        plt.plot(path[n,:,0],path[n,:,1],'--')
        plt.plot(path[n,-1,0],path[n,-1,1],'*',label=str(n))
    print(angle)
    # print(env_info)
    # T,sum_dis,min_distance = plan.index(path)
    # print(T,sum_dis,min_distance.shape)
    # k = [k for k in range(min_distance.shape[0])]
    # plt.plot(k,min_distance,'-')
    plt.legend()
    plt.show()
    