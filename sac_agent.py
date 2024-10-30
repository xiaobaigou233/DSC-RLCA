from copy import copy
import paddle
import numpy as np
import math
from paddle.distribution import Normal
from scipy.spatial import distance
from collections import deque
from scipy.spatial import KDTree
np.set_printoptions(precision=5,suppress=True)
class Agent(): 
    def __init__(self, model,uav_id):

        self.act_dim = 2                        #动作空间维数
        self.obs_N = 6                          # 输入到网络的其他无人机数目
        self.frames = 3                         #连续状态帧数
        self.safe_distance = 1                  #安全消解半径
        self.end_distance = 0.4                 #判断到达目标点的距离
        self.action_threshold = 10 * math.pi / 180   #单次角速度限制
        self.action_low  = np.array([-self.action_threshold,0],dtype=np.float32)
        self.action_high = np.array([ self.action_threshold,1],dtype=np.float32)
        self.perception_time = 20            #冲突消解提前时间
        self.perception_distance = 20         #感知距离

        self.sample_num = 10                    #潜在冲突采样次数
        self.sample_v_threshold = [-0.2, 0.2]  #采样加速范围限制
        self.sample_angle_threshold = [-self.action_threshold, self.action_threshold]  #采样转角范围限制
        self.potential_time = 20            #潜在冲突判断时间
        self.potential_distance = 20         #潜在冲突判断距离
        self.model = model
        
        self.relative_distance = None
        self.obs_state = None
        #前向点参数
        self.ind_near = None
        self.distance_near = None
        self.angle_difference = None
        self.plan= None                 #存储每个无人机的路径
        self.plan_tree = None           #存储每个无人机的路径KDTree 

    def reset(self,plan):
        self.plan = plan
        self.plan_tree = KDTree(plan)
        self.obs_state = []         #观察状态空间
        self.ind_near = np.zeros(3)
        self.is_end = False
        self.distance_near = 0
        self.angle_difference = 0

    def predict(self, obs):

        obs = paddle.to_tensor([obs], dtype='float32')
        act_mean, _ = self.model.policy(obs)
        action = paddle.tanh(act_mean)
        action_numpy = action.cpu().numpy()[0]
        
        action_numpy = self._tf_action(action_numpy)

        return action_numpy

    def sample(self, obs):
        obs = paddle.to_tensor([obs], dtype='float32')
        act_mean, act_log_std = self.model.policy(obs)
        normal = Normal(act_mean, act_log_std.exp())
        # for reparameterization trick  (mean + std*N(0,1))
        x_t = normal.sample([1])
        action = paddle.tanh(x_t)
        action_numpy = action[0].cpu().numpy()[0]

        action_numpy = self._tf_action(action_numpy)
        return action_numpy


    def _tf_action(self,action):
        action = action*0.5*(self.action_high-self.action_low)+0.5*(self.action_high+self.action_low)           ##将网络输出调整为动作大小
        return action
    
    def warmup(self):
        action = np.random.uniform(-1, 1, self.act_dim)
        action = self._tf_action(action)
        return action


    def _conflict_detection(self,self_state,others_state,relative_distance,perception_distance,perception_time,safe_R):        
        """ #冲突检测
        功能:根据自身状态与其他飞机状态判断冲突，计算冲突时间
        输入：
            relative_distance -> 无人机间的距离,外部输入,节约计算
            self_state -> 自身状态信息 numpy([x,y,theta,v])
            others_state -> 其他飞机状态 numpy([x,y,theta,v]*N)
            perception_distance -> 判断距离，在距离内的飞机进行冲突检测
            perception_time -> 判断时间，冲突时间在该时间内的判定为冲突
            safe_R -> 安全半径
        输出：
            conflict_list -> 冲突列表,与输入的其他飞机状态矩阵发生冲突的序号
            time -> 冲突时间,无冲突输出为-1
        """
        # print(self_state,self_state.shape,others_state,others_state.shape)
        self_point = copy(self_state[:2])                 #自身坐标
        others_point = copy(others_state[:,:2])             #周围无人机坐标
        conflict_list = []
        time = -np.ones(len(others_state))
        # relative_distance = self.relative_distance              #获取无人机间的距离  
        perception_index = np.array(np.where(relative_distance <= perception_distance)[0])      #感知范围内的无人机索引 
        # print(perception_index,type(perception_index))
        rough = np.zeros(len(perception_index))                                         #粗选接近的无人机
        rough_para = np.zeros((len(perception_index),4))                               #粗选参数,无人机dat x,dat y,dat vx,dat vy
        for x in range(len(perception_index)):
            rough_para[x,0] = others_point[perception_index[x], 0]-self_point[0]
            rough_para[x,1] = others_point[perception_index[x], 1]-self_point[1]
            rough_para[x,2] = others_state[perception_index[x],3]*math.cos(others_state[perception_index[x],2])-self_state[3]*math.cos(self_state[2])
            rough_para[x,3] = others_state[perception_index[x],3]*math.sin(others_state[perception_index[x],2])-self_state[3]*math.sin(self_state[2])
        
            rough[x] = rough_para[x,0]*rough_para[x,2]+rough_para[x,1]*rough_para[x,3]
        rough_index = np.array(np.where(rough<0)[0])                       #粗选正在接近，可能存在冲突的索引
        # print(rough_para,rough)                        
        judge = np.zeros(len(rough_index))
        judge_para = np.zeros((len(rough_index),3))                                 #方程参数,A,B,C
        for y in range(len(rough_index)):
            judge_para[y,0] = rough_para[rough_index[y],2]**2+rough_para[rough_index[y],3]**2
            judge_para[y,1] = rough[rough_index[y]]
            judge_para[y,2] = rough_para[rough_index[y],0]**2+rough_para[rough_index[y],1]**2-(safe_R)**2
            judge[y] = judge_para[y,1]**2 - (judge_para[y,0])*(judge_para[y,2])
            # if judge_para[y,2] > 0:
            #     judge[y] = judge_para[y,1]**2 - (judge_para[y,0])*(judge_para[y,2])
            # else:
            #     judge[y] = -1
        judge_index = np.array(np.where(judge>0)[0])                                             #判断存在冲突
        # print(judge,judge_para)
        collision_time = np.zeros(len(judge_index))                                 #计算冲突时间
        # print(judge_index)
        for z in range(len(judge_index)):
            collision_time[z] = (-judge_para[judge_index[z],1]-math.sqrt(judge[judge_index[z]]))/judge_para[judge_index[z],0]
        collision_index = np.array(np.where(collision_time < perception_time)[0])
        # print(collision_time)
        index = perception_index[rough_index[judge_index[collision_index]]]
        for ind in range(len(index)):
            time[index[ind]] = collision_time[collision_index[ind]] 
        conflict_list = index
        
        return conflict_list,time

    def _sample_action(self,sample_num):
        sample_low  = np.array([self.sample_angle_threshold[0],self.sample_v_threshold[0]],dtype=np.float32)
        sample_high = np.array([self.sample_angle_threshold[1],self.sample_v_threshold[1]],dtype=np.float32)
        temp_action = np.zeros((sample_num,2,2))               #此处动作为角速度，加速度
        for num in range(sample_num):
            for k in range(2):
                temp_action[num,k,:2] =np.random.uniform(-1, 1, 2)*0.5*(sample_high-sample_low)+0.5*(sample_high+sample_low)        #随机采样动作
            temp_action[num,0,1] = 0
        return temp_action

    def _full_conflict_detection(self,self_state,others_state,relative_distance):
        # print(self_state,self_state.shape,others_state,others_state.shape)
        sample_num = self.sample_num
        full_conflict_list = []
        # sample_low  = np.array([self.sample_angle_threshold[0],self.sample_v_threshold[0]],dtype=np.float32)
        # sample_high = np.array([self.sample_angle_threshold[1],self.sample_v_threshold[1]],dtype=np.float32)
        conflict_list,time = self._conflict_detection(self_state,others_state,relative_distance,self.perception_distance,self.perception_time,self.safe_distance)
        
        for j in range(len(others_state)):
            if others_state[j,3] <=0.05:
                #如果对方静止则不用判断潜在冲突
                break
            elif relative_distance[j] > self.potential_distance:
                #距离过远不用判断潜在冲突
                break
            elif len(np.where(conflict_list==j)[0])==0:       #若与第j无人机无既有冲突，则检测是否有潜在冲突
                # ind = np.array([i,j])
                # temp_action = np.zeros((2,2))               #此处动作为角速度，加速度
                temp_action = self._sample_action(sample_num)
                temp_state = np.append([self_state],[others_state[j]],axis=0)
                conflict_count = 0
                conflict_time = 0
                for num in range(sample_num):
                    # for k in range(2):
                    #     temp_action[k,:2] =np.random.uniform(-1, 1, 2)*0.5*(sample_high-sample_low)+0.5*(sample_high+sample_low)        #随机采样动作
                    # temp_action[0,1] = 0
                    next_state = self._run(temp_state,temp_action[num]) #随机动作采样检测是否碰撞。
                    next_relative_distance = np.array([distance.cdist([next_state[0,0:2]],[next_state[1,0:2]]).flatten()])          #计算新的相对距离
                    next_conflict_list,next_time = self._conflict_detection(next_state[0],next_state[1].reshape(1,4),next_relative_distance.reshape(1,1),self.potential_time,self.potential_distance,self.safe_distance)
                    # print(next_state,next_conflict_list)
                    # print(num,"采样",j)
                    if len(next_conflict_list) !=0:
                        conflict_count = conflict_count+1
                        conflict_time = conflict_time + next_time[0]        #error:此处无冲突时time为-1
                if conflict_count>0:
                    # print("潜在冲突："+str(conflict_count))
                    conflict_list = np.append(conflict_list,j)
                    #计算潜在冲突平均时间
                    time[j] = ((conflict_time-self.perception_time*conflict_count)/sample_num)+self.perception_time
                    # time[j] = self.perception_time-conflict_count/sample_num
                    # print(time[j])
        # print(time)
        full_conflict_list = conflict_list[np.lexsort((time[conflict_list],))]
        # print(full_conflict_list)
        return full_conflict_list,time 
    
    def _obs_transform(self,self_state,others_state,near_point):         #观察空间向量转换
        """对传入的无人机状态处理为神经网络所需的格式
        参数：自身全部状态、其他无人机外部状态
        返回：拼接好的状态量
        """
        self_point = copy(self_state[:2])                 #自身坐标
        others_point = copy(others_state[:,:2])             #周围无人机坐标
        
        self.relative_distance = distance.cdist([self_point],others_point).flatten()              #获取无人机间的距离
        
        conflict_list,time = self._full_conflict_detection(self_state,others_state,self.relative_distance)
        # print(conflict_list)

        obs_single = self_state[2:4]

        obs_single = np.append(near_point,obs_single,axis=0)
        index = conflict_list
        for j in range(self.obs_N):
            if j < len(index):
                point_distance = self.relative_distance[index[j]] -self.safe_distance
                point_angle = np.arctan2(others_point[index[j], 1]-self_point[1],others_point[index[j], 0]-self_point[0])
                point_angle_v = others_state[index[j],2:4]
                point_t = time[index[j]]
            else:
                point_distance = 0
                point_angle = 0
                point_angle_v = np.array([0,0])
                point_t = -1
            point_point = np.array([point_distance,point_angle,point_angle_v[0],point_angle_v[1],point_t])
            obs_single = np.append(obs_single,point_point,axis=0)

        if len(self.obs_state) == 0:
            self.obs_state = deque([obs_single for x in range(self.frames)])
        left = self.obs_state.popleft()
        self.obs_state.append(obs_single)
        obs_state = np.asarray(copy(self.obs_state))

        return obs_state,conflict_list,time
    
    def _calc_self(self,self_state):
        """自身信息计算
        功能：计算轨迹最近点，轨迹前向点，最近距离，前向点角度，终点判断，奖励计算 
        输入：
            self.plan -> 内部轨迹,使用self保存
            self.plan_tree -> 内部轨迹KDTree,使用self保存
            self.ind_near -> ind_near[0]最近点序号,ind_near[1]已经到达的序号,ind_near[2]前向点序号,每次调用更新,需要使用self保存
            self.distance_near -> 调用distance_near用于计算单步奖励
            self.is_end -> 终点判断,self保存,一次判断到达终点
            self_state -> 外部输入当前状态
        输出：
            self.ind_near -> 更新self内部参数,不直接输出
            self.distance_near -> 保存distance_near用于计算单步奖励,不直接输出
            self.is_end -> 终点判断,更新self内部参数,并输出
            near_point -> 最近点距离，前瞻点角度
            step_reward -> 计算自身运动奖励（用于训练）
        """
        
        # 搜索轨迹最近点序号
        point = self_state[:2]      #当前飞机位置
        distance_near, self.ind_near[0] = self.plan_tree.query(point)
        if self.ind_near[0]< self.ind_near[1]: 
            self.ind_near[0] = self.ind_near[1]             #ind_near[0]最近点序号，ind_near[1]已经到达的序号
            distance_near = np.sqrt(np.sum(np.square(point-self.plan[int(self.ind_near[0])])))
        else: self.ind_near[1] = self.ind_near[0]
        # self.near = self.plan[int(self.ind_near[0])]     #最近点坐标

        #计算是否end
        end = False
        if self.ind_near[0] == (self.plan.shape[0]-1) and distance_near<= self.end_distance:
            end = True
            self.is_end = end
        # 搜索前向目标点序号
        k = 0.0  # look forward gain
        Lfc = 2  # look-ahead distance
        v = self_state[3]
        self.Lf = k * v + Lfc
        L = 0.0            
        ind = copy(self.ind_near[0])
        while self.Lf > L and (ind + 1) < len(self.plan):
            L += np.linalg.norm(self.plan[int(ind + 1)]-self.plan[int(ind)])
            ind += 1
        self.ind_near[2] = ind                    #ind_near[2]前向点序号
        look_angle = np.arctan2(self.plan[int(ind),1] - point[1],self.plan[int(ind),0] - point[0])
        near_point = np.array([distance_near,look_angle])       #最近点距离，前瞻点角度
        #计算单步奖励
        ##距离奖励
        last_near_distance = copy(self.distance_near)
        if (last_near_distance-distance_near) >0.05:
            # print("奖励",last_near_distance-self.distance_near)
            distance_reward = 0.5*(last_near_distance-distance_near)# 运行时获得奖励每步速度越大奖励越大
            # distance_reward = 0.0
        elif (last_near_distance-distance_near) <-0.05:
            # print("惩罚",last_near_distance-self.distance_near)
            distance_reward = 0.5*(last_near_distance-distance_near)     # 运行时获得奖励           
        else:
            distance_reward = 0.0
        # distance_reward = -0.01*distance_near
        self.distance_near = copy(distance_near)            #保存到最近点距离
        ##角度奖励
        angle_difference = abs(self_state[2]-look_angle)        #计算角度差
        last_angle_difference = copy(self.angle_difference)
        if (last_angle_difference-angle_difference) >0.05:
            # print("角度奖励",last_angle_difference-self.angle_difference)
            angle_reward = 0.5*(last_angle_difference-angle_difference)# 运行时获得奖励每步速度越大奖励越大
            # angle_reward = 0.0
        elif (last_angle_difference-angle_difference) <-0.05:
            # print("角度惩罚",last_angle_difference-self.angle_difference)
            angle_reward = 0.5*(last_angle_difference-angle_difference)     # 运行时获得奖励           
        else:
            angle_reward = 0.0
        # angle_reward = -0.01*angle_difference
        self.angle_difference = copy(angle_difference)            #保存到最近点距离
        
        step_reward = distance_reward + angle_reward
        return near_point,self.is_end,step_reward,distance_reward,angle_reward
    
    #位置更新
    def _run(self,state,action):
        from Aircraft_math_model import Aircraft_math_model
        #使用数学模型预测状态，用于潜在冲突检测
        next_state = copy(state)
        for i in range(len(action)):
            position = Aircraft_math_model(state[i,0], state[i,1], 0, state[i,3], 0, state[i,2], 0, control1=action[i,1], control2=0.0, control3=action[i,0])  #方向速度控制

            next_state[i,0]= position[-1,0]
            next_state[i,1]= position[-1,1]
            next_state[i,2]= self._tf_theta(position[-1,5])   #当前角度,转换范围
            next_state[i,3]= position[-1,3]   #当前速度
        return next_state
    
    #角度转换
    def _tf_theta(self,theta):
        theta = theta - 2*math.pi*np.floor((theta+math.pi)/(2*math.pi)) #角度区间转换进-pi_pi
        return theta
    def _action_select(self,self_state,obs_state,near_point,action_type = "predict"):
        if action_type == "predict":
            action = self.predict(obs_state)
        elif action_type == "sample":
            action = self.sample(obs_state)
        elif action_type == "warmup":
            action = self.warmup()    
        action_next = copy(action)

        action_next[1] = 0.5*(np.sign(action[1]-0.5)+1)
        if action_next[1] == 1 and self_state[3]!=0:    #若不存在冲突则使用期望值
            action_next[0] = self._tf_theta(2*self_state[3]*np.sin(near_point[1] - self_state[2])/self.Lf).clip(self.action_low[0], self.action_high[0])#方向控制
            # action_next[0] = self._tf_theta(self.look_angle - state[2]).clip(self.action_low[0], self.action_high[0])#方向控制
        else:
            action_next[0] = action[0]
        return action_next,action           #调试，输出处理后的动作与原始动作
    
    def action(self,self_state,others_state,action_type = "predict"):
        """ 最终动作输出 
        输入:
        self_state -> numpy[x,y,theta,v]
        others_state -> numpy[[x,y,theta,v]*N]
        输出：
        action -> numpy[angle_speed,track_flag]
        """
        # print(self_state,self_state.shape,others_state,others_state.shape)
        self_state[2] = self._tf_theta(self_state[2])
        others_state[:,2] = self._tf_theta(others_state[:,2])       #将角度转到范围内
        action = np.zeros(self.act_dim)
        near_point,is_end,step_reward,distance_reward,angle_reward = self._calc_self(self_state)

        if is_end:
            return action,is_end,step_reward,{},action
        else:
            obs_state,conflict_list_test,time= self._obs_transform(self_state,others_state,near_point)
            # print(conflict_list_test)
            action,action_o = self._action_select(self_state,obs_state,near_point,action_type)

            return action,is_end,step_reward,obs_state,action_o   #调试



