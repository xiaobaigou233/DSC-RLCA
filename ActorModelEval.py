import paddle
import numpy as np
import math
import time

import matplotlib.pyplot as plt
from Path_planning import Plan
from plot2DEnv import plot2DEnv
from sac_model import Model
from sac_agent import Agent   #导入智能体
from sac import SAC           #导入算法
N = 9


class Plot():
    def __init__(self,detail=False):
        self.detail = detail
        fig1 = plt.figure('min_distance',figsize=(4, 4))        
        self.ax1 = fig1.subplots()
        if self.detail == True:
            fig2 = plt.figure('theta')
            fig3 = plt.figure('v')
            self.ax2 = fig2.subplots()
            self.ax3 = fig3.subplots()    
    def fig(self,min_distance,theta_v_agent,min_distance_plan):
        k = np.array([k for k in range(min_distance.shape[0])])
        line1 = [0.5 for _ in range(min_distance.shape[0])]
        line2 = [1.0 for _ in range(min_distance.shape[0])]
        if min_distance_plan.shape[0] < min_distance.shape[0]:
            for x in range(min_distance.shape[0]-min_distance_plan.shape[0]):
                min_distance_plan = np.append(min_distance_plan,min_distance_plan[-1])
        self.ax1.clear()
        self.ax1.plot(k,min_distance_plan[k],'b')
        self.ax1.plot(k,min_distance[k],'g')        
        self.ax1.plot(k,line1,'r')
        self.ax1.plot(k,line2,'y')
        # self.ax1.plot(k,line2,'r')
        if self.detail == True:
            self.ax2.clear()
            self.ax3.clear()
            for i in range(theta_v_agent.shape[1]):
                for j in k[:-1]:
                    if theta_v_agent[j,i,0] - theta_v_agent[0,i,0] >= math.pi:
                        theta_v_agent[j,i,0] = theta_v_agent[j,i,0] - 2*math.pi
                    if theta_v_agent[j,i,0] - theta_v_agent[0,i,0] <= -math.pi:
                        theta_v_agent[j,i,0] = theta_v_agent[j,i,0] + 2*math.pi
                self.ax2.plot(k[:-1],theta_v_agent[k[:-1],i,0]*180/np.pi)
                self.ax3.plot(k[:-1],theta_v_agent[k[:-1],i,1])
            self.ax2.grid(True, linestyle="--", alpha=0.5)
            self.ax3.grid(True, linestyle="--", alpha=0.5)
        self.ax1.grid(True, linestyle="--", alpha=0.5)
        
        plt.show(block= False)
    def close(self):
        # plt.close()
        plt.show(block= True)


def run_evaluate_episodes(model, env, eval_episodes):
    avg_reward = 0.0
    # avg_reward = np.zeros(env.N)
    Success_rate_list,T_list,Distance_sum_list = [],[],[]
    for eps in range(eval_episodes):
        #生成路径plan
        mission_plan = Plan(9)
        env_message,plan = mission_plan.env_unchanged(1)
        T_p,distance_sum_p,min_distance_plan = mission_plan.index(plan)

        #初始化
        obs = env.reset(env_message[0],env_message[1],env_message[2],env_message[3],env_message[4])
        agents = [[] for x in range(env.N)]
        for i in range(env.N):
            agents[i] = Agent(model,i)      
            agents[i].reset(plan[i])
        action = np.zeros((env.N,act_space))
        rpm_obs = [[] for x in range(env.N)]
        env_reward = np.zeros(env.N)
        done = np.full((N, ), False, dtype=bool)
        episode_steps = 0
        while not done.all():
            env.render()
            episode_steps += 1
            for i in range(env.N):
                # start = time.time()
                action[i],_,action_reward,rpm_obs[i],_ = agents[i].action(obs[i],np.delete(obs,i,axis=0),action_type = "predict")
                # end = time.time()
                # print("cal_time",end - start)
            reward = env_reward + action_reward
            # action[1] = np.zeros(2)       #不合作飞机
            # action[3] = np.zeros(2)       #不合作飞机
            # action[5] = np.zeros(2)       #不合作飞机
            # action[7] = np.zeros(2)       #不合作飞机
            obs, env_reward, done,  _ = env.step(action)
            avg_reward += np.mean(reward)
            if episode_steps >= 200:
                break

        env.close()
        print(avg_reward)
        Success_rate,T,Distance_sum,min_distance_agent,theta_v_agent = env.index()              ##min_distance_agent,theta_v_agent,min_distance_plan单次运行数据记录
        
        # data_save_path ='D:\HANJIALE\code\ORCA1\\agent_orca\paper_data\\traj\data_rl_fixed_random.csv'
        # # data_save_path = f'D:\\HANJIALE\\code\\ORCA1\\agent_orca\\data_statistics\\RL_circle_8\\data_rl_{eps}.csv'
        # env.data_save(data_save_path)     #保存数据
        
        Success_rate_list.append(Success_rate)
        T_list.append(np.mean(T))
        Distance_sum_list.append(np.mean(Distance_sum))
        print('Successful, Episode %d \t EpSuccess %.3f \t EpLen %.3f \t EpDistance  %.3f'%(eps, Success_rate, np.mean(T), np.mean(Distance_sum)))
        print(Success_rate,T,Distance_sum,done)
        print(min_distance_plan.shape,min_distance_agent.shape)

        plot.fig(min_distance_agent,theta_v_agent,min_distance_plan)
    print(' successful rate: {:.2%}'.format(np.mean(Success_rate_list)), "average EpLen:", np.round(np.mean(T_list),2), 'std length', np.round(np.std(T_list),2), 'average distance:', np.round(np.mean(Distance_sum_list),2), 'std distance', np.round(np.std(Distance_sum_list),2))    
    
    avg_reward /= eval_episodes
    return avg_reward

if __name__ == "__main__":

    env = plot2DEnv(N=N)

    obs_N = 6                          # 观察到其他无人机数目
    frames = 3
    obs_space = 4+5*obs_N     # 状态空间维数
    act_space = 2
    model = Model(obs_space, frames ,act_space)
    
    save_path = "inference_model_cnn_L"
    param_dict = paddle.load(save_path)
    model.set_state_dict(param_dict)

    plot = Plot(detail=True)
    
    avg_reward = run_evaluate_episodes(model, env, 1)
    
    plot.close()
    # print(avg_reward)
