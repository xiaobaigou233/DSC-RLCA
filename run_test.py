import paddle
import numpy as np
import math
import time

from Path_planning import Plan
from plot2DEnv import plot2DEnv
from sac_model import Model
from sac_agent import Agent   #导入智能体
from sac import SAC           #导入算法
N = 9


def run_evaluate_episodes(model, env, eval_episodes):


    for eps in range(eval_episodes):
        #生成路径plan
        mission_plan = Plan(9)
        env_message,plan = mission_plan.env_unchanged(1)

        #初始化
        obs = env.reset(env_message[0],env_message[1],env_message[2],env_message[3],env_message[4])
        agents = [[] for x in range(env.N)]
        for i in range(env.N):
            agents[i] = Agent(model,i)      
            agents[i].reset(plan[i])
        action = np.zeros((env.N,act_space))

        done = np.full((N, ), False, dtype=bool)
        episode_steps = 0
        while not done.all():
            env.render()
            episode_steps += 1
            for i in range(env.N):
                # start = time.time()
                action[i],_,_,_,_ = agents[i].action(obs[i],np.delete(obs,i,axis=0),action_type = "predict")
                # end = time.time()
                # print("cal_time",end - start)
            # action[1] = np.zeros(2)       #不合作飞机
            # action[3] = np.zeros(2)       #不合作飞机
            # action[5] = np.zeros(2)       #不合作飞机
            # action[7] = np.zeros(2)       #不合作飞机
            obs, _, done,  _ = env.step(action)

            if episode_steps >= 200:
                break

        env.close()
        
        # data_save_path ='D:\HANJIALE\code\ORCA1\\agent_orca\paper_data\\traj\data_rl_fixed_random.csv'
        # # data_save_path = f'D:\\HANJIALE\\code\\ORCA1\\agent_orca\\data_statistics\\RL_circle_8\\data_rl_{eps}.csv'
        # env.data_save(data_save_path)     #保存数据
        

    return

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

    run_evaluate_episodes(model, env, 1)
    

