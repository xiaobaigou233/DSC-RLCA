import paddle
import random
import gym
import argparse
import numpy as np
from copy import copy
from parl.utils import logger, summary      #数据记录

from sac import SAC           #导入算法
from sac_model import Model   #导入网络
from sac_agent import Agent   #导入智能体

from parl.utils import ReplayMemory         #导入经验池
from Path_planning import Plan
from plot2DEnv import plot2DEnv

N = 9               #无人机数目，用来设置合适大小的参数

WARMUP_STEPS = N*300
EVAL_EPISODES = 5
MEMORY_SIZE = int(N*1500)
BATCH_SIZE = N*300
# BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 1e-4
CRITIC_LR = 1e-4
EXPL_NOISE = 0.1  # Std of Gaussian exploration noise

class Train():
    def __init__(self, algorithm):
        self.alg = algorithm
        self.act_space = 2
        self.global_train_step = 0
        self.alg.sync_target(decay=0)

    def rpm_append(self,obs, action, reward, next_obs, done, last_done, rpm):        
        if not done:
            rpm.append(obs, action, reward, next_obs, done)
            # print("not done:"+str(done))
        elif done != last_done:
            rpm.append(obs, action, reward, next_obs, done)
            # print("done!=last_done:"+str(done))
        else:
            pass    

    def learn(self,obs, action, reward, next_obs, terminal):
            
            
            terminal = np.expand_dims(terminal, -1)
            reward = np.expand_dims(reward, -1)

            obs = paddle.to_tensor(obs, dtype='float32')
            action = paddle.to_tensor(action, dtype='float32')
            reward = paddle.to_tensor(reward, dtype='float32')
            next_obs = paddle.to_tensor(next_obs, dtype='float32')
            terminal = paddle.to_tensor(terminal, dtype='float32')
            critic_loss, actor_loss = self.alg.learn(obs, action, reward, next_obs,
                                                    terminal)
            return critic_loss, actor_loss

    # Run episode for training
    def run_train_episode(self, env, rpm):
        mission_plan = Plan(N)
        env_message,plan = mission_plan.env_random_change()
        obs = env.reset(env_message[0],env_message[1],env_message[2],env_message[3],env_message[4])
        agents = [[] for x in range(env.N)]
        for i in range(env.N):
            agents[i] = Agent(self.alg.model,i)
            agents[i].reset(plan[i])
        action = np.zeros((env.N,self.act_space))
        action_reward = np.zeros(env.N)
        env_reward = np.zeros(env.N)
        reward = np.zeros(env.N)
        done = np.full((env.N, ), False, dtype=bool)
        last_done = copy(done)
        rpm_obs = [[] for x in range(env.N)]
        rpm_last_obs = [[] for x in range(env.N)]
        rpm_action = np.zeros((env.N,self.act_space))
        
        self.global_train_step = 0

        episode_reward = np.zeros(env.N)
        episode_steps = 0
        while True:
            # env.render()
            episode_steps += 1
            # Select action randomly or according to policy
            for i in range(env.N):
                if rpm.size() < WARMUP_STEPS:
                    action[i],is_end,action_reward[i],rpm_obs[i],_ = agents[i].action(obs[i],np.delete(obs,i,axis=0),action_type = "warmup")
                else:
                    action[i],is_end,action_reward[i],rpm_obs[i],_ = agents[i].action(obs[i],np.delete(obs,i,axis=0),action_type = "sample")
                reward[i] = env_reward[i] + action_reward[i]
                if (not is_end) and len(rpm_last_obs[i]) != 0:
                    self.rpm_append(rpm_last_obs[i], rpm_action[i], reward[i], rpm_obs[i], done[i], last_done[i], rpm)
                    # print("UAV:"+str(i)+"   done:"+str(done[i]))
            episode_reward += reward
            if done.all():
                break
            last_done = copy(done)
            next_obs, env_reward, done, _ = env.step(action)
            
            rpm_last_obs = copy(rpm_obs)
            rpm_action = copy(action)

            obs = copy(next_obs)

            # Train agent after collecting sufficient data
            if rpm.size() >= WARMUP_STEPS:
                self.global_train_step += 1

                # only update parameter every 10 steps     #调整每100次更新一下网络参数
                if self.global_train_step % 5 != 0:
                    batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
                        BATCH_SIZE)
                    self.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                                batch_terminal)

            if episode_steps >= 200:
                break
        env.close()
        return episode_reward, episode_steps


    # Runs policy for 5 episodes by default and returns average reward
    # A fixed seed is used for the eval environment
    def run_evaluate_episodes(self, env, eval_episodes):
        avg_reward = 0
        
        for _ in range(eval_episodes):
            mission_plan = Plan(N)
            env_message,plan = mission_plan.env_random_change()
            obs = env.reset(env_message[0],env_message[1],env_message[2],env_message[3],env_message[4])
            agents = [[] for x in range(env.N)]
            for i in range(env.N):
                agents[i] = Agent(self.alg.model,i)
                agents[i].reset(plan[i])
            action = np.zeros((env.N,self.act_space))
            done = np.full((env.N, ), False, dtype=bool)
            episode_steps = 0
            action_reward = np.zeros(env.N)
            env_reward = np.zeros(env.N)
            while True:
                env.render()
                episode_steps += 1
                for i in range(env.N):
                    action[i],_,action_reward[i],_,_ = agents[i].action(obs[i],np.delete(obs,i,axis=0),action_type = "predict")
                reward = env_reward + action_reward
                if done.all():
                    break
                # print(action[:,1])
                obs, env_reward, done,  _ = env.step(action)
                avg_reward += np.mean(reward)
                if episode_steps >= 200:
                    break

            env.close()
        avg_reward /= eval_episodes
        return avg_reward


def main():

    env = plot2DEnv(N=N)
    obs_N = 6
    obs_space = 4+5*obs_N     # 状态空间维数
    act_space = 2
    frames = 3

    model = Model(obs_space, frames ,act_space)
    # load_inference_path = './inference_model_cnn_I_3499'
    # param_dict = paddle.load(load_inference_path)
    # model.set_state_dict(param_dict)            #加载网络参数
    save_inference_path = './inference_model_cnn_M'  

    algorithm = SAC(model, gamma=GAMMA, tau=TAU, alpha=0.2, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)

    
    
    rpm = ReplayMemory(
        max_size=MEMORY_SIZE, obs_dim=obs_space, frames= frames ,act_dim=act_space)
    train = Train(algorithm)

    total_steps = 0
    test_flag = 0
    save_flag = 0
    while total_steps < args.train_total_steps:
        # Train episode
        episode_reward, episode_steps = train.run_train_episode(env, rpm)
        # total_steps += episode_steps
        total_steps += 1

        summary.add_scalar('train/episode_reward', np.mean(episode_reward), total_steps)
        logger.info('Total Steps: {} Reward: {}'.format(
            total_steps, episode_reward))

        # Evaluate episode
        if (total_steps + 1) // args.test_every_steps >= test_flag:
            while (total_steps + 1) // args.test_every_steps >= test_flag:
                test_flag += 1
            avg_reward = train.run_evaluate_episodes(env, EVAL_EPISODES)
            summary.add_scalar('eval/episode_reward', np.mean(avg_reward), total_steps)
            logger.info('Evaluation over: {} episodes, Reward: {}'.format(
                EVAL_EPISODES, avg_reward))
        # save_inference
        if (total_steps + 1) // args.save_every_steps >= save_flag:
            while (total_steps + 1) // args.save_every_steps >= save_flag:
                save_flag += 1
            save_inference_path_step = save_inference_path + '_' + str(total_steps)
            paddle.save(model.state_dict(), save_inference_path_step)       #保存网络参数
            # agent.save(save_inference_path_step)                     
    paddle.save(model.state_dict(), save_inference_path)       #保存网络参数


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_total_steps",
        default=8000,
        type=int,
        help='Max time steps to run environment')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=int(50),
        help='The step interval between two consecutive evaluations')
    parser.add_argument(
        '--save_every_steps',
        type=int,
        default=int(500),
        help='save_every_steps')
    args = parser.parse_args()
    
    main()
