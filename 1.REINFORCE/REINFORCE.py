import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# from moviepy.editor import ImageSequenceClip


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_width):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(state_dim, hidden_width)
        self.l2 = nn.Linear(hidden_width, action_dim)

    def forward(self, s):
        s = F.relu(self.l1(s))
        a_prob = F.softmax(self.l2(s), dim=1)
        return a_prob


class REINFORCE(object):
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_width = 64  # The number of neurons in hidden layers of the neural network
        self.lr = 1e-3  # learning rate
        self.GAMMA = 0.99  # discount factor
        self.episode_s, self.episode_a, self.episode_r = [], [], []

        self.policy = Policy(state_dim, action_dim, self.hidden_width)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def choose_action(self, s, deterministic):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        prob_weights = self.policy(s).detach().numpy().flatten()  # probability distribution(numpy)
        if deterministic:  # We use the deterministic policy during the evaluating
            a = np.argmax(prob_weights)  # Select the action with the highest probability
            return a
        else:  # We use the stochastic policy during the training
            a = np.random.choice(range(self.action_dim), p=prob_weights)  # Sample the action according to the probability distribution
            return a

    def store(self, s, a, r):
        self.episode_s.append(s)
        self.episode_a.append(a)
        self.episode_r.append(r)

    def learn(self, ):
        G = []
        g = 0
        for r in reversed(self.episode_r):  # calculate the return G reversely
            g = self.GAMMA * g + r
            G.insert(0, g)

        G = torch.tensor(G, dtype=torch.float)
        G = (G - G.mean()) / (G.std() + 1e-5)  # Normalize returns

        for t in range(len(self.episode_r)):
            s = torch.unsqueeze(torch.tensor(self.episode_s[t], dtype=torch.float), 0)
            a = self.episode_a[t]
            g = G[t]

            a_prob = self.policy(s).flatten()
            policy_loss = -g * torch.log(a_prob[a])
            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

        # Clean the buffer
        self.episode_s, self.episode_a, self.episode_r = [], [], []


class CustomMountainCarEnv(gym.Wrapper):
    def __init__(self, env):
        super(CustomMountainCarEnv, self).__init__(env)
    
    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        
        # # 自定义奖励：鼓励小车向右爬行
        position, velocity = state
        reward = position
        # if velocity > 0:
        #     reward = (position + 0.5) * -1 # 根据位置增加奖励
        # else:
        #     reward = (position + 0.5) * 1 # 根据位置增加奖励
        # if position >= 0.5:
        #     reward += 10  # 到达目标位置给予额外奖励
        
        return state, reward, done, truncated, info



def evaluate_policy(env, agent):
    times = 3  # Perform three evaluations and calculate the average
    evaluate_reward = 0
    for _ in range(times):
        s, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            a = agent.choose_action(s, deterministic=True)  # We use the deterministic policy during the evaluating
            s_, r, done, truncated, _ = env.step(a)
            done = done or truncated
            episode_reward += r
            s = s_
            # MountainCar 终止条件
            if s_[0] >= 0.5:
                done = True
            else:
                done = done or truncated
        evaluate_reward += episode_reward

    return int(evaluate_reward / times)


if __name__ == '__main__':
    env_name = ['CartPole-v0', 'CartPole-v1', 'MountainCar-v0']
    env_index = 2  # 环境索引
    env = gym.make(env_name[env_index])
    env = CustomMountainCarEnv(env)  # 使用自定义奖励的环境
    env_evaluate = gym.make(env_name[env_index])  # 评估时需要重新构建环境
    env_evaluate = CustomMountainCarEnv(env_evaluate)
    number = 1
    seed = 500

    # 设置随机种子
    env.reset(seed=seed)
    env_evaluate.reset(seed=seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 增加 max_episode_steps
    max_episode_steps = 200  # 例如，将最大步数增加到 500
    env._max_episode_steps = max_episode_steps
    env_evaluate._max_episode_steps = max_episode_steps

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    max_episode_steps = env._max_episode_steps  # 每个episode的最大步数
    print("state_dim={}".format(state_dim))
    print("action_dim={}".format(action_dim))
    print("max_episode_steps={}".format(max_episode_steps))

    agent = REINFORCE(state_dim, action_dim)
    writer = SummaryWriter(log_dir='1.REINFORCE/runs/REINFORCE/REINFORCE_env_{}_number_{}_seed_{}'.format(env_name[env_index], number, seed))  # 构建tensorboard

    max_train_steps = 1e6  # 最大训练步数
    evaluate_freq = 1e3  # 每隔evaluate_freq步评估一次策略
    evaluate_num = 0  # 记录评估次数
    evaluate_rewards = []  # 记录评估奖励
    total_steps = 0  # 记录训练的总步数

    frames = []

    while total_steps < max_train_steps:
        episode_steps = 0
        s, _ = env.reset()
        done = False
        while not done:
            # if total_steps % 1000 == 0:  # 每10步渲染一次
            #     env.render()

            episode_steps += 1
            a = agent.choose_action(s, deterministic=False)
            s_, r, done, truncated, _ = env.step(a)
            done = done or truncated
            agent.store(s, a, r)
            s = s_

            # 每隔evaluate_freq步评估一次策略
            if (total_steps + 1) % evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(env_evaluate, agent)
                # env_evaluate.render()
                evaluate_rewards.append(evaluate_reward)
                print(f"evaluate_num:{evaluate_num} \t evaluate_reward:{evaluate_reward} \t")
                writer.add_scalar(f'step_rewards_{env_name[env_index]}', evaluate_reward, global_step=total_steps)
                if evaluate_num % 10 == 0:
                    np.save(f'1.REINFORCE/data_train/REINFORCE_env_{env_name[env_index]}_number_{number}_seed_{seed}.npy', 
                            np.array(evaluate_rewards))

            total_steps += 1
            # MountainCar 终止条件
            if s_[0] >= 0.5:
                done = True
            else:
                done = done or truncated

        # 一个episode结束后进行更新
        agent.learn()
