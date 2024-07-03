import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

# 定义策略网络
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

# 定义REINFORCE算法
class REINFORCE:
    def __init__(self, state_dim, action_dim, lr=0.05, gamma=0.99):
        self.policy_net = PolicyNet(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.action_dim = action_dim

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy_net(state)
        m = Categorical(probs)
        action = m.sample()
        return action.item()

    def learn(self, rewards, states, actions):
        rewards = torch.tensor(rewards)
        states = torch.from_numpy(np.stack(states)).float()
        actions = torch.tensor(actions)

        qvals = []
        G = 0
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            qvals.insert(0, G)
        qvals = torch.tensor(qvals)

        log_probs = []
        for state, action in zip(states, actions):
            probs = self.policy_net(state.unsqueeze(0))
            m = Categorical(probs)
            log_prob = m.log_prob(action)
            log_probs.append(log_prob)
        log_probs = torch.stack(log_probs)

        loss = (-log_probs * qvals).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)



if __name__ == "__main__":
    env = gym.make('MountainCar-v0', max_episode_steps=None) #render_mode = 'human', 
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCE(state_dim, action_dim)

    max_episodes = 1000
    max_steps = 200
    max_reward = 0
    for episode in range(max_episodes):
        state, _ = env.reset()
        rewards, states, actions = [], [], []
        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            rewards.append(reward)
            states.append(state)
            actions.append(action)
            state = next_state
            if done:
                break
        agent.learn(rewards, states, actions)
        total_reward = sum(rewards)

        if total_reward > max_reward:
            # 保存模型
            agent.save_model(f'models/model_{total_reward}.pth')
            max_reward = total_reward
        
        # 渲染环境
        if (episode + 1) % 100 == 0:
            # env.render()  # 渲染环境
            print(f'Episode {episode+1}/{max_episodes}, Total Reward: {total_reward}')
            
