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
    def __init__(self, state_dim, action_dim, lr=0.005, gamma=0.99):
        self.policy_net = PolicyNet(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.action_dim = action_dim

    def choose_action(self, state):
        # 向左推(0)、不推(1)、向右推(2)
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

def sign_compare(a, b):
    if a == 0 or b == 0:
        if a == 0 and b == 0:
            return 1
        else:
            return -1
    elif (a > 0 and b > 0) or (a < 0 and b < 0):
        return 1
    else:
        return -1


def train(max_reward = -float('inf')):
    env = gym.make('MountainCar-v0', max_episode_steps=None) #render_mode = 'human', 
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCE(state_dim, action_dim)

    max_episodes = 1000
    max_steps = 500
    better_model_path = ''
    
    for episode in range(max_episodes):
        state, _ = env.reset()

        rewards, states, actions = [], [], []
        for step in range(max_steps):
            action = agent.choose_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            # 1. abs(next_state[0] + 0.5)
            # 2. abs(next_state[0] + 0.5) - abs(state[0] + 0.5)
            # 3. abs(next_state[1]) - abs(state[1])
            # 4. sign_compare(action - 1, state[1])
            reward = 0.5 - next_state[0]
            
            rewards.append(reward)
            states.append(state)
            actions.append(action)
            state = next_state
            if done or truncated:
                break
        agent.learn(rewards, states, actions)
        total_reward = sum(rewards)

        if total_reward > max_reward:
            # 保存模型
            better_model_path = f'1.REINFORCE/models/model_{total_reward:.5f}.pth'
            agent.save_model(better_model_path)
            max_reward = total_reward
        
        # 渲染环境
        if (episode + 1) % 100 == 0:
            # env.render()  # 渲染环境
            print(f'Episode {episode+1}/{max_episodes}, Total Reward: {total_reward}')

    return better_model_path
            
def test(model_path, num_episodes=10):
    env = gym.make('MountainCar-v0', render_mode = 'human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = PolicyNet(state_dim, action_dim)
    policy_net.load_state_dict(torch.load(model_path))

    total_rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            temp_state = torch.from_numpy(state).float().unsqueeze(0)
            probs = policy_net(temp_state)
            action_probs = probs.squeeze().detach().numpy()
            action = np.random.choice(action_dim, p=action_probs)
            
            # if state[1] > 0:
            #     action = 2
            # elif state[1] < 0:
            #     action = 0
            # else:
            #     action = 1
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            state = next_state
            
            if done or truncated:
                break

        total_rewards.append(episode_reward)
        print(f'Episode {episode+1}/{num_episodes}, Reward: {episode_reward}')

    print(f'Average Reward: {sum(total_rewards) / num_episodes}')

if __name__ == '__main__':
    model_path = '1.REINFORCE/models/model_60.00000.pth'

    model_path = train()
    print(model_path)
    test(model_path)