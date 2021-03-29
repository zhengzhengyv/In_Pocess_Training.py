import random
import time
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt

from DDPG import Robot, STATE_DIM, ACTION_DIM, ACTION_BOUND, Workspace, w

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w,init_w)
        self.linear3.bias.data.uniform_(-init_w,init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.conv(x.unsqueeze(1)).squeeze(1))
        x = self.linear3(x)

        return x

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w = 3e-3):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        # uniform_将tensor用从均匀分布中抽样得到的值填充。参数初始化
        self.linear3.weight.data.uniform_(-init_w, init_w)
        #也用用normal_(0, 0.1) 来初始化的，高斯分布中抽样填充，这两种都是比较有效的初始化方式
        self.linear3.bias.data.uniform_(-init_w, init_w)
        #其意义在于我们尽可能保持 每个神经元的输入和输出的方差一致。
        #使用 RELU（without BN） 激活函数时，最好选用 He 初始化方法，将参数初始化为服从高斯分布或者均匀分布的较小随机数
        #使用 BN 时，减少了网络对参数初始值尺度的依赖，此时使用较小的标准差(eg：0.01)进行初始化即可

        #但是注意DRL中不建议使用BN

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.conv(x.unsqueeze(1)).squeeze(1))
        x = F.tanh(self.linear3(x))

        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.forward(state)  # shape[1,3]
        return action.detach().cpu().numpy()[0]  # hong_modified

class OUNoise(object):
    def __init__(self, action_dim=3, low=-1, high=1, mu=0.0, theta = 0.15, max_sigma = 0.3, min_sigma = 0.3, decay_period = 100000):#decay_period要根据迭代次数合理设置
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.low = low
        self.high = high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) *self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta* (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


# class NormalizedActions(gym.ActionWrapper):
#
#     def action(self, action):
#         low_bound = self.action_space.low
#         upper_bound = self.action_space.high
#
#         action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
#         #将经过tanh输出的值重新映射回环境的真实值内
#         action = np.clip(action, low_bound, upper_bound)
#
#         return action
#
#     def reverse_action(self, action):
#         low_bound = self.action_space.low
#         upper_bound = self.action_space.high
#
#         #因为激活函数使用的是tanh，这里将环境输出的动作正则化到（-1，1）
#
#         action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
#         action = np.clip(action, low_bound, upper_bound)

        # return action


class DDPG(object):
    def __init__(self, action_dim, state_dim, hidden_dim):
        super(DDPG,self).__init__()
        self.action_dim, self.state_dim, self.hidden_dim = action_dim, state_dim, hidden_dim
        self.batch_size = 128
        self.gamma = 0.99
        self.min_value = -np.inf
        self.max_value = np.inf
        self.soft_tau = 1e-2
        self.replay_buffer_size = 5000
        self.value_lr = 1e-6
        self.policy_lr = 1e-5

        self.value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        self.target_value_net = ValueNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.policy_lr)

        self.value_criterion = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

    def ddpg_update(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(device)

        policy_loss = self.value_net(state, self.policy_net(state))
        policy_loss = -policy_loss.mean()

        next_action = self.target_policy_net(next_state)
        target_value = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, self.min_value, self.max_value)

        value = self.value_net(state, action)
        value_loss = self.value_criterion(value, expected_value.detach())

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        print(dict(policy_loss=policy_loss.item()))
        print(dict(value_loss=value_loss.item()))
        w.observe(event_name='value_loss',value_loss=value_loss.item())
        w.observe(event_name='loss_policy', loss_policy=-policy_loss.item())


        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

# def plot(frame_idx, rewards):
#     plt.figure(figsize=(20,5))
#     plt.subplot(131)
#     plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
#     plt.plot(rewards)
#     plt.show()


def main():
    # env = gym.make("Pendulum-v0")
    # env = NormalizedActions(env)
    ur_robot = Robot()
    ur_robot.connection()
    ur_robot.read_object_handle()

    # ou_noise = OUNoise(env.action_space)
    # from DDPG import Robot, STATE_DIM, ACTION_DIM, ACTION_BOUND, Workspace
    ou_noise = OUNoise(action_dim=ACTION_DIM, low=-1, high=1)

    # state_dim = env.observation_space.shape[0]
    # action_dim = env.action_space.shape[0]
    state_dim = STATE_DIM
    action_dim = ACTION_DIM  # Workspace.shape[0]

    hidden_dim = 256

    ddpg = DDPG(action_dim, state_dim, hidden_dim)

    # max_frames = 12000
    max_episodes = 12000
    max_steps = 600
    frame_idx = 0
    rewards = []
    batch_size = 256

    # while frame_idx < max_frames:
    for episode in range(max_episodes):
        # state = env.reset()
        ur_robot.reset_target()
        state = ur_robot.get_state()

        ou_noise.reset()
        episode_reward = 0

        for step in range(max_steps):
            print(dict(episode=episode, step=step))
            # time.sleep(0.2)
            # env.render()

            action = ddpg.policy_net.get_action(state)  # should be 3-d vector ranging from (-1,1)
            for i in range(action_dim):  # apply transform here
                action[i] = np.clip(action[i], -1, 1)
            action = ou_noise.get_action(action, step)
            print(dict(action=action))
            w.observe(event_name='action', action=np.mean(action))

            for i in range(action_dim):  # apply transform here
                action[i] = action[i] * (Workspace[i][1] - Workspace[i][0]) / 2 + (Workspace[i][1] + Workspace[i][0]) / 2

            ur_robot.conduct_action(action)
            next_state = ur_robot.get_state()
            reward = ur_robot.get_reward()
            done = ur_robot.tip_target_dis <= 2
            # next_state, reward, done, _ = env.step(action)
            print('r',reward)
            # w.observe(event_name='state', state=np.mean(state))
            w.observe(event_name='reward', reward=reward)


            ddpg.replay_buffer.push(state, action, reward, next_state, done)
            if len(ddpg.replay_buffer) > batch_size:
                print('train')
                ddpg.ddpg_update()

            state = next_state
            episode_reward += reward
            frame_idx += 1

            # if frame_idx % max(1000, max_steps + 1) == 0:
            #     plot(frame_idx, rewards)

            if done:
                break

        rewards.append(episode_reward)
        torch.save({
            'ddpg': ddpg,
            'episode_reward':episode_reward
        }, 'torch_model.pt')
    # env.close()


if __name__ == '__main__':
    main()