import numpy as np
import random
from collections import deque


class replay_buffer(object):
    def __init__(self, capacity, gamma, lam):
        self.capacity = capacity
        self.gamma = gamma
        self.lam = lam
        self.memory = deque(maxlen=self.capacity)

    def store(self, observation, action, reward, done, value):
        # obs, act, D(判别器打分，越接近专家数据越高), done, ppo_value网络打分
        observation = np.expand_dims(observation, 0)
        self.memory.append([observation, action, reward, done, value])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, dones, values, returns, advantages = zip(* batch)
        # obs act 判别器分值 done ppo-value网络 value网络值-蒙特卡洛-求和  优势值: 判别器值-value值
        return np.concatenate(observations, 0), actions, returns, advantages

    def process(self):
        R = 0
        Adv = 0
        Value_previous = 0
        for traj in reversed(list(self.memory)):
            R = self.gamma * R * (1 - traj[3]) + traj[2]
            traj.append(R)
            # * the generalized advantage estimator(GAE)
            delta = traj[2] + Value_previous * self.gamma * (1 - traj[3]) - traj[4]
            Adv = delta + (1 - traj[3]) * Adv * self.gamma * self.lam
            traj.append(Adv)
            Value_previous = traj[4]

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()