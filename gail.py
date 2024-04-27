from replay_buffer import replay_buffer
from net import disc_policy_net, value_net, discriminator, unity_policy_net
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import gym
import random
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
import pandas as pd 


class gail(object):
    def __init__(self, env, episode, capacity, gamma, lam, is_disc, value_learning_rate, policy_learning_rate, discriminator_learning_rate, batch_size, file, policy_iter, disc_iter, value_iter, epsilon, entropy_weight, train_iter, clip_grad, render):
        self.env = env
        self.episode = episode
        self.capacity = capacity
        self.gamma = gamma
        self.lam = lam
        self.is_disc = is_disc
        self.value_learning_rate = value_learning_rate
        self.policy_learning_rate = policy_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.batch_size = batch_size
        self.file = file
        self.policy_iter = policy_iter
        self.disc_iter = disc_iter
        self.value_iter = value_iter
        self.epsilon = epsilon
        self.entropy_weight = entropy_weight
        self.train_iter = train_iter
        self.clip_grad = clip_grad
        self.render = render

        # self.observation_dim = self.env.observation_space.shape[0]
        # if is_disc:
        #     self.action_dim = self.env.action_space.n
        # else:
        #     self.action_dim = self.env.action_space.shape[0]
        if is_disc:
            self.policy_net = disc_policy_net(self.observation_dim, self.action_dim)
        else:
            self.policy_net = unity_policy_net(64, 9)
        # self.value_net = value_net(self.observation_dim, 1)
        # self.discriminator = discriminator(self.observation_dim + self.action_dim)
        self.buffer = replay_buffer(self.capacity, self.gamma, self.lam)
        self.pool = pickle.load(self.file)  #专家数据经验
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.policy_learning_rate)
        # self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.value_learning_rate)
        # self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.discriminator_learning_rate)
        self.disc_loss_func = nn.MSELoss()
        self.weight_reward = None
        self.weight_custom_reward = None

    def sampleData(self):
        self.pool = random.sample(self.pool, 2000)
        #print(self.pool.__len__())
    '''
    [[array([]),0],[array([]),1]]
    '''
    def policy_train(self,ep):
        #print(self.pool.__len__())
        expert_batch = random.sample(self.pool, self.batch_size)
        expert_batch= np.vstack(expert_batch)
        expert_batch = torch.FloatTensor(expert_batch)
        expert_observations = expert_batch[:,:21168] #21168
        expert_actions = expert_batch[:,21168:]
        # expert_observations, expert_actions = zip(* expert_batch)
        # expert_observations = np.vstack(expert_observations)
        # expert_observations = torch.FloatTensor(expert_observations)
        # if self.is_disc: # 离散环境
        #     expert_actions_index = torch.LongTensor(expert_actions).unsqueeze(1)
        #     expert_actions = torch.zeros(self.batch_size, self.action_dim)
        #     expert_actions.scatter_(1, expert_actions_index, 1)
        # else:
        #     expert_actions = torch.FloatTensor(expert_actions).unsqueeze(1)
        for i in range(3):
            y_predict = self.policy_net.forward(expert_observations)
            loss = self.disc_loss_func(y_predict, expert_actions)
            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()
            if(ep%500 == 0):
                print(ep,':',loss)
        if ep%2000 == 0:
            torch.save(self.policy_net.state_dict(), './traj/unity_net_params.pkl')

        #expert_trajs = torch.cat([expert_observations, expert_actions], 1)
        # 专家数据标签为 0
    
    def load_model(self):
        self.policy_net.load_state_dict(torch.load('./traj/unity_net_params.pkl')) 
    
    def cartpoleTest(self,ep):
        episode_traj = [] # 每个episode数据
        obs = np.array(self.env.reset()).ravel()
        total_reward = 0
        while True:
            action = self.policy_net(torch.FloatTensor(np.expand_dims(obs, 0)))
            action = action.detach().numpy()
            next_obs, reward, done, _ = self.env.step(action)
            total_reward = total_reward + reward
            self.env.render()
            # s_a = np.append(obs,action) # 存每个episode
            # episode_traj.extend([s_a])# 存每个episode
            obs = np.array(next_obs).ravel()
            if done:
                print('test:',ep,':',total_reward)
                break
        #name = str(ep)+'.csv'
        #pd.DataFrame(episode_traj).to_csv(name)
