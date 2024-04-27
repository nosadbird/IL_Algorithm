import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

# ppo policy 离散环境 discrete
class disc_policy_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(disc_policy_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, 1)
    # 离散环境下，根据各可能动作的概率进行选择
    def act(self, input):
        probs = self.forward(input)
        dist = Categorical(probs)
        action = dist.sample()
        action = action.detach().item()
        return action

# ppo policy 连续环境 continuous
class cont_policy_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(cont_policy_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)

    def forward(self, input):
        x = torch.tanh(self.fc1(input))
        x = torch.tanh(self.fc2(x))
        mu = self.fc3(x)
        return mu

    def act(self, input):
        mu = self.forward(input)
        sigma = torch.ones_like(mu)
        dist = Normal(mu, sigma)
        action = dist.sample().detach().item()
        return action

    def get_distribution(self, input):
        mu = self.forward(input)
        sigma = torch.ones_like(mu)
        dist = Normal(mu, sigma)
        return dist

# ppo valueNet
class value_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(value_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, self.output_dim)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 判别器 0 - 1
class discriminator(nn.Module):
    def __init__(self, input_dim):
        super(discriminator, self).__init__()
        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.model(input)

# ppo policy 连续环境 continuous 128
class unity_policy_net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(unity_policy_net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, 5292)
        self.fc2 = nn.Linear(5292, 1323)
        self.fc3 = nn.Linear(1323, self.output_dim)

    def forward(self, input):
        x = torch.tanh(self.fc1(input))
        x = torch.tanh(self.fc2(x))
        mu = self.fc3(x)
        return mu

    def act(self, input):
        mu = self.forward(input)
        sigma = torch.ones_like(mu)
        dist = Normal(mu, sigma)
        action = dist.sample().detach().item()
        return action

    def get_distribution(self, input):
        mu = self.forward(input)
        sigma = torch.ones_like(mu)
        dist = Normal(mu, sigma)
        return dist
