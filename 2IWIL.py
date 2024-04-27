import argparse
from itertools import count

# import gym
import gym
import gym.spaces
import scipy.optimize
import numpy as np
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
from loss import *

"""python 2IWIL.py --env Ant-v2 --num-epochs 5000 --traj-size 600 --weight
2IWIL: proposed method (--weight)python 2IWIL.py --env Ant-v2 --num-epochs 5000 --traj-size 600 --weight --only --noconf
GAIL (U+C): no need to specify option
GAIL (Reweight): use only confidence data (--weight --only)
GAIL (C): use only confidence data without reweighting (--weight --only --noconf)
"""
# import wandb

# wandb.init(project="2IWIL", entity="nxbb")

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env', type=str, default="Reacher-v1", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value (default: 1e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=1111, metavar='N',
                    help='random seed (default: 1111')
parser.add_argument('--batch-size', type=int, default=5000, metavar='N',
                    help='size of a single batch')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--fname', type=str, default='expert', metavar='F',
                    help='the file name to save trajectory')
parser.add_argument('--num-epochs', type=int, default=500, metavar='N',
                    help='number of epochs to train an expert')
parser.add_argument('--hidden-dim', type=int, default=100, metavar='H',
                    help='the size of hidden layers')
parser.add_argument('--lr', type=float, default=1e-3, metavar='L',
                    help='learning rate')
parser.add_argument('--weight', action='store_true',
                    help='consider confidence into loss')
parser.add_argument('--only', action='store_true',
                    help='only use labeled samples')
parser.add_argument('--noconf', action='store_true',
                    help='use only labeled data but without conf')
parser.add_argument('--vf-iters', type=int, default=30, metavar='V',
                    help='number of iterations of value function optimization iterations per each policy optimization step')
parser.add_argument('--vf-lr', type=float, default=3e-4, metavar='V',
                    help='learning rate of value network')
parser.add_argument('--noise', type=float, default=0.0, metavar='N')
parser.add_argument('--eval-epochs', type=int, default=3, metavar='E',
                    help='epochs to evaluate model')
parser.add_argument('--prior', type=float, default=0.2,
                    help='ratio of confidence data')
parser.add_argument('--traj-size', type=int, default=600) #原本2000
parser.add_argument('--ofolder', type=str, default='log')
parser.add_argument('--ifolder', type=str, default='demonstrations')
args = parser.parse_args()

#env = gym.make("hopper-v3")
env = gym.make("MiniGrid-Empty-5x5-v0")

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

policy_net = Policy(num_inputs, num_actions, args.hidden_dim)
value_net = Value(num_inputs, args.hidden_dim).to(device)
discriminator = Discriminator(num_inputs + num_actions, args.hidden_dim).to(device)
disc_criterion = nn.BCEWithLogitsLoss()
value_criterion = nn.MSELoss()
disc_optimizer = optim.Adam(discriminator.parameters(), args.lr)
value_optimizer = optim.Adam(value_net.parameters(), args.vf_lr)
mse_loss = nn.MSELoss()

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def update_params(batch):
    rewards = torch.Tensor(batch.reward).to(device)
    masks = torch.Tensor(batch.mask).to(device)
    actions = torch.Tensor(np.concatenate(batch.action, 0)).to(device)
    states = torch.Tensor(batch.state).to(device)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1).to(device)
    deltas = torch.Tensor(actions.size(0),1).to(device)
    advantages = torch.Tensor(actions.size(0),1).to(device)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    batch_size = math.ceil(states.shape[0] / args.vf_iters)
    idx = np.random.permutation(states.shape[0])
    for i in range(args.vf_iters):
        smp_idx = idx[i * batch_size: (i + 1) * batch_size]
        smp_states = states[smp_idx, :]
        smp_targets = targets[smp_idx, :]
        
        value_optimizer.zero_grad()
        value_loss = value_criterion(value_net(Variable(smp_states)), smp_targets)
        value_loss.backward()
        value_optimizer.step()

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states.cpu()))
    fixed_log_prob = normal_log_density(Variable(actions.cpu()), action_means, action_log_stds, action_stds).data.clone()

    def get_loss():
        action_means, action_log_stds, action_stds = policy_net(Variable(states.cpu()))
        log_prob = normal_log_density(Variable(actions.cpu()), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages.cpu()) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()


    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states.cpu()))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)

def expert_reward(states, actions):
    states = np.concatenate(states)
    actions = np.concatenate(actions)
    state_action = torch.Tensor(np.concatenate([states, actions], 1)).to(device)
    return -F.logsigmoid(discriminator(state_action)).cpu().detach().numpy()

# 满5000评估一次
def evaluate(episode):
    avg_reward = 0.0
    for _ in range(args.eval_epochs):
        state = env.reset()
        for _ in range(10000): # Don't infinite loop while learning
            state = torch.from_numpy(state).unsqueeze(0)
            action, _, _ = policy_net(Variable(state))
            action = action.data[0].numpy()
            next_state, reward, done, _ = env.step(action)
            avg_reward += reward
            if done:
                break
            state = next_state
    #writer.log(episode, avg_reward / args.eval_epochs)

# plabel = ''
# try:
#     expert_traj = np.load("./{}/{}_mixture.npy".format(args.ifolder, "HalfCheetah-v2"))
#     expert_conf = np.load("./{}/{}_mixture_conf.npy".format(args.ifolder, "HalfCheetah-v2"))
#     expert_conf += (np.random.randn(*expert_conf.shape) * args.noise)
#     expert_conf = np.clip(expert_conf, 0.0, 1.0)
# except: #13000
#     print('Mixture demonstrations not loaded successfully.')
#     assert False
# #2000条  600
# idx = np.random.choice(expert_traj.shape[0], args.traj_size, replace=False)
# expert_traj = expert_traj[idx, :]
# expert_conf = expert_conf[idx, :]


#if not args.only and args.weight:  # 加权的执行这个
for i_episode in range(5000):#500 args.num_epochs
    memory = Memory()

    num_steps = 0
    num_episodes = 0
    
    reward_batch = []
    states = []
    actions = []
    mem_actions = []
    mem_mask = []
    mem_next = []
    env_reward = []
    next_states = []

    while num_steps < args.batch_size:#5000 不管几个episode，够5000就向下
        state = env.reset()
   

        reward_sum = 0
        for t in range(10000): # 一个episode最多10000步
            action = select_action(state)
            action = action.data[0].numpy()
            states.append(np.array([state]))
            actions.append(np.array([action]))
            next_state, true_reward, done, _ = env.step(action)
            reward_sum += true_reward
            env.render()

            mask = 1
            if done:
                mask = 0

            mem_mask.append(mask)
            mem_next.append(next_state)
            env_reward.append(true_reward)

            if done:
                break

            state = next_state
        num_steps += (t-1)
        num_episodes += 1

        reward_batch.append(reward_sum)

    evaluate(i_episode)

    #rewards = expert_reward(states, actions)
    env_reward = np.array(env_reward)
    env_reward = env_reward.reshape(-1,1)
    rewards = env_reward
    for idx in range(len(states)):
        memory.push(states[idx][0], actions[idx], mem_mask[idx], mem_next[idx], \
                    rewards[idx][0])
    batch = memory.sample()
    update_params(batch)

    ### update discriminator ###
    # actions = torch.from_numpy(np.concatenate(actions))
    # states = torch.from_numpy(np.concatenate(states))
    
    # idx = np.random.randint(0, expert_traj.shape[0], num_steps)
    
    # expert_state_action = expert_traj[idx, :]
    # expert_pvalue = expert_conf[idx, :]
    # expert_state_action = torch.Tensor(expert_state_action).to(device)
    # expert_pvalue = torch.Tensor(expert_pvalue / Z).to(device)

    # state_action = torch.cat((states, actions), 1).to(device)
    # fake = discriminator(state_action)
    # real = discriminator(expert_state_action)

    # disc_optimizer.zero_grad()
    # weighted_loss = nn.BCEWithLogitsLoss(weight=expert_pvalue)
    # if True: #args.weight:
    #     disc_loss = disc_criterion(fake, torch.ones(states.shape[0], 1).to(device)) + \
    #                 weighted_loss(real, torch.zeros(expert_state_action.size(0), 1).to(device))
    # else:
    #     disc_loss = disc_criterion(fake, torch.ones(states.shape[0], 1).to(device)) + \
    #                 disc_criterion(real, torch.zeros(expert_state_action.size(0), 1).to(device))
    # disc_loss.backward()
    # disc_optimizer.step()
    ############################

   # if i_episode % args.log_interval == 0:
        #wandb.log({"Avg_owner": np.mean(reward_batch)})
        #print('Episode {}\tAverage reward: {:.2f}\tMax reward: {:.2f}\tLoss (disc): {:.2f}'.format(i_episode, np.mean(reward_batch), max(reward_batch), disc_loss.item()))

