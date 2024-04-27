from mlagents.trainers import demo_loader
import numpy as np
import os
import pickle
import pandas as pd 


def LoadDataFromUnityDemo(demo_path):
    _, info_action_pairs, _ = demo_loader.load_demonstration(demo_path)

    observations = []
    actions = []
    rewards = []
    dones = []
    next_observations = []

    traj_pool = [] # 轨迹库 compressed_data

    obs_pre = np.array(info_action_pairs[0].agent_info.observations[0].float_data, dtype=np.float32)
    done_pre = info_action_pairs[0].agent_info.done

    # print(len(info_action_pairs))
    for info_action_pair in info_action_pairs[1:]:
        agent_info = info_action_pair.agent_info
        action_info = info_action_pair.action_info

        episode_traj = [] # 每个episode数据

        obs = np.array(agent_info.observations[0].compressed_data, dtype=np.float32)
        rew = agent_info.reward
        act = np.array(action_info.continuous_actions, dtype=np.float32)
        done = agent_info.done
        if not done_pre:
            observations.append(obs_pre) # 202
            actions.append(act) # 2
            rewards.append(rew)
            dones.append(done)  #
            next_observations.append(obs)

            s_a = np.append(obs_pre,act) # 存每个episode
            #s_ns = np.append(obs_pre,obs)
            episode_traj.extend([s_a])

        obs_pre = obs
        done_pre = done

        traj_pool.extend(episode_traj)
        del episode_traj[:]

    #pd.DataFrame(traj_pool).to_csv('sample.csv')
    file = open('D:/Desktop/bc/traj/easycar8-2_0.pkl', 'wb')
    pickle.dump(traj_pool, file)
    data = dict(
                obs=np.array(observations, dtype=np.float32),
                acts=np.array(actions, dtype=np.float32),
                rews=np.array(rewards, dtype=np.float32),
                next_obs=np.array(next_observations, dtype=np.float32),
                done=np.array(dones, dtype=np.float32),
            )
    return data


if __name__ == '__main__':
    path = "D:/app/Unity/workspace/rlenvironments/Assets/Demonstrations/imagecar8-2_0.demo"
    data = LoadDataFromUnityDemo(path)
    print("obs: ",data["obs"].shape)
    print("acts: ",data["acts"].shape)
    print("rews: ",data["rews"].shape)
    print("next_obs: ",data["next_obs"].shape)
    print("done: ",data["done"].shape)