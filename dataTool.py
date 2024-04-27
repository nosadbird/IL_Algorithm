import numpy as np
from mlagents.trainers import demo_loader



def LoadDataFromUnityDemo(demo_path):
    _, info_action_pairs, _ = demo_loader.load_demonstration(demo_path)

    # print(len(info_action_pairs))
    # print(info_action_pairs[0])
    #
    # print(info_action_pairs[0].agent_info.observations[0].float_data.data)
    # print(info_action_pairs[0].action_info.continuous_actions)

    observations = []
    actions = []
    rewards = []
    dones = []
    next_observations = []

    obs_pre = np.array(info_action_pairs[0].agent_info.observations[0].float_data.data, dtype=np.float32)
    done_pre = info_action_pairs[0].agent_info.done

    # print(len(info_action_pairs))
    for info_action_pair in info_action_pairs[1:]:
        agent_info = info_action_pair.agent_info
        action_info = info_action_pair.action_info

        obs = np.array(agent_info.observations[0].float_data.data, dtype=np.float32)
        rew = agent_info.reward
        act = np.array(action_info.continuous_actions, dtype=np.float32)
        done = agent_info.done
        if not done_pre:
            observations.append(obs_pre)
            actions.append(act)
            rewards.append(rew)
            dones.append(done)
            next_observations.append(obs)

        obs_pre = obs
        done_pre = done

    data = dict(
                obs=np.array(observations, dtype=np.float32),
                acts=np.array(actions, dtype=np.float32),
                rews=np.array(rewards, dtype=np.float32),
                next_obs=np.array(next_observations, dtype=np.float32),
                done=np.array(dones, dtype=np.float32),
            )

    return data

if __name__ == '__main__':
    path = "D:/app/Unity/workspace/ML-Agent-Project/Assets/Demonstrations/ball-test_3.demo"
    data = LoadDataFromUnityDemo(path)
    print("obs: ",data["obs"].shape)
    print("acts: ",data["acts"].shape)
    print("rews: ",data["rews"].shape)
    print("next_obs: ",data["next_obs"].shape)
    print("done: ",data["done"].shape)
