import gym
from gail import gail
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment


if __name__ == '__main__':
    # * make the performance improve evidently
    #env = gym.make('CartPole-v0')#None
    env_path = "./Worm/UnityEnvironment.exe"
    #env_path = None
    unity_env = UnityEnvironment(env_path)
    env = UnityToGymWrapper(unity_env, allow_multiple_obs=True)
    file = open('./traj/imgcar8-2.pkl', 'rb')
    test = gail(
        env=env,
        episode=5000,
        capacity=1000,
        gamma=0.99,
        lam=0.95,
        is_disc=False,
        value_learning_rate=3e-4,
        policy_learning_rate=3e-4,
        discriminator_learning_rate=3e-4,
        batch_size=64,
        file=file,
        policy_iter=1,
        disc_iter=10,
        value_iter=1,
        epsilon=0.2,
        entropy_weight=1e-4,
        train_iter=500,
        clip_grad=40,
        render=False
    )
    # test.sampleData()

    # for i in range(10000):
    #     test.policy_train(i)
    #     if i%10 == 0:
    #         for j in range(2):
    #             test.cartpoleTest(i)

    test.load_model()
    for i in range(5):
        test.cartpoleTest(i)