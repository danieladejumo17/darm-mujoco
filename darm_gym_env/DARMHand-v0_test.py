import gym
# from darm_gym_env import DARMEnv
from darm_gym import DARMEnv
import time
from pprint import pprint
import numpy as np

# env = gym.make("darm/DarmHand-v0", render_mode="human", hand_name="hand1")
env = DARMEnv(render_mode="human", hand_name="hand1", 
              digits=["iii"],
              start_state_file="DARMHand_diii_start_state.npy")
env = gym.wrappers.TimeLimit(env, max_episode_steps=200)

done = False
obs = env.reset()

start = time.time()
episode_return = 0

while not done:
    ac = np.array([0, 0, 1, 0, 0]) # env.action_space.sample()
    # print("-----------------")
    # old_len = env.data.actuator(2).length[0]
    # print(f"Old Length: {old_len}")
    obs, rew, done, info = env.step(ac)
    # new_len = env.data.actuator(2).length[0]
    # print(f"New Length: {new_len}")
    # print(f"Change: {new_len - old_len}")
    # print("-----------------")
    episode_return += rew

    # time_prev = time.time()
    # while time.time() - time_prev < 1:
    env.render()
    # print(ac > 0)
    # for i in range(env.nact):
    #     print(f"{env.data.actuator(i).name} force: {env.data.actuator(i).force}")
    #     print(f"{env.data.actuator(i+env.nact).name} force: {env.data.actuator(i+env.nact).force}")
    # time.sleep(5)

    if done:
        obs = env.reset()
        done = False

        print(f"Duration: {time.time() - start}")
        pprint(info)
        print(f"Episode Return: {episode_return} \n\n")
        episode_return = 0

        start = time.time()
    # TODO: Address links going beyond their limits

env.close()