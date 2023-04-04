import gym
# from darm_gym_env import DARMEnv
from darm_gym import DARMEnv
import time
from pprint import pprint

# env = gym.make("darm/DarmHand-v0", render_mode="human", hand_name="hand1")
env = DARMEnv(render_mode="human", hand_name="hand1", 
              digits=["ii"], start_state_file="DARMHand_SF_start_state.npy")
env = gym.wrappers.TimeLimit(env, max_episode_steps=200)

done = False
obs = env.reset()

start = time.time()
episode_return = 0

while not done:
    ac = env.action_space.sample()
    obs, rew, done, info = env.step(ac)
    episode_return += rew

    env.render()

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