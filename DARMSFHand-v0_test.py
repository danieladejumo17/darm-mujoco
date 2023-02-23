import gym
from darm_gym_env import DARMSFEnv
import time

env = gym.make("darm/DarmSFHand-v0", render_mode="human", hand_name="hand1")

done = False
obs = env.reset()

start = time.time()
episode_return = 0
while not done:
    ac = env.action_space.sample()
    obs, rew, done, info = env.step(ac)
    # obs is in m and m/s. It can be scaled before passing into model in cm and cm/s
    episode_return += rew
    env.render()

    if done:
        print(f"Duration: {time.time() - start}")
        print(info)
        print(f"Return: {episode_return}")
        episode_return = 0
        start = time.time()

        obs = env.reset()
        done = False

    # TODO: Address links going beyond their limits
    # TODO: Add trucation signal from closing window

env.close()