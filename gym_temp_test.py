import gym
from darm_gym_env import DARMEnv
import time
import random

env = gym.make("Ant-v3")

done = False
obs = env.reset()

start = time.time()
while not done:
    ac = env.action_space.sample()
    obs, rew, done, info = env.step(ac)
    env.render()

    if done:
        env.reset()
        done = False
env.close()