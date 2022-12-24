import gym
from gym_env import DARMEnv
import time
import random

env = gym.make("darm/DarmHand-v0", render_mode="human")

done = False
obs, info = env.reset()

start = time.time()
while not done:
    ac = env.action_space.sample()
    if random.random() > 0.5: 
        ac[0:4] = [0.0, 0.0, 20.0, 20.0]
    obs, rew, term, trunc, info = env.step(ac)
    done = term or trunc

    # TODO: Address links going beyond their limits
    # TODO: Add trucation signal from closing window

print(f"Duration: {time.time() - start}")
print(info)
env.close()