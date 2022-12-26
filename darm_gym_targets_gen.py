import gym
from gym_env import DARMEnv
import time
import random
import numpy as np
import os

# [AdAb: rand; FlEx: rand]
# [AdAb: 0; FlEx: 0]

TARGETS_FILE = "darm_targets.npy"

targets = []
if os.path.exists(TARGETS_FILE):
    print("Loading saved target")
    with open(TARGETS_FILE, 'rb') as f:
        targets = [i for i in np.load(f)]
        print(f"Targets: {targets[-2:]}, ...")
else:
    f = open(TARGETS_FILE, 'x')
    f.close()

env = gym.make("darm/DarmHand-v0", render_mode=None, hand_name="hand1")
N_EPISODES = 100

for _ in range(N_EPISODES):
    done = False
    obs, info = env.reset()

    start = time.time() 
    while not done:
        ac = env.action_space.sample()
        if random.random() > 0.5: 
            ac[0:4] = [0.0, 0.0, 20.0, 20.0]
        obs, rew, term, trunc, info = env.step(ac)
        targets.append(obs[:15])

        done = term or trunc

    # Print Statistics
    print(f"Duration: {time.time() - start}")
    print(info)

# Close the environment
env.close()

# save the targets
print("Saving Targets")
print(f"Num of Targets: {len(targets)}")
print(f"Targets: {targets[-2:]}, ...")
with open(TARGETS_FILE, 'wb') as f:
    np.save(f, np.array(targets))

# TODO: Get Targets from other wrist pose