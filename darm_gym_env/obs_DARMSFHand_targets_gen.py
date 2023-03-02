import gym
from darm_gym_env import DARMSFEnv
import time
import random
import numpy as np
import os


TARGETS_FILE = "darm_sf_joint_space_targets.npy"

targets = []
if os.path.exists(TARGETS_FILE):
    print("Loading saved target")
    with open(TARGETS_FILE, 'rb') as f:
        targets = [i for i in np.load(f)]
        print(f"Targets: {targets[-2:]}, ...")
else:
    f = open(TARGETS_FILE, 'x')
    f.close()

env = gym.make("darm/DarmSFHand-v0", render_mode=None, hand_name="hand1")
env.reset()
N_SAMPLES = 10_000

for i in range(N_SAMPLES):
    max_vals = np.array([20, 90, 90, 90])*(np.pi/180)
    min_vals = np.array([-20, -45, -10, -10])*(np.pi/180)

    joint_space = gym.spaces.Box(low=min_vals, high=max_vals, 
                                    shape=(4,), dtype=float)
    
    target = joint_space.sample()
    while(target[3] > target[2]):
        target = joint_space.sample()

    obs = env.forward(target)
    targets.append(obs[:3])

    # env.render()
    # time.sleep(1)

    if i % 500 == 0:
        print(f"Sampled {i} targets")

# Close the environment
env.close()

# save the targets
print("Saving Targets")
print(f"Num of Targets: {len(targets)}")
print(f"Targets: {targets[-2:]}, ...")
with open(TARGETS_FILE, 'wb') as f:
    np.save(f, np.array(targets))



# all_obs = []
# valid_obs_idx = []

# def filter_obs(obs) -> bool:
#     all_obs.append(obs)
#     # if len(all_obs) % 70 == 0: print(f"Random Observation {len(all_obs)}: {obs}")
#     max_vals = np.array([20, 90, 90, 90])*(np.pi/180)
#     min_vals = np.array([-20, -45, -10, -10])*(np.pi/180)
#     # max_vals = np.array([0.3491, 1.571, 1.571, 1.571])
#     # min_vals = np.array([-0.3491, -0.7854, -0.1745, -0.1745])

#     lm = obs <= max_vals
#     gm = obs >= min_vals
#     lm = lm.all(); gm = gm.all()
#     valid = lm and gm

#     if valid:
#         print(f"Valid Observation: Total Valid: {len(valid_obs_idx)} All: {len(all_obs)} || {obs}")
#         valid_obs_idx.append(len(all_obs) - 1)
#         return True
#     return False