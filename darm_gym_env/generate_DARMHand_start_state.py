import gym
from darm_gym_env import DARMEnv
import time
import random
import numpy as np
import os
from tqdm import tqdm

# ================================= TODO: CHECKLIST =================================
# Change file path
# Change single_finger_env
# Change Mujoco XML
# ================================= TODO: CHECKLIST =================================

# Choose single_finger or multi-fingers
START_STATE_FILE = f"{os.getenv('DARM_MUJOCO_PATH')}/darm_gym_env/DARMHand_SF_start_state.npy"
# START_STATE_FILE = f"{os.getenv('DARM_MUJOCO_PATH')}/darm_gym_env/DARMHand_MFNW_start_state.npy"

env = gym.make("darm/DarmHand-v0", render_mode=None, hand_name="hand1",
               single_finger_env=True,
               ignore_load_start_states=True)

targets = []
N_OBERVATIONS = int(1e4)

try:
    for _ in tqdm(range(N_OBERVATIONS)):
        _, joint_state, target_obs = env.generate_start_state()
        targets.append(np.array([joint_state, target_obs], dtype=object))
except Exception as e:
    print(e)
finally:
    # Close the environment
    env.close()

    # save the targets
    print("Saving start state Observation")
    print(f"Num of observations: {len(targets)}")
    print(f"Last two observations: {targets[-2:]}, ...")

    if os.path.exists(START_STATE_FILE):
        print("Loading saved observations ...")
        with open(START_STATE_FILE, 'rb') as f:
            prev_targets = [i for i in np.load(f, allow_pickle=True)]
            print("Appending new observations ...")
            targets = prev_targets + targets
            print(f"Total observations: {len(targets)}")
    else:
        # Create targets file
        f = open(START_STATE_FILE, 'x')
        f.close()

    with open(START_STATE_FILE, 'wb') as f:
        np.save(f, np.array(targets))

    # TODO: Get Targets from other wrist pose
