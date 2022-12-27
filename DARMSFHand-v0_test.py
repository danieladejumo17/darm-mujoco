import gym
from darm_gym_env import DARMSFEnv
import time

env = gym.make("darm/DarmSFHand-v0", render_mode="human", hand_name="hand1")

done = False
obs = env.reset()

start = time.time()
while not done:
    ac = env.action_space.sample()
    obs, rew, done, info = env.step(ac)
    env.render()

    if done:
        print(f"Duration: {time.time() - start}")
        print(info)
        start = time.time()

        env.reset()
        done = False

    # TODO: Address links going beyond their limits
    # TODO: Add trucation signal from closing window

env.close()