from gym_env.darm_gym import DARMEnv

from gym.envs.registration import register

register(
    id="darm/DarmHand-v0",
    entry_point="gym_env:DARMEnv",
    max_episode_steps=100
)
