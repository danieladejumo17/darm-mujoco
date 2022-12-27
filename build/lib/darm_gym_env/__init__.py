from darm_gym_env.darm_gym import DARMEnv
from darm_gym_env.darm_sf_gym import DARMSFEnv

from gym.envs.registration import register

register(
    id="darm/DarmHand-v0",
    entry_point="darm_gym_env:DARMEnv",
    max_episode_steps=100
)

register(
    id="darm/DarmSFHand-v0",
    entry_point="darm_gym_env:DARMSFEnv",
    max_episode_steps=100
)
