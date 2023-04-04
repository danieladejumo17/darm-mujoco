from setuptools import setup

setup(
    name="darm_gym_env",
    version="0.0.1",
    install_requires=["mujoco==2.3.3", "gym==0.21.0"],
    packages=["darm_gym_env"]
)

# 392c8a47eb0658eb5c71190757a69110e2140f4a

# install gcc
# update python
# read number of cpus

# nans at the beginning means no episode has been completed as yet
# avoid shared cpus - low utilization
# avoid low bandwidth - long install times
# only checkpoints from the latest run is resumed
# always check pwd before running 