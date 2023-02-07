from setuptools import setup

setup(
    name="darm_gym_env",
    version="0.0.1",
    install_requires=["mujoco==2.2.2", "gym==0.21.0"],
    packages=["darm_gym_env"]
)