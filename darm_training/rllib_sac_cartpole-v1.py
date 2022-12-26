from ray.rllib.algorithms.sac import SACConfig
from ray.tune.logger import pretty_print

config = SACConfig().training(gamma=0.9, lr=0.01)  
config = config.resources(num_gpus=0)  
config = config.rollouts(num_rollout_workers=4, num_envs_per_worker=4)
config = config.framework("torch")  
print(config.to_dict())  
# Build a Algorithm object from the config and run 1 training iteration.
algo = config.build(env="CartPole-v1")  
print(pretty_print(algo.train()))
