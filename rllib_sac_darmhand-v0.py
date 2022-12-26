from ray.rllib.algorithms.sac import SACConfig
from ray.tune.logger import pretty_print
import ray.tune as tune
import ray

import gym
from gym_env import DARMEnv

env = gym.make("darm/DarmHand-v0", render_mode=None, hand_name="hand1")
tune.register_env("darm/DarmHand-v0", lambda env_ctx: env)

env = gym.make("darm/DarmHand-v0")
ray.rllib.utils.check_env(env)

algo = (
    SACConfig()
    .rollouts(num_rollout_workers=4, rollout_fragment_length=1)
    .resources(num_gpus=0)
    .environment(env="darm/DarmHand-v0", normalize_actions=True)
    .framework("torch")
    .reporting(min_sample_timesteps_per_iteration=1000)
    .training(q_model_config={"fcnet_activation": "relu", "fcnet_hiddens": [256, 256]},
                policy_model_config={"fcnet_activation": "relu", "fcnet_hiddens": [256, 256]},
                tau=0.005,
                n_step=1,
                train_batch_size=256,
                target_network_update_freq=1,
                replay_buffer_config={"type": "MultiAgentPrioritizedReplayBuffer"},
                num_steps_sampled_before_learning_starts=10000,
                optimization_config={"actor_learning_rate": 0.0003, "critic_learning_rate": 0.0003, 
                                    "entropy_learning_rate": 0.0003},
                clip_actions=False
            )
    .build()
    # .evaluation(evaluation_num_workers=1)
)

for i in range(1):
    result = algo.train()

    if i % 5 == 0 or i == 9:
        print(pretty_print(result))
        checkpoint_dir = algo.save()
        print(f"Checkpoint saved in directory {checkpoint_dir}")
