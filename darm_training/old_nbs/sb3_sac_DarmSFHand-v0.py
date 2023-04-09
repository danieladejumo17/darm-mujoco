import gym
from darm_gym_env import DARMSFEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from datetime import datetime

if __name__ == "__main__":
    def create_env():
        return gym.make("darm/DarmSFHand-v0", render_mode=None, hand_name="hand1")

    env = SubprocVecEnv(env_fns=[create_env, create_env, create_env, create_env])
    # env = VecNormalize(env) # FIXME: Remember to save norm params
    env = VecMonitor(env)

    policy_kwargs = dict(net_arch=dict(pi=[32, 256, 256, 64], qf=[32, 256, 256, 64]))
    model = SAC("MlpPolicy", env, verbose=1,
                learning_starts=40_000,
                gradient_steps=4, # num of envs
                policy_kwargs=policy_kwargs,
                tensorboard_log="./results/darm_sf_hand")
    

    try:
        model.learn(total_timesteps=10_000_000, log_interval=8, tb_log_name="PlainDarmSFEnv")
        # Add calbacks
    except Exception as e:
        print("Exception caught:")
        print(e)
    finally:
        timestamp = f"{datetime.now().date()}__{datetime.now().time()}"
        model_name = f"./checkpoints/darm_sf_hand_{timestamp}"
        env_norm_name = f"./checkpoints/darm_sf_hand_env_norm_{timestamp}"
        model.save(model_name)
        # env.save(env_norm_name) # FIXME: Remember to save norm params
        env.close



  # # DONE TRAINING
  # model_name = "darm_training/checkpoints/darm_sf_hand_2022-12-28__10:10:05.637581"
  # env_norm_name = "darm_training/checkpoints/darm_sf_hand_env_norm_2022-12-28__10:10:05.637581"
  
  # env = DummyVecEnv([lambda: gym.make("darm/DarmSFHand-v0", render_mode="human", hand_name="hand1")])
  # env = VecNormalize.load(env_norm_name, env)
  # env.training = False
  
  # # del model # remove to demonstrate saving and loading

  # model = SAC.load(model_name)

  # obs = env.reset()
  # episode_return = 0
  # N_EPISODES = 10

  # import numpy as np
  # def norm_to_target(obs):
  #   """
  #   Returns the norm of each fingertip to the target position
  #   obs: an observation from the observation space [...fingertip_pos, ...target_pos]
  #   """
  #   obs = obs.reshape((-1, 3))
  #   n_fingertips = len(obs)//2

  #   fingertip_poses = obs[0:n_fingertips]
  #   target_poses = obs[n_fingertips:]

  #   return np.linalg.norm(fingertip_poses-target_poses, ord=2, axis=-1)


  # for i in range(N_EPISODES):
  #     obs = env.reset()
  #     done = False
  #     episode_return = 0
  #     episode_return_norm = 0

  #     while not done:
  #       # print("Observation: ", env.unnormalize_obs(obs))
  #       old_norm = norm_to_target(env.unnormalize_obs(obs))
        
  #       action, _states = model.predict(obs, deterministic=True)
  #       # print("Action: ", action)
        
  #       obs, reward, done, info = env.step(action)
  #       new_norm = norm_to_target(env.unnormalize_obs(obs))

  #       # Get actual reward
  #       unnormalized_reward = env.unnormalize_reward(reward)
  #       episode_return += unnormalized_reward
  #       episode_return_norm += reward
  #       # print(f"Reward: {unnormalized_reward}; Normalized: {reward}")

  #       # print(f"Next Observation: {env.unnormalize_obs(obs)}")
  #       # print(f"Change in Norm: {new_norm - old_norm}")
  #       # print("-----------------------------------------------------")

  #       # render
  #       env.render()

  #     print(f"Episode Return: {episode_return}")
  #     print(f"Episode Return Norm: {episode_return_norm}")
  #     print("Zero Norm: ", env.unnormalize_reward(-0.47959065))
  #     if episode_return > -70: 
  #       print("Goal Reached!")
  #     print("\n\n")