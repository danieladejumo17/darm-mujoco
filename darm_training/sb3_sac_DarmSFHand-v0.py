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

  # env = gym.make("darm/DarmSFHand-v0", render_mode=None, hand_name="hand1")
  env = SubprocVecEnv(env_fns=[create_env, create_env, create_env, create_env])
  env = VecNormalize(env)
  env = VecMonitor(env)


  model = SAC("MlpPolicy", env, verbose=1, 
              tensorboard_log="./results/darm_sf_hand",
              learning_starts=40_000)
  model.learn(total_timesteps=10_000_000, log_interval=100, tb_log_name="VecNormEnv")

  timestamp = f"{datetime.now().date()}__{datetime.now().time()}"
  model_name = f"./checkpoints/darm_sf_hand_{timestamp}"
  env_norm_name = f"./checkpoints/darm_sf_hand_env_norm_{timestamp}"
  model.save(model_name)
  env.save(env_norm_name)

  # DONE TRAINING

  # env = gym.make("darm/DarmSFHand-v0", render_mode="human", hand_name="hand1")
  # env = DummyVecEnv([create_env])
  # env = VecNormalize.load(env_norm_name, env)

  # del model # remove to demonstrate saving and loading

  # model = SAC.load(model_name)

  # print(env)
  # obs = env.reset()
  # while True:
  #     action, _states = model.predict(obs, deterministic=True)
  #     obs, reward, done, info = env.step(action)
  #     print("Action: ", action)
  #     print("Observation: ", obs)
  #     env.render()
  #     if done:
  #       obs = env.reset()