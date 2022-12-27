import gym
from darm_gym_env import DARMSFEnv
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from datetime import datetime

if __name__ == "__main__":
  def create_env():
    return gym.make("darm/DarmSFHand-v0", render_mode=None, hand_name="hand1")

  # env = gym.make("darm/DarmSFHand-v0", render_mode=None, hand_name="hand1")
  env = SubprocVecEnv(env_fns=[create_env, create_env, create_env, create_env],
                      start_method="forkserver")

  model = SAC("MlpPolicy", env, verbose=1, 
              tensorboard_log="./results/darm_sf_hand",
              learning_starts=30_000)
  model.learn(total_timesteps=10_000_000, log_interval=4)
  model.save(f"darm_sf_hand_{datetime.now().date()}__{datetime.now().time()}")

  # del model # remove to demonstrate saving and loading

  # model = SAC.load("sac_pendulum")

  # obs = env.reset()
  # while True:
  #     action, _states = model.predict(obs, deterministic=True)
  #     obs, reward, done, info = env.step(action)
  #     env.render()
  #     if done:
  #       obs = env.reset()