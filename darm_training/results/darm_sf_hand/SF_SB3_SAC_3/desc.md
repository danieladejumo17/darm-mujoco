# RUN:

This run used the TransformObservation wrapper to scale observations from m and m/s to cm and cm/s
- Observation includes fingertip_position size(3,), target_position size(3,), and fingertip_velocity size(3,)
- Mujoco action range is [0, 5]
- Model action range is [-1, 1]


Details:
"""
- The environment was updated such that the target is within a range from the start point
- Velocity penalty was removed and only effort penalty was used
- The reward function was updated according to the reach task reward used in facebookresearch/myosuite [https://github.com/facebookresearch/myosuite/blob/main/myosuite/envs/myo/reach_v0.py]
- The done signal is trigerred only when the fingertip goes beyond a threshold. The episode continues to the maximum timestep otherwise.
- The friction and damping coefficient of the environment is updated. Values are inspired from Deepmind's Mujoco Menagerie [https://github.com/deepmind/mujoco_menagerie/blob/main/shadow_hand/right_hand.xml]
- The range of action from the model was changed to [-1, 1]. This action is mapped to the actual action sent to mujoco e.g [0, 2]]. This change is inspired from values used in OpenAI's Gym Mujoco environments.
- max_episode_steps was updated to 200.
- Velocity vector (size [3,]) was added to observation. Observation size is now (9,)
- Action range was increased to [0, 5]
- Observation warpper to scale observation from m and m/s to cm and cm/s was applied

- This run was trained on vast_ai using SB3's SAC algo.
"""


# RESULTS:

- 'model' is the last model saved from training
- 'best_model' is the best model saved from evaluation which lags training
- 'model_wandb_last' is the last model uploaded to wandb

- 'model' can still be fine-tuned with another SAC run
