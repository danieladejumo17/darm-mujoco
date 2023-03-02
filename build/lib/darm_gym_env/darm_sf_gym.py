import os
import numpy as np
import collections

import gym
import mujoco as mj
from mujoco.glfw import glfw

from pathlib import Path


# TARGETS_FILE = f"{os.getenv('DARM_MUJOCO_PATH')}/darm_sf_joint_space_targets.npy"
DARM_XML_FILE = f"{os.getenv('DARM_MUJOCO_PATH')}/mujoco_env/darm.xml"

class DARMSFEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None, reaction_time=0.08, hand_name="hand1",
                    target_joint_state_delta = [4, 8, 8, 8],
                    min_th = 0.004,
                    min_target_th = 2*0.004,
                    max_target_th = 10*0.004,
                    min_joint_vals = [-20, -45, -10, -10],
                    max_joint_vals = [20, 90, 90, 90]) -> None:
        super().__init__()

        # Env Parameters
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.hand_name = hand_name
        self.reaction_time = reaction_time
        self.ep_start_time = 0  # episode start time
        
        # abs increament of joint state from starting state to target state
        self.target_joint_state_delta = np.array(target_joint_state_delta)*(np.pi/180)  
        self.min_th = min_th    # norm threshold in metres at which env is solved
        self.min_target_th = min_target_th  # min norm to target state
        self.max_target_th = max_target_th  # max norm to target state
        # TODO: The following data should be read from mujoco
        self.min_joint_vals = np.array(min_joint_vals)*(np.pi/180)
        self.max_joint_vals = np.array(max_joint_vals)*(np.pi/180)

        self.rwd_keys_wt = dict(
            reach = 1.0,
            bonus = 4.0,
            penalty = 50,
            act_reg = 0.1,
            # sparse = 1,
            # solved = 1, # review - weight should not be assigned to this?
            # done = 1 # review - weight should not be assigned to this?
        )

        # Load the Model
        self._load_model()
        if not (self.model and self.data):
            raise "Error loading model"
        self._get_fingertip_indices()

        # Get Ref Position
        # The ref pos will remain fixed since it was taken before simulation started
        mj.mj_forward(self.model, self.data)
        finger_idx = "ii"
        ref_body_idx = mj.mj_name2id(self.model, int(mj.mjtObj.mjOBJ_BODY), f"{self.hand_name}_mcp_centre_block_{finger_idx}")
        self.ref_pos = np.array(self.data.xpos[ref_body_idx])

        # Load targets
        # with open(TARGETS_FILE, 'rb') as f:
        #     # np.array([np.random.random((15)) for _ in range(5)])
        #     self.targets = np.load(f)
        # self.targets_len = len(self.targets)

        # Define Observation Space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, 
                                                shape=(3*3*len(self.fingertip_indices),), 
                                                dtype=np.float32)
        
        # Define Action Space
        # NOTE: Watch out for Box upper limit if Carpal Actuators are involved
        # FIXME: Fix action range in reward and step functions. Current ==> [0,2] after denorm
        # Define a mujoco action range array used for scaling
        self.action_space = gym.spaces.Box(low=np.array([-1.0]*self.model.nu), 
                                            high=np.array([1.0]*self.model.nu), 
                                            shape=(self.model.nu,), dtype=np.float32)

        # For Human Rendering
        self.window = None
        self.window_size = 1200, 900

    def _load_model(self):
        xml_path = DARM_XML_FILE
        self.model = mj.MjModel.from_xml_path(xml_path)

        if self.model: 
            print("Loaded XML file successfully") 
        else:
            print(f"Error Loading XML file: {xml_path}")
            return
        
        self.data = mj.MjData(self.model)

    def _init_controller(self):
        pass

    def _controller_cb(self, model, data):
        pass

    def _get_fingertip_indices(self):
        indices = ["ii"]
        self.fingertip_indices = [mj.mj_name2id(self.model, int(mj.mjtObj.mjOBJ_SITE), f"{self.hand_name}_fingertip_{i}") for i in indices]
    
    def _get_obs(self, prev_obs, new_obs, action_time=None):
        if not action_time:
            # if no action time velocity is zero. i.e. after reset
            vel_obs = np.zeros((3*len(self.fingertip_indices),))
        elif action_time: 
            prev_fingertip_pos = prev_obs[:3*len(self.fingertip_indices)]
            new_fingertip_pos = new_obs[:3*len(self.fingertip_indices)]
            # BUG: FIXED: (prev - new) to (new - prev)
            vel_obs = (new_fingertip_pos - prev_fingertip_pos)/action_time

        return np.concatenate((np.array([(np.array(self.data.site_xpos[i]) - self.ref_pos) for i in self.fingertip_indices]).flatten(),
                             self.target_obs,
                             vel_obs))

    def _get_info(self):
        return {"sim_time": self.data.time - self.ep_start_time}

    def _norm_to_target(self, obs):
        """
        Returns the norm of each fingertip to the target position
        obs: an observation from the observation space [...fingertip_pos, ...target_pos, ...fingertip_vel]
        """
        obs = obs.reshape((-1, 3))
        n_fingertips = len(self.fingertip_indices)

        fingertip_poses = obs[0:n_fingertips]
        target_poses = obs[n_fingertips:2*n_fingertips]

        return np.linalg.norm(fingertip_poses-target_poses, ord=2, axis=-1)

    def _get_reward(self, state, action, new_state, time_delta):
        """
        Reward function to compute reward given state, action, and new state.
        R = R(S, a, S')

        If norm to target reduces: -1 else (-1 + x) where x is a neg. number 
                proportional to number of fingers with increased norms
        // Punish high velocity according to the eqution: -0.3 + 0.3*np.exp(-1*vel): DEPR
        Punish high torque according to the equation: -0.5 + 0.5*np.exp(-1*action)
        Reward reaching target with a tolerance of 4mm: 250
        """

        reach_dist = self._norm_to_target(new_state)    # NOTE: Single finger
        near_th = self.min_th
        far_th = 2*self.max_target_th
        # Use 0.5 to scale down reward to [0 1] from [0 2]
        # TODO: Let action range of model be limited to 1. Scale action input in mujoco model
        # NOTE: Some of the fingers in five fingered hand have more than five actuators
        act_mag = np.linalg.norm((1/5)*action.reshape(-1, 5)) # reshape action to (-1,5), ensure nu is ordered from mujoco
        
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach',   -1.*reach_dist),
            ('bonus',   1.*(reach_dist<2*near_th) + 1.*(reach_dist<near_th)),
            ('act_reg', -1.*act_mag),
            ('penalty', -1.*(reach_dist>far_th)),
            # Must keys
            ('sparse',  -1.*reach_dist),
            ('solved',  reach_dist<near_th),
            ('done',    reach_dist > far_th),
        ))

        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict


        # # Change in Norm to Target
        # prev_norm = self._norm_to_target(state)
        # new_norm = self._norm_to_target(new_state)
        # # norm_reward = -1 + sum(new_norm > prev_norm)*(-1/len(self.fingertip_indices))
        # norm_reward = 1 if new_norm < prev_norm else -1

        # # agent takes advantage of +1 by turning all arounf
        # # implement far_th and near_th
        # # essentially use the myosuite reward function

        # # norm_reward = -40*self._norm_to_target(state)   # max: -3.2

        # # Velocity Correction
        # # previous_pos = state[:3*len(self.fingertip_indices)].reshape((-1,3))
        # # new_pos = new_state[:3*len(self.fingertip_indices)].reshape((-1,3))
        # # vel = np.linalg.norm(new_pos-previous_pos, ord=2, axis=-1) / time_delta
        # # vel_reward = (-0.3 + 0.3*np.exp(-1*vel)).mean() # scale vel term in exp beween [-1,-5]
        
        # # Effort Correction
        # # action_reward = (-0.5 + 0.5*np.exp(-1*action)).mean()

        # ctrl_reward = -0.1*np.sum((0.5*action)**2)  # max: -0.4

        # # Reach Target Reward
        # reach_reward = 100 if self._get_done(new_state) else 0

        # reward_info = {"norm_reward": norm_reward, "ctrl_reward": ctrl_reward, "reach_reward": reach_reward}
        # return (norm_reward + ctrl_reward + reach_reward), reward_info

    def _get_done(self, new_state):
        return all(self._norm_to_target(new_state) < self.min_th)

    # def reset(self, seed=None, options=None):
        # super().reset()
    def reset(self, **kwargs):
        # Go to a random valid pose
        # Sample from Joint Space
        joint_state = np.random.uniform(low=self.min_joint_vals, high=self.max_joint_vals)

        # FIXME: Create utility to check for collision in multifingered case
        
        # Create a new goal
        self.target_obs = np.random.random(3)
        while True:
            joint_state_delta = self.target_joint_state_delta*np.random.choice(a=[-1,1], size=(4,), replace=True)
            target_joint_state = np.clip(a=joint_state + joint_state_delta, 
                                        a_min=self.min_joint_vals, 
                                        a_max=self.max_joint_vals)
            self.target_obs = self.forward(target_joint_state)[:3*len(self.fingertip_indices)]
            # self.target_obs = self.targets[np.random.randint(0, self.targets_len)]
            
            # MOCAPS for visualization
            if self.render_mode == "human":
                self.data.mocap_pos = self.target_obs.reshape(len(self.fingertip_indices),3) + self.ref_pos

            # Go forward to Joint State
            observation = self.forward(joint_state) #self._get_obs()

            # Verify distance is within limits
            norm = self._norm_to_target(observation)
            if norm >= self.min_target_th and norm <= self.max_target_th:
                break

        # Render Frame
        if self.render_mode == "human":
            self._render_frame()

        self.ep_start_time = self.data.time
        return observation #, self._get_info()

    def step(self, action):
        prev_obs = self._get_obs(prev_obs=None, new_obs=None, action_time=None)

        # FIXME: READ Action range from self.<>
        # action from model is in the range [-1,1]
        # action + 1 === [0, 2]
        # action * x === [0, 2x]
        action = (action + 1)*2.5  # [0, 5]
        action = np.clip(action, 0, 5)
        self.data.ctrl[0 : self.model.nu] = action
        time_prev = self.data.time   # simulation time in seconds

        # Perform action    
        while (self.data.time - time_prev < self.reaction_time):
            mj.mj_step(self.model, self.data)
        time_after = self.data.time # time after performing action

        # Get new observation
        new_obs = self._get_obs(prev_obs=None, new_obs=None, action_time=None)
        # include velocity in new obs
        new_obs = self._get_obs(prev_obs=prev_obs,
                                new_obs=new_obs, 
                                action_time=time_after-time_prev)

        if self.render_mode == "human":
            self._render_frame()

        # Get Reward
        rwd_dict = self._get_reward(prev_obs, action, new_obs, time_after-time_prev)
        reward = rwd_dict["dense"].mean()
        done = any(rwd_dict["done"])    # all(rwd_dict["solved"]) or any(rwd_dict["done"])

        # if self.render_mode == "human":
        #     self._render_frame()
        
        return new_obs, reward, done, {**self._get_info(), "action": action, "reward": {**rwd_dict}}   # False, self._get_info()
        # return new_obs, reward, self._get_done(new_obs), {**self._get_info(), "reward": {**reward_info}}   # False, self._get_info()

    def forward(self, joint_conf):
        self.data.qpos = joint_conf
        mj.mj_forward(self.model, self.data)
        return self._get_obs(prev_obs=None, new_obs=None, action_time=None)

    def render(self, mode, **kwargs):
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        if self.render_mode == "human" and not self.window:
            # Init GLFW, create window, make OpenGL context current, request v-sync
            glfw.init()
            self.window = glfw.create_window(self.window_size[0], self.window_size[1], "DARM", None, None)
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)

            # Visualization
            self.cam = mj.MjvCamera()    # abstract camera
            self.opt = mj.MjvOption()    # visualization options
            mj.mjv_defaultCamera(self.cam)
            mj.mjv_defaultOption(self.opt)
            self.scene = mj.MjvScene(self.model, maxgeom=10000)
            self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

            self.cam.azimuth = 110
            self.cam.elevation = -24
            self.cam.distance = 0.36
            self.cam.lookat = np.array([0.006, -0.004,  0.215])

            # For callback functions TODO:
            self.window_button_left = False
            self.window_button_middle = False
            self.window_button_right = False
            self.window_lastx = 0
            self.window_lasty = 0

            def mouse_button(window, button, act, mods):
                # update button state
                self.window_button_left = (glfw.get_mouse_button(
                    window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
                self.window_button_middle = (glfw.get_mouse_button(
                    window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
                self.window_button_right = (glfw.get_mouse_button(
                    window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

                # update mouse position
                glfw.get_cursor_pos(window) # TODO: Why is this needed again

            def mouse_move(window, xpos, ypos):
                # compute mouse displacement, save
                dx = xpos - self.window_lastx
                dy = ypos - self.window_lasty
                self.window_lastx = xpos
                self.window_lasty = ypos

                # no buttons down: nothing to do
                if (not self.window_button_left) and (not self.window_button_middle) and (not self.window_button_right):
                    return

                # get current window size
                width, height = glfw.get_window_size(window)

                # get shift key state
                PRESS_LEFT_SHIFT = glfw.get_key(
                    window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
                PRESS_RIGHT_SHIFT = glfw.get_key(
                    window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
                mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

                # determine action based on mouse button
                if self.window_button_right:
                    if mod_shift:
                        action = mj.mjtMouse.mjMOUSE_MOVE_H
                    else:
                        action = mj.mjtMouse.mjMOUSE_MOVE_V
                elif self.window_button_left:
                    if mod_shift:
                        action = mj.mjtMouse.mjMOUSE_ROTATE_H
                    else:
                        action = mj.mjtMouse.mjMOUSE_ROTATE_V
                else:
                    action = mj.mjtMouse.mjMOUSE_ZOOM

                mj.mjv_moveCamera(self.model, action, dx/width,
                                dy/height, self.scene, self.cam)    # TODO: Look into this, height/width issue

            def scroll(window, xoffset, yoffset):
                action = mj.mjtMouse.mjMOUSE_ZOOM
                mj.mjv_moveCamera(self.model, action, 0.0, -0.05 *
                                yoffset, self.scene, self.cam)

            glfw.set_cursor_pos_callback(self.window, mouse_move)
            glfw.set_mouse_button_callback(self.window, mouse_button)
            glfw.set_scroll_callback(self.window, scroll)
        
        # Get Framebuffer Viewport
        vp_width, vp_height = glfw.get_framebuffer_size(self.window)
        viewport = mj.MjrRect(0, 0, vp_width, vp_height)

        # Update scene and render
        mj.mjv_updateScene(self.model, self.data, self.opt, None, self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        mj.mjr_render(viewport, self.scene, self.context)

        # swap OpenGL buffers (blocking call due to v-sync)
        glfw.swap_buffers(self.window)

        # process pending GUI events, call GLFW callbacks
        glfw.poll_events()

    def close(self):
        glfw.terminate()