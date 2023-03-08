import os
import numpy as np
import collections

import gym
import mujoco as mj
from mujoco.glfw import glfw

from pathlib import Path


DARM_XML_FILE = f"{os.getenv('DARM_MUJOCO_PATH')}/mujoco_env/darm.xml"
SF_START_STATE_FILE = f"{os.getenv('DARM_MUJOCO_PATH')}/darm_gym_env/DARMHand_SF_start_state.npy"
MF_START_STATE_FILE = f"{os.getenv('DARM_MUJOCO_PATH')}/darm_gym_env/DARMHand_MFNW_start_state.npy"

class DARMEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 1}

    def __init__(self, render_mode=None, action_time=0.08, hand_name="hand1",
                min_th = 0.004,
                min_target_th = 2*0.004,
                max_target_th = 5*0.004, # 20 mm
                target_joint_state_delta = [],
                min_joint_vals = [],
                max_joint_vals = [],
                max_tendon_tension = [],
                single_finger_env = False,
                ignore_load_start_states = False
                ) -> None:
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        

        # ========================== Env Parameters ==========================
        self.render_mode = render_mode
        self.hand_name = hand_name
        self.single_finger_env = single_finger_env
        self.action_time = action_time
        self.ep_start_time = 0  # episode start time


        # ========================== Load the Model ==========================
        self._load_model()
        if not (self.model and self.data):
            raise "Error loading model"
        self._get_fingertip_indices()


        # ========================== Load targets ==========================
        if not ignore_load_start_states:
            self._load_start_states()


        # ========================== Mujoco Model Simulation Parameters ==========================
        self.min_joint_vals = min_joint_vals or self._get_joint_limits("min")   # degrees
        self.max_joint_vals = max_joint_vals or self._get_joint_limits("max")   # degress
        # abs increament of joint state from starting state to target state
        self.target_joint_state_delta = target_joint_state_delta or self._compute_target_joint_state_delta()   # degrees
        self.max_tendon_tension = max_tendon_tension or self._get_actuator_ctrlrange("max")

        self.min_joint_vals = self.min_joint_vals*(np.pi/180)
        self.max_joint_vals = self.max_joint_vals*(np.pi/180)
        self.target_joint_state_delta = self.target_joint_state_delta*(np.pi/180)

        self.min_th = min_th    # norm threshold in metres at which env is solved
        self.min_target_th = min_target_th  # min norm to target state
        self.max_target_th = max_target_th  # max norm to target state

        # Initialize target observation
        self.target_obs = np.zeros(3*len(self.fingertip_indices))


        # ========================== Reward Components Weights ==========================
        self.rwd_keys_wt = dict(
            reach = 1.0,
            bonus = 4.0,
            penalty = 1,  # done = all(...)
            act_reg = 0.1,
            # sparse = 1,
            # solved = 1, # review - weight should not be assigned to this?
            # done = 1 # review - weight should not be assigned to this?
        )


        # ========================== Get Ref Position ==========================
        # Reference Position is at the centre of the wrist
        # The ref pos will remain fixed since it was taken before simulation started
        mj.mj_forward(self.model, self.data)
        ref_body_idx = mj.mj_name2id(self.model, int(mj.mjtObj.mjOBJ_BODY), f"{self.hand_name}_rc_centre_block")
        self.ref_pos = np.array(self.data.xpos[ref_body_idx])


        # ========================== Define Observation and Action Space ==========================
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, 
                                                shape=(3*3*len(self.fingertip_indices),), 
                                                dtype=np.float32)
        

        # NOTE: Watch out for Box upper limit if Carpal Actuators are involved
        # FIXME: Fix action range in reward and step functions. Current ==> [0,2] after denorm
        # Define a mujoco action range array used for scaling
        self.action_space = gym.spaces.Box(low=np.array([-1.0]*self.model.nu), 
                                            high=np.array([1.0]*self.model.nu), 
                                            shape=(self.model.nu,), dtype=np.float32)


        # ========================== For Human Rendering ==========================
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

    def _load_start_states(self):
        if self.single_finger_env:
            filename = SF_START_STATE_FILE
        else:
            filename = MF_START_STATE_FILE

        with open(filename, 'rb') as f:
            self.start_states = np.load(f, allow_pickle=True)
            self.start_states_len = len(self.start_states)

    def _get_joint_limits(self, type = None):
        joint_limits = []
        for i in range(self.model.njnt):
            joint_limits.append(self.model.jnt_range[i]*(180/np.pi))
        
        joint_limits = np.asarray(joint_limits)
        
        if type == "min":
            return joint_limits[:, 0]
        if type == "max":
            return joint_limits[:, 1]
        
        return joint_limits[:, 0], joint_limits[:, 1]

    def _compute_target_joint_state_delta(self):
        # return (self.max_joint_vals - self.min_joint_vals)//10  # for every range of 10 deg, have a delta of 1 deg
        joint_state_delta =  ((self.max_joint_vals - self.min_joint_vals)//40) * 2  # for every range of 40 deg, have a delta of 2 deg
        return np.clip(joint_state_delta, a_min=2, a_max=10)    # a minimum delta of 2 degrees, max of 10 degrees

    def _get_actuator_ctrlrange(self, type = None):
        if type == "min":
            return np.array([self.model.actuator_ctrlrange[i][0] for i in range(self.model.nu)])
        if type == "max":
            return np.array([self.model.actuator_ctrlrange[i][1] for i in range(self.model.nu)])
    
        ctrl_range = np.array([self.model.actuator_ctrlrange[i] for i in range(self.model.nu)])
        return ctrl_range[:, 0], ctrl_range[:, 1] # (min, max)

    def _init_controller(self):
        pass

    def _controller_cb(self, model, data):
        pass

    def _get_fingertip_indices(self):
        # NOTE: Remember to set the mocap index properly in reset()
        if self.single_finger_env:
            indices = ["ii"]
        else:
            indices = ["i", "ii", "iii", "iv", "v"]

        self.fingertip_indices = [mj.mj_name2id(self.model, int(mj.mjtObj.mjOBJ_SITE), f"{self.hand_name}_fingertip_{i}") for i in indices]
    
    def _get_obs(self, prev_obs, new_obs, action_time=None):
        if not action_time:
            # if no action time velocity is zero. i.e. after reset
            vel_obs = np.zeros((3*len(self.fingertip_indices),))
        elif action_time: 
            prev_fingertip_pos = prev_obs[:3*len(self.fingertip_indices)]
            new_fingertip_pos = new_obs[:3*len(self.fingertip_indices)]
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

    def _get_reward(self, action, new_state):
        """
        Reward function to compute reward given action, and new state.
        R = R(a, S')

        Agent is punished for being far from target
        Agent is punished for going farther than a threshold from the target
        Agent is punished for high action magnitude
        Agent is rewarded for being close to the target
        Agent is rewarded for coming close to the target beyond a threshold 
        """

        reach_dist = self._norm_to_target(new_state)    # NOTE: Single finger
        near_th = self.min_th
        far_th = 2*self.max_target_th

        # Scale action down to [0, 1] from [0, max_tendon_tension]
        action = action / self.max_tendon_tension
        
        # NOTE: Some of the fingers in five fingered hand have more than five actuators
        # act_mag = np.linalg.norm(action.reshape(-1, 5)) # reshape action to (-1,5), ensure nu is ordered from mujoco
        # TODO: Consider scaling down this act_mag to be equiv. to a single finger with nu=5
        act_mag = np.linalg.norm(action)/np.sqrt(self.model.nu/1) # action magnitude is not measured per finger but as a whole
        act_mag = np.array([act_mag]*len(self.fingertip_indices))
        # by dividing by sqrt(nu/5), the norm is similar to when computing with nu==5. Check it out.
        # by dividing by sqrt(nu) act_mag will have a max value in the order of the max_value of action now => 1
        
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

            # Weights:
            # reach = 1.0,
            # bonus = 4.0,
            # penalty = 50,
            # act_reg = 0.1,
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def _get_done(self, new_state):
        return all(self._norm_to_target(new_state) < self.min_th)

    def _check_collision(self):
        """Returns True if there is collision, otherwise False"""
        return len(self.data.contact.geom1) > 0

    def generate_start_state(self):
        while True:
            # ========================== Sample valid start_state from Joint Space ==========================
            joint_state = np.random.uniform(low=self.min_joint_vals, high=self.max_joint_vals)
            # normal_sampling = np.random.normal(loc=0.5, scale=0.5/3, size=self.min_joint_vals.shape)
            # normal_sampling = np.clip(normal_sampling, 0, 1)
            # joint_state = self.min_joint_vals + normal_sampling*(self.max_joint_vals - self.min_joint_vals)
            self.forward(joint_state)

            if self._check_collision(): # returns True if there is collision
                # ensure there is no collision at the start state
                continue
            
        
            # ========================== Create a valid target ==========================
            joint_state_delta = self.target_joint_state_delta*np.random.choice(a=[-1,1], size=(self.model.njnt,), replace=True)
            target_joint_state = np.clip(a=joint_state + joint_state_delta, 
                                        a_min=self.min_joint_vals, 
                                        a_max=self.max_joint_vals)
            self.target_obs = self.forward(target_joint_state)[:3*len(self.fingertip_indices)]
            if self._check_collision(): # returns True if there is collision
                # ensure there is no collision at the target state
                continue
            
            # Return to start state
            observation = self.forward(joint_state)

            # Verify distance of start state to target state is within limits
            norm = self._norm_to_target(observation)
            if not (all(norm >= self.min_target_th) and all(norm <= self.max_target_th)):
                continue
            
            # If all checks are positive, break random search loop
            return observation, joint_state, self.target_obs      

    def sample_saved_start_states(self):
        # Sample a start state from saved start states
        # start_state = [joint_state, target_obs]
        start_state = self.start_states[np.random.randint(self.start_states_len)]

        # Set Target Obs
        self.target_obs = start_state[1]

        # Go forward to start state
        observation = self.forward(start_state[0])

        # Return Observation
        return observation

    # def reset(self, seed=None, options=None):
        # super().reset()
    def reset(self, **kwargs):
        # ========================== Get a random valid pose and target ==========================
        # observation, _, _ = self.generate_start_state()
        observation = self.sample_saved_start_states()

        # ========================== Render Frame ==========================
        if self.render_mode == "human":
            # Update target visualization mocaps pos
            self.data.mocap_pos = self.target_obs.reshape(len(self.fingertip_indices),3) + self.ref_pos
            # Go Forward
            mj.mj_forward(self.model, self.data)
            self._render_frame()

        self.ep_start_time = self.data.time
        return observation

    def step(self, action):
        prev_obs = self._get_obs(prev_obs=None, new_obs=None, action_time=None)

        # action from model is in the range [-1,1]
        # action + 1 === [0, 2]
        # action * x === [0, 2x]
        action = (action + 1)*(self.max_tendon_tension/2)
        action = np.clip(action, 0, self.max_tendon_tension)
        self.data.ctrl[0 : self.model.nu] = action
        time_prev = self.data.time   # simulation time in seconds

        # Perform action  
        while (self.data.time - time_prev < self.action_time):
            mj.mj_step(self.model, self.data)
        time_after = self.data.time # time after performing action


        # Get new observation (fingertips_pos)
        new_obs = self._get_obs(prev_obs=None, new_obs=None, action_time=None)
        # include velocity in new obs
        new_obs = self._get_obs(prev_obs=prev_obs,
                                new_obs=new_obs, 
                                action_time=time_after-time_prev)

        if self.render_mode == "human":
            self._render_frame()

        # Get Reward
        rwd_dict = self._get_reward(action, new_obs)
        reward = rwd_dict["dense"].mean()
        done = all(rwd_dict["done"])
        
        return new_obs, reward, done, {**self._get_info(), "action": action, "reward": {**rwd_dict}}

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