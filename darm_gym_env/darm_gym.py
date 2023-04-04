import os
import numpy as np
import collections

import darm_gym_env.darm_render
import gym
import mujoco as mj

from pathlib import Path


DARM_XML_FILE = f"{os.getenv('DARM_MUJOCO_PATH')}/mujoco_env/darm.xml"
START_STATE_FILE_DIR = Path(f"{os.getenv('DARM_MUJOCO_PATH')}/darm_gym_env/start_states")

class DARMEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None, action_time=0.08, hand_name="hand1",
                min_th = 0.4,
                angle_min_th = 10, # degrees
                min_target_th = 0.8,
                max_target_th = 2.0,
                distance_metres_scale = 100,
                target_joint_state_delta = [],
                min_joint_vals = [],
                max_joint_vals = [],
                max_tendon_tension = [],
                start_state_file = "DARMHand_MFNW_start_state.npy",
                ignore_load_start_states = False,
                digits = ["i", "ii", "iii", "iv", "v"]
                ) -> None:
        super().__init__()
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        
        # ========================== Others ==========================
        self.distance_scale = distance_metres_scale # scaling factor for distance measurements from meteres

        # ========================== Env Parameters ==========================
        self.render_mode = render_mode
        self.hand_name = hand_name
        self.start_state_file = start_state_file
        self.action_time = action_time
        self.ep_start_time = 0  # episode start time


        # ========================== Load the Model ==========================
        self._load_model()
        if not (self.model and self.data):
            raise "Error loading model"
        

        # ========================== Setup Rendering ==========================
        if self.render_mode == "human":
            self.darm_render = darm_gym_env.darm_render.DARMRender(self.model, self.data, (1200,900))
            self.darm_render.init_window_render()


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

        self.min_th = min_th   # norm threshold in metres at which env is solved
        self.angle_min_th = np.radians(angle_min_th)
        self.min_target_th = min_target_th # min norm to target state
        self.max_target_th = max_target_th # max norm to target state

        # Initialize target observation
        self.target_obs = np.zeros((5,7))
        self.prev_fingertip_pose = np.zeros((5,7))


        # ========================== Reward Components Weights ==========================
        self.rwd_keys_wt = dict(
            reach = 1.0,
            contact = 1.0,
            bonus = 4.0,
            penalty = 50,
            act_reg = 0.01,
            # sparse = 1,
            # solved = 1, # review - weight should not be assigned to this?
            # done = 1 # review - weight should not be assigned to this?
        )


        # ========================== Get Ref Position ==========================
        # Reference Position is at the centre of the wrist
        # The ref pos will remain fixed since it was taken before simulation started
        mj.mj_forward(self.model, self.data)
        self.ref_pos = self.data.body(f"{self.hand_name}_rc_centre_block").xpos

        # ========================== Others ==========================
        self.index_str_mapping = {"i":0, "ii":1, "iii":2, "iv":3, "v":4}
        self.index_int_mapping = {0:"i", 1:"ii", 2:"iii", 3:"iv", 4:"v"}
        self.all_digits = ["i", "ii", "iii", "iv", "v"]
        self.digits = digits
        self.digits_indices = np.array([self.index_str_mapping[idx_str] for idx_str in self.digits])


        # ========================== Define Observation and Action Space ==========================
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, 
                                                shape=(len(self._get_obs()),), 
                                                dtype=np.float32)
        

        # NOTE: Watch out for Box upper limit if Carpal Actuators are involved
        # FIXME: Fix action range in reward and step functions. Current ==> [0,2] after denorm
        # Define a mujoco action range array used for scaling
        self.action_space = gym.spaces.Box(low=np.array([-1.0]*self.model.nu), 
                                            high=np.array([1.0]*self.model.nu), 
                                            shape=(self.model.nu,), dtype=np.float32)

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
        filename = START_STATE_FILE_DIR/self.start_state_file

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
    
    def transform_distance_obs(self, obs):
        """
        - Transforms the distance to be in the frame of the wrist (RC Joint)
        - Convert the distance reading to cm
        """
        return (obs - self.ref_pos)*self.distance_scale
    
    def remove_distance_obs_transform(self, obs):
        """Removes the transformation applied to distance observations"""
        return (obs/self.distance_scale) + self.ref_pos

    def get_finger_frames_pos(self, idx_str):
        # get Proximal Phalanx frame
        def pp_frame():
            # BODY: ${hand_name}_proximal_phalanx_${index}
            frame_pos = self.data.body(f"{self.hand_name}_proximal_phalanx_{idx_str}").xpos
            return self.transform_distance_obs(frame_pos)
            
        # get Middle Phalanx frame
        def mp_frame():
            # BODY: ${hand_name}_middle_phalanx_${index} [ii to v]
            frame_pos = self.data.body(f"{self.hand_name}_middle_phalanx_{idx_str}").xpos
            return self.transform_distance_obs(frame_pos)

        # get Distal Phalanx frame
        def dp_frame():
            # BODY: ${hand_name}_distal_phalanx_${index}
            frame_pos = self.data.body(f"{self.hand_name}_distal_phalanx_{idx_str}").xpos
            return self.transform_distance_obs(frame_pos)

        # get fingertip frame
        def fingertip_frame():
            # SITE: ${hand_name}_fingertip_${index}
            frame_pos = self.data.site(f"{self.hand_name}_fingertip_{idx_str}").xpos
            return self.transform_distance_obs(frame_pos)

        if idx_str == "i":
            return np.concatenate((pp_frame(), dp_frame(),
                                   fingertip_frame()))
        else:
            return np.concatenate((pp_frame(), mp_frame(),
                                   dp_frame(), fingertip_frame()))

    def get_fingertip_pose(self, idx_str):
        """Returns the position and orientation of the fingertip `idx_str`"""

        fingertip_pos = self.data.site(f"{self.hand_name}_fingertip_{idx_str}").xpos
        fingertip_pos = self.transform_distance_obs(fingertip_pos)

        fingertip_orient = self.data.body(f"{self.hand_name}_distal_phalanx_{idx_str}").xquat

        return np.concatenate((fingertip_pos, fingertip_orient))        

    def get_finger_contacts(self, index):
        contact_geoms1 = self.data.contact.geom1
        contact_geoms2 = self.data.contact.geom2

        colliding_digits = []
        colliding_with_palm = False

        def get_digit_str(name):
            if name.endswith("_i"): return "i"
            if name.endswith("_ii"): return "ii"
            if name.endswith("_iii"): return "iii"
            if name.endswith("_iv"): return "iv"
            if name.endswith("_v"): return "v"
            
            return None

        def filter_contacts(contacts1, contacts2):
            for idx in range(len(contacts1)):
                geom_idx = contacts1[idx]
                # Ignore Wraps
                if "_wrap" in self.model.geom(geom_idx).name:
                    continue
                
                # Get the Body name
                bodyid = self.model.geom(geom_idx).bodyid[0]
                bodyname = self.model.body(bodyid).name

                # If body is in current phalanges group
                if bodyname.endswith(f"_phalanx_{self.index_int_mapping[index]}"):
                    coll_geom_idx = contacts2[idx]
                    # Ignore Wraps
                    if "_wrap" in self.model.geom(coll_geom_idx).name:
                        continue

                    # Get the body collided with
                    coll_bodyid = self.model.geom(coll_geom_idx).bodyid[0]
                    coll_bodyname = self.model.body(coll_bodyid).name

                    if coll_bodyname == "hand1_carpals_metacarpals":
                        nonlocal colliding_with_palm
                        colliding_with_palm = True

                    digit = get_digit_str(coll_bodyname)
                    if digit and (digit != self.index_int_mapping[index]):
                        colliding_digits.append(digit)

        filter_contacts(contact_geoms1, contact_geoms2)
        filter_contacts(contact_geoms2, contact_geoms1)

        collision_obs = np.zeros(6)
        if colliding_with_palm: collision_obs[0] = 1
        for idx_str in colliding_digits:
            collision_obs[self.index_str_mapping[idx_str]+1] = 1

        return collision_obs

    def digits_in_contact(self):
        """Returns True if there is any collision between the digits of the hand"""
        contacts = np.concatenate([self.get_finger_contacts(index) for index in self.digits_indices])
        return sum(contacts) > 0

    def _get_obs(self, action_time=None):
        def get_target_pose(index):
            return self.target_obs[index]

        def get_kinematic_chain_obs(index):
            return self.get_finger_frames_pos(self.index_int_mapping[index])

        def get_vel_obs(index):
            if not action_time:
                # if no action time, velocity is zero. i.e. after reset
                return np.zeros(3)
            
            prev_fingertip_pos = self.prev_fingertip_pose[index][:3]
            new_fingertip_pos = self.get_fingertip_pose(self.index_int_mapping[index])[:3]
            vel_obs = (new_fingertip_pos - prev_fingertip_pos)/action_time
            return vel_obs

        def get_contact_obs(index):
            return self.get_finger_contacts(index)

        def get_finger_obs(index):
            return np.concatenate((get_target_pose(index),
                            get_kinematic_chain_obs(index),
                            get_vel_obs(index),
                            get_contact_obs(index)))
            
        obs = np.concatenate([get_finger_obs(index) for index in self.digits_indices])
        return obs

    def _get_info(self):
        return {"sim_time": self.data.time - self.ep_start_time}

    def position_norm(self, pos1, pos2):
        """Returns the distance between two position vectors"""
        return np.linalg.norm(pos1-pos2, ord=2, axis=-1)

    def orientation_norm(self, quat1, quat2):
        """Returns the angular distance between two quaternion orientation"""
        return 2*np.arccos(np.abs(np.dot(quat1, np.transpose(quat2)).diagonal()))

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

        near_th = self.min_th
        angle_near_th = self.angle_min_th
        far_th = 2*self.max_target_th

        fingertip_obs = np.zeros_like(self.target_obs)
        for idx_str in self.digits:
            fingertip_obs[self.index_str_mapping[idx_str]] = self.get_fingertip_pose(idx_str)
        reach_dist_all = self.position_norm(fingertip_obs[:, :3], self.target_obs[:, :3])
        angle_dist_all = self.orientation_norm(fingertip_obs[:, 3:7], self.target_obs[:, 3:7])
        # Compute reward only for active digits
        reach_dist = reach_dist_all[self.digits_indices]
        angle_dist = angle_dist_all[self.digits_indices]
        contact = np.array([sum(self.get_finger_contacts(i)) for i in self.digits_indices])

        # Scale action down to [0, 1] from [0, max_tendon_tension]
        action = action / self.max_tendon_tension
        
        # NOTE: Some of the fingers in five fingered hand have more than five actuators
        # act_mag = np.linalg.norm(action.reshape(-1, 5)) # reshape action to (-1,5), ensure nu is ordered from mujoco
        # TODO: Consider scaling down this act_mag to be equiv. to a single finger with nu=5
        act_mag = np.linalg.norm(action)/np.sqrt(self.model.nu/1) # action magnitude is not measured per finger but as a whole
        act_mag = np.array([act_mag]*len(self.digits))
        # by dividing by sqrt(nu/5), the norm is similar to when computing with nu==5. Check it out.
        # by dividing by sqrt(nu) act_mag will have a max value in the order of the max_value of action now => 1
        
        # reach dist scaled from cm to m
        reach_rwd = -1*0.01*reach_dist - 0.05*angle_dist
        bonus_rwd = (1.*(np.logical_and((reach_dist<2*near_th), (angle_dist<2*angle_near_th))) + 
                 1.*(np.logical_and((reach_dist<near_th), (angle_dist<angle_near_th))))
        rwd_dict = collections.OrderedDict((
            # Optional Keys
            ('reach',   reach_rwd),
            ('bonus',   bonus_rwd),
            ('contact', -1*contact),
            ('act_reg', -1.*act_mag),
            ('penalty', -1.*(reach_dist>far_th)),
            # Must keys
            ('sparse',  -1.*reach_dist),
            ('solved',  reach_dist<near_th),
            ('done',    reach_dist > far_th),
        ))

            # Weights:
            # reach = 1.0,
            # contact = 1.0
            # bonus = 4.0,
            # penalty = 50,
            # act_reg = 0.1,
        rwd_dict['dense'] = np.sum([wt*rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)
        return rwd_dict

    def generate_start_state(self):
        while True:
            # ========================== Sample valid start_state from Joint Space ==========================
            joint_state = np.random.uniform(low=self.min_joint_vals, high=self.max_joint_vals)
            # normal_sampling = np.random.normal(loc=0.5, scale=0.5/3, size=self.min_joint_vals.shape)
            # normal_sampling = np.clip(normal_sampling, 0, 1)
            # joint_state = self.min_joint_vals + normal_sampling*(self.max_joint_vals - self.min_joint_vals)
            self.forward(joint_state)

            if self.digits_in_contact(): # returns True if there is collision between fingers
                # ensure there is no collision at the start state
                continue
            
        
            # ========================== Create a valid target ==========================
            joint_state_delta = self.target_joint_state_delta*np.random.choice(a=[-1,1], size=(self.model.njnt,), replace=True)
            target_joint_state = np.clip(a=joint_state + joint_state_delta, 
                                        a_min=self.min_joint_vals, 
                                        a_max=self.max_joint_vals)
            self.forward(target_joint_state)
            if self.digits_in_contact(): # returns True if there is collision
                # ensure there is no collision at the target state
                continue
            
            for idx_str in self.digits:
                self.target_obs[self.index_str_mapping[idx_str]] = self.get_fingertip_pose(idx_str)
            
            # Return to start state
            observation = self.forward(joint_state)

            # Verify distance of start state to target state is within limits
            fingertip_obs = np.zeros_like(self.target_obs)
            for idx_str in self.digits:
                fingertip_obs[self.index_str_mapping[idx_str]] = self.get_fingertip_pose(idx_str)
            norm_all = self.position_norm(fingertip_obs[:, :3], self.target_obs[:, :3])
            norm = norm_all[self.digits_indices]
            if not (all(norm >= self.min_target_th) and all(norm <= self.max_target_th)):
                continue
            
            # If all checks are positive, break random search loop
            return observation, joint_state, self.target_obs.copy()      

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

    def reset(self, **kwargs):
        # ========================== Get a random valid pose and target ==========================
        # observation, _, _ = self.generate_start_state()
        observation = self.sample_saved_start_states()
        
        for idx_str in self.digits:
            self.prev_fingertip_pose[self.index_str_mapping[idx_str]] = self.get_fingertip_pose(idx_str)

        # ========================== Render Frame ==========================
        if self.render_mode == "human":
            # Update target visualization mocaps pos and quat
            target_pos = self.target_obs[self.digits_indices, :3]
            self.data.mocap_pos = self.remove_distance_obs_transform(target_pos)
            self.data.mocap_quat = self.target_obs[self.digits_indices, 3:]
            # Go Forward
            mj.mj_forward(self.model, self.data)
            self._render_frame()

        self.ep_start_time = self.data.time
        return observation

    def step(self, action):
        for idx_str in self.digits:
            self.prev_fingertip_pose[self.index_str_mapping[idx_str]] = self.get_fingertip_pose(idx_str)

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


        # Get observation
        obs = self._get_obs(action_time=time_after-time_prev)

        if self.render_mode == "human":
            self._render_frame()

        # Get Reward
        rwd_dict = self._get_reward(action, obs)
        reward = rwd_dict["dense"].mean()
        done = any(rwd_dict["done"])  # all(rwd_dict["done"])
        
        return obs, reward, done, {**self._get_info(), "action": action, "reward": {**rwd_dict}}

    def forward(self, joint_conf):
        self.data.qpos = joint_conf
        mj.mj_forward(self.model, self.data)
        return self._get_obs(action_time=None)

    def render(self, mode="human", **kwargs):
        if mode=="human" and self.render_mode == "human":
            return self._render_frame()
        # else:
        #     T = 1/DARMEnv.metadata["render_fps"]    # period
        #     if (self.data.time - self.last_frame_time) >= T:
        #         self.renderer.update_scene(self.data, scene_option=self.scene_option)
        #         self.last_frame = self.renderer.render()
        #     return self.last_frame.copy()

    def _render_frame(self):
        self.darm_render.window_render()

    def close(self):
        if self.render_mode == "human":
            self.darm_render.close_window()


if __name__ == "__main__":
    env = DARMEnv(render_mode="human")
    env.reset()
    while True:
        env.render(mode="human")
        