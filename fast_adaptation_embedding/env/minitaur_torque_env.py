"""This file implements the gym environment of minitaur.

"""
import math
import time

import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet
from fast_adaptation_embedding.env.assets.pybullet_envs import bullet_client
from fast_adaptation_embedding.env.assets import pybullet_data
from fast_adaptation_embedding.env.assets.pybullet_envs import minitaur
from fast_adaptation_embedding.env.assets.pybullet_envs import minitaur_logging
from fast_adaptation_embedding.env.assets.pybullet_envs import minitaur_logging_pb2
from fast_adaptation_embedding.env.assets.pybullet_envs import motor
from pkg_resources import parse_version

NUM_MOTORS = 8
MOTOR_ANGLE_OBSERVATION_INDEX = 0
MOTOR_VELOCITY_OBSERVATION_INDEX = MOTOR_ANGLE_OBSERVATION_INDEX + NUM_MOTORS
MOTOR_TORQUE_OBSERVATION_INDEX = MOTOR_VELOCITY_OBSERVATION_INDEX + NUM_MOTORS
BASE_ORIENTATION_OBSERVATION_INDEX = MOTOR_TORQUE_OBSERVATION_INDEX + NUM_MOTORS
ACTION_EPS = 0.01
OBSERVATION_EPS = 0.01
RENDER_HEIGHT = 1080
RENDER_WIDTH =  1920
SENSOR_NOISE_STDDEV = minitaur.SENSOR_NOISE_STDDEV
DEFAULT_URDF_VERSION = "default"
DERPY_V0_URDF_VERSION = "derpy_v0"
RAINBOW_DASH_V0_URDF_VERSION = "rainbow_dash_v0"
NUM_SIMULATION_ITERATION_STEPS = 300

MINIATUR_URDF_VERSION_MAP = {
    DEFAULT_URDF_VERSION: minitaur.Minitaur,
    # DERPY_V0_URDF_VERSION: minitaur_derpy.MinitaurDerpy,
    # RAINBOW_DASH_V0_URDF_VERSION: minitaur_rainbow_dash.MinitaurRainbowDash,
}


def convert_to_list(obj):
    try:
        iter(obj)
        return obj
    except TypeError:
        return [obj]


class MinitaurTorqueEnv(gym.Env):
    """The gym environment for the minitaur.

  It simulates the locomotion of a minitaur, a quadruped robot. The state space
  include the angles, velocities and torques for all the motors and the action
  space is the desired motor angle for each motor. The reward function is based
  on how far the minitaur walks in 1000 steps and penalizes the energy
  expenditure.

  """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self,
                 urdf_root=pybullet_data.getDataPath(),
                 urdf_version=None,
                 distance_weight=1.0,
                 desired_speed=1.0,
                 high_weight=0.0,
                 roll_weight=0.0,
                 pitch_weight=0.0,
                 yaw_weight=0.0,
                 action_weight=0.0,
                 action_vel_weight=0.0,
                 action_acc_weight=0.0,
                 action_jerk_weight=0.0,
                 action_space="Motor",
                 distance_limit=float("inf"),
                 observation_noise_stdev=SENSOR_NOISE_STDDEV,
                 self_collision_enabled=True,
                 motor_velocity_limit=np.inf,
                 pd_control_enabled=False,
                 leg_model_enabled=True,
                 accurate_motor_model_enabled=False,
                 remove_default_joint_damping=False,
                 motor_kp=1.0,
                 motor_kd=0.02,
                 control_latency=0.0,
                 pd_latency=0.0,
                 torque_control_enabled=False,
                 motor_overheat_protection=False,
                 hard_reset=True,
                 on_rack=False,
                 render=False,
                 num_steps_to_log=1000,
                 action_repeat=1,
                 control_time_step=None,
                 env_randomizer=None,
                 forward_reward_cap=float("inf"),
                 reflection=True,
                 log_path=None,
                 partial_torque_control=False,
                 vkp=0.
                 ):
        """Initialize the minitaur gym environment.

    Args:
      urdf_root: The path to the urdf data folder.
      urdf_version: [DEFAULT_URDF_VERSION, DERPY_V0_URDF_VERSION,
        RAINBOW_DASH_V0_URDF_VERSION] are allowable
        versions. If None, DEFAULT_URDF_VERSION is used. DERPY_V0_URDF_VERSION
        is the result of first pass system identification for derpy.
        We will have a different URDF and related Minitaur class each time we
        perform system identification. While the majority of the code of the
        class remains the same, some code changes (e.g. the constraint location
        might change). __init__() will choose the right Minitaur class from
        different minitaur modules based on
        urdf_version.
      distance_weight: The weight of the distance term in the reward.
      energy_weight: The weight of the energy term in the reward.
      shake_weight: The weight of the vertical shakiness term in the reward.
      drift_weight: The weight of the sideways drift term in the reward.
      distance_limit: The maximum distance to terminate the episode.
      observation_noise_stdev: The standard deviation of observation noise.
      self_collision_enabled: Whether to enable self collision in the sim.
      motor_velocity_limit: The velocity limit of each motor.
      pd_control_enabled: Whether to use PD controller for each motor.
      leg_model_enabled: Whether to use a leg motor to reparameterize the action
        space.
      accurate_motor_model_enabled: Whether to use the accurate DC motor model.
      remove_default_joint_damping: Whether to remove the default joint damping.
      motor_kp: proportional gain for the accurate motor model.
      motor_kd: derivative gain for the accurate motor model.
      control_latency: It is the delay in the controller between when an
        observation is made at some point, and when that reading is reported
        back to the Neural Network.
      pd_latency: latency of the PD controller loop. PD calculates PWM based on
        the motor angle and velocity. The latency measures the time between when
        the motor angle and velocity are observed on the microcontroller and
        when the true state happens on the motor. It is typically (0.001-
        0.002s).
      torque_control_enabled: Whether to use the torque control, if set to
        False, pose control will be used.
      motor_overheat_protection: Whether to shutdown the motor that has exerted
        large torque (OVERHEAT_SHUTDOWN_TORQUE) for an extended amount of time
        (OVERHEAT_SHUTDOWN_TIME). See ApplyAction() in minitaur.py for more
        details.
      hard_reset: Whether to wipe the simulation and load everything when reset
        is called. If set to false, reset just place the minitaur back to start
        position and set its pose to initial configuration.
      on_rack: Whether to place the minitaur on rack. This is only used to debug
        the walking gait. In this mode, the minitaur's base is hanged midair so
        that its walking gait is clearer to visualize.
      render: Whether to render the simulation.
      num_steps_to_log: The max number of control steps in one episode that will
        be logged. If the number of steps is more than num_steps_to_log, the
        environment will still be running, but only first num_steps_to_log will
        be recorded in logging.
      action_repeat: The number of simulation steps before actions are applied.
      control_time_step: The time step between two successive control signals.
      env_randomizer: An instance (or a list) of EnvRandomizer(s). An
        EnvRandomizer may randomize the physical property of minitaur, change
          the terrrain during reset(), or add perturbation forces during step().
      forward_reward_cap: The maximum value that forward reward is capped at.
        Disabled (Inf) by default.
      log_path: The path to write out logs. For the details of logging, refer to
        minitaur_logging.proto.
    Raises:
      ValueError: If the urdf_version is not supported.
    """
        # Set up logging.
        self._log_path = log_path
        self.logging = minitaur_logging.MinitaurLogging(log_path)
        # PD control needs smaller time step for stability.
        if control_time_step is not None:
            self.control_time_step = control_time_step
            self._action_repeat = action_repeat
            self._time_step = control_time_step / action_repeat
        else:
            # Default values for time step and action repeat
            if accurate_motor_model_enabled or pd_control_enabled:
                self._time_step = 0.002
                self._action_repeat = 5
            else:
                self._time_step = 0.01
                self._action_repeat = 1
            self.control_time_step = self._time_step * self._action_repeat
        self._num_bullet_solver_iterations = int(NUM_SIMULATION_ITERATION_STEPS / self._action_repeat)
        self._urdf_root = urdf_root
        self._self_collision_enabled = self_collision_enabled
        self._motor_velocity_limit = motor_velocity_limit
        self._observation = []
        self._true_observation = []
        self._objectives = []
        self._objective_weights = [distance_weight, high_weight, roll_weight, pitch_weight, yaw_weight,
                                   action_weight, action_vel_weight, action_acc_weight, action_jerk_weight]
        self._env_step_counter = 0
        self._num_steps_to_log = num_steps_to_log
        self._is_render = render
        self._last_base_position = [0, 0, 0]
        self._distance_weight = distance_weight
        self._desired_speed = desired_speed
        self._high_weight = high_weight
        self._roll_weight = roll_weight
        self._pitch_weight = pitch_weight
        self._yaw_weight = yaw_weight
        self._past_actions = np.zeros((4, 8))
        assert action_space in ['Motor', 'Motor2', 'velocity', 'velocity2', 'S&E', 'Torque', "Torque2"], "Action space not implemented"
        self._action_space = action_space
        self._vkp = vkp
        self._vkd = 0.
        self._distance_limit = distance_limit
        self._observation_noise_stdev = observation_noise_stdev
        self._pd_control_enabled = pd_control_enabled
        self._leg_model_enabled = leg_model_enabled
        self._accurate_motor_model_enabled = accurate_motor_model_enabled
        self._remove_default_joint_damping = remove_default_joint_damping
        self._motor_kp = motor_kp
        self._motor_kd = motor_kd
        self._torque_control_enabled = torque_control_enabled
        self._partial_torque_control = partial_torque_control
        self._motor_overheat_protection = motor_overheat_protection
        self._on_rack = on_rack
        self._cam_dist = 1.0
        self._cam_yaw = 0
        self._cam_pitch = -30
        self._forward_reward_cap = forward_reward_cap
        self._hard_reset = True
        self._last_frame_time = 0.0
        self._control_latency = control_latency
        self._pd_latency = pd_latency
        self._urdf_version = urdf_version
        self._ground_id = None
        self._reflection = reflection
        self._env_randomizers = convert_to_list(env_randomizer) if env_randomizer else []
        self._episode_proto = minitaur_logging_pb2.MinitaurEpisode()
        self._slope_degree = 0
        self._friction = 1
        self._g = 9.8
        self._unblocked_steering = True
        self._fc = 10
        self._C = 1/(np.tan(np.pi*self._fc*self.control_time_step))
        self._A = 1/(1+self._C)
        self._past_velocity = None
        self._init_orientation = [0, 0, 0, 1]
        if self._is_render:
            self._pybullet_client = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self._pybullet_client = bullet_client.BulletClient()
        if self._urdf_version is None:
            self._urdf_version = DEFAULT_URDF_VERSION
        self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)
        self.seed()
        self.reset()
        observation_high = (self._get_observation_upper_bound() + OBSERVATION_EPS)
        observation_low = (self._get_observation_lower_bound() - OBSERVATION_EPS)
        action_dim = 8  # [steering, step_size, leg_extension, leg_extension_offset]
        self._action_bound = np.pi
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        self.observation_space = spaces.Box(observation_low, observation_high)
        self.viewer = None
        self._hard_reset = hard_reset  # This assignment need to be after reset()

    def steering(self, unblock):
        self._unblocked_steering = unblock

    def close(self):
        if self._env_step_counter > 0:
            self.logging.save_episode(self._episode_proto)
        self.minitaur.Terminate()

    def add_env_randomizer(self, env_randomizer):
        self._env_randomizers.append(env_randomizer)

    def set_init_orientation(self, orientation):
        self._init_orientation = orientation

    def reset(self, initial_motor_angles=None, reset_duration=1.0):
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 0)
        if self._env_step_counter > 0:
            self.logging.save_episode(self._episode_proto)
        self._episode_proto = minitaur_logging_pb2.MinitaurEpisode()
        minitaur_logging.preallocate_episode_proto(self._episode_proto, self._num_steps_to_log)
        if self._hard_reset:
            self._pybullet_client.resetSimulation()
            self._pybullet_client.setPhysicsEngineParameter(
                numSolverIterations=int(self._num_bullet_solver_iterations))
            self._pybullet_client.setTimeStep(self._time_step)
            self._ground_id = self._pybullet_client.loadURDF("%s/plane.urdf" % self._urdf_root)
            self._pybullet_client.changeDynamics(self._ground_id, linkIndex=-1, lateralFriction=self._friction)
            if (self._reflection):
                self._pybullet_client.changeVisualShape(self._ground_id, -1, rgbaColor=[1, 1, 1, 0.8])
                self._pybullet_client.configureDebugVisualizer(
                    self._pybullet_client.COV_ENABLE_PLANAR_REFLECTION, self._ground_id)
            alpha = np.pi / 180 * self._slope_degree
            self._pybullet_client.setGravity(-self._g * np.sin(alpha), 0, -self._g * np.cos(alpha))
            acc_motor = self._accurate_motor_model_enabled
            motor_protect = self._motor_overheat_protection
            if self._urdf_version not in MINIATUR_URDF_VERSION_MAP:
                raise ValueError("%s is not a supported urdf_version." % self._urdf_version)
            else:
                self.minitaur = (MINIATUR_URDF_VERSION_MAP[self._urdf_version](
                    pybullet_client=self._pybullet_client,
                    action_repeat=self._action_repeat,
                    urdf_root=self._urdf_root,
                    time_step=self._time_step,
                    self_collision_enabled=self._self_collision_enabled,
                    motor_velocity_limit=self._motor_velocity_limit,
                    pd_control_enabled=self._pd_control_enabled,
                    accurate_motor_model_enabled=acc_motor,
                    remove_default_joint_damping=self._remove_default_joint_damping,
                    motor_kp=self._motor_kp,
                    motor_kd=self._motor_kd,
                    control_latency=self._control_latency,
                    pd_latency=self._pd_latency,
                    observation_noise_stdev=self._observation_noise_stdev,
                    torque_control_enabled=self._torque_control_enabled,
                    motor_overheat_protection=motor_protect,
                    on_rack=self._on_rack,
                    partial_torque_control=self._partial_torque_control,
                    vkd=self._vkd,
                    vkp=self._vkp))
        self.minitaur.SetInitOrientation(self._init_orientation)
        self.minitaur.Reset(reload_urdf=False,
                            default_motor_angles=initial_motor_angles,
                            reset_time=reset_duration)
        # self.minitaur.SetFootFriction(self._friction)
        # Loop over all env randomizers.
        for env_randomizer in self._env_randomizers:
            env_randomizer.randomize_env(self)

        self._pybullet_client.setPhysicsEngineParameter(enableConeFriction=0)
        self._env_step_counter = 0
        self._last_base_position = [0, 0, 0]
        self._past_velocity = [self.minitaur.GetMotorVelocities()]*2
        self._objectives = []
        self._pybullet_client.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw,
                                                         self._cam_pitch, [0, 0, 0])
        self._pybullet_client.configureDebugVisualizer(self._pybullet_client.COV_ENABLE_RENDERING, 1)
        return self._get_observation()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _transform_action_to_motor_command(self, action):
        if self._leg_model_enabled:
            for i, action_component in enumerate(action):
                if not (-self._action_bound - ACTION_EPS <= action_component <=
                        self._action_bound + ACTION_EPS):
                    raise ValueError("{}th action {} out of bounds.".format(i, action_component))
            action = self.minitaur.ConvertFromLegModel(action)
        return action

    def step(self, action):
        """Step forward the simulation, given the action.

    Args:
      action: A list of desired motor angles for eight motors.

    Returns:
      observations: The angles, velocities and torques of all motors.
      reward: The reward for the current state-action pair.
      done: Whether the episode has ended.
      info: A dictionary that stores diagnostic information.

    Raises:
      ValueError: The action dimension is not the same as the number of motors.
      ValueError: The magnitude of actions is out of bounds.
    """
        self._last_base_position = self.minitaur.GetBasePosition()

        if self._is_render:
            # Sleep, otherwise the computation takes less time than real time,
            # which will make the visualization like a fast-forward video.
            time_spent = time.time() - self._last_frame_time
            self._last_frame_time = time.time()
            time_to_sleep = self.control_time_step - time_spent
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            base_pos = self.minitaur.GetBasePosition()
            # Keep the previous orientation of the camera set by the user.
            [yaw, pitch, dist] = self._pybullet_client.getDebugVisualizerCamera()[8:11]
            self._pybullet_client.resetDebugVisualizerCamera(dist, yaw, pitch, base_pos)

        for env_randomizer in self._env_randomizers:
            env_randomizer.randomize_step(self)

        a = np.copy(action)

        if self._action_space == "velocity":
            """For action in delta motor space"""
            a = self.minitaur.GetMotorAngles() + a * np.pi/4
            if self._partial_torque_control:
                a = 8 * (a - self.minitaur.GetMotorAngles())
        elif self._action_space == "velocity2":
            a = self.minitaur.GetMotorAngles() + a
            if self._partial_torque_control:
                a = 8 * (a - self.minitaur.GetMotorAngles())
        elif self._action_space == "Motor":
            a = np.ones(8) * 1.5 + 0.8 * a * np.pi / 4
            if self._partial_torque_control:
                a = 8 * (a - self.minitaur.GetMotorAngles())
        elif self._action_space == "Motor2":
            """For action in motor space"""
            a = a + 1.5
            if self._partial_torque_control:
                a = 8 * (a - self.minitaur.GetMotorAngles())
        elif self._action_space == "S&E":
            """For a in swing & extension"""
            # ub = 0.8*[0.75, 0.75, 0.75, 0.75, 0.85, 0.85, 0.85, 0.85]) and lb = -[0.75, 0.75, 0.75, 0.75, 0.1, 0.1, 0.1, 0.1])*0.8
            a[:4] = a[:4] * 0.6
            a[4:] = (a[4:] * 0.4 + 0.5) * 0.75 + 0.10
            a = self._transform_action_to_motor_command(a)
            if self._partial_torque_control:
                a = 8 * (a - self.minitaur.GetMotorAngles())
        elif self._action_space == "Torque":
            a = 8 * a
        elif self._action_space == "Torque2":
            a = a
        observed_torques, desired_torques = self.minitaur.Step(a)

        """ for smoothness rewards """
        self._past_actions[:-1] = self._past_actions[1:]
        self._past_actions[-1] = np.copy(action)

        reward = self._reward()
        done = self._termination()
        if self._log_path is not None:
            minitaur_logging.update_episode_proto(self._episode_proto, self.minitaur, a,
                                                  self._env_step_counter)
        self._env_step_counter += 1
        if done:
            self.minitaur.Terminate()

        return np.array(self._get_observation()), reward, done, {'desired_torques': desired_torques, 'rewards': self._objectives, 'observed_torques': observed_torques}

    def set_desired_speed(self, vd):
        self._desired_speed = vd

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos = self.minitaur.GetBasePosition()
        view_matrix = self._pybullet_client.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=-self._cam_dist,
            yaw=self._cam_yaw + self._slope_degree,
            pitch=-80,
            roll=0,
            upAxisIndex=1)
        proj_matrix = self._pybullet_client.computeProjectionMatrixFOV(fov=60,
                                                                       aspect=float(RENDER_WIDTH) /
                                                                              RENDER_HEIGHT,
                                                                       nearVal=0.1,
                                                                       farVal=100.0)
        (_, _, px, _, _) = self._pybullet_client.getCameraImage(
            width=RENDER_WIDTH,
            height=RENDER_HEIGHT,
            renderer=self._pybullet_client.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def set_mismatch(self, mismatch):
        slope = mismatch[0]
        friction = mismatch[1]
        g = mismatch[2]
        self._g = 10 + 10 * g
        self._slope_degree = 25 * slope
        self._friction = (1 + 2 * friction) if friction >= 0 else (1 + friction)

    def get_foot_contact(self):
        """
      FR r3 - l6
      BR r9 - l12
      FL l16 -r19
      BL l22 - r25
      :return: [FR, BR, FL, BL] boolean contact with the floor
      """
        contacts = []
        for foot_id in self.minitaur._foot_link_ids:
            contact = self.pybullet_client.getContactPoints(0, 1, -1, foot_id)
            if contact != ():
                contacts.append(foot_id)
        FR = 3 in contacts or 6 in contacts
        BR = 9 in contacts or 12 in contacts
        FL = 16 in contacts or 19 in contacts
        BL = 22 in contacts or 25 in contacts
        return [FR, BR, FL, BL]

    def get_foot_position(self):
        """
      FR r3 - l6
      BR r9 - l12
      FL l16 -r19
      BL l22 - r25
      :return: [FR, BR, FL, BL] boolean contact with the floor
      """
        Positions = []
        for foot_id in self.minitaur._foot_link_ids:
            pos = self.pybullet_client.getLinkState(self.minitaur.quadruped, foot_id)
            if foot_id in [3, 6, 9, 12, 16, 19, 22, 25]:
                Positions.append(pos[0])
        return Positions

    def get_minitaur_motor_angles(self):
        """Get the minitaur's motor angles.

    Returns:
      A numpy array of motor angles.
    """
        return np.array(self._observation[MOTOR_ANGLE_OBSERVATION_INDEX:MOTOR_ANGLE_OBSERVATION_INDEX +
                                                                        NUM_MOTORS])

    def get_minitaur_motor_velocities(self):
        """Get the minitaur's motor velocities.

    Returns:
      A numpy array of motor velocities.
    """
        return np.array(
            self._observation[MOTOR_VELOCITY_OBSERVATION_INDEX:MOTOR_VELOCITY_OBSERVATION_INDEX +
                                                               NUM_MOTORS])

    def get_minitaur_motor_torques(self):
        """Get the minitaur's motor torques.

    Returns:
      A numpy array of motor torques.
    """
        return np.array(
            self._observation[MOTOR_TORQUE_OBSERVATION_INDEX:MOTOR_TORQUE_OBSERVATION_INDEX +
                                                             NUM_MOTORS])

    def get_minitaur_base_orientation(self):
        """Get the minitaur's base orientation, represented by a quaternion.

    Returns:
      A numpy array of minitaur's orientation.
    """
        return np.array(self._observation[BASE_ORIENTATION_OBSERVATION_INDEX:])

    def is_fallen(self):
        """Decide whether the minitaur has fallen.

    If the up directions between the base and the world is larger (the dot
    product is smaller than 0.85) or the base is very low on the ground
    (the height is smaller than 0.13 meter), the minitaur is considered fallen.

    Returns:
      Boolean value that indicates whether the minitaur has fallen.
    """
        orientation = self.minitaur.GetBaseOrientation()
        rot_mat = self._pybullet_client.getMatrixFromQuaternion(orientation)
        local_up = rot_mat[6:]
        pos = self.minitaur.GetBasePosition()
        return (np.dot(np.asarray([0, 0, 1]), np.asarray(local_up)) < 0.70 or pos[2] < 0.13)

    def _termination(self):
        position = self.minitaur.GetBasePosition()
        distance = math.sqrt(position[0] ** 2 + position[1] ** 2)
        return self.is_fallen() or distance > self._distance_limit

    def _reward(self):
        current_base_position = self.minitaur.GetBasePosition()
        current_base_velocity = self.minitaur.GetBaseVelocity()
        current_base_orientation = self.minitaur.GetBaseRollPitchYaw()
        distance_reward = -(current_base_velocity[0] - self._desired_speed) ** 2
        high_reward = -(current_base_position[2] - 0.17) ** 2
        roll_reward = -(current_base_orientation[0]) ** 2
        pitch_reward = -(current_base_orientation[1]) ** 2
        yaw_reward = -(current_base_orientation[2]) ** 2
        action_reward = -np.sum((self._past_actions[3]) ** 2)
        action_vel_reward = -np.sum((self._past_actions[3] - self._past_actions[2]) ** 2)
        action_acc_reward = -np.sum((self._past_actions[3] - 2 * self._past_actions[2] + self._past_actions[1]) ** 2)
        action_jerk_reward = -np.sum((self._past_actions[3] - 3 * self._past_actions[2] + 3 * self._past_actions[1] + self._past_actions[0]) ** 2)
        rewards = [distance_reward, high_reward, roll_reward, pitch_reward, yaw_reward, action_reward,
                   action_vel_reward, action_acc_reward, action_jerk_reward]
        self._objectives = np.copy(rewards)
        reward = np.sum([np.exp(rewards[i] * self._objective_weights[i]) for i in range(len(rewards))])
        return reward

    def get_objectives(self):
        return self._objectives

    @property
    def objective_weights(self):
        """Accessor for the weights for all the objectives.

    Returns:
      List of floating points that corresponds to weights for the objectives in
      the order that objectives are stored.
    """
        return self._objective_weights

    def _get_observation(self):
        """Get observation of this environment, including noise and latency.

    The minitaur class maintains a history of true observations. Based on the
    latency, this function will find the observation at the right time,
    interpolate if necessary. Then Gaussian noise is added to this observation
    based on self.observation_noise_stdev.

    Returns:
      The noisy observation with latency.
    """
        observation = []
        observation.extend(self.minitaur.GetMotorAngles().tolist())
        observation.extend(self._filter_velocities(self.minitaur.GetMotorVelocities()).tolist())
        # observation.extend(self.minitaur.GetMotorTorques().tolist())
        observation.extend(self.minitaur.GetBaseRollPitchYaw().tolist())
        observation.extend(self.minitaur.GetBaseRollPitchYawRate().tolist())
        observation.extend(self.minitaur.GetBaseVelocity().tolist()[:2])
        # observation.extend([self.minitaur.GetBasePosition()[2]])
        self._observation = observation
        return self._observation

    def _filter_velocities(self, x):
        y = self._A * (x + self._past_velocity[0]) + (1-self._C)*self._A*self._past_velocity[1]
        self._past_velocity = np.copy([x, y])
        return y

    def _get_true_observation(self):
        """Get the observations of this environment.

    It includes the angles, velocities, torques and the orientation of the base.

    Returns:
      The observation list. observation[0:8] are motor angles. observation[8:16]
      are motor velocities, observation[16:24] are motor torques.
      observation[24:28] is the orientation of the base, in quaternion form.
    """
        observation = []
        observation.extend(self.minitaur.GetTrueMotorAngles().tolist())
        observation.extend(self.minitaur.GetTrueMotorVelocities().tolist())
        observation.extend(self.minitaur.GetTrueMotorTorques().tolist())
        observation.extend(list(self.minitaur.GetTrueBaseOrientation()))

        self._true_observation = observation
        return self._true_observation

    def _get_observation_upper_bound(self):
        """Get the upper bound of the observation.

    Returns:
      The upper bound of an observation. See GetObservation() for the details
        of each element of an observation.
    """
        upper_bound = np.zeros(self._get_observation_dimension())
        num_motors = self.minitaur.num_motors
        upper_bound[0:num_motors] = math.pi  # Joint angle.
        upper_bound[num_motors:2 * num_motors] = (motor.MOTOR_SPEED_LIMIT)  # Joint velocity.
        upper_bound[2 * num_motors:3 * num_motors] = (motor.OBSERVED_TORQUE_LIMIT)  # Joint torque.
        upper_bound[3 * num_motors:] = 1.0  # Quaternion of base orientation.
        return upper_bound

    def _get_observation_lower_bound(self):
        """Get the lower bound of the observation."""
        return -self._get_observation_upper_bound()

    def _get_observation_dimension(self):
        """Get the length of the observation list.

    Returns:
      The length of the observation list.
    """
        return len(self._get_observation())

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed
        _step = step

    def set_time_step(self, control_step, simulation_step=0.001):
        """Sets the time step of the environment.

    Args:
      control_step: The time period (in seconds) between two adjacent control
        actions are applied.
      simulation_step: The simulation time step in PyBullet. By default, the
        simulation step is 0.001s, which is a good trade-off between simulation
        speed and accuracy.
    Raises:
      ValueError: If the control step is smaller than the simulation step.
    """
        if control_step < simulation_step:
            raise ValueError("Control step should be larger than or equal to simulation step.")
        self.control_time_step = control_step
        self._time_step = simulation_step
        self._action_repeat = int(round(control_step / simulation_step))
        self._num_bullet_solver_iterations = (NUM_SIMULATION_ITERATION_STEPS / self._action_repeat)
        self._pybullet_client.setPhysicsEngineParameter(
            numSolverIterations=self._num_bullet_solver_iterations)
        self._pybullet_client.setTimeStep(self._time_step)
        self.minitaur.SetTimeSteps(action_repeat=self._action_repeat, simulation_step=self._time_step)

    def get_timesteps(self):
        return self.control_time_step, self._time_step

    @property
    def pybullet_client(self):
        return self._pybullet_client

    @property
    def ground_id(self):
        return self._ground_id

    @ground_id.setter
    def ground_id(self, new_ground_id):
        self._ground_id = new_ground_id

    @property
    def env_step_counter(self):
        return self._env_step_counter

    def add_obstacles(self, orientation):
        colBoxId = self._pybullet_client.createCollisionShape(self._pybullet_client.GEOM_BOX,
                                                              halfExtents=[0.01, 2, 0.1])
        self._pybullet_client.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=[0, 0, 1.],
                                              baseOrientation=[0, 0.6427876, 0, 0.7660444])

        # visBoxId = self._pybullet_client.createVisualShape(self._pybullet_client.GEOM_BOX, halfExtents=[(obstacle[1]-obstacle[0])/2.0+0.005, (obstacle[3]-obstacle[2])/2.0+0.005, 0.4], rgbaColor=[0.1,0.7,0.6, 1.0], specularColor=[0.3, 0.5, 0.5, 1.0])
        # self._pybullet_client.createMultiBody(baseMass=0, baseInertialFramePosition=[0, 0, 0], baseVisualShapeIndex=visBoxId, useMaximalCoordinates=True, basePosition=[(obstacle[0]+obstacle[1])/2.0, (obstacle[2]+obstacle[3])/2.0, 0.04])


if __name__ == "__main__":
    import gym
    import time
    import pickle
    from tqdm import tqdm
    import fast_adaptation_embedding.env
    from gym.wrappers.monitoring.video_recorder import VideoRecorder
    from pybullet_envs.bullet import minitaur_gym_env


    # render = False
    render = True
    action_space = ['Motor', 'Motor2', 'velocity', 'velocity2', 'S&E', 'Torque',  'Torque2'][1]

    dt = ctrl_time_step = 1 / 250 if action_space in ["Torque", "Torque2"] else 1/50
    env = gym.make("MinitaurTorqueEnv_fastAdapt-v0", render=render, on_rack=0,
                   control_time_step=ctrl_time_step,
                   action_repeat=int(250 * ctrl_time_step),
                   accurate_motor_model_enabled=0,
                   pd_control_enabled=1,
                   partial_torque_control=(action_space in ["Torque", "Torque2"])+1,
                   vkp=20,
                   env_randomizer=None,
                   action_space=action_space,
                   )

    recorder = None
    env.metadata['video.frames_per_second'] = 1 / ctrl_time_step
    # recorder = VideoRecorder(env, "test.mp4")
    Obs, Action, Reward = [], [], []
    env.set_mismatch([0., 0., 0.])
    ub = 1
    lb = -ub
    degree = -1
    max_vel, max_acc, max_jerk, max_torque_jerk = 10, 100, 10000, 25
    for k in range(1):
        previous_obs = env.reset()
        obs, action, reward = [previous_obs], [], []

        past = np.zeros((3, 8))
        # if action_space == "Torque":
        #     past[-1] = np.full(8, 0.1)+np.array([0, 0, 0, 0, 0, 0, 0, 0])

        for i in tqdm(range(3, int(10/dt) + 3)):
            if recorder is not None:
                recorder.capture_frame()

            if action_space in ["Motor", "Motor2", "velocity2", "S&E", 'Torque2']:
                if degree == 0:
                    amax = [ub] * 8
                    amin = [lb] * 8
                elif degree == 1:
                    amax = past[i - 1] + max_vel * dt
                    amin = past[i - 1] - max_vel * dt
                elif degree == 2:
                    amax = np.min((past[i - 1] + max_vel * dt, 2 * past[i - 1] - past[i - 2] + max_acc * dt ** 2), axis=0)
                    amin = np.max((past[i - 1] - max_vel * dt, 2 * past[i - 1] - past[i - 2] - max_acc * dt ** 2), axis=0)
                else:
                    amax = np.min((past[i - 1] + max_vel * dt,
                                   2 * past[i - 1] - past[i - 2] + max_acc * dt ** 2,
                                   3 * past[i - 1] - 3 * past[i - 2] + past[i - 3] + max_jerk * dt ** 3),
                                  axis=0)
                    amin = np.max((past[i - 1] - max_vel * dt,
                                   2 * past[i - 1] - past[i - 2] - max_acc * dt ** 2,
                                   3 * past[i - 1] - 3 * past[i - 2] + past[i - 3] - max_jerk * dt ** 3),
                                  axis=0)
                amax, amin = np.clip(amax, lb, ub), np.clip(amin, lb, ub)
                x = np.random.uniform(amin, amax)
                a = np.copy(x)
            elif action_space == "velocity":
                x = a = np.random.uniform(lb, ub, 8)
            elif action_space == "Torque":
                if degree == 0:
                    amax = [ub] * 8
                    amin = [lb] * 8
                else:
                    amax = past[i - 1] + max_torque_jerk * dt
                    amin = past[i - 1] - max_torque_jerk * dt
                amax, amin = np.clip(amax, lb, ub), np.clip(amin, lb, ub)
                x = np.random.uniform(amin, amax)
                a = np.copy(x)
            if action_space == "Torque2":
                a[:4] = a[:4] * 0.6
                a[4:] = (a[4:] * 0.4 + 0.5) * 0.75 + 0.10
                a = env.minitaur.ConvertFromLegModel(a)
                a = 8 * (a - env.minitaur.GetMotorAngles())
            elif action_space == "velocity2":
                a[:4] = a[:4] * 0.6
                a[4:] = (a[4:] * 0.4 + 0.5) * 0.75 + 0.10
                a = env.minitaur.ConvertFromLegModel(a)
                a = (a - env.minitaur.GetMotorAngles())
            elif action_space == "Motor2":
                a[:4] = a[:4] * 0.6
                a[4:] = (a[4:] * 0.4 + 0.5) * 0.75 + 0.10
                a = env.minitaur.ConvertFromLegModel(a)-1.5
            """ env step """
            o, r, done, info = env.step(a)
            past = np.append(past, [np.copy(x)], axis=0)
            obs.append(o)
            action.append(a)
            reward.append(info['rewards'])
            time.sleep(ctrl_time_step)
            if done:
                break
        Obs.append(obs)
        Action.append(action)
        Reward.append(reward)

        # if k % 1000 == 0:
        #     with open("data/random_s&e_"+str(k)+".pk", 'wb') as f:
        #         pickle.dump([Obs, Action, Reward], f)
    if recorder is not None:
        recorder.capture_frame()
        recorder.close()
