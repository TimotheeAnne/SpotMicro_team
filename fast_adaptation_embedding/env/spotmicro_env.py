"""
SpotMicroAI Simulation
"""
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet_data
import time
import pybullet as p
from pybullet import DIRECT, GUI
import math
import numpy as np
import gym
from gym import spaces
from fast_adaptation_embedding.env.assets.pybullet_envs import bullet_client

# from pynput.keyboard import Key, Listener

gym.logger.set_level(40)

RENDER_HEIGHT = 360
RENDER_WIDTH = 480

MOTOR_NAMES = ['front_left_shoulder_joint', 'front_left_thigh_joint', 'front_left_calf_joint',
               'front_right_shoulder_joint', 'front_right_thigh_joint', 'front_right_calf_joint',
               'rear_left_shoulder_joint', 'rear_left_thigh_joint', 'rear_left_calf_joint',
               'rear_right_shoulder_joint', 'rear_right_thigh_joint', 'rear_right_calf_joint',
               ]

blue = [0., 0.7, 1, 1]
black = [0.1, 0.1, 0.1, 1]
grey = [0.6, 0.6, 0.6, 1]

MOTORS_COLORS = [black, blue, grey]*4


class SpotMicroEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self,
                 render=False,
                 on_rack=False,
                 action_space="Motor",
                 ctrl_time_step=0.02,
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
                 init_joint=None,
                 ub=None,
                 lb=None,
                 kp=None,
                 kd=None,
                 urdf_model="basic",
                 inspection=False,
                 normalized_action=True,
                 faulty_motors=[],
                 faulty_joints=[],
                 load_weight=0,
                 load_pos=0
                 ):

        self.is_render = render
        if self.is_render:
            self.pybullet_client = bullet_client.BulletClient(connection_mode=GUI)
        else:
            self.pybullet_client = bullet_client.BulletClient(connection_mode=DIRECT)
        # Simulation Configuration
        self.fixedTimeStep = 1. / 1000  # 250
        self.action_repeat = int(ctrl_time_step / self.fixedTimeStep)
        self.numSolverIterations = 200
        self.useFixeBased = on_rack
        self.init_oritentation = self.pybullet_client.getQuaternionFromEuler([0, 0, np.pi/2])
        self.reinit_position = [0, 0, 0.3]
        self.init_position = [0, 0, 0.23]
        self.kp = np.array([10, 5, 3] * 4) if kp is None else kp
        self.kd = np.array([0.1, 0.1, 0.04] * 4) if kd is None else kd
        self.maxForce = 3
        self._motor_direction = [-1, 1, 1] * 4
        self.shoulder_to_knee = 0.2
        self.knee_to_foot = 0.1
        self.normalized_action = normalized_action
        if init_joint is None:
            init_joint = [0., 0.6, -1.] * 4
        self.init_joint = init_joint
        self.lateral_friction = 0.8
        self.urdf_model = urdf_model
        self.fc = 10
        self.C = 1 / (np.tan(np.pi * self.fc * self.fixedTimeStep))
        self.A = 1 / (1 + self.C)
        self.inspection = inspection
        assert action_space in ["S&E", "Motor"], "Control mode not implemented yet"
        self.action_space = action_space
        if (ub is not None) and (lb is not None):
            self.ub = np.array(ub)
            self.lb = np.array(lb)
        elif self.action_space == "Motor":
            self.ub = np.array(
                [0.2, 0.3, -1.2] * 4)  # max [0.548, 1.548, 0.1]
            # [0.2, -0.5, 1.6]*4)
            self.lb = np.array(
                [-0.2, 0.9, -0.8] * 4)  # min [-0.548, -2.666, -2.59]
            # [-0.2, -0.9, 1.1]*4)
        elif self.action_space == "S&E":
            """ abduction - swing - extension"""
            self.ub = np.array([0.2, 0.4, 0.25] * 4)  # [0.2, 0.4, 0.25]
            self.lb = np.array([-0.2, -0.6, 0.2] * 4)  # [-0.2, -0.6, 0.2]

        self._objective_weights = [distance_weight, high_weight, roll_weight, pitch_weight, yaw_weight,
                                   action_weight, action_vel_weight, action_acc_weight, action_jerk_weight]
        self.desired_speed = desired_speed
        self.t = 0

        self.pybullet_client.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 1)

        self.faulty_motors = faulty_motors
        self.faulty_joints = faulty_joints
        self.load_weight = load_weight
        self.load_pos = load_pos

        self.wind_force = 0
        self.friction = 0.8
        self.mismatch = {'friction': self.friction,
                         'wind_force': self.wind_force,
                         'load_weight': self.load_weight,
                         'load_pos': self.load_pos,
                         'faulty_motors': self.faulty_motors,
                         'faulty_joints': self.faulty_joints}

        self.texture_id = self.pybullet_client.loadTexture(currentdir + "/assets/checker_blue.png")
        self.quadruped = self.loadModels()
        self._BuildJointNameToIdDict()
        self._BuildMotorIdList()
        self.jointNameToId = self.getJointNames(self.quadruped)
        self.pybullet_client.setPhysicsEngineParameter(numSolverIterations=self.numSolverIterations,
                                                       fixedTimeStep=self.fixedTimeStep)
        self.pybullet_client.resetDebugVisualizerCamera(1, 85.6, 0, [-0.61, 0.12, 0.25])

        action_dim = 12  # [steering, step_size, leg_extension, leg_extension_offset]
        self._past_actions = np.zeros((4, action_dim))
        self._past_velocity = np.zeros((2, action_dim))

        self._action_bound = 1
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        observation_high = (self.get_observation_upper_bound())
        observation_low = (self.get_observation_lower_bound())
        self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)
        self.set_mismatch(self.mismatch)

    def _BuildJointNameToIdDict(self):
        num_joints = self.pybullet_client.getNumJoints(self.quadruped)
        self._joint_name_to_id = {}
        for i in range(num_joints):
            joint_info = self.pybullet_client.getJointInfo(self.quadruped, i)
            self._joint_name_to_id[joint_info[1].decode("UTF-8")] = joint_info[0]

    def _BuildMotorIdList(self):
        self._motor_id_list = [self._joint_name_to_id[motor_name] for motor_name in MOTOR_NAMES]

    def loadModels(self):
        self.pybullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.pybullet_client.setGravity(0, 0, -9.81)

        orn = self.pybullet_client.getQuaternionFromEuler([0, 0, 0])
        self.pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.planeUid = self.pybullet_client.loadURDF("plane_transparent.urdf", [0, 0, 0.], orn)
        self.pybullet_client.changeVisualShape(self.planeUid, -1, textureUniqueId=self.texture_id)
        self.pybullet_client.changeDynamics(self.planeUid, -1, lateralFriction=self.lateral_friction)

        flags = self.pybullet_client.URDF_USE_SELF_COLLISION

        urdf_model = 'spot_micro_urdf_v2/urdf/spot_micro_urdf_v2.urdf.xml'

        quadruped = self.pybullet_client.loadURDF(currentdir + "/assets/" + urdf_model, self.init_position,
                                                  self.init_oritentation,
                                                  useFixedBase=self.useFixeBased,
                                                  useMaximalCoordinates=False,
                                                  flags=flags)
        self.pybullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        self.wind_arrow = self.pybullet_client.loadURDF(currentdir + "/assets/urdf/arrow.urdf.xml",
                                                   [0, 0, -2],
                                                   p.getQuaternionFromEuler((0, 0, 0)))
        self.load_visual = self.pybullet_client.loadURDF(currentdir + "/assets/urdf/load.urdf.xml",
                                                   [0, 0, -2],
                                                   p.getQuaternionFromEuler((0, 0, 0)))

        return quadruped

    def set_kd(self, kd):
        self.kd = kd

    def set_kp(self, kp):
        self.kp = kp

    def reset(self, hard_reset=False):
        if hard_reset:
            self.pybullet_client.resetSimulation()
            self.pybullet_client.setPhysicsEngineParameter(numSolverIterations=self.numSolverIterations,
                                                           fixedTimeStep=self.fixedTimeStep)

            self.quadruped = self.loadModels()
            self._BuildJointNameToIdDict()
            self._BuildMotorIdList()
        if self.useFixeBased:
            init_pos = [0, 0, 0.5]
            reinit_pos = [0, 0, 0.5]
        else:
            init_pos = [0, 0, 0.23]
            reinit_pos = [0, 0, 0.5]
        self.pybullet_client.resetBasePositionAndOrientation(self.quadruped, reinit_pos, self.init_oritentation)
        self.pybullet_client.resetBaseVelocity(self.quadruped, [0, 0, 0], [0, 0, 0])

        for _ in range(40):
            self.pybullet_client.setJointMotorControlArray(
                self.quadruped,
                self._motor_id_list,
                self.pybullet_client.POSITION_CONTROL,
                targetPositions=np.multiply(self.apply_faulty_motors(self.init_joint), self._motor_direction),
                forces=np.ones(12) * 1000
            )
            # time.sleep(self.fixedTimeStep*20)
            self.pybullet_client.stepSimulation()
        p.setJointMotorControlArray(
            self.quadruped,
            self._motor_id_list,
            p.POSITION_CONTROL,
            targetPositions=np.multiply(self.apply_faulty_motors(self.init_joint), self._motor_direction),
            forces=np.zeros(12)
        )

        self.pybullet_client.stepSimulation()
        self.pybullet_client.resetBasePositionAndOrientation(self.quadruped, init_pos, self.init_oritentation)
        self.pybullet_client.resetBaseVelocity(self.quadruped, [0, 0, 0], [0, 0, 0])
        for _ in range(300):
            self.apply_action(self.init_joint)
            # time.sleep(self.fixedTimeStep*10)
        self.t = 0
        self._past_velocity = np.zeros((2, 12))
        return self.get_obs()

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])
        bodyPos, _ = self.pybullet_client.getBasePositionAndOrientation(self.quadruped)
        base_pos = bodyPos

        if self.inspection:
            base_pos = base_pos[0], base_pos[1], base_pos[2] - 0.1,
            view_matrix = self.pybullet_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos,
                distance=-0.4,
                yaw=0,
                pitch=-80,
                roll=0,
                upAxisIndex=1)
        else:
            view_matrix = self.pybullet_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos,
                distance=-0.5,
                yaw=0,
                pitch=-80,
                roll=0,
                upAxisIndex=1)

        proj_matrix = self.pybullet_client.computeProjectionMatrixFOV(fov=60,
                                                                      aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                                      nearVal=0.1,
                                                                      farVal=100.0)
        (_, _, px, _, _) = self.pybullet_client.getCameraImage(
            width=RENDER_WIDTH,
            height=RENDER_HEIGHT,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def changeDynamics(self, quadruped):
        nJoints = self.pybullet_client.getNumJoints(quadruped)
        for i in range(nJoints):
            self.pybullet_client.changeDynamics(quadruped, i, localInertiaDiagonal=[0.000001, 0.000001, 0.000001])

    def getJointNames(self, quadruped):
        nJoints = self.pybullet_client.getNumJoints(quadruped)
        jointNameToId = {}

        for i in range(nJoints):
            jointInfo = self.pybullet_client.getJointInfo(quadruped, i)
            jointNameToId[jointInfo[1].decode('UTF-8')] = jointInfo[0]
        return jointNameToId

    def getPos(self):
        bodyPos, _ = self.pybullet_client.getBasePositionAndOrientation(self.quadruped)
        return bodyPos

    def getIMU(self):
        _, bodyOrn = self.pybullet_client.getBasePositionAndOrientation(self.quadruped)
        linearVel, angularVel = self.pybullet_client.getBaseVelocity(self.quadruped)
        return bodyOrn, linearVel, angularVel

    def step(self, action):
        a = np.copy(action)
        if self.normalized_action:
            a = a * (self.ub - self.lb) / 2 + (self.ub + self.lb) / 2
        if self.action_space == "S&E":
            """ abduction - swing - extension"""
            a = self.from_leg_to_motor(a)

        observed_torques = []
        for _ in range(self.action_repeat):
            obs_torque = self.apply_action(a)
            observed_torques.append(obs_torque)

        """ for smoothness rewards """
        self._past_actions[:-1] = self._past_actions[1:]
        self._past_actions[-1] = np.copy(a)

        obs = self.get_obs()
        reward, rewards = self.get_reward()
        info = {'rewards': rewards, 'observed_torques': observed_torques}
        done = self.fallen()
        return np.array(obs), reward, done, info

    def fallen(self):
        _, bodyOrn = self.pybullet_client.getBasePositionAndOrientation(self.quadruped)
        return self.checkSimulationReset(bodyOrn)

    def checkSimulationReset(self, bodyOrn):
        (xr, yr, _) = self.pybullet_client.getEulerFromQuaternion(bodyOrn)
        return abs(xr) > math.pi / 4 or abs(yr) > math.pi / 4

    def apply_faulty_motors(self, action):
        faulty_action = np.copy(action)
        for index in range(len(self.faulty_motors)):
            faulty_action[self.faulty_motors[index]] = self.faulty_joints[index]
        return faulty_action

    def apply_action(self, action):
        action = self.apply_faulty_motors(action)
        self.t = self.t + self.fixedTimeStep
        q = self.GetMotorAngles()
        q_vel = self.GetMotorVelocities()
        PD_torque = self.kp * (action - q) - self.kd * q_vel
        PD_torque = np.clip(PD_torque, -self.maxForce, self.maxForce)
        self.pybullet_client.setJointMotorControlArray(bodyIndex=self.quadruped,
                                                       jointIndices=self._motor_id_list,
                                                       controlMode=self.pybullet_client.TORQUE_CONTROL,
                                                       forces=np.multiply(PD_torque, self._motor_direction))

        if self.wind_force > 0:
            external_force = [self.wind_force * np.cos(self.wind_angle), self.wind_force * np.sin(self.wind_angle), 0]

            self.pybullet_client.applyExternalForce(objectUniqueId=self.quadruped,
                                                    linkIndex=0,
                                                    forceObj=external_force,
                                                    posObj=self.get_body_xyz(),
                                                    flags=p.WORLD_FRAME)
            [x, y, _] = self.get_body_xyz()
            self.pybullet_client.resetBasePositionAndOrientation(self.wind_arrow, [x, y + 0.025, 0.25],
                                                                 p.getQuaternionFromEuler(
                                                                     [1.57, 0, 1.48 + self.wind_angle]))

        if self.load_weight > 0:
            external_force = [0, 0, -9.81*self.load_weight]
            [x_b, y_b, z_b], body_orient = self.pybullet_client.getBasePositionAndOrientation(self.quadruped)
            x = x_b + self.load_pos
            y = y_b
            z = z_b
            self.pybullet_client.applyExternalForce(objectUniqueId=self.quadruped,
                                                    linkIndex=0,
                                                    forceObj=external_force,
                                                    posObj=[x, y, z],
                                                    flags=p.WORLD_FRAME)

            self.pybullet_client.resetBasePositionAndOrientation(self.load_visual, [x, y, z + 0.07], [0, 0, 0, 1])

        self.pybullet_client.stepSimulation()
        return np.copy(PD_torque)

    def _filter_velocities(self, x):
        y = self.A * (x + self._past_velocity[0]) + (1 - self.C) * self.A * self._past_velocity[1]
        self._past_velocity = np.copy([x, y])
        return y

    def get_body_xyz(self):
        bodyPos, _ = self.pybullet_client.getBasePositionAndOrientation(self.quadruped)
        return bodyPos

    def get_body_rpy(self):
        _, bodyOrn = self.pybullet_client.getBasePositionAndOrientation(self.quadruped)
        bodyOrn = self.pybullet_client.getEulerFromQuaternion(bodyOrn)
        yaw = bodyOrn[2]-np.pi/2
        bodyOrn = [bodyOrn[0], bodyOrn[1], yaw if yaw >= -np.pi else yaw + 2 * np.pi]
        return bodyOrn

    def get_linear_velocity(self):
        linearVel, _ = self.pybullet_client.getBaseVelocity(self.quadruped)
        return linearVel

    def get_angular_velocity(self):
        _, angularVel = self.pybullet_client.getBaseVelocity(self.quadruped)
        return angularVel

    def get_obs(self):
        Obs = []
        Obs.extend(self.GetMotorAngles().tolist())
        Obs.extend(self._filter_velocities(self.GetMotorVelocities()).tolist())
        Obs.extend(self.get_body_rpy())
        Obs.extend(self.get_angular_velocity())
        Obs.extend(self.get_linear_velocity()[:2])  # x and y velocity
        Obs.extend(self.get_body_xyz()[2:3])  # z height
        return Obs

    def set_desired_speed(self, speed):
        self.desired_speed = speed

    def get_reward(self):
        obs = self.get_obs()
        distance_reward = -(obs[-2] - self.desired_speed) ** 2
        high_reward = -(obs[-1] - 0.1855) ** 2
        roll_reward = -(obs[24]) ** 2
        pitch_reward = -(obs[25]) ** 2
        yaw_reward = -(obs[26]) ** 2
        action_reward = -np.sum((self._past_actions[3]) ** 2)
        action_vel_reward = -np.sum((self._past_actions[3] - self._past_actions[2]) ** 2)
        action_acc_reward = -np.sum((self._past_actions[3] - 2 * self._past_actions[2] + self._past_actions[1]) ** 2)
        action_jerk_reward = -np.sum((self._past_actions[3] - 3 * self._past_actions[2] + 3 * self._past_actions[1] +
                                      self._past_actions[0]) ** 2)
        rewards = [distance_reward, high_reward, roll_reward, pitch_reward, yaw_reward, action_reward,
                   action_vel_reward, action_acc_reward, action_jerk_reward]
        reward_sum = 0
        n_rewards = 0
        for i in range(len(rewards)):
            if self._objective_weights != 0:
                reward_sum += np.exp(rewards[i]) / 500
                n_rewards += 1
        reward_sum /= n_rewards
        return reward_sum, np.copy(rewards)

    def set_mismatch(self, mismatch):
        """ a hard reset is required after setting_mismatch"""
        for (key, val) in mismatch.items():
            self.mismatch[key] = val
        assert 0 <= self.mismatch['friction'], 'friction must be non-negative'
        for x in self.mismatch['faulty_motors']:
            assert 0 <= x < 12, 'faulty motor must be None or an int in [0;11]'
        assert len(self.mismatch['faulty_motors']) == len(self.mismatch['faulty_joints']), "must specify a joint for each faulty motor"
        assert 0 <= self.mismatch['load_weight'], 'load weight must be >= 0'
        assert -0.07 <= self.mismatch['load_pos'] <= 0.07, 'load pos must be in [-0.07,0.07]'

        self.lateral_friction = self.mismatch['friction']
        self.wind_angle = np.pi / 2 if self.mismatch['wind_force'] > 0 else -np.pi / 2
        self.wind_force = self.mismatch['wind_force']
        self.faulty_motors = self.mismatch['faulty_motors']
        self.faulty_joints = self.mismatch['faulty_joints']
        self.load_weight = self.mismatch['load_weight']
        self.load_pos = self.mismatch['load_pos']

        for motor in range(12):
            if motor in self.faulty_motors:
                self.pybullet_client.changeVisualShape(self.quadruped, motor, rgbaColor=[1, 0, 0, 1])
            else:
                self.pybullet_client.changeVisualShape(self.quadruped, motor, rgbaColor=MOTORS_COLORS[motor])

        if self.lateral_friction > 0.8:
            friction_level = "striped"
        elif self.lateral_friction == 0.8:
            friction_level = "checker_blue"
        elif self.lateral_friction <= 0.2:
            friction_level = "checker_red"
        else:
            friction_level = "checker_purple"
        self.texture_id = self.pybullet_client.loadTexture(currentdir + "/assets/" + friction_level + ".png")

    def get_observation_upper_bound(self):
        """Get the upper bound of the observation.

    Returns:
      The upper bound of an observation. See GetObservation() for the details
        of each element of an observation.
    """
        upper_bound = np.full(self.get_observation_dimension(), np.inf)
        return upper_bound

    def get_observation_lower_bound(self):
        """Get the lower bound of the observation."""
        return -self.get_observation_upper_bound()

    def get_observation_dimension(self):
        """Get the length of the observation list.

        Returns:
          The length of the observation list.
        """
        return len(self.get_obs())

    def GetMotorAngles(self):
        """Gets the twelve motor angles at the current moment, mapped to [-pi, pi].

        Returns:
          Motor angles, mapped to [-pi, pi].
        """
        motor_angles = [
            self.pybullet_client.getJointState(self.quadruped, motor_id)[0]
            for motor_id in self._motor_id_list
        ]
        motor_angles = np.multiply(motor_angles, self._motor_direction)
        return motor_angles

    def GetMotorVelocities(self):
        """Get the velocity of all eight motors.

        Returns:
          Velocities of all eight motors.
        """
        motor_velocities = [
            self.pybullet_client.getJointState(self.quadruped, motor_id)[1]
            for motor_id in self._motor_id_list
        ]
        motor_velocities = np.multiply(motor_velocities, self._motor_direction)
        return motor_velocities


if __name__ == "__main__":
    import gym
    import fast_adaptation_embedding.env
    from tqdm import tqdm, trange
    from gym.wrappers.monitoring.video_recorder import VideoRecorder
    import matplotlib.pyplot as plt
    import pickle

    render = True
    # render = False

    on_rack = 0

    # (init_joint, real_ub, real_lb) = None, None, None

    run = 0

    maxis = [[10, 10, 1000]]
    bounds = [[[0.1, 0.8, -0.8], [-0.1, 0.4, -1.2]]]
    config = []
    for maxi in maxis:
        for bound in bounds:
            config.append({'maxi': maxi, 'real_ub': bound[0], 'real_lb': bound[1]})

    init_joint = np.array([0., 0.6, -1.] * 4)
    real_ub = np.array(config[run]['real_ub'] * 4)  # max [0.548, 1.548, 2.59]
    real_lb = np.array(config[run]['real_lb'] * 4)  # min [-0.548, -2.666, -0.1]

    max_vel, max_acc, max_jerk = config[run]['maxi']

    max_vel = max_vel * 2 / (real_ub - real_lb)
    max_acc = max_acc * 2 / (real_ub - real_lb)
    max_jerk = max_jerk * 2 / (real_ub - real_lb)

    env = gym.make("SpotMicroEnv-v0",
                   render=render,
                   on_rack=on_rack,
                   action_space=["S&E", "Motor"][1],
                   init_joint=init_joint,
                   ub=real_ub,
                   lb=real_lb,
                   urdf_model='sphere',
                   inspection=True,
                   normalized_action=1,
                   ctrl_time_step=0.02,
                   faulty_motors=[],
                   faulty_joints=[],
                   load_weight=0,
                   load_pos=0,
                   )

    O, A = [], []
    for iter in tqdm(range(1)):
        env.set_mismatch({})
        init_obs = env.reset(hard_reset=0)
        # time.sleep(100)

        recorder = None
        # recorder = VideoRecorder(env, "test.mp4")

        ub = 1
        lb = -1

        past = [(env.init_joint - (env.ub + env.lb) / 2) * 2 / (env.ub - env.lb)] * 3
        # past = [env.init_joint]*3

        dt = 0.02
        R = 0
        Obs, Acs = [init_obs], []

        # f = "/home/haretis/Documents/SpotMicro_team/exp/results/spot_micro_03/23_03_2020_12_40_33_experiment/run_0/logs.pk"
        # with open(f, 'rb') as f:
        #     data = pickle.load(f)

        actions = None
        # actions = data['actions'][0][250]

        degree = 3
        # t = trange(3, 500 + 3, desc='', leave=True)
        t = range(3, 500 * 4 + 3)
        for i in t:
            if recorder is not None:
                recorder.capture_frame()
            if degree == 0:
                amax = [ub] * 12
                amin = [lb] * 12
            elif degree == 1:
                amax = past[i - 1] + max_vel * dt
                amin = past[i - 1] - max_vel * dt
                amax, amin = np.clip(amax, lb, ub), np.clip(amin, lb, ub)
            elif degree == 2:
                amax = np.min((past[i - 1] + max_vel * dt,
                               2 * past[i - 1] - past[i - 2] + max_acc * dt ** 2),
                              axis=0)
                amin = np.max((past[i - 1] - max_vel * dt,
                               2 * past[i - 1] - past[i - 2] - max_acc * dt ** 2),
                              axis=0)
                amax, amin = np.clip(amax, lb, ub), np.clip(amin, lb, ub)
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
            # x = past[-1]
            # x[:6] = past[-1][:6]
            action = np.copy(x)
            if actions is not None:
                action = actions[(i - 3)]
            obs, reward, done, info = env.step(action)
            Obs.append(obs)
            Acs.append(action)
            R += reward
            past = np.append(past, [np.copy(x)], axis=0)
            if done:
                break
            if render and recorder is None:
                time.sleep(0.02)
        O.append(np.copy(Obs))
        A.append(np.copy(Acs))

        if recorder is not None:
            recorder.capture_frame()
            recorder.close()
