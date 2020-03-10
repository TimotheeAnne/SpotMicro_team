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
import math
import numpy as np
import gym
from gym import spaces
from fast_adaptation_embedding.env.assets.pybullet_envs import bullet_client
from pynput.keyboard import Key, Listener

NUM_SIMULATION_ITERATION_STEPS = 300
RENDER_HEIGHT = 360
RENDER_WIDTH = 480

MOTOR_NAMES = ["front_left_shoulder", "front_left_leg", "front_left_foot",
               "front_right_shoulder", "front_right_leg", "front_right_foot",
               "rear_left_shoulder", "rear_left_leg", "rear_left_foot",
               "rear_right_shoulder", "rear_right_leg", "rear_right_foot"]


class SpotMicroEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(self,
                 render=False,
                 on_rack=False,
                 action_space="Motor",
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
                 urdf_model="basic",
                 inspection=False,
                 ):

        self.is_render = render
        if self.is_render:
            self.pybullet_client = bullet_client.BulletClient(connection_mode=p.GUI)
        else:
            self.pybullet_client = bullet_client.BulletClient(connection_mode=p.DIRECT)

        # Simulation Configuration
        self.fixedTimeStep = 1. / 250  # 550
        self.action_repeat = 5
        self.numSolverIterations = int(NUM_SIMULATION_ITERATION_STEPS / self.action_repeat)
        self.useFixeBased = on_rack
        self.init_oritentation = self.pybullet_client.getQuaternionFromEuler([0, 0, np.pi])
        self.reinit_position = [0, 0, 0.3]
        self.init_position = [0, 0, 0.23]
        self.kp = 0.045
        self.kd = .4
        self.maxForce = 12.5
        self._motor_direction = [-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1]
        self.mismatch = [0]
        self.shoulder_to_knee = 0.2
        self.knee_to_foot = 0.1
        self.init_joint = self.from_leg_to_motor([0.2, 0.1, 0.25]*2+[0, -0.1, 0.25]*2) if init_joint is None else self.from_leg_to_motor(init_joint)
        self.lateral_friction = 0.8
        self.urdf_model = urdf_model
        self.fc = 10
        self.C = 1/(np.tan(np.pi*self.fc*self.fixedTimeStep))
        self.A = 1/(1+self.C)
        self._inspection = inspection
        assert action_space in ["S&E", "Motor"], "Control mode not implemented yet"
        self.action_space = action_space
        if (ub is not None) and (lb is not None):
            self.ub = np.array(ub)
            self.lb = np.array(lb)
        elif self.action_space == "Motor":
            self.ub = np.array([0.4, -0., 1.65] * 2 + [0.02, -0.45, 1.65] * 2)  # [0.02, -0.45, 1.65] max [0.548, 1.548, 2.59]
            self.lb = np.array([0, -0.75, 1.2] * 2 + [-0.09, -1.2, 1.2] * 2)  # [-0.09, -1.2, 1.2] min [-0.548, -2.666, -0.1]
        elif self.action_space == "S&E":
            """ abduction - swing - extension"""
            self.ub = np.array([0.2, 0.4, 0.25] * 4)  # [0.2, 0.4, 0.25]
            self.lb = np.array([-0.2, -0.6, 0.2] * 4)  # [-0.2, -0.6, 0.2]

        self._objective_weights = [distance_weight, high_weight, roll_weight, pitch_weight, yaw_weight,
                                   action_weight, action_vel_weight, action_acc_weight, action_jerk_weight]
        self.desired_speed = desired_speed
        self.t = 0

        self.pybullet_client.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 1)

        self.quadruped = self.loadModels()
        self._BuildJointNameToIdDict()
        self. _BuildMotorIdList()
        self.jointNameToId = self.getJointNames(self.quadruped)
        self.pybullet_client.setPhysicsEngineParameter(numSolverIterations=self.numSolverIterations)
        self.pybullet_client.resetDebugVisualizerCamera(1, 85.6, 0, [-0.61, 0.12, 0.25])

        action_dim = 12  # [steering, step_size, leg_extension, leg_extension_offset]
        self._past_actions = np.zeros((4, action_dim))
        self._past_velocity = np.zeros((2, action_dim))

        self._action_bound = 1
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        observation_high = (self.get_observation_upper_bound())
        observation_low = (self.get_observation_lower_bound())
        self.observation_space = spaces.Box(observation_low, observation_high)

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
        self.planeUid = self.pybullet_client.loadURDF("plane_transparent.urdf", [0, 0, 0], orn)
        self.pybullet_client.changeDynamics(self.planeUid, -1, lateralFriction=self.lateral_friction)

        flags = self.pybullet_client.URDF_USE_SELF_COLLISION
        urdf_model = 'spotmicroai_sphere_feet.xml' if self.urdf_model == 'sphere' else 'spotmicroai_gen.urdf.xml'
        quadruped = self.pybullet_client.loadURDF(currentdir + "/assets/urdf/"+ urdf_model, self.init_position,
                               self.init_oritentation,
                               useFixedBase=self.useFixeBased,
                               useMaximalCoordinates=False,
                               flags=flags)
        self.pybullet_client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        return quadruped

    def reset(self):
        if self.useFixeBased:
            init_pos = [0, 0, 0.5]
            reinit_pos = [0, 0, 0.5]
        else:
            init_pos = [0, 0, 0.22]
            reinit_pos = [0, 0, 0.5]
        self.pybullet_client.resetBasePositionAndOrientation(self.quadruped, reinit_pos, self.init_oritentation)
        self.pybullet_client.resetBaseVelocity(self.quadruped, [0, 0, 0], [0, 0, 0])
        for _ in range(40):
            self.apply_action(self.init_joint)
        self.pybullet_client.resetBasePositionAndOrientation(self.quadruped, init_pos, self.init_oritentation)
        self.pybullet_client.resetBaseVelocity(self.quadruped, [0, 0, 0], [0, 0, 0])
        self.t = 0
        self._past_velocity = [[0]*12]*2
        return self.get_obs()

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])
        bodyPos, _ = self.pybullet_client.getBasePositionAndOrientation(self.quadruped)
        base_pos = bodyPos

        if self.inspection:
            base_pos = base_pos[0],  base_pos[1],  base_pos[2]-0.1,
            view_matrix = self.pybullet_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=base_pos,
                distance=-0.4,
                yaw=0,
                pitch=-90,
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
                                                   aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
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
        a = a * (self.ub - self.lb) / 2 + (self.ub + self.lb) / 2
        if self.action_space == "S&E":
            """ abduction - swing - extension"""
            a = self.from_leg_to_motor(a)

        for _ in range(self.action_repeat):
            self.apply_action(a)

        """ for smoothness rewards """
        self._past_actions[:-1] = self._past_actions[1:]
        self._past_actions[-1] = np.copy(a)

        obs = self.get_obs()
        reward, rewards = self.get_reward()
        info = {'rewards': rewards}
        done = self.fallen()
        return np.array(obs), reward, done, info

    def fallen(self):
        _, bodyOrn = self.pybullet_client.getBasePositionAndOrientation(self.quadruped)
        return self.checkSimulationReset(bodyOrn)

    def checkSimulationReset(self, bodyOrn):
        (xr, yr, _) = self.pybullet_client.getEulerFromQuaternion(bodyOrn)
        return abs(xr) > math.pi / 3 or abs(yr) > math.pi / 3

    def apply_action(self, action):
        self.t = self.t + self.fixedTimeStep
        action = np.multiply(action, self._motor_direction)
        for lx, leg in enumerate(['front_left', 'front_right', 'rear_left', 'rear_right']):
            for px, part in enumerate(['shoulder', 'leg', 'foot']):
                j = self.jointNameToId[leg + "_" + part]
                self.pybullet_client.setJointMotorControl2(bodyIndex=self.quadruped,
                                        jointIndex=j,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=action[lx * 3 + px],
                                        positionGain=self.kp,
                                        velocityGain=self.kd,
                                        force=self.maxForce)

        self.pybullet_client.stepSimulation()

    def _filter_velocities(self, x):
        y = self.A * (x + self._past_velocity[0]) + (1-self.C)*self.A*self._past_velocity[1]
        self._past_velocity = np.copy([x, y])
        return y

    def get_body_xyz(self):
        bodyPos, _ = self.pybullet_client.getBasePositionAndOrientation(self.quadruped)
        return bodyPos

    def get_body_rpy(self):
        _, bodyOrn = self.pybullet_client.getBasePositionAndOrientation(self.quadruped)
        bodyOrn = self.pybullet_client.getEulerFromQuaternion(bodyOrn)
        yaw = bodyOrn[2]+np.pi
        bodyOrn = [bodyOrn[0], bodyOrn[1], yaw if yaw <= np.pi else yaw-2*np.pi]
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
        Obs.extend(self.GetMotorVelocities().tolist())
        Obs.extend(self.get_body_rpy())
        Obs.extend(self.get_angular_velocity())
        Obs.extend(self.get_linear_velocity()[:2])  # x and y velocity
        # Obs.extend(self.get_body_xyz()[2:3])  # z height
        return Obs

    def set_desired_speed(self, speed):
        self.desired_speed = speed

    def get_reward(self):
        obs = self.get_obs()
        distance_reward = -(obs[-2] - self.desired_speed) ** 2
        high_reward = -0*(obs[-1] - 0.17) ** 2
        roll_reward = -(obs[24]) ** 2
        pitch_reward = -(obs[25]) ** 2
        yaw_reward = -(obs[26]) ** 2
        action_reward = -np.sum((self._past_actions[3]) ** 2)
        action_vel_reward = -np.sum((self._past_actions[3] - self._past_actions[2]) ** 2)
        action_acc_reward = -np.sum((self._past_actions[3] - 2 * self._past_actions[2] + self._past_actions[1]) ** 2)
        action_jerk_reward = -np.sum((self._past_actions[3] - 3 * self._past_actions[2] + 3 * self._past_actions[1] + self._past_actions[0]) ** 2)
        rewards = [distance_reward, high_reward, roll_reward, pitch_reward, yaw_reward, action_reward,
                   action_vel_reward, action_acc_reward, action_jerk_reward]
        reward_sum = np.sum([np.exp(rewards[i] * self._objective_weights[i]) for i in range(len(rewards))])
        return reward_sum, np.copy(rewards)

    def set_mismatch(self, mismatch):
        self.mismatch = mismatch
        [friction] = mismatch
        self.lateral_friction = 0.8*friction + 0.8
        self.pybullet_client.changeDynamics(self.planeUid, -1, lateralFriction=self.lateral_friction)
        if self.lateral_friction >= 0.8:
            friction_level = 2
        elif self.lateral_friction <= 0.2:
            friction_level = 0
        else:
            friction_level = 1
        texUid = self.pybullet_client.loadTexture(currentdir + "/assets/"+["checker_red", "checker_purple", "striped"][friction_level]+".png")
        self.pybullet_client.changeVisualShape(self.planeUid, -1, textureUniqueId=texUid)

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

    def from_motor_to_effector_position(self, joints):
        sin = np.sin
        cos = np.cos
        l0, l1, l2 = [0.1, 0.2, 0.15]
        x0, y0, z0 = [-0.1, 0.03, 0]
        [o0, o1, o2] = joints
        x = x0 - l1*sin(o1) - l2*sin(o1+o2)
        y = y0 - l0*cos(o0) + sin(o0)*(l1*cos(o1) + l2*cos(o1+o2))
        z = z0 + l0*sin(o0) + cos(o0)*(l1*cos(o1) + l2*cos(o1+o2))
        return [x, y, z]

    def from_motor_to_leg(self, motor):
        """ leg is [abduction, swing, extension] * 4 """
        l1 = self.shoulder_to_knee
        l2 = self.knee_to_foot
        leg = np.zeros(12)
        for i in range(4):
            e = np.sqrt(l1**2+l2**2+2*l1*l2*np.cos(motor[3*i+2]))  # extension
            leg[3*i+2] = np.copy(e)
            leg[3*i+1] = motor[3*i+1] + np.arccos((l1**2+e**2-l2**2)/(2*l1*e))  # swing
            leg[3*i] = motor[3*i]  # abduction
        return leg

    def from_leg_to_motor(self, leg):
        """
        leg is [abduction, swing, extension] * 4
        e = extension in [0.1, 0.3]
        """
        l1 = self.shoulder_to_knee
        l2 = self.knee_to_foot
        motor = np.zeros(12)
        for i in range(4):
            motor[3*i] = leg[3*i]
            e = leg[3*i+2]
            motor[3*i+1] = leg[3*i+1] - np.arccos((l1**2+e**2-l2**2)/(2*l1*e))
            motor[3*i+2] = np.arccos((e**2-l1**2-l2**2)/(2*l1*l2))
        return motor


if __name__ == "__main__":
    import gym
    import fast_adaptation_embedding.env
    from tqdm import tqdm
    from gym.wrappers.monitoring.video_recorder import VideoRecorder
    import matplotlib.pyplot as plt
    import pickle
    render = True
    # render = False

    on_rack = 0

    # (init_joint, real_ub, real_lb) = np.array([0.2, 0.1, 0.25] * 4), np.array([0.548, 1.548, 2.59] * 4), np.array([-0.548, -2.666, -0.1] * 4)
    (init_joint, real_ub, real_lb) = None, None, None
    env = gym.make("SpotMicroEnv-v0",
                   render=render,
                   on_rack=on_rack,
                   action_space=["S&E", "Motor"][1],
                   init_joint=init_joint,
                   ub=real_ub,
                   lb=real_lb,
                   urdf_model='sphere'
    )

    env.set_mismatch([0.])
    init_obs = env.reset()

    recorder = None
    # recorder = VideoRecorder(env, "test.mp4")
    # env.metadata["video.frames_per_second"] = 12

    ub = 1
    lb = -1

    past = [(env.init_joint-(env.ub+env.lb)/2)*2/(env.ub-env.lb)]*3
    max_vel, max_acc, max_jerk = 10, 100, 10000
    dt = 0.02

    actions = None

    f = "/home/timothee/Documents/SpotMicro_team/exp/results/spot_micro_03/09_03_2020_12_03_21_experiment/run_0/logs.pk"
    with open(f, 'rb') as f:
        data = pickle.load(f)

    actions = Noneactions = data['actions'][0][-1]

    degree = 0
    for i in tqdm(range(3, 500+3)):
        if recorder is not None:
            recorder.capture_frame()
        if degree == 0:
            amax = [ub]*12
            amin = [lb]*12
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
        action = np.copy(x)
        if actions is not None:
            action = actions[i-3]
        obs, reward, done, info = env.step(action)
        past = np.append(past, [np.copy(x)], axis=0)
        if done:
            break
        time.sleep(0.02)

    if recorder is not None:
        recorder.capture_frame()
        recorder.close()
