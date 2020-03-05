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
                 ):

        # Simulation Configuration
        self.fixedTimeStep = 1. / 250  # 550
        self.action_repeat = 5
        self.numSolverIterations = int(NUM_SIMULATION_ITERATION_STEPS / self.action_repeat)
        self.useFixeBased = on_rack
        self.is_render = render
        self.init_oritentation = p.getQuaternionFromEuler([0, 0, 0])
        self.reinit_position = [0, 0, 0.3]
        self.init_position = [0, 0, 0.23]
        self.kp = 0.045
        self.kd = .4
        self.maxForce = 12.5
        self._motor_direction = [-1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1]
        self.mismatch = None
        self.shoulder_to_knee = 0.2
        self.knee_to_foot = 0.1
        self.init_joint = self.from_leg_to_motor([0, -0.2, 0.25]*4)
        self.fc = 10
        self.C = 1/(np.tan(np.pi*self.fc*self.fixedTimeStep))
        self.A = 1/(1+self.C)
        assert action_space in ["S&E", "Motor"], "Control mode not implemented yet"
        self.control_mode = action_space
        if self.is_render:
            self.pybullet_client = bullet_client.BulletClient(connection_mode=p.GUI)
        else:
            self.pybullet_client = bullet_client.BulletClient(connection_mode=p.DIRECT)
        self._objective_weights = [distance_weight, high_weight, roll_weight, pitch_weight, yaw_weight,
                                   action_weight, action_vel_weight, action_acc_weight, action_jerk_weight]
        self.desired_speed = desired_speed
        self.t = 0

        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 1)

        self.quadruped = self.loadModels()
        self._BuildJointNameToIdDict()
        self. _BuildMotorIdList()
        self.jointNameToId = self.getJointNames(self.quadruped)
        p.setPhysicsEngineParameter(numSolverIterations=self.numSolverIterations)
        p.resetDebugVisualizerCamera(1, 85.6, 0, [-0.61, 0.12, 0.25])

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
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.setGravity(0, 0, -9.81)

        orn = p.getQuaternionFromEuler([math.pi / 30 * 0, 0 * math.pi / 50, 0])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeUid = p.loadURDF("plane_transparent.urdf", [0, 0, 0], orn)
        p.changeDynamics(planeUid, -1, lateralFriction=1)
        texUid = p.loadTexture(currentdir + "/assets/images/concrete.png")
        p.changeVisualShape(planeUid, -1, textureUniqueId=texUid)

        flags = p.URDF_USE_SELF_COLLISION
        quadruped = p.loadURDF(currentdir + "/assets/urdf/spotmicroai_gen.urdf.xml", self.init_position,
                               self.init_oritentation,
                               useFixedBase=self.useFixeBased,
                               useMaximalCoordinates=False,
                               flags=flags)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.changeDynamics(quadruped, -1, lateralFriction=0.8)

        return quadruped

    def reset(self):
        if self.useFixeBased:
            init_pos = [0, 0, 0.5]
            reinit_pos = [0, 0, 0.5]
        else:
            init_pos = [0, 0, 0.22]
            reinit_pos = [0, 0, 0.5]
        p.resetBasePositionAndOrientation(self.quadruped, reinit_pos, self.init_oritentation)
        p.resetBaseVelocity(self.quadruped, [0, 0, 0], [0, 0, 0])
        for _ in range(40):
            self.apply_action(self.init_joint)
        p.resetBasePositionAndOrientation(self.quadruped, init_pos, self.init_oritentation)
        p.resetBaseVelocity(self.quadruped, [0, 0, 0], [0, 0, 0])
        self.t = 0
        self._past_velocity = [[0]*12]*2
        return self.get_obs()

    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])
        bodyPos, _ = p.getBasePositionAndOrientation(self.quadruped)
        base_pos = bodyPos
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=-0.5,
            yaw=0,
            pitch=-80,
            roll=0,
            upAxisIndex=1)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60,
                                                   aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
                                                   nearVal=0.1,
                                                   farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(
            width=RENDER_WIDTH,
            height=RENDER_HEIGHT,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def changeDynamics(self, quadruped):
        nJoints = p.getNumJoints(quadruped)
        for i in range(nJoints):
            p.changeDynamics(quadruped, i, localInertiaDiagonal=[0.000001, 0.000001, 0.000001])

    def getJointNames(self, quadruped):
        nJoints = p.getNumJoints(quadruped)
        jointNameToId = {}

        for i in range(nJoints):
            jointInfo = p.getJointInfo(quadruped, i)
            jointNameToId[jointInfo[1].decode('UTF-8')] = jointInfo[0]
        return jointNameToId

    def getPos(self):
        bodyPos, _ = p.getBasePositionAndOrientation(self.quadruped)
        return bodyPos

    def getIMU(self):
        _, bodyOrn = p.getBasePositionAndOrientation(self.quadruped)
        linearVel, angularVel = p.getBaseVelocity(self.quadruped)
        return bodyOrn, linearVel, angularVel

    def step(self, action):

        a = np.copy(action)
        if self.control_mode == "Motor":
            ub = np.array([0.2, -0.4, 1.5] * 4)  # [0.548, 1.548, 2.59]
            lb = np.array([-0.2, -0.8, 1] * 4)  # [-0.548, -2.666, -0.1]
            a = a*(ub-lb)/2+(ub+lb)/2
        elif self.control_mode == "S&E":
            """ abduction - swing - extension"""
            ub = np.array([0.2, 0.4, 0.25] * 4)  # [0.2, 0.4, 0.25]
            lb = np.array([-0.2, -0.6, 0.2] * 4)  # [-0.2, -0.6, 0.2]
            a = a*(ub-lb)/2+(ub+lb)/2
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
        _, bodyOrn = p.getBasePositionAndOrientation(self.quadruped)
        return self.checkSimulationReset(bodyOrn)

    def checkSimulationReset(self, bodyOrn):
        (xr, yr, _) = p.getEulerFromQuaternion(bodyOrn)
        return abs(xr) > math.pi / 3 or abs(yr) > math.pi / 3

    def apply_action(self, action):
        self.t = self.t + self.fixedTimeStep
        action = np.multiply(action, self._motor_direction)
        for lx, leg in enumerate(['front_left', 'front_right', 'rear_left', 'rear_right']):
            for px, part in enumerate(['shoulder', 'leg', 'foot']):
                j = self.jointNameToId[leg + "_" + part]
                p.setJointMotorControl2(bodyIndex=self.quadruped,
                                        jointIndex=j,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=action[lx * 3 + px],
                                        positionGain=self.kp,
                                        velocityGain=self.kd,
                                        force=self.maxForce)

        p.stepSimulation()

    def _filter_velocities(self, x):
        y = self.A * (x + self._past_velocity[0]) + (1-self.C)*self.A*self._past_velocity[1]
        self._past_velocity = np.copy([x, y])
        return y

    def get_obs(self):
        Obs = []
        bodyPos, bodyOrn = p.getBasePositionAndOrientation(self.quadruped)
        linearVel, angularVel = p.getBaseVelocity(self.quadruped)
        Obs.extend(self.GetMotorAngles().tolist())
        Obs.extend(self.GetMotorVelocities().tolist())
        Obs.extend(p.getEulerFromQuaternion(bodyOrn))
        Obs.extend(angularVel)
        Obs.extend(linearVel[:2])  # x and y velocity
        Obs.extend(bodyPos[2:3])  # z height
        return Obs

    def get_reward(self):
        obs = self.get_obs()
        distance_reward = -(obs[-3] - self.desired_speed) ** 2
        high_reward = -(obs[-1] - 0.17) ** 2
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

    def set_mismatch(self, mismatch):
        self.mismatch = mismatch

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

    render = True
    # render = False

    on_rack = 0

    env = gym.make("SpotMicroEnv-v0",
                   render=render,
                   on_rack=on_rack,
                   action_space=["S&E", "Motor"][1])
    init_obs = env.reset()

    recorder = None
    # recorder = VideoRecorder(env, "test.mp4")
    ub = 1
    lb = -1

    past = np.array([[0, -0.2, 1]*4]*3) if env.action_space == ["S&E"] else np.array([[0, 0, 0]*4]*3)
    max_vel, max_acc, max_jerk = 10, 100, 10000
    dt = 0.02

    degree = 0
    for i in tqdm(range(3, 500+3)):
        if recorder is not None:
            recorder.capture_frame()

        if degree == 0:
            amax = [ub]*12
            amin = [lb]*12
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
        # action = np.copy(x)
        action = past[-1]
        obs, reward, done, info = env.step(action)
        past = np.append(past, [np.copy(x)], axis=0)
        if done:
            break
        time.sleep(0.02)

    if recorder is not None:
        recorder.capture_frame()
        recorder.close()
