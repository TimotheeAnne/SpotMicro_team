import gym
from gym import spaces
import pybullet as p
import numpy as np
from os import path
from enum import IntEnum, unique
from fast_adaptation_embedding.env.assets.pybullet_envs import bullet_client
import matplotlib.pyplot as plt
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)



@unique
class Joints(IntEnum):
    LF_HAA = 1
    LF_HFE = 2
    LF_KFE = 3
    RF_HAA = 7
    RF_HFE = 8
    RF_KFE = 9
    LH_HAA = 13
    LH_HFE = 14
    LH_KFE = 15
    RH_HAA = 19
    RH_HFE = 20
    RH_KFE = 21


@unique
class FootLinks(IntEnum):
    LF_FOOT = 6
    RF_FOOT = 12
    LH_FOOT = 18
    RH_FOOT = 24


SIMULATIONFREQUENCY = 500
PGAIN_ORIGIN = 65
DGAIN_ORIGIN = 2
MAXTORQUE = 25.0

JOINT_POSITIONS = [0.1, 0.7, -1.2,
                   -0.1, 0.7, -1.2,
                   0.1, -0.7, 1.2,
                   -0.1, -0.7, 1.2]

BASE_ORIENTATION = np.array([0.0, 0.0, 0.0, 1.0])
BASE_ORIENTATION /= np.linalg.norm(BASE_ORIENTATION)

BASE_POSITION = [0, 0, 0.33]
BASE_ON_RACK_POSITION = [0, 0, 1]

REST_JOINT_POSITIONS = [-0.05, 1.45, -2.65,
                        0.05, 1.45, -2.65,
                        -0.05, -1.45, 2.65,
                        0.05, -1.45, 2.65]

EAGLE_JOINT_POSITIONS = [0.0, -1.6, 0.0,
                         0.0, -1.6, 0.0,
                         0.0, 1.6, 0.0,
                         0.0, 1.6, 0.0]


class ANYmalEnv(gym.Env):
    def __init__(self,
                 on_rack=False,
                 render=False,
                 ctrl_frequency=50,
                 ):

        self.on_rack = on_rack
        self.render = render
        self.ctrl_frequency = ctrl_frequency
        self.action_repeat = int(SIMULATIONFREQUENCY/self.ctrl_frequency)
        self.Kp = PGAIN_ORIGIN
        self.Kd = DGAIN_ORIGIN

        if self.render:
            self.pybullet_client = bullet_client.BulletClient(connection_mode=p.GUI)
        else:
            self.pybullet_client = bullet_client.BulletClient()

        self.action_space = spaces.Box(
            low=-2 * np.pi * np.ones(12),
            high=2 * np.pi * np.ones(12)
        )

        maxPosition = np.pi * np.ones(12)
        maxVelocity = 2 * np.ones(12)
        maxBasePositionAndOrientation = np.ones(7)
        maxBaseVelocity = 1 * np.ones(3)
        maxBaseAngularVelocity = 2 * np.pi * np.ones(3)
        observationUpperBound = np.concatenate([
            maxPosition,
            maxVelocity,
            maxBasePositionAndOrientation,
            maxBaseVelocity,
            maxBaseAngularVelocity
        ])
        self.observation_space = spaces.Box(
            low=-observationUpperBound,
            high=observationUpperBound
        )

        self.t = 0.0
        self.lastVelocity = np.zeros(12)

    def reset(self):

        if self.on_rack:
            init_position = BASE_ON_RACK_POSITION
        else:
            init_position = BASE_POSITION

        p.setTimeStep(1.0 / SIMULATIONFREQUENCY)
        p.setGravity(0, 0, -9.81)
        self.ground = p.loadURDF(
            path.join(path.dirname(__file__), "assets/plane.urdf"),
            [0, 0, 0]
        )
        self.anymal = p.loadURDF(
            "/assets/anymal_bedi_urdf/anymal.urdf",
            init_position,
            useFixedBase=self.on_rack
        )

        p.resetBasePositionAndOrientation(
            self.anymal,
            init_position,
            BASE_ORIENTATION
        )

        jointNum = 0
        for joint in Joints:
            initPositionNoise = np.random.uniform(0, 0)
            positionTarget = JOINT_POSITIONS[jointNum]
            p.resetJointState(self.anymal, joint.value, positionTarget + initPositionNoise, 0.0)
            jointNum += 1

        observation, _ = self._getObservation()
        self.t = 0.0

        p.setJointMotorControlArray(
            self.anymal,
            [joint.value for joint in Joints],
            p.POSITION_CONTROL,
            targetPositions=np.zeros(12),
            forces=np.zeros(12)
        )

        return observation

    def step(self, action):
        for _ in range(self.action_repeat):
            self.apply_action(action)
        observation, observationAsDict = self._getObservation()
        reward = 0
        info = {}
        done = 0
        return observation, reward, done, info

    def apply_action(self, action):
        _, measurement = self._getObservation()
        PD_torque = self.Kp * (action - measurement["position"]) - self.Kd * measurement["velocity"]
        joint_torque = np.clip(PD_torque, -MAXTORQUE, MAXTORQUE)

        p.setJointMotorControlArray(
            self.anymal,
            [j.value for j in Joints],
            p.TORQUE_CONTROL,
            forces=joint_torque)
        p.stepSimulation()
        self.t += 1/SIMULATIONFREQUENCY


    def render(self, mode="rgb"):
        pass

    def _getObservation(self):
        allJoints = [j.value for j in Joints]
        jointStates = p.getJointStates(self.anymal, allJoints)
        position = np.array([js[0] for js in jointStates])
        velocity = np.array([js[1] for js in jointStates])
        acceleration = (velocity - self.lastVelocity) * SIMULATIONFREQUENCY
        self.lastVelocity = np.copy(velocity)
        basePosition, baseOrientation = p.getBasePositionAndOrientation(self.anymal)
        baseVelocity, baseAngularVelocity = p.getBaseVelocity(self.anymal)
        observationAsArray = np.concatenate([
            position,
            velocity,
            basePosition,
            baseOrientation,
            baseVelocity,
            baseAngularVelocity,
            acceleration
        ])
        observationAsDict = {
            "position": position,
            "velocity": velocity,
            "basePosition": basePosition,
            "baseOrientation": baseOrientation,
            "baseVelocity": baseVelocity,
            "baseAngularVelocity": baseAngularVelocity,
            "acceleration": acceleration
        }
        return observationAsArray, observationAsDict

    def close(self):
        p.disconnect()


def from_q_to_end_effector(q):
    [a0, a1, a2] = q
    beta = a1
    gamma = a2 - np.pi/2
    cos = np.cos
    sin = np.sin
    [l1, l2, l3, l4, l5] = [0.1, 0.3, 0.1, 0.15, 0.3]
    xf = l2*cos(beta)+l4*cos(beta+gamma)-l5*sin(beta+gamma)
    yf = l2*sin(beta)+l4*sin(beta+gamma)+l5*cos(beta+gamma)
    xk = l2*cos(beta)
    yk = l2*sin(beta)
    return xf, yf, xk, yk


if __name__ == "__main__":
    import time
    import fast_adaptation_embedding.env

    render = True

    ctrl_frequency = 50
    env = gym.make("ANYmalEnv-v0", on_rack=1, render=render, ctrl_frequency=ctrl_frequency)
    IDa0 = env.pybullet_client.addUserDebugParameter("a0", -np.pi, np.pi, 0)
    IDa1 = env.pybullet_client.addUserDebugParameter("a1", -np.pi, np.pi, 0)
    IDa2 = env.pybullet_client.addUserDebugParameter("a2", -np.pi, np.pi, 0)

    IDa = env.pybullet_client.addUserDebugParameter("a", -np.pi, np.pi, 0)
    IDb = env.pybullet_client.addUserDebugParameter("b", -np.pi, np.pi, 0)
    IDc = env.pybullet_client.addUserDebugParameter("c", -np.pi, np.pi, 0)

    ub = np.pi
    lb = -ub

    prev_obs = env.reset()
    for n in range(100000*ctrl_frequency):
        a0 = env.pybullet_client.readUserDebugParameter(IDa0)
        a1 = env.pybullet_client.readUserDebugParameter(IDa1)
        a2 = env.pybullet_client.readUserDebugParameter(IDa2)

        a = env.pybullet_client.readUserDebugParameter(IDa)
        b = env.pybullet_client.readUserDebugParameter(IDb)
        c = env.pybullet_client.readUserDebugParameter(IDc)

        xf, yf, xk, yk = from_q_to_end_effector([a0, a1, a2])
        base_pos = prev_obs[24:27]
        env.pybullet_client.addUserDebugText(".KNEE", [base_pos[0]+0.3-yk, base_pos[1]+0.2, base_pos[2]+0.05 - xk], [0, 0, 0],lifeTime=0.1)
        env.pybullet_client.addUserDebugText(".FOOT", [base_pos[0]+0.3-yf, base_pos[1]+0.2, base_pos[2]+0.05-xf], [0, 0, 0], lifeTime=0.1)
        # action = np.random.uniform(lb, ub, 12)
        action = [a0, a1, a2] + [0]*9
        observation, reward, done, measurement = env.step(action)
        time.sleep(1/ctrl_frequency)
        if done:
            break
    env.close()
