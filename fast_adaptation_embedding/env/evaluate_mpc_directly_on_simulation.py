import gym
import time
import pickle
from tqdm import tqdm, trange
import numpy as np
import fast_adaptation_embedding.env
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from pybullet_envs.bullet import minitaur_gym_env


def sample_candidates(past, N, H, action_dimension, max_vel, max_acc, max_jerk, dt):
    candidates = np.zeros((N, H+3, action_dimension))
    candidates[:, :3] = np.array(list([past])*N)
    for t in range(3, 3+H):
        amax = np.min((candidates[:, t-1] + max_vel * dt,
                       2 * candidates[:, t-1] - candidates[:, t-2] + max_acc * dt ** 2,
                       3 * candidates[:, t-1] - 3 * candidates[:, t-2] + candidates[:, t-3] + max_jerk * dt ** 3),
                      axis=0)
        amin = np.max((candidates[:, t-1] - max_vel * dt,
                       2 * candidates[:, t-1] - candidates[:, t-2] - max_acc * dt ** 2,
                       3 * candidates[:, t-1] - 3 * candidates[:, t-2] + candidates[:, t-3] - max_jerk * dt ** 3),
                      axis=0)
        amax, amin = np.clip(amax, -1, 1), np.clip(amin, -1, 1)
        candidates[:, t] = np.random.uniform(amin, amax, size=(N, action_dimension))
    return candidates[:, 3:]


def evaluate_candidates(candidates, A):
    return np.random.random(len(candidates))


if __name__ == "__main__":
    # render = False
    render = True
    N = 100
    H = 25
    action_dimension = 8
    max_vel, max_acc, max_jerk = 10, 100, 10000
    dt = ctrl_time_step = 1 / 50

    config = {
        "render": render,
        "on_rack": 0,
        "control_time_step": ctrl_time_step,
        "action_repeat": int(250 * ctrl_time_step),
        "accurate_motor_model_enabled": 1,
        "pd_control_enabled": 1,
        "partial_torque_control": False,
        "vkp": 20,
        "env_randomizer": None,
        "action_space": 'S&E'
    }

    env = gym.make("MinitaurTorqueEnv_fastAdapt-v0", **config)
    env.metadata['video.frames_per_second'] = 1 / ctrl_time_step

    recorder = None
    # recorder = VideoRecorder(env, "test.mp4")

    S, A = [], []
    distance = 0
    S.append(env.reset())
    A = [np.zeros(8) for _ in range(3)]

    tbar = trange(500, desc='', leave=True)
    for t in tbar:
        """ Sample action candidate """
        candidates = sample_candidates(A[-3:], N, H, action_dimension, max_vel, max_acc, max_jerk, dt)

        """ Evaluate each candidate """
        evaluations = evaluate_candidates(candidates, A)

        """ Select best candidates """
        best_candidate = candidates[np.argmax(evaluations)]

        """ apply the best candidate """
        action = best_candidate[0]
        current_state, _, _, _ = env.step(action)
        A.append(np.copy(action))
        S.append(np.copy(current_state))
        distance += ctrl_time_step * current_state[-2]

        tbar.set_description(" Distance " + "{:1.2f}".format(distance) + "m")

