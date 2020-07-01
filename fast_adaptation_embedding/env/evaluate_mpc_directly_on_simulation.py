import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
import time
import pickle
from tqdm import tqdm, trange
import numpy as np
import fast_adaptation_embedding.env
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from pybullet_envs.bullet import minitaur_gym_env
from datetime import datetime
import json


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


def reward(s, desired_speed, weights):
    distance_reward = np.exp(-(s[-2] - desired_speed) ** 2)
    roll_reward = np.exp(-(s[16]) ** 2)
    pitch_reward = np.exp(-(s[17]) ** 2)
    yaw_reward = np.exp(-(s[18]) ** 2)
    r = distance_reward*weights[0] + roll_reward*weights[1] + pitch_reward*weights[2] + yaw_reward*weights[3]
    return r


def evaluate_candidates(candidates, A, config, desired_speed, reward_weights):
    config['render'] = False
    evaluations = []
    for i in range(len(candidates)):
        eval_env = gym.make("MinitaurTorqueEnv_fastAdapt-v0", **config)
        eval_env.reset()
        evalutation = 0
        for action in A:
            eval_env.step(action)
        for action in candidates[i]:
            s, _, _, _ = eval_env.step(action)
            evalutation += reward(s, desired_speed, reward_weights)
        evaluations.append(evalutation)
    return evaluations


def add_comments_to_video(texts, distance, path, video_name, dt=0.02):
    steps = len(texts)
    """ create subtitle """
    comments = ""
    for i in range(steps):
        comments += str(i) + '\n'
        s = int(dt * i)
        s1 = '0' + str(s) if s < 10 else str(s)
        ms = (i % int(1 / dt)) * int(dt * 1000)
        if ms < 1:
            ms1 = '000'
        elif ms < 10:
            ms1 = '00' + str(ms)
        elif ms < 100:
            ms1 = '0' + str(ms)
        else:
            ms1 = str(ms)

        s = int((i + 1) * dt)
        s2 = '0' + str(s) if s < 10 else str(s)
        ms = ((i + 1) % int(1 / dt)) * int(dt * 1000)
        if ms < 1:
            ms2 = '000'
        elif ms < 10:
            ms2 = '00' + str(ms)
        elif ms < 100:
            ms2 = '0' + str(ms)
        else:
            ms2 = str(ms)

        comments += "00:00:" + s1 + "," + ms1 + " --> 00:00:" + s2 + "," + ms2 + "\n"
        comments += texts[i] + "\n\n"
    with open(path+'/subtitles.srt', 'w') as f:
        f.write(comments)
    os.system('ffmpeg -y -i ' + path + '/subtitles.srt ' + path + '/subtitles.ass')
    """ add subtitle to mp4"""
    os.system('ffmpeg -y -i ' + path + "/" + video_name + '.mp4 -vf ass=' + path + '/subtitles.ass ' + path + "/" + video_name + '_' + "{:1.2f}".format(distance) + 'm.mp4')


if __name__ == "__main__":
    render = False
    # render = True

    config = {
        "n_run": 12,
        "episode_length": 500,

        "N": '10**(run//3)',
        "H": 25,
        'action_dimension': 8,
        "max_vel": 10,
        "max_acc": 100,
        "max_jerk": 1000,
        "dt": 1 / 50,
        "desired_speed": 0.5,
        "reward_weights": [1, 1, 1, 1]
    }

    env_config = {
        "render": render,
        "on_rack": 0,
        "control_time_step": config['dt'],
        "action_repeat": int(250 * config['dt']),
        "accurate_motor_model_enabled": 1,
        "pd_control_enabled": 1,
        "partial_torque_control": False,
        "vkp": 20,
        "env_randomizer": None,
        "action_space": 'S&E'
    }

    now = datetime.now()
    timestamp = now.strftime("%d_%m_%Y_%H_%M_%S")
    folder_path = "results/"+"eval_MPC_on_simulation_"+timestamp
    os.makedirs(folder_path)

    with open(folder_path + '/config.json', 'w') as fp:
        json.dump(config, fp)

    with open(folder_path + '/env_config.json', 'w') as fp:
        json.dump(env_config, fp)

    for run in range(config["n_run"]):
        config['N'] = int(10**(run//3))

        env = gym.make("MinitaurTorqueEnv_fastAdapt-v0", **env_config)
        env.metadata['video.frames_per_second'] = 1 / config['dt']

        run_path = folder_path + "/run_" + str(run)
        os.makedirs(run_path)

        # recorder = None
        video_name = "eval_MPC"
        recorder = VideoRecorder(env, run_path+"/"+video_name+".mp4")

        S, A = [], []
        distance = 0
        distance_text = [" Distance " + "{:1.2f}".format(distance) + "m"]
        S.append(env.reset())
        A = [np.zeros(config["action_dimension"]) for _ in range(3)]

        tbar = trange(config["episode_length"], desc='', leave=True)
        for t in tbar:
            """ Sample action candidate """
            candidates = sample_candidates(A[-3:], config["N"], config["H"], config["action_dimension"], config["max_vel"],
                                           config["max_acc"], config["max_jerk"], config['dt'])

            """ Evaluate each candidate """
            evaluations = evaluate_candidates(candidates, A[3:], env_config,
                                              config["desired_speed"], config["reward_weights"])

            """ Select best candidates """
            best_candidate = candidates[np.argmax(evaluations)]

            """ apply the best candidate """
            if recorder is not None:
                recorder.capture_frame()
            action = best_candidate[0]
            current_state, _, _, _ = env.step(action)
            A.append(np.copy(action))
            S.append(np.copy(current_state))
            distance += config['dt'] * current_state[-2]
            distance_text.append(" Distance " + "{:1.2f}".format(distance) + "m")
            tbar.set_description(distance_text[-1])

        if recorder is not None:
            recorder.capture_frame()
            recorder.close()
            add_comments_to_video(distance_text, distance, run_path, video_name)

        logs = {
            "actions": A,
            "states": S,
        }

        with open(run_path + '/logs.pk', 'wb') as f:
            pickle.dump(logs, f)
