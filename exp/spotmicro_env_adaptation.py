import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import fast_adaptation_embedding.env
from fast_adaptation_embedding.models.ffnn import FFNN_Ensemble_Model, load_model
from fast_adaptation_embedding.controllers.cem import CEM_opt
from fast_adaptation_embedding.controllers.random_shooting import RS_opt
import torch
import numpy as np
import copy
import gym
import time
from datetime import datetime
import pickle
import os
from os import path
import argparse
from utils import ProgBar
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from tqdm import tqdm, trange
from pynput.keyboard import Key, Listener


class Cost(object):
    def __init__(self, ensemble_model, init_state, horizon, action_dim, goal, pred_high, pred_low, config, last_action,
                 speed=None):
        self.__ensemble_model = ensemble_model
        self.__init_state = init_state
        self.__horizon = horizon
        self.__action_dim = action_dim
        self.__goal = goal
        self.__models = self.__ensemble_model.get_models()
        self.__pred_high = pred_high
        self.__pred_low = pred_low
        self.__obs_dim = len(init_state)
        self.__discount = config['discount']
        self.__batch_size = config['ensemble_batch_size']
        self.__xreward = config['xreward']
        self.__zreward = config['zreward']
        self.__rollreward = config['rollreward']
        self.__pitchreward = config['pitchreward']
        self.__yawreward = config['yawreward']
        self.__desired_speed = config['desired_speed'] if speed is None else speed
        self.__last_action = last_action
        self.__action_norm_w = config['action_norm_weight']
        self.__action_vel_w = config['action_vel_weight']
        self.__action_acc_w = config['action_acc_weight']
        self.__action_jerk_w = config['action_jerk_weight']
        self.__soft_smoothing = config['soft_smoothing']
        self.__hard_smoothing = config['hard_smoothing']
        self.__max_iters = config["max_iters"]  # 20
        self.__lb, self.__ub = config["lb"], config["ub"]  # -1, 1
        self.__popsize = config["popsize"]  # 200
        self.__max_vel = config["max_action_velocity"]
        self.__max_acc = config["max_action_acceleration"]
        self.__max_jerk = config["max_action_jerk"]
        self.__max_torque_jerk = config["max_torque_jerk"]
        self.__sol_dim = config["sol_dim"]  # 2*10 #action dim*horizon
        self.__action_space = config['action_space']
        self.__ctrl_time_step = config['ctrl_time_step']

    def transform_action_to_motor_command(self, actions):
        motor_angle = torch.zeros([self.__popsize, self.__action_dim]).cuda()
        scale_for_singularity = 1
        offset_for_singularity = 1.5
        num_motors = 8
        half_num_motors = int(num_motors / 2)
        quater_pi = np.pi / 4
        for i in range(num_motors):
            action_idx = int(i // 2)
            forward_backward_component = (-scale_for_singularity * quater_pi *
                                          (actions[:, action_idx + half_num_motors] + offset_for_singularity))
            extension_component = (-1) ** i * quater_pi * actions[:, action_idx]
            if i >= half_num_motors:
                extension_component = -extension_component
            motor_angle[:, i] = (np.pi + forward_backward_component + extension_component)
        return motor_angle

    def cost_fn(self, samples):
        a = torch.FloatTensor(samples).cuda() \
            if self.__ensemble_model.CUDA \
            else torch.FloatTensor(samples)
        # compute the samples for hard smoothing
        ad = self.__action_dim
        if self.__hard_smoothing:
            for t in range(3, int(self.__sol_dim / ad) + 3):
                amax = torch.min(a[:, (t - 1) * ad:t * ad] + self.__max_vel,
                                 2 * a[:, (t - 1) * ad:t * ad] - a[:, (t - 2) * ad:(t - 1) * ad] + self.__max_acc,
                                 out=None)
                amax = torch.min(amax,
                                 3 * a[:, (t - 1) * ad:t * ad] - 3 * a[:, (t - 2) * ad:(t - 1) * ad] + a[:, (
                                                                                                                    t - 3) * ad:(
                                                                                                                                        t - 2) * ad] + self.__max_jerk,
                                 out=None)
                amin = torch.max(a[:, (t - 1) * ad:t * ad] - self.__max_vel,
                                 2 * a[:, (t - 1) * ad:t * ad] - a[:, (t - 2) * ad:(t - 1) * ad] - self.__max_acc,
                                 out=None)
                amin = torch.max(amin,
                                 3 * a[:, (t - 1) * ad:t * ad] - 3 * a[:, (t - 2) * ad:(t - 1) * ad] + a[:, (
                                                                                                                    t - 3) * ad:(
                                                                                                                                        t - 2) * ad] - self.__max_jerk,
                                 out=None)
                amax, amin = torch.clamp(amax, self.__lb, self.__ub), torch.clamp(amin, self.__lb,
                                                                                  self.__ub)
                if torch.sum(amax > amin) != self.__popsize * ad:
                    amax = torch.max(amin, amax, out=None)
                    amin = torch.min(amin, amax, out=None)
                m = torch.distributions.uniform.Uniform(amin - 0.001, amax + 0.001,
                                                        (self.__max_iters * self.__popsize, ad))
                a[:, t * ad:(t + 1) * ad] = m.sample()
        # evaluate the samples
        init_states = torch.FloatTensor(np.repeat([self.__init_state], len(samples), axis=0)).cuda() \
            if self.__ensemble_model.CUDA \
            else torch.FloatTensor(np.repeat([self.__init_state], len(samples), axis=0))
        all_costs = torch.FloatTensor(np.zeros(len(samples))).cuda() \
            if self.__ensemble_model.CUDA \
            else torch.FloatTensor(np.zeros(len(samples)))

        n_batch = max(1, int(len(samples) / self.__batch_size))
        per_batch = len(samples) / n_batch
        index_offset = int(len(samples[0]) / self.__action_dim) - self.__horizon
        for i in range(n_batch):
            start_index = int(i * per_batch)
            end_index = len(samples) if i == n_batch - 1 else int(i * per_batch + per_batch)
            action_batch = a[start_index:end_index]
            start_states = init_states[start_index:end_index]
            dyn_model = self.__models[np.random.randint(0, len(self.__models))]
            for h in range(index_offset, self.__horizon + index_offset):
                actions = action_batch[:, h * self.__action_dim: h * self.__action_dim + self.__action_dim].clone()
                model_input = torch.cat((start_states, actions), dim=1)
                diff_state = dyn_model.predict_tensor(model_input)
                start_states += diff_state
                x_vel_cost = (start_states[:, -3] - self.__desired_speed) ** 2 * self.__xreward
                z_cost = (start_states[:, -1] - 0.21) ** 2 * self.__zreward
                roll_cost = (start_states[:, 24]) ** 2 * self.__rollreward
                pitch_cost = (start_states[:, 25]) ** 2 * self.__pitchreward
                yaw_cost = (start_states[:, 26]) ** 2 * self.__yawreward

                if self.__soft_smoothing:
                    action_norm_cost = torch.sum(-torch.exp(-self.__action_norm_w * (
                    action_batch[:, h * self.__action_dim: (h + 1) * self.__action_dim]) ** 2))
                    all_costs[start_index: end_index] += action_norm_cost * self.__discount ** h
                    action_vel_cost = -torch.exp(-self.__action_vel_w * (
                                action_batch[:, h * self.__action_dim: (h + 1) * self.__action_dim] - action_batch[:, (
                                                                                                                                  h - 1) * self.__action_dim: h * self.__action_dim]) ** 2)
                    all_costs[start_index: end_index] += torch.sum(action_vel_cost, axis=1) * self.__discount ** h
                    action_acc_cost = -torch.exp(
                        -self.__action_acc_w * (action_batch[:, h * self.__action_dim: (h + 1) * self.__action_dim]
                                                - 2 * action_batch[:,
                                                      (h - 1) * self.__action_dim: h * self.__action_dim]
                                                + action_batch[:,
                                                  (h - 2) * self.__action_dim: (h - 1) * self.__action_dim]) ** 2)
                    all_costs[start_index: end_index] += torch.sum(action_acc_cost, axis=1) * self.__discount ** h
                    action_jerk_cost = -torch.exp(
                        -self.__action_jerk_w * (action_batch[:, h * self.__action_dim: (h + 1) * self.__action_dim]
                                                 - 3 * action_batch[:,
                                                       (h - 1) * self.__action_dim: h * self.__action_dim]
                                                 + 3 * action_batch[:,
                                                       (h - 2) * self.__action_dim: (h - 1) * self.__action_dim]
                                                 - action_batch[:,
                                                   (h - 3) * self.__action_dim: (h - 2) * self.__action_dim]) ** 2)
                    all_costs[start_index: end_index] += torch.sum(action_jerk_cost, axis=1) * self.__discount ** h

                # all_costs[start_index: end_index] += x_vel_cost * self.__discount ** h \
                #                                      + yaw_cost * self.__discount ** h \
                #                                      + pitch_cost * self.__discount ** h \
                #                                      + roll_cost * self.__discount ** h \
                #                                      + z_cost * self.__discount ** h
                all_costs[start_index: end_index] += -torch.exp(-x_vel_cost) * self.__discount ** h \
                                                     + -torch.exp(-yaw_cost) * self.__discount ** h \
                                                     + -torch.exp(-pitch_cost) * self.__discount ** h \
                                                     + -torch.exp(-roll_cost) * self.__discount ** h \
                                                     + -torch.exp(-z_cost) * self.__discount ** h
                # all_costs[start_index: end_index] += -torch.exp(-x_vel_cost) \
                #                                      * torch.exp(-yaw_cost)  \
                #                                      * torch.exp(-pitch_cost)  \
                #                                      * torch.exp(-roll_cost)  \
                #                                      * torch.exp(-z_cost) * self.__discount ** h
        if self.__hard_smoothing:
            return a[torch.argmin(all_costs)].cpu().detach().numpy()
        else:
            return all_costs.cpu().detach().numpy()


def train_ensemble_model(train_in, train_out, sampling_size, config, model=None):
    network = model
    if network is None:
        network = FFNN_Ensemble_Model(dim_in=config["ensemble_dim_in"],
                                      hidden=config["ensemble_hidden"],
                                      hidden_activation=config["hidden_activation"],
                                      dim_out=config["ensemble_dim_out"],
                                      CUDA=config["ensemble_cuda"],
                                      SEED=config["ensemble_seed"],
                                      output_limit=config["ensemble_output_limit"],
                                      dropout=config["ensemble_dropout"],
                                      n_ensembles=config["n_ensembles"])
    network.train(epochs=config["ensemble_epoch"], training_inputs=train_in, training_targets=train_out,
                  batch_size=config["ensemble_batch_size"], logInterval=config["ensemble_log_interval"],
                  sampling_size=sampling_size)
    return copy.deepcopy(network)


def process_data(data):
    """Assuming dada: an array containing [state, action, state_transition, cost] """
    training_in = []
    training_out = []
    for d in data:
        s = d[0]
        a = d[1]
        training_in.append(np.concatenate((s, a)))
        training_out.append(d[2])
    return np.array(training_in), np.array(training_out), np.max(training_in, axis=0), np.min(training_in, axis=0)


def execute_random(env, steps, init_state, K, index_iter, res_dir, samples, config):
    current_state = env.reset()
    max_vel, max_acc, max_jerk, max_torque_jerk = config['max_action_velocity'], config['max_action_acceleration'], \
                                                  config[
                                                      'max_action_jerk'], config['max_torque_jerk']
    lb, ub = config['lb'], config['ub']
    trajectory = []
    traject_cost = 0
    recorder = None
    # recorder = VideoRecorder(env, config['logdir'] + "/videos/random_"+str(index_iter)+".mp4")
    obs, acs, reward, rewards, desired_torques, observed_torques = [current_state], [], [], [], [], []
    past = np.zeros((3, 12))
    for i in range(3, steps + 3):
        if config['hard_smoothing']:
            amax = np.min((past[i - 1] + max_vel, 2 * past[i - 1] - past[i - 2] + max_acc,
                           3 * past[i - 1] - 3 * past[i - 2] + past[i - 3] + max_jerk), axis=0)
            amin = np.max((past[i - 1] - max_vel, 2 * past[i - 1] - past[i - 2] - max_acc,
                           3 * past[i - 1] - 3 * past[i - 2] + past[i - 3] - max_jerk), axis=0)
            amax, amin = np.clip(amax, lb, ub), np.clip(amin, lb, ub)
            x = np.random.uniform(amin, amax)
            a = np.copy(x)
        else:
            a = np.random.uniform(lb, ub, 12)
        next_state, r = 0, 0
        for k in range(K):
            next_state, rew, done, info = env.step(a)
            r += rew
            if recorder is not None:
                recorder.capture_frame()
        obs.append(next_state)
        acs.append(a)
        if config['hard_smoothing']:
            past = np.append(past, [np.copy(x)], axis=0)
        reward.append(r)
        rewards.append(info['rewards'])
        desired_torques = []
        observed_torques = []
        trajectory.append([current_state.copy(), a.copy(), next_state - current_state, -r])
        current_state = next_state
        traject_cost += -r
        if done:
            break
    if recorder is not None:
        recorder.capture_frame()
        recorder.close()
    samples['acs'].append(np.copy(acs))
    samples['obs'].append(np.copy(obs))
    samples['reward'].append(np.copy(reward))
    samples['rewards'].append(np.copy(rewards))
    samples['desired_torques'].append(np.copy(desired_torques))
    samples['observed_torques'].append(np.copy(observed_torques))
    return np.array(trajectory), traject_cost


def execute(env, init_state, steps, init_mean, init_var, model, config, last_action_seq, pred_high,
            pred_low, K, index_iter, final_iter, samples, env_index, n_task, test=False):
    index_iter = index_iter // n_task if test else (index_iter // n_task - config["random_episodes"])
    if config['record_video']:
        if test and (index_iter % config['video_recording_frequency'] == 0 or index_iter == final_iter - 1):
            recorder = VideoRecorder(env, config['logdir'] + "/videos/" + "test_env_" + str(env_index) + "_run_" + str(
                index_iter) + ".mp4")
        elif not test and ((index_iter % config['video_recording_frequency'] == 0) or (
                index_iter == final_iter - config["random_episodes"] - 1)):
            recorder = VideoRecorder(env, config['logdir'] + "/videos/env_" + str(env_index) + "_run_" + str(
                index_iter) + ".mp4")
        else:
            recorder = None
    else:
        recorder = None
    current_state = env.reset()
    trajectory = []
    traject_cost = 0
    model_error = 0
    sliding_mean = np.zeros(config["sol_dim"])
    obs, acs, reward, rewards, desired_torques, observed_torques = [current_state], [], [], [], [], []
    past = np.zeros((3, 12))
    for t in range(steps):
        virtual_acs = list(past[-3]) + list(past[-2]) + list(past[-1])
        cost_object = Cost(ensemble_model=model, init_state=current_state, horizon=config["horizon"],
                           action_dim=env.action_space.shape[0], goal=config["goal"], pred_high=pred_high,
                           pred_low=pred_low, config=config, last_action=virtual_acs, speed=None)

        config["cost_fn"] = cost_object.cost_fn
        optimizer = RS_opt(config)
        sol = optimizer.obtain_solution(acs=virtual_acs)
        x = sol[0:env.action_space.shape[0]]
        a = np.copy(x)
        next_state, r = 0, 0
        for k in range(K):
            next_state, rew, done, info = env.step(a)
            r += rew
            if recorder is not None:
                recorder.capture_frame()
        obs.append(next_state)
        acs.append(np.copy(a))
        reward.append(r)
        rewards.append(info['rewards'])
        past = np.append(past, [np.copy(x)], axis=0)
        desired_torques = []
        observed_torques = []
        trajectory.append([current_state.copy(), a.copy(), next_state - current_state, -r])
        model_error += test_model(model, current_state.copy(), a.copy(), next_state - current_state)
        current_state = next_state
        traject_cost += -r
        sliding_mean[0:-len(a)] = sol[len(a)::]
        if done:
            break
    if recorder is not None:
        recorder.capture_frame()
        recorder.close()
    # print("Model error: ", model_error)
    samples['acs'].append(np.copy(acs))
    samples['obs'].append(np.copy(obs))
    samples['reward'].append(np.copy(reward))
    samples['rewards'].append(np.copy(rewards))
    samples['desired_torques'].append(np.copy(desired_torques))
    samples['observed_torques'].append(np.copy(observed_torques))
    return np.array(trajectory), traject_cost


def test_model(ensemble_model, init_state, action, state_diff, to_print=False):
    x = np.concatenate(([init_state], [action]), axis=1)
    y = state_diff.reshape(1, -1)
    y_pred = ensemble_model.get_models()[0].predict(x)
    if to_print:
        print(y_pred)
    return np.power(y - y_pred, 2).sum()


def extract_action_seq(data):
    actions = []
    for d in data:
        actions += d[1].tolist()
    return np.array(actions)


def main(gym_args, mismatches, config, gym_kwargs={}):
    """---------Prepare the directories------------------"""
    if config['exp_dir'] is None:
        now = datetime.now()
        timestamp = now.strftime("%d_%m_%Y_%H_%M_%S")
        experiment_name = timestamp + "_" + config["exp_suffix"]
        config['exp_dir'] = os.path.join(os.getcwd(), config["result_dir"], config["env_name"], experiment_name)
    try:
        i = 0
        while True:
            res_dir = os.path.join(config['exp_dir'], "run_" + str(i))
            i += 1
            if not os.path.isdir(res_dir):
                os.makedirs(res_dir)
                os.makedirs(res_dir + "/videos")
                os.makedirs(res_dir + "/models")
                config['logdir'] = res_dir
                break
    except:
        print("Could not make the result directory!!!")

    with open(res_dir + "/details.txt", "w+") as f:
        f.write(config["exp_details"])

    with open(res_dir + '/config.json', 'w') as fp:
        import json
        json.dump(config, fp)

    # **********************************
    n_task = len(mismatches)
    data = n_task * [None]
    models = n_task * [None]
    best_action_seq = np.random.rand(config["sol_dim"]) * 2.0 - 1.0
    best_cost = 10000
    last_action_seq = None
    all_action_seq = []
    all_costs = []
    traj_obs, traj_acs, traj_reward, traj_rewards = [[] for _ in range(n_task)], [[] for _ in range(n_task)], \
                                                    [[] for _ in range(n_task)], [[] for _ in range(n_task)]
    traj_observed_torques, traj_desired_torques = [[] for _ in range(n_task)], [[] for _ in range(n_task)]
    best_reward, best_distance = -np.inf, -np.inf
    '''-------------Attempt to load saved data------------------'''
    if config['pretrained_model'] is not None:
        mismatches = np.load(config['pretrained_model'] + "/mismatches.npy")
        n_task = len(mismatches)
        models = n_task * [None]
        device = torch.device("cuda") if config["cuda"] else torch.device("cpu")
        for i in range(n_task):
            models[i] = load_model(config['pretrained_model']+"/models/ensemble_"+str(i)+"/", device=device)
        config['iterations'] = 0
        print("Found pretrained model. Passing directly to testing model.")

    elif path.exists(config["data_dir"] + "/trajectories.npy") and path.exists(config["data_dir"] + "/mismatches.npy"):
            print("Found stored data. Setting random trials to zero.")
            data = np.load(config["data_dir"] + "/trajectories.npy")
            mismatches = np.load(config["data_dir"] + "/mismatches.npy")
            config["random_episodes"] = 0
            n_task = len(mismatches)

    envs = [gym.make(*gym_args, **gym_kwargs) for i in range(n_task)]
    for i, e in enumerate(envs):
        e.set_mismatch(mismatches[i])
        e.metadata['video.frames_per_second'] = 1 / config['ctrl_time_step']

    for i in range(n_task):
        with open(res_dir + "/costs_task_" + str(i) + ".txt", "w+") as f:
            f.write("")

    np.save(res_dir + '/mismatches.npy', mismatches)
    t = trange(config["iterations"] * n_task, desc='', leave=True)
    for index_iter in t:
        env_index = int(index_iter % n_task)
        env = envs[env_index]

        samples = {'acs': [], 'obs': [], 'reward': [], 'rewards': [], 'desired_torques': [], 'observed_torques': []}
        if data[env_index] is None or index_iter < config["random_episodes"] * n_task:
            trajectory, c = execute_random(env=env, steps=config["episode_length"], init_state=config["init_state"],
                                           K=config['K'], index_iter=index_iter, res_dir=res_dir, samples=samples,
                                           config=config)
            if data[env_index] is None:
                data[env_index] = trajectory
            else:
                data[env_index] = np.concatenate((data[env_index], trajectory), axis=0)
            # print("Cost : ", c)

            if c < best_cost:
                best_cost = c
                best_action_seq = []
                for d in trajectory:
                    best_action_seq += d[1].tolist()
                best_action_seq = np.array(best_action_seq)
                last_action_seq = best_action_seq
            all_action_seq.append(extract_action_seq(trajectory))
            all_costs.append(c)
        else:
            '''------------Update models------------'''
            x, y, high, low = process_data(data[env_index])
            models[env_index] = train_ensemble_model(train_in=x, train_out=y, sampling_size=-1, config=config,
                                                     model=models[env_index])
            trajectory, c = execute(env=env,
                                    init_state=config["init_state"],
                                    model=models[env_index],
                                    steps=config["episode_length"],
                                    init_mean=best_action_seq[0:config["sol_dim"]],
                                    init_var=0.1 * np.ones(config["sol_dim"]),
                                    config=config,
                                    last_action_seq=best_action_seq,
                                    pred_high=high,
                                    pred_low=low,
                                    K=config['K'],
                                    index_iter=index_iter,
                                    final_iter=config["iterations"],
                                    samples=samples,
                                    n_task=n_task,
                                    env_index=env_index)
            data[env_index] = np.concatenate((data[env_index], trajectory), axis=0)

            if c < best_cost:
                best_cost = c
                best_action_seq = []
                for d in trajectory:
                    best_action_seq += d[1].tolist()
                best_action_seq = np.array(best_action_seq)
                last_action_seq = extract_action_seq(trajectory)

            all_action_seq.append(extract_action_seq(trajectory))
            all_costs.append(c)

        traj_obs[env_index].extend(samples["obs"])
        traj_acs[env_index].extend(samples["acs"])
        traj_reward[env_index].extend(samples["reward"])
        traj_rewards[env_index].extend(samples["rewards"])
        traj_desired_torques[env_index].extend(samples["desired_torques"])
        traj_observed_torques[env_index].extend(samples["observed_torques"])

        if (len(samples['reward'][0])) == config['episode_length']:
            best_reward = max(best_reward, np.sum(samples["reward"]))
        best_distance = max(best_distance, np.sum(np.array(samples['obs'][0])[:, -3]) * config['ctrl_time_step'])
        t.set_description(("Reward " + str(int(best_reward * 100) / 100) if best_reward != -np.inf else str(
            -np.inf)) + " Distance " + str(int(100 * best_distance) / 100))
        t.refresh()

        with open(os.path.join(config['logdir'], "logs.pk"), 'wb') as f:
            pickle.dump({
                "observations": traj_obs,
                "actions": traj_acs,
                "reward": traj_reward,
                "rewards": traj_rewards,
                "observed_torques": traj_observed_torques,
                "desired_torques": traj_desired_torques,
            }, f)
        # if index_iter % config["dump_trajects"] == 0 and index_iter > 0:
        #     np.save(res_dir + "/trajectories.npy", data)
        with open(res_dir + "/costs_task_" + str(env_index) + ".txt", "a+") as f:
            f.write(str(c) + "\n")

    print("Finally Saving trajectories..")
    np.save(res_dir + "/trajectories.npy", data)

    print("Finally Saving model..")

    for env_index in range(n_task):
        os.makedirs(res_dir + "/models/ensemble_"+str(env_index))
        models[env_index].save(file_path=res_dir+"/models/ensemble_"+str(env_index)+"/")

    if config['test_mismatches'] is not None:
        test_mismatches = config['test_mismatches']
        n_task = len(test_mismatches)

        traj_obs, traj_acs, traj_reward, traj_rewards = [[] for _ in range(n_task)], [[] for _ in range(n_task)], \
                                                        [[] for _ in range(n_task)], [[] for _ in range(n_task)]
        traj_observed_torques, traj_desired_torques = [[] for _ in range(n_task)], [[] for _ in range(n_task)]
        best_reward, best_distance = -np.inf, -np.inf

        envs = [gym.make(*gym_args, **gym_kwargs) for i in range(n_task)]
        for i, e in enumerate(envs):
            e.set_mismatch(test_mismatches[i])
            # e.render("human")

        for i in range(n_task):
            with open(res_dir + "/test_costs_task_" + str(i) + ".txt", "w+") as f:
                f.write("")

        np.save(res_dir + '/test_mismatches.npy', mismatches)
        t = trange(config["test_iterations"] * n_task, desc='', leave=True)
        for index_iter in t:
            '''Pick a random environment'''
            env_index = int(index_iter % n_task)  # np.random.randint(0, n_task)
            env = envs[env_index]

            samples = {'acs': [], 'obs': [], 'reward': [], 'rewards': [], 'desired_torques': [], 'observed_torques': []}

            '''------------Update models------------'''
            _, c = execute(env=env,
                           init_state=config["init_state"],
                           model=models[0],
                           steps=config["episode_length"],
                           init_mean=best_action_seq[0:config["sol_dim"]],
                           init_var=0.1 * np.ones(config["sol_dim"]),
                           config=config,
                           last_action_seq=best_action_seq,
                           pred_high=None,
                           pred_low=None,
                           K=config['K'],
                           index_iter=index_iter,
                           final_iter=config["test_iterations"],
                           samples=samples,
                           env_index=env_index,
                           n_task=n_task,
                           test=True)

            with open(res_dir + "/test_costs_task_" + str(env_index) + ".txt", "a+") as f:
                f.write(str(c) + "\n")

            traj_obs[env_index].extend(samples["obs"])
            traj_acs[env_index].extend(samples["acs"])
            traj_reward[env_index].extend(samples["reward"])
            traj_rewards[env_index].extend(samples["rewards"])
            traj_desired_torques[env_index].extend(samples["desired_torques"])
            traj_observed_torques[env_index].extend(samples["observed_torques"])

            if (len(samples['reward'][0])) == config['episode_length']:
                best_reward = max(best_reward, np.sum(samples["reward"]))
            best_distance = max(best_distance, np.sum(np.array(samples['obs'][0])[:, -3]) * 0.02)
            t.set_description(("Reward " + str(int(best_reward * 100) / 100) if best_reward != -np.inf else str(
                -np.inf)) + " Distance " + str(int(100 * best_distance) / 100))
            t.refresh()

            with open(os.path.join(config['logdir'], "test_logs.pk"), 'wb') as f:
                pickle.dump({
                    "observations": traj_obs,
                    "actions": traj_acs,
                    "reward": traj_reward,
                    "rewards": traj_rewards,
                    "observed_torques": traj_observed_torques,
                    "desired_torques": traj_desired_torques,
                }, f)


def execute_online(env, steps, model, config, pred_high, pred_low, K):
    current_state = env.reset()
    past = np.zeros((3, 8))
    event = [None]
    max_vel, max_acc, max_jerk, max_torque_jerk = config['max_action_velocity'], config['max_action_acceleration'], \
                                                  config['max_action_jerk'], config['max_torque_jerk']
    lb, ub = config['lb'], config['ub']

    def on_press(key):
        if key == Key.left:
            env.reset()
            past[:3] = np.zeros((3, 8))
        elif key == Key.right:
            event[0] = None
        elif key == Key.down:
            event[0] = 'pause'

    def on_release(key):
        if key == Key.esc:
            return False

    IDvd = env.pybullet_client.addUserDebugParameter("Desired velocity", -0.5, 0.5, 1)
    IDcontrol = env.pybullet_client.addUserDebugParameter("Random - MPC", 0, 1, 1)
    IDsmooth = env.pybullet_client.addUserDebugParameter("Smooth", 0, 1, 1)
    IDpopsize = env.pybullet_client.addUserDebugParameter("population size (10^x)", 0, 5, 2)
    IDhorizon = env.pybullet_client.addUserDebugParameter("Horizon", 1, 50, 25)
    IDorientation = env.pybullet_client.addUserDebugParameter("orientation", -180, 180, 0)
    # Collect events until released
    listener = Listener(on_press=on_press, on_release=on_release)
    listener.start()
    while True:
        if event[0] != 'pause':
            vd = env.pybullet_client.readUserDebugParameter(IDvd)
            env.set_desired_speed(vd)
            MPC = env.pybullet_client.readUserDebugParameter(IDcontrol) > 0
            smooth = env.pybullet_client.readUserDebugParameter(IDsmooth) > 0
            popsize = env.pybullet_client.readUserDebugParameter(IDpopsize)
            horizon = env.pybullet_client.readUserDebugParameter(IDhorizon)
            yaw = env.pybullet_client.readUserDebugParameter(IDorientation)
            config['popsize'] = int(10**popsize)
            config['horizon'] = int(horizon)
            config['sol_dim'] = config['horizon'] * config['action_dim']
            config['hard_smoothing'] = smooth
            env.set_init_orientation(env.pybullet_client.getQuaternionFromEuler([0, 0, yaw*np.pi/180]))
            if MPC:
                virtual_acs = list(past[-3]) + list(past[-2]) + list(past[-1])
                cost_object = Cost(ensemble_model=model, init_state=current_state, horizon=config["horizon"],
                                   action_dim=env.action_space.shape[0], goal=config["goal"], pred_high=pred_high,
                                   pred_low=pred_low, config=config, last_action=virtual_acs, speed=vd)

                config["cost_fn"] = cost_object.cost_fn
                optimizer = RS_opt(config)

                sol = optimizer.obtain_solution(acs=virtual_acs)
                x = sol[0:env.action_space.shape[0]]
                a = np.copy(x)
            else:
                if smooth:
                    amax = np.min((past[- 1] + max_vel, 2 * past[- 1] - past[- 2] + max_acc,
                                   3 * past[- 1] - 3 * past[- 2] + past[- 3] + max_jerk), axis=0)
                    amin = np.max((past[- 1] - max_vel, 2 * past[- 1] - past[- 2] - max_acc,
                                   3 * past[- 1] - 3 * past[- 2] + past[- 3] - max_jerk), axis=0)
                    amax, amin = np.clip(amax, lb, ub), np.clip(amin, lb, ub)
                    x = np.random.uniform(amin, amax)
                    a = np.copy(x)
                else:
                    x = np.random.random(8)*2-1
                    a = np.copy(x)
            if config['action_space'] == "Torque2":
                a[:4] = a[:4] * 0.6
                a[4:] = (a[4:] * 0.4 + 0.5) * 0.75 + 0.10
                a = env.minitaur.ConvertFromLegModel(a)
                a = 8 * (a - env.minitaur.GetMotorAngles())
            elif config['action_space'] == "velocity2":
                a[:4] = a[:4] * 0.6
                a[4:] = (a[4:] * 0.4 + 0.5) * 0.75 + 0.10
                a = env.minitaur.ConvertFromLegModel(a)
                a = (a - env.minitaur.GetMotorAngles())
            elif config['action_space'] == "Motor2":
                a[:4] = a[:4] * 0.6
                a[4:] = (a[4:] * 0.4 + 0.5) * 0.75 + 0.10
                a = env.minitaur.ConvertFromLegModel(a) - 1.5

            if not MPC:
                txt = "Random"
            elif vd < 0:
                txt = "Backward"
            elif vd == 0:
                txt = "Stationary"
            elif vd > 0:
                txt = "Forward"
            base_pos = env.minitaur.GetBasePosition()
            env.pybullet_client.addUserDebugText(txt, base_pos + np.array([0, 0, 0.2]), [0, 0, 0], 4, 0.04)
            for k in range(K):
                obs, rew, done, info = env.step(a)
            past = np.append(past, [np.copy(x)], axis=0)
            if done:
                env.reset()
                past = np.zeros((3, 8))


def online_test(gym_args, mismatches, config, gym_kwargs={}):
    """---------Prepare the directories------------------"""
    assert config['pretrained_model'] is not None, "must give an existing logdir"
    res_dir = config['pretrained_model']
    device = torch.device("cuda") if config["cuda"] else torch.device("cpu")
    models = load_model(res_dir+"/models/ensemble_0/", device=device)
    # test_model(models, np.zeros(25), np.zeros(8), np.zeros(25), True)
    gym_kwargs['render'] = True
    env = gym.make(*gym_args, **gym_kwargs)
    env.set_mismatch(mismatches[0])
    env.metadata['video.frames_per_second'] = 1 / config['ctrl_time_step']
    env.render("human")
    execute_online(env=env, model=models, steps=config["episode_length"], config=config, pred_high=None, pred_low=None, K=config['K'])

################################################################################


config = {
    # exp parameters:
    "horizon": 1,  # NOTE: "sol_dim" must be adjusted
    "iterations": 2,
    "random_episodes": 1,  # per task
    "episode_length": 50,  # number of times the controller is updated
    "test_mismatches": None,
    "test_iterations": 0,
    "init_state": None,  # Must be updated before passing config as param
    "action_dim": 12,
    "action_space": ['S&E', 'Motor'][1],
    "partial_torque_control": 0,
    "vkp": 0,
    # choice of action space between Motor joint, swing and extension of each leg and delta motor joint
    "goal": None,  # Sampled during env reset
    "ctrl_time_step": 0.02,
    "K": 1,  # number of control steps with the same controller
    "desired_speed": 0.8,
    "xreward": 1,
    "zreward": 1,
    "rollreward": 1,
    "pitchreward": 1,
    "yawreward": 1,
    "action_norm_weight": 0.0,
    "action_vel_weight": 0,  # 0.05 seems working
    "action_acc_weight": 0,  # 0.05 seems working
    "action_jerk_weight": 0,  # 0.05 seems working
    "soft_smoothing": 0,
    "hard_smoothing": 0,

    # logging
    "record_video": 0,
    "video_recording_frequency": 25,
    "result_dir": "results",
    "env_name": "spot_micro_03",
    "exp_suffix": "experiment",
    "exp_dir": None,
    "exp_details": "SpotMicro evaluate from scratch",
    "dump_trajects": 1,
    "data_dir": "",
    "logdir": None,
    "pretrained_model": None,

    # Ensemble model params
    "cuda": True,
    "ensemble_epoch": 5,
    "ensemble_dim_in": 33 + 12,
    "ensemble_dim_out": 33,
    "ensemble_hidden": [256, 256],
    "hidden_activation": "relu",
    "ensemble_cuda": True,
    "ensemble_seed": None,
    "ensemble_output_limit": None,
    "ensemble_dropout": 0.0,
    "n_ensembles": 1,
    "ensemble_batch_size": 8192 * 2,
    "ensemble_log_interval": 500,

    # Optimizer parameters
    "max_iters": 1,
    "epsilon": 0.0001,
    "lb": -1,
    "ub": 1,
    "max_action_velocity": 10,  # 10 from working controller
    "max_action_acceleration": 100,  # 100 from working controller
    "max_action_jerk": 1000,  # 1000 from working controller
    "max_torque_jerk": 25,
    "popsize": 10000,
    "sol_dim": None,  # NOTE: Depends on Horizon
    "num_elites": 50,
    "cost_fn": None,
    "alpha": 0.1,
    "discount": 1.
}

# optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("--iterations",
                    help='Total episodes.',
                    type=int)
parser.add_argument("--episode_length",
                    help='Total time steps in Episodes.',
                    type=int)
parser.add_argument("--random_episodes",
                    help='Random Episodes.',
                    type=int)
parser.add_argument("--exp_details",
                    help='Details about the experiment',
                    type=str)
parser.add_argument("--data_dir",
                    help='To load trajectories from',
                    type=str)
parser.add_argument("--dump_trajects",
                    help='Create trajectory.npy after every dump_trajects iterations',
                    type=int)
parser.add_argument('-c', '--config', action='append', nargs=2, default=[])
parser.add_argument('-logdir', type=str, default='None')

arguments = parser.parse_args()
if arguments.iterations is not None: config['iterations'] = arguments.iterations
if arguments.episode_length is not None: config['episode_length'] = arguments.episode_length
if arguments.random_episodes is not None: config['random_episodes'] = arguments.random_episodes
if arguments.exp_details is not None: config['exp_details'] = arguments.exp_details
if arguments.dump_trajects is not None: config['dump_trajects'] = arguments.dump_trajects
if arguments.data_dir is not None: config['data_dir'] = arguments.data_dir

logdir = arguments.logdir
config['logdir'] = None if logdir == 'None' else logdir


def check_config(config):
    config['sol_dim'] = config['horizon'] * config['action_dim']
    config["max_action_velocity"] = config["max_action_velocity"] * config['ctrl_time_step']
    config["max_action_acceleration"] = config["max_action_acceleration"] * config['ctrl_time_step'] ** 2
    config["max_action_jerk"] = config["max_action_jerk"] * config['ctrl_time_step'] ** 3


for (key, val) in arguments.config:
    if key in ['horizon', 'K', 'popsize', 'iterations', 'n_ensembles', 'episode_length']:
        config[key] = int(val)
    elif key in ['load_data', 'hidden_activation', 'data_size', 'save_data', 'script']:
        config[key] = val
    else:
        config[key] = float(val)

mismatches = np.array([
    [0., 0., 0],
])

test_mismatches = None
# test_mismatches = [
#     [0., 0., 0],
#     [0., -0.8, 0],
# ]

config['test_mismatches'] = test_mismatches

args = ["SpotMicroEnv-v0"]

config_params = None

# config['exp_suffix'] = "without_Z"
# config_params = []
# # popsize = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
# for i in range(1):
#     for _ in range(10):
#         config_params.append({
#         })


def apply_config_params(conf, params):
    for (key, val) in list(params.items()):
        conf[key] = val
    return conf


def env_args_from_config(config):
    return {
        "action_space": config['action_space'],
        'distance_weight': config["xreward"],
        'desired_speed': config["desired_speed"],
        'high_weight': config["zreward"],
        'roll_weight': config["rollreward"],
        'pitch_weight': config["pitchreward"],
        'yaw_weight': config["yawreward"],
        'action_weight': config["action_norm_weight"],
        'action_vel_weight': config["action_vel_weight"],
        'action_acc_weight': config["action_acc_weight"],
        'action_jerk_weight': config["action_jerk_weight"],
    }


online = False
# online = True

if online:
    config["xreward"] = 1
    config["zreward"] = 0
    config["rollreward"] = 0
    config["pitchreward"] = 0
    config["yawreward"] = 0
    config["popsize"] = 100
    path = "/home/timothee/Documents/LARSEN/fast_adaptation_embedding/exp/results/"
    directory = ['03_03_2020_10_15_03_experiment', '03_03_2020_11_54_28_experiment'][0]
    config['pretrained_model'] = path + config['env_name'] + "/"+directory+"/run_0"
    check_config(config)
    kwargs = env_args_from_config(config)
    online_test(gym_args=args, gym_kwargs=kwargs, mismatches=mismatches, config=config)
else:
    n_run = 1 if config_params is None else len(config_params)
    if n_run == 1:
        # For one-run experiment
        check_config(config)
        kwargs = env_args_from_config(config)
        main(gym_args=args, gym_kwargs=kwargs, mismatches=mismatches, config=config)
    else:
        # For multi-run experiment
        exp_dir = None
        for run in range(n_run):
            conf = copy.copy(config)
            conf['exp_dir'] = exp_dir
            conf = apply_config_params(conf, config_params[run])
            print("Params: ", config_params[run])
            check_config(conf)
            kwargs = env_args_from_config(conf)
            main(gym_args=args, gym_kwargs=kwargs, mismatches=mismatches, config=conf)
            exp_dir = conf['exp_dir']

            if run == 0:
                with open(exp_dir + "/config.txt", "w") as f:
                    f.write(str(config_params))

