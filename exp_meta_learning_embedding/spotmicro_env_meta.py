import torch
import numpy as np
import copy
import gym
from datetime import datetime
import pickle
import os
from os import path
from gym.wrappers.monitoring.video_recorder import VideoRecorder
import argparse
from tqdm import trange
import os, inspect
import json

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import fast_adaptation_embedding.env
import fast_adaptation_embedding.models.embedding_nn_normalized_v2 as nn_model
from fast_adaptation_embedding.controllers.random_shooting import RS_opt


class Cost(object):
    def __init__(self, model, init_state, horizon, action_dim, goal, config, last_action,
                 task_likelihoods, speed=None):
        self.__models = model
        self.__init_state = init_state
        self.__horizon = horizon
        self.__action_dim = action_dim
        self.__goal = goal
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
        self.__task_likelihoods = task_likelihoods

    def cost_fn(self, samples):
        a = torch.FloatTensor(samples).cuda() \
            if self.__models[0].cuda_enabled \
            else torch.FloatTensor(samples)

        # compute the samples for hard smoothing
        ad = self.__action_dim
        max_vel = torch.FloatTensor(self.__max_vel).cuda()
        max_acc = torch.FloatTensor(self.__max_acc).cuda()
        max_jerk = torch.FloatTensor(self.__max_jerk).cuda()
        if self.__hard_smoothing:
            for t in range(3, int(self.__sol_dim / ad) + 3):
                amax = torch.min(a[:, (t - 1) * ad:t * ad] + max_vel,
                                 2 * a[:, (t - 1) * ad:t * ad] - a[:, (t - 2) * ad:(t - 1) * ad] + max_acc,
                                 out=None)
                amax = torch.min(amax,
                                 3 * a[:, (t - 1) * ad:t * ad] - 3 * a[:, (t - 2) * ad:(t - 1) * ad] + a[:,
                                                                                                       (t - 3) * ad:(
                                                                                                                            t - 2) * ad] + max_jerk,
                                 out=None)
                amin = torch.max(a[:, (t - 1) * ad:t * ad] - max_vel,
                                 2 * a[:, (t - 1) * ad:t * ad] - a[:, (t - 2) * ad:(t - 1) * ad] - max_acc,
                                 out=None)
                amin = torch.max(amin,
                                 3 * a[:, (t - 1) * ad:t * ad] - 3 * a[:, (t - 2) * ad:(t - 1) * ad] + a[:, (
                                                                                                                    t - 3) * ad:(
                                                                                                                                        t - 2) * ad] - max_jerk,
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
            if self.__models[0].cuda_enabled \
            else torch.FloatTensor(np.repeat([self.__init_state], len(samples), axis=0))
        all_costs = torch.FloatTensor(np.zeros(len(samples))).cuda() \
            if self.__models[0].cuda_enabled \
            else torch.FloatTensor(np.zeros(len(samples)))

        n_batch = max(1, int(len(samples) / self.__batch_size))
        per_batch = len(samples) / n_batch
        index_offset = int(len(samples[0]) / self.__action_dim) - self.__horizon
        for i in range(n_batch):
            start_index = int(i * per_batch)
            end_index = len(samples) if i == n_batch - 1 else int(i * per_batch + per_batch)
            action_batch = a[start_index:end_index]
            start_states = init_states[start_index:end_index]
            dyn_model = self.__models[np.argmax(self.__task_likelihoods)]
            for h in range(index_offset, self.__horizon + index_offset):
                actions = action_batch[:, h * self.__action_dim: h * self.__action_dim + self.__action_dim].clone()
                model_input = torch.cat((start_states, actions), dim=1)
                diff_state = dyn_model.predict_tensor(model_input)
                start_states += diff_state
                x_vel_cost = (start_states[:, 30] - self.__desired_speed) ** 2 * self.__xreward
                z_cost = (start_states[:, -1] - 0.1855) ** 2 * self.__zreward
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

                all_costs[start_index: end_index] += -torch.exp(-x_vel_cost) * self.__discount ** h \
                                                     + -torch.exp(-yaw_cost) * self.__discount ** h \
                                                     + -torch.exp(-pitch_cost) * self.__discount ** h \
                                                     + -torch.exp(-roll_cost) * self.__discount ** h \
                                                     + -torch.exp(-z_cost) * self.__discount ** h

        if self.__hard_smoothing:
            return a[torch.argmin(all_costs)].cpu().detach().numpy()
        else:
            return a[torch.argmin(all_costs)].cpu().detach().numpy()


def train_meta(tasks_in, tasks_out, config, valid_in=[], valid_out=[]):
    model = nn_model.Embedding_NN(dim_in=config["dim_in"],
                                  hidden=config["hidden_layers"],
                                  dim_out=config["dim_out"],
                                  embedding_dim=config["embedding_size"],
                                  num_tasks=len(tasks_in),
                                  CUDA=config["cuda"],
                                  SEED=None,
                                  output_limit=config["output_limit"],
                                  dropout=0.0,
                                  hidden_activation=config["hidden_activation"])
    task_losses, valid_losses, saved_embeddings = nn_model.train_meta(model,
                                                                      tasks_in,
                                                                      tasks_out,
                                                                      valid_in=valid_in,
                                                                      valid_out=valid_out,
                                                                      meta_iter=config["meta_iter"],
                                                                      inner_iter=config["inner_iter"],
                                                                      inner_step=config["inner_step"],
                                                                      meta_step=config["meta_step"],
                                                                      minibatch=config["meta_batch_size"],
                                                                      inner_sample_size=config["inner_sample_size"])
    return model, task_losses, saved_embeddings, valid_losses


def train_model(model, train_in, train_out, task_id, config):
    cloned_model = copy.deepcopy(model)
    nn_model.train(cloned_model,
                   train_in,
                   train_out,
                   task_id=task_id,
                   inner_iter=config["epoch"],
                   inner_lr=config["learning_rate"],
                   minibatch=config["minibatch_size"])
    return cloned_model


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


def execute(env, init_state, steps, init_mean, init_var, model, config, last_action_seq, task_likelihoods,
            K, index_iter, final_iter, samples, env_index=0, n_task=1, n_model=1, model_index=0,
            test=False, ):
    index_iter = index_iter // (n_task * n_model)
    if config['record_video']:
        if (index_iter % config['video_recording_frequency'] == 0) or (index_iter == final_iter - 1):
            recorder = VideoRecorder(env, config['logdir'] + "/videos/env_" + str(env_index) + "_run_" + str(
                index_iter) + ".mp4")
        else:
            recorder = None
    else:
        recorder = None
    current_state = env.reset()
    trajectory = []
    traject_cost = 0

    obs, acs, reward, rewards, desired_torques, observed_torques = [current_state], [], [], [], [], []
    past = np.array([(env.init_joint - (env.ub + env.lb) / 2) * 2 / (env.ub - env.lb) for _ in range(3)])
    for t in range(steps):
        virtual_acs = list(past[-3]) + list(past[-2]) + list(past[-1])
        cost_object = Cost(model=model, init_state=current_state, horizon=config["horizon"],
                           action_dim=env.action_space.shape[0], goal=config["goal"],
                           config=config, last_action=virtual_acs, speed=None,
                           task_likelihoods=task_likelihoods)

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
    return trajectory, traject_cost


def execute_online(env, steps, model, config, task_likelihoods, K, samples,
                   current_state, recorder, trajectory, traject_cost, past):
    for t in range(steps):
        virtual_acs = list(past[-3]) + list(past[-2]) + list(past[-1])
        cost_object = Cost(model=model, init_state=current_state, horizon=config["horizon"],
                           action_dim=env.action_space.shape[0], goal=config["goal"],
                           config=config, last_action=virtual_acs, speed=None,
                           task_likelihoods=task_likelihoods)

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
        trajectory.append([current_state.copy(), a.copy(), next_state - current_state, -r])
        current_state = next_state
        traject_cost += -r
        past = np.append(past, [np.copy(x)], axis=0)
        samples['acs'].append(np.copy(a))
        samples['obs'].append(np.copy(next_state))
        samples['reward'].append(r)
        samples['rewards'].append(info['rewards'])
        if done:
            break
    return traject_cost, done, past, current_state.copy()


def extract_action_seq(data):
    actions = []
    for d in data:
        actions += d[1].tolist()
    return np.array(actions)


def compute_likelihood(data, models, beta=1.0):
    """
    Computes MSE loss and then softmax to have a probability
    """
    data_size = config['adapt_steps']
    if data_size is None: data_size = len(data)
    lik = np.zeros(len(models))
    x, y, _, _ = process_data(data[-data_size::])
    for i, m in enumerate(models):
        y_pred = m.predict(x)
        lik[i] = np.exp(- beta * m.loss_function_numpy(y, y_pred) / len(x))
    if np.sum(lik) == 0:
        return np.ones(len(models)) / len(models)
    else:
        return lik / np.sum(lik)


def sample_model_index(likelihoods):
    cum_sum = np.cumsum(likelihoods)
    num = np.random.rand()
    for i, cum_prob in enumerate(cum_sum):
        if num <= cum_prob: return i


def main(gym_args, config, test_mismatch, index, gym_kwargs={}):
    """---------Prepare the directories------------------"""
    if config['logdir'] is None:
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
                    config['logdir'] = res_dir
                    break
        except:
            print("Could not make the result directory!!!")
    else:
        res_dir = config['logdir']
        os.makedirs(res_dir + "/videos")

    with open(res_dir + "/details.txt", "w+") as f:
        f.write(config["exp_details"])

    with open(res_dir + '/config.json', 'w') as fp:
        import json
        json.dump(config, fp)

    '''---------Prepare the test environment---------------'''
    env = gym.make(*gym_args, **gym_kwargs)
    list_data_dir = os.listdir(config["data_dir"])
    if config['training_tasks_index'] is None:
        n_training_tasks = 0
        for name in list_data_dir:
            if "run" in name:
                n_training_tasks += 1
    else:
        n_training_tasks = len(config['training_tasks_index'])
    try:
        s = os.environ['DISPLAY']
        # env.render(mode="rgb_array")
        env.render(mode="human")
        env.reset()
    except:
        # print("Display not available")
        env.reset()

    '''---------Initialize global variables------------------'''
    data = []
    all_action_seq = []
    all_costs = []
    with open(res_dir + "/costs_" + str(index) + ".txt", "w+") as f:
        f.write("mismatches" + str(test_mismatch) + "\n")

    '''--------------------Meta learn the models---------------------------'''
    if not path.exists(config["data_dir"] + "/"+config['meta_model_name'] + "/" + config["model_name"] + ".pt"):
        print("Model not found. Learning from data...")
        tasks_in, tasks_out = [], []
        valid_in, valid_out = [], []
        tasks_list = range(n_training_tasks) if config['training_tasks_index'] is None else config[
            'training_tasks_index']
        for n in tasks_list:
            meta_data = np.load(config["data_dir"] + "/run_" + str(n) + "/trajectories.npy", allow_pickle=True)
            x, y, high, low = process_data(meta_data[0])
            tasks_in.append(x)
            tasks_out.append(y)
            print("task ", n, " data: ", len(tasks_in[n]), len(tasks_out[n]))
        if config['valid_dir'] is not None:
            for n in tasks_list:
                meta_data = np.load(config["valid_dir"] + "/run_" + str(n) + "/trajectories.npy", allow_pickle=True)
                x, y, high, low = process_data(meta_data[0])
                valid_in.append(x)
                valid_out.append(y)
        meta_model, task_losses, saved_embeddings, valid_losses = train_meta(tasks_in, tasks_out, config, valid_in,
                                                                             valid_out)
        os.mkdir(config["data_dir"] + "/"+config['meta_model_name'])
        meta_model.save(config["data_dir"] + "/"+config['meta_model_name'] + "/" + config["model_name"] + ".pt")
        np.save(config["data_dir"] + "/"+config['meta_model_name'] + "/" + config["model_name"] + "_task_losses.npy", task_losses)
        np.save(config["data_dir"] + "/"+config['meta_model_name'] + "/" + config["model_name"] + "_valid_losses.npy", valid_losses)
        np.save(config["data_dir"] + "/"+config['meta_model_name'] + "/" + config["model_name"] + "_embeddings.npy", saved_embeddings)
        with open(config["data_dir"] + "/"+config['meta_model_name'] + '/config.json', 'w') as fp:
            json.dump(config, fp)
    else:
        print("Model found. Loading from '.pt' file...")
        device = torch.device("cuda") if config["cuda"] else torch.device("cpu")
        meta_model = nn_model.load_model(config["data_dir"] + "/"+config['meta_model_name'] + "/" + config["model_name"] + ".pt", device)

    raw_models = [copy.deepcopy(meta_model) for _ in range(n_training_tasks)]
    models = [copy.deepcopy(meta_model) for _ in range(n_training_tasks)]
    for task_id, m in enumerate(raw_models):
        m.fix_task(task_id)

    for task_id, m in enumerate(models):
        m.fix_task(task_id)

    '''------------------------Test time------------------------------------'''

    traj_obs, traj_acs, traj_reward, traj_rewards, traj_likelihood = [], [], [], [], []
    best_reward, best_distance = -np.inf, 0
    tbar = trange(config["iterations"], desc='', leave=True)

    alpha = (np.array(config['real_ub']) - np.array(config['real_lb'])) / 2
    config['max_action_velocity'] = config['max_action_velocity'] / alpha * config['ctrl_time_step']
    config['max_action_acceleration'] = config['max_action_acceleration'] / alpha * config['ctrl_time_step'] ** 2
    config['max_action_jerk'] = config['max_action_jerk'] / alpha * config['ctrl_time_step'] ** 3

    for index_iter in tbar:
        samples = {'acs': [], 'obs': [], 'reward': [], 'rewards': [], 'likelihood': []}
        task_likelihoods = np.random.rand(n_training_tasks)

        if config['online']:
            if config['record_video'] and (
                    (index_iter + 1) % config['video_recording_frequency'] == 0 or index_iter == config[
                "iterations"] - 1):
                recorder = VideoRecorder(env, config['logdir'] + "/videos/run_" + str(index_iter) + ".mp4")
            else:
                recorder = None
            c = 0
            # init_mismatch = test_mismatch[1][0]
            init_mismatch = {}
            env.set_mismatch(init_mismatch)
            current_state = env.reset(hard_reset=True)
            past = np.array([(env.init_joint - (env.ub + env.lb) / 2) * 2 / (env.ub - env.lb) for _ in range(3)])
            n_adapt_steps = int(config["episode_length"] / config['successive_steps'])
            for adapt_steps in range(n_adapt_steps):
                steps = adapt_steps * config['successive_steps']
                if steps in test_mismatch[0]:
                    mismatch_index = test_mismatch[0].index(steps)
                    env.set_mismatch(test_mismatch[1][mismatch_index])
                trajectory = []
                c, done, past, current_state = execute_online(env=env,
                                                              steps=config['successive_steps'],
                                                              model=models,
                                                              config=config,
                                                              task_likelihoods=task_likelihoods,
                                                              samples=samples,
                                                              current_state=current_state,
                                                              recorder=recorder,
                                                              K=config['K'],
                                                              trajectory=trajectory,
                                                              traject_cost=c,
                                                              past=past
                                                              )
                if done:
                    break
                data = trajectory
                '''-----------------Compute likelihood before relearning the models-------'''
                if steps < config['stop_adapatation_step']:
                    task_likelihoods = compute_likelihood(data, raw_models)
                    samples['likelihood'].append(np.copy(task_likelihoods))
                    task_index = np.argmax(task_likelihoods)
                    task_likelihoods = task_likelihoods * 0
                    task_likelihoods[task_index] = 1.0
                    if config['adapt_steps'] != 0:
                        x, y, high, low = process_data(data)
                        data_size = config['adapt_steps']
                        if data_size is None:
                            data_size = len(x)

                        models[task_index] = train_model(model=copy.deepcopy(raw_models[task_index]),
                                                         train_in=x[-data_size::],
                                                         train_out=y[-data_size::], task_id=task_index, config=config)

            if recorder is not None:
                recorder.close()

            if (len(samples['reward'])) == config['episode_length']:
                best_reward = max(best_reward, np.sum(samples["reward"]))
            best_distance = max(best_distance, np.sum(np.array(samples['obs'])[:, -3]) * 0.02)
            tbar.set_description(
                ("Reward " + str(int(best_reward * 100) / 100) if best_reward != -np.inf else str(-np.inf)) +
                " Distance " + str(int(100 * best_distance) / 100))
            tbar.refresh()

            traj_obs.append(samples["obs"])
            traj_acs.append(samples["acs"])
            traj_reward.append(samples["reward"])
            traj_rewards.append(samples["rewards"])
            traj_likelihood.append(samples["likelihood"])
        else:
            new_mismatch = test_mismatch
            env.set_mismatch(new_mismatch)
            trajectory, c = execute(env=env,
                                    init_state=config["init_state"],
                                    model=models,
                                    steps=config["episode_length"],
                                    init_mean=np.zeros(config["sol_dim"]),
                                    init_var=0.01 * np.ones(config["sol_dim"]),
                                    config=config,
                                    last_action_seq=None,
                                    task_likelihoods=task_likelihoods,
                                    K=config['K'],
                                    index_iter=index_iter,
                                    final_iter=config["iterations"],
                                    samples=samples
                                    )

            data += trajectory
            '''-----------------Compute likelihood before relearning the models-------'''
            task_likelihoods = compute_likelihood(data, raw_models)
            samples['likelihood'].append(np.copy(task_likelihoods))
            if (len(samples['reward'][0])) == config['episode_length']:
                best_reward = max(best_reward, np.sum(samples["reward"]))
            best_distance = max(best_distance, np.sum(np.array(samples['obs'][0])[:, -3]) * 0.02)
            tbar.set_description(
                ("Reward " + str(int(best_reward * 100) / 100) if best_reward != -np.inf else str(-np.inf)) +
                " Distance " + str(int(100 * best_distance) / 100) +
                " likelihoods: " + str(task_likelihoods))
            tbar.refresh()

            x, y, high, low = process_data(data)
            task_index = sample_model_index(task_likelihoods) if config["sample_model"] else np.argmax(task_likelihoods)

            task_likelihoods = task_likelihoods * 0
            task_likelihoods[task_index] = 1.0
            data_size = config['adapt_steps']
            if data_size is None:
                data_size = len(x)

            models[task_index] = train_model(model=copy.deepcopy(raw_models[task_index]), train_in=x[-data_size::],
                                             train_out=y[-data_size::], task_id=task_index, config=config)

            traj_obs.extend(samples["obs"])
            traj_acs.extend(samples["acs"])
            traj_reward.extend(samples["reward"])
            traj_rewards.extend(samples["rewards"])
            traj_likelihood.append(samples["likelihood"])

        """ save logs """
        with open(res_dir + "/costs_" + str(index) + ".txt", "a+") as f:
            f.write(str(c) + "\n")

        with open(os.path.join(config['logdir'], "logs.pk"), 'wb') as f:
            pickle.dump({
                "observations": traj_obs,
                "actions": traj_acs,
                "reward": traj_reward,
                "rewards": traj_rewards,
                "likelihood": traj_likelihood,
            }, f)

        all_action_seq.append(extract_action_seq(trajectory))
        all_costs.append(c)

        # np.save(res_dir + "/trajectories_" + str(index) + ".npy", data)


#######################################################################################################


config = {
    # exp parameters:
    "horizon": 25,  # NOTE: "sol_dim" must be adjusted
    "iterations": 10,
    # "random_episodes": 1,  # per task
    "episode_length": 500,  # number of times the controller is updated
    "online": True,
    "adapt_steps": None,
    "successive_steps": 50,
    "stop_adapatation_step": 10000,
    "init_state": None,  # Must be updated before passing config as param
    "action_dim": 12,
    "action_space": ['S&E', 'Motor'][1],
    "init_joint": [0., 0.6, -1.] * 4,
    "real_ub": [0.1, 0.8, -0.8] * 4,
    "real_lb": [-0.1, 0.4, -1.2] * 4,
    "goal": None,  # Sampled during env reset
    "ctrl_time_step": 0.02,
    "K": 1,  # number of control steps with the same control1ler
    "desired_speed": 0.5,
    "xreward": 1,
    "zreward": 1,
    "rollreward": 1,
    "pitchreward": 1,
    "yawreward": 1,
    "action_norm_weight": 0.0,
    "action_vel_weight": 0.,
    "action_acc_weight": 0.,
    "action_jerk_weight": 0.,
    "soft_smoothing": 0,
    "hard_smoothing": 1,
    "record_video": 1,
    "video_recording_frequency": 20,
    "online_damage_probability": 0.0,
    "sample_model": False,

    # logging
    "result_dir": "results",
    "data_dir": "data/spotmicro/motor_damaged",
    "valid_dir": None,
    'training_tasks_index': [0, 1, 2, 4, 5, 6, 7, 8],
    "model_name": "spotmicro_meta_embedding_model",
    "meta_model_name": "meta_model",
    "env_name": "meta_spotmicro_04",
    "exp_suffix": "experiment",
    "exp_dir": None,
    "exp_details": "Default experiment.",
    "logdir": None,

    # Model_parameters
    "dim_in": 33 + 12,
    "dim_out": 33,
    "hidden_layers": [256, 256],
    "embedding_size": 5,
    "cuda": True,
    "output_limit": 10.0,
    "ensemble_batch_size": 16384,

    # Meta learning parameters
    "meta_iter": 5000,  # 5000,
    "meta_step": 0.3,
    "inner_iter": 10,  # 10,
    "inner_step": 0.0001,
    "meta_batch_size": 32,
    "inner_sample_size": 500,

    # Model learning parameters
    "epoch": 20,
    "learning_rate": 1e-4,
    "minibatch_size": 32,
    "hidden_activation": "relu",

    # Optimizer parameters
    "max_iters": 1,
    "epsilon": 0.0001,
    "lb": -1,
    "ub": 1,
    "popsize": 10000,
    "sol_dim": 12 * 25,  # NOTE: Depends on Horizon
    "max_action_velocity": 8,  # 10 from working controller
    "max_action_acceleration": 80,  # 100 from working controller
    "max_action_jerk": 8000,  # 10000 from working controller
    "max_torque_jerk": 25,
    "num_elites": 30,
    "cost_fn": None,
    "alpha": 0.1,
    "discount": 1.0
}

# optional arguments
parser = argparse.ArgumentParser()
parser.add_argument("--iterations",
                    help='Total episodes in episodic learning. Total MPC steps in the experiment.',
                    type=int)
parser.add_argument("--data_dir",
                    help='Path to load dynamics data and/or model',
                    type=str)
parser.add_argument("--exp_details",
                    help='Details about the experiment',
                    type=str)
parser.add_argument("--online",
                    action='store_true',
                    help='Will not reset back to init position', )
parser.add_argument("--adapt_steps",
                    help='Past steps to be used to learn a new model from the meta model',
                    type=int)
parser.add_argument("--control_steps",
                    help='Steps after which learn a new model => Learning frequency.',
                    type=int)
parser.add_argument("--rand_motor_damage",
                    action='store_true',
                    help='Sample a random joint damage.')
parser.add_argument("--rand_orientation_fault",
                    action='store_true',
                    help='Sample a random orientation estimation fault.')
parser.add_argument("--sample_model",
                    action='store_true',
                    help='Sample a model (task-id) using the likelihood information. Default: Picks the most likely model.')
parser.add_argument("--online_damage_probability",
                    help='Sample probabilistically random mismatch during mission. NOT used for episodic testing',
                    default=0.0,
                    type=float)
parser.add_argument('-c', '--config', action='append', nargs=2, default=[])
parser.add_argument('-logdir', type=str, default='None')

arguments = parser.parse_args()
if arguments.data_dir is not None: config['data_dir'] = arguments.data_dir
if arguments.iterations is not None: config['iterations'] = arguments.iterations
if arguments.exp_details is not None: config['exp_details'] = arguments.exp_details
if arguments.online is True:
    config['online'] = True
    if arguments.adapt_steps is not None: config['adapt_steps'] = arguments.adapt_steps
    if arguments.control_steps is not None: config['episode_length'] = arguments.control_steps
    if arguments.online_damage_probability is not None: config[
        'online_damage_probability'] = arguments.online_damage_probability
    print("Online learning with adaptation steps: ", config['adapt_steps'], " control steps: ",
          config['episode_length'])
else:
    print("Episodic learning with episode length: ", config['episode_length'])

if arguments.rand_motor_damage is not None: config['rand_motor_damage'] = arguments.rand_motor_damage
if arguments.rand_orientation_fault is not None: config['rand_orientation_fault'] = arguments.rand_orientation_fault
if arguments.sample_model is not None: config['sample_model'] = arguments.sample_model

logdir = arguments.logdir
config['logdir'] = None if logdir == 'None' else logdir

for (key, val) in arguments.config:
    if key in ['horizon', 'K', 'popsize', 'iterations', 'n_ensembles', 'episode_length']:
        config[key] = int(val)
    elif key in ['load_data', 'hidden_activation', 'data_size', 'save_data', 'script']:
        config[key] = val
    else:
        config[key] = float(val)

config['sol_dim'] = config['horizon'] * config['action_dim']
'''----------- Environment specific setup --------------'''

args = ["SpotMicroEnv-v0"]


def check_config(config):
    config['sol_dim'] = config['horizon'] * config['action_dim']


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
        'on_rack': False,
        "init_joint": np.array(config["init_joint"]),
        "ub": np.array(config["real_ub"]),
        "lb": np.array(config["real_lb"]),
        "normalized_action": True,
    }


def apply_config_params(conf, params):
    for (key, val) in list(params.items()):
        conf[key] = val
    return conf


kwargs = env_args_from_config(config)

exp_dir = None

# mismatches = ([0, 250], [{}, {'faulty_motors': [4], 'faulty_joints': [0]}])
# mismatches = ([0], [{'faulty_motors': [4], 'faulty_joints': [0]}])
test_mismatches = None

mismatches = [
    # ([0, 250], [{}, {'faulty_motors': [1], 'faulty_joints': [0]}]),
    # ([0, 250], [{}, {'faulty_motors': [2], 'faulty_joints': [-1]}]),
    ([0, 250], [{}, {'faulty_motors': [4], 'faulty_joints': [0]}]),
    # ([0, 250], [{}, {'faulty_motors': [5], 'faulty_joints': [-1]}]),
    # ([0, 250], [{}, {'faulty_motors': [7], 'faulty_joints': [0.45]}]),
    # ([0, 250], [{}, {'faulty_motors': [8], 'faulty_joints': [0]}]),
    # ([0, 250], [{}, {'faulty_motors': [10], 'faulty_joints': [0.5]}]),
    # ([0, 250], [{}, {'faulty_motors': [11], 'faulty_joints': [0]}]),
]
test_mismatches = []
config_params = []

adapt_steps = [10, 20, 50, 100, 200]
embedding_sizes = [1, 2, 5, 10]
epochs = [10, 20, 50, 100]
for a in adapt_steps:
    for embedding_size in embedding_sizes:
        for epoch in epochs:
            for i in range(len(mismatches)):
                config_params.append({"adapt_steps": a, 'successive_steps': 1, "epoch": epoch, "embedding_size": embedding_size,
                                      "meta_model_name": "damaged_without_FLT_embedding_size_"+str(embedding_size)})
                test_mismatches.append(mismatches[i])

n_run = len(config_params)
exp_dir = None
assert test_mismatches is None or len(test_mismatches) == n_run
for index in range(n_run):
    conf = copy.copy(config)
    conf['exp_dir'] = exp_dir
    conf = apply_config_params(conf, config_params[index])
    print('run ', index + 1, " on ", n_run)
    print("Params: ", config_params[index])
    check_config(conf)
    kwargs = env_args_from_config(conf)
    current_mismatches = mismatches if test_mismatches is None else test_mismatches[index]
    main(gym_args=args, gym_kwargs=kwargs, test_mismatch=current_mismatches, index=index, config=conf)
    exp_dir = conf['exp_dir']

    if index == 0:
        with open(exp_dir + "/test_mismatches.txt", "w") as f:
            f.write(str(test_mismatches))
        with open(exp_dir + "/config_params.txt", "w") as f:
            f.write(str(config_params))
    print("\n")
