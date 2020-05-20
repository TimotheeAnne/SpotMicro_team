import torch
import numpy as np
import copy
import pickle
import os
from tqdm import trange
import inspect
import json
import matplotlib.pyplot as plt
from datetime import datetime
import itertools


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import fast_adaptation_embedding.env
import fast_adaptation_embedding.models.embedding_nn_normalized_v2 as nn_model

pi = np.pi
sin = np.sin
colors = ['#332288', '#117733',
          '#88CCEE', '#DDCC77', '#CC6677',
          '#AA4499', '#882255']


def init_model(tasks_in, config):
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
    return model


def train_meta(model, tasks_in, tasks_out, config, valid_in=[], valid_out=[], order=1):
    if order == 1:
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
    else:
        task_losses, valid_losses, saved_embeddings = nn_model.train_meta2(model,
                                                                           tasks_in,
                                                                           tasks_out,
                                                                           meta_iter=config["meta_iter"],
                                                                           inner_iter=config["inner_iter"],
                                                                           inner_step=config["inner_step"],
                                                                           meta_step=config["meta_step"],
                                                                           normalization=config["normalization"],
                                                                           inner_n_tasks=config['inner_n_tasks'],
                                                                           K=config["K"],
                                                                           M=config["M"])
    return model, task_losses, saved_embeddings, valid_losses


def train_model(model, train_in, train_out, task_id, config):
    cloned_model = copy.deepcopy(model)
    loss = nn_model.train(cloned_model,
                          train_in,
                          train_out,
                          task_id=task_id,
                          inner_iter=config["epoch"],
                          inner_lr=config["learning_rate"],
                          normalization=config["normalization"],
                          minibatch=config["minibatch_size"])
    return cloned_model, loss


def param_to_sin(a, phi):
    return lambda x: a*sin(x + phi)


base_config = {
                # experience param
                "env_param": [(1, 0), (0.5, pi/2), (2.5, 3*pi / 4)],
                "total_n_tasks": 1000,
                "inner_n_tasks": 16,
                "meta_train_size": 100,
                "adapt_size": 25,
                "exp_suffix": "toy_env",

                # model
                "dim_in": 1,
                "dim_out": 1,
                "hidden_layers": [64, 64],
                "embedding_size": 0,
                "cuda": True,
                "output_limit": None,
                "hidden_activation": "relu",

                # meta training
                "normalization": False,
                "order": 1,
                "meta_iter": 20,  # 5000,
                "meta_step": 1e-3,
                "inner_iter": 500,  # 10,
                "inner_step": 1e-3,
                "meta_batch_size": 32,
                "inner_sample_size": 500,
                "K": 25,
                "M": None,  # =adapt_size

                # meta_adaptation
                "epoch": 32,
                "learning_rate": 1e-4,
                "minibatch_size": 32,
                "inner_optimize": 'SGD',
}


def apply_config_params(old_config, params):
    new_config = copy.copy(old_config)
    for (key, val) in list(params.items()):
        new_config[key] = val
    new_config['M'] = new_config['adapt_size']
    new_config['total_n_tasks'] = new_config['meta_iter']
    return new_config


""" creating logs folder """
now = datetime.now()
timestamp = now.strftime("%d_%m_%Y_%H_%M_%S")
experiment_name = timestamp + "_" + base_config["exp_suffix"]
path = os.path.join(os.getcwd(), "results", "toy_env", experiment_name)
os.makedirs(path)

meta_step = [1]
inner_step = [1e-3]
inner_iter = [512]
epoch = [1024]
meta_iter = [1000]
adapt_size = [128]
K = [64]
meta_train_size = [128]
keys = ['meta_step', 'inner_step', 'inner_iter', 'epoch', 'meta_iter', 'adapt_size', 'meta_train_size']
somelists = [meta_step, inner_step, inner_iter, epoch, meta_iter, adapt_size, meta_train_size]
config_params = []
for element in itertools.product(*somelists):
    param = {}
    for i, key in enumerate(keys):
        param[key] = element[i]
    config_params.append(copy.copy(param))

for i, config_param in enumerate(config_params):
    print("\n run ", i+1, " on ", len(config_params))
    print(config_param)
    config = apply_config_params(base_config, config_param)
    run_path = path + "/run_"+str(i)
    os.makedirs(run_path)
    with open(run_path + '/config.pk', 'wb') as f:
        pickle.dump(config, f)
    with open(run_path + "/config.txt", "w") as f:
        f.write(str(config))
    """ Creating training and testing data """
    base_in = np.arange(-5, 5, 0.1)
    torch_base_in = torch.FloatTensor(base_in.reshape(-1, 1)).cuda()
    base_outs = []
    N = len(config['env_param'])
    train_loss, test_loss = [[] for _ in range(N)], [[] for _ in range(N)]
    meta_test_train_loss, meta_test_test_loss = [[] for _ in range(N)], [[] for _ in range(N)]
    predictions = [[] for _ in range(N)]

    train_in, train_out = [], []
    for _ in range(config['total_n_tasks']):
        a = np.random.uniform(0.1, 5)
        phi = np.random.uniform(0, pi)
        func = param_to_sin(a, phi)
        x = np.random.uniform(-5, 5, (config['meta_train_size'], 1))
        train_in.append(np.copy(x))
        train_out.append(func(x))

    adapt_in, adapt_out = [], []
    for (a, phi) in config['env_param']:
        func = param_to_sin(a, phi)
        base_outs.append(func(base_in))
        x = np.random.uniform(-5, 5, (config['adapt_size'], 1))
        adapt_in.append(np.copy(x))
        adapt_out.append(func(x))

    model = init_model(tasks_in=train_in, config=config)
    """ Before meta-training """
    models = [copy.deepcopy(model) for _ in range(N)]
    for task_id, m in enumerate(models):
        m.fix_task(task_id)
    for i in range(N):
        predictions[i].append(models[i].predict_tensor_without_normalization(torch_base_in).cpu().detach().numpy())
        adapted_model, loss = train_model(model=models[i], train_in=adapt_in[i], train_out=adapt_out[i], task_id=i, config=config)
        train_loss[i] = np.copy(loss)
        predictions[i].append(adapted_model.predict_tensor_without_normalization(torch_base_in).cpu().detach().numpy())
        test_loss[i] = np.mean((predictions[i][-1][:, 0] - base_outs[i]) ** 2)

    # """ Meta-training """
    # trained_model, meta_train_test_loss, _, _ = train_meta(model, train_in, train_out, config, order=config['order'])
    # plt.subplots(figsize=(16, 9))
    # if len(meta_train_test_loss[i]) == N:
    #     for i in range(N):
    #         plt.plot(meta_train_test_loss[i], label=str(i))
    # else:
    #     res = []
    #     for i in range(config['meta_iter']):
    #         i1 = i % config['total_n_tasks']
    #         i2 = i//config['total_n_tasks']
    #         res.append(meta_train_test_loss[i1][i2])
    #     plt.plot(res)
    # plt.yscale('log')
    # # plt.legend()
    # plt.xlabel("outer-loop iterations")
    # plt.ylabel('loss')
    # plt.title("Meta-training testing loss")
    # plt.savefig(run_path+"/training_testing.svg")
    # plt.close()
    # """ Meta-adaptation """
    # if config['meta_iter'] > 0:
    #     models = [copy.deepcopy(trained_model) for _ in range(N)]
    #     for i, m in enumerate(models):
    #         m.fix_task(i)
    #         adapted_model, loss = train_model(model=m, train_in=adapt_in[i], train_out=adapt_out[i], task_id=i,
    #                                     config=config)
    #         meta_test_train_loss[i] = np.copy(loss)
    #         predictions[i].append(adapted_model.predict_tensor_without_normalization(torch_base_in).cpu().detach().numpy())
    #         meta_test_test_loss[i] = np.mean((predictions[i][-1][:, 0] - base_outs[i]) ** 2)
    #
    # """ Plot """
    plt.subplots(figsize=(16, 9))
    for i in range(len(train_loss)):
        plt.plot(train_loss[i], colors[i], lw=3, ls=":", label="before meta-learning "+str(i), alpha=0.5)
        # plt.plot(meta_test_train_loss[i], colors[i], label="after meta-learning", lw=3, alpha=0.5)
    plt.yscale('log')
    plt.xlabel("adaptation iterations")
    plt.ylabel('loss')
    plt.legend()
    plt.title("Adaptation loss comparison with and without meta-learning")
    plt.savefig(run_path+"/testing_testing_loss.svg")
    plt.close()
    plt.subplots(figsize=(16, 9))
    for i, base_out in enumerate(base_outs):
        plt.plot(base_in, base_out, colors[i], label="base " + str(i), lw=3)
        plt.scatter(adapt_in[i], adapt_out[i], marker="o", s=100, c=colors[i])
        plt.plot(base_in, predictions[i][0], ls=":", c=colors[i], label="init model")
        plt.plot(base_in, predictions[i][1], ls=":", c=colors[i], label="adaptation without meta-learning")
        # if config['meta_iter'] > 0:
        #     plt.plot(base_in, predictions[i][1], ls="--", lw=3, c=colors[i], label="with meta-learning")

    plt.legend()
    plt.xlabel("input")
    plt.ylabel('output')
    plt.title('Regression after adaptation with and without meta-learning')
    plt.savefig(run_path+"/regression.svg")
    plt.close()
    # """ Saving logs """
    # log = dict()
    # log['base_in'] = base_in
    # log['base_outs'] = base_outs
    # log['adapt_in'] = adapt_in
    # log['adapt_out'] = adapt_out
    # log['predictions'] = predictions
    # log['train_loss'] = train_loss
    # log['meta_test_train_loss'] = meta_test_train_loss
    # log['meta_test_test_loss'] = meta_test_test_loss
    # log['meta_train_test_loss'] = meta_train_test_loss
    #
    # with open(run_path + '/logs.pk', 'wb') as f:
    #     pickle.dump(log, f)
    # trained_model.save(file_path=run_path + "/meta_model.pt")
