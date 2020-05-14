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
                          minibatch=config["minibatch_size"])
    return cloned_model, loss


def param_to_sin(f, phi):
    return lambda x: sin(f * x + phi)


base_config = {
                # experience param
                "env_param": [(1, 0), (1, pi), (1, pi / 2), (1, 3 * pi / 2)],
                "meta_train_size": 100,
                "adapt_size": 25,
                "exp_suffix": "toy_env",

                # model
                "dim_in": 1,
                "dim_out": 1,
                "hidden_layers": [128, 128],
                "embedding_size": 5,
                "cuda": True,
                "output_limit": None,
                "hidden_activation": "relu",

                # meta training
                "order": 1,
                "meta_iter": 20,  # 5000,
                "meta_step": 1e-3,
                "inner_iter": 500,  # 10,
                "inner_step": 1e-3,
                "meta_batch_size": 32,
                "inner_sample_size": 500,
                "K": 100,
                "M": None,  # =adapt_size

                # meta_adaptation
                "epoch": None,  # =inner_iter
                "learning_rate": 1e-4,
                "minibatch_size": 32,
}


def apply_config_params(old_config, params):
    new_config = copy.copy(old_config)
    for (key, val) in list(params.items()):
        new_config[key] = val
    new_config['epoch'] = new_config['inner_iter']
    new_config['M'] = new_config['adapt_size']
    return new_config


""" creating logs folder """
now = datetime.now()
timestamp = now.strftime("%d_%m_%Y_%H_%M_%S")
experiment_name = timestamp + "_" + base_config["exp_suffix"]
path = os.path.join(os.getcwd(), "results", "toy_env", experiment_name)
os.makedirs(path)

meta_step = [1e-3, 1e-4]
inner_step = [1e-3, 1e-4]
inner_iter = [100, 500]
meta_iter = [100, 1000]
adapt_size = [16, 32]
meta_train_size = [64, 128]
keys = ['meta_step', 'inner_step', 'inner_iter', 'meta_iter', 'adapt_size', 'meta_train_size']
somelists = [meta_step, inner_step, inner_iter, meta_iter, adapt_size, meta_train_size]
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
    base_in = np.arange(0, 2 * pi, 0.1)
    torch_base_in = torch.FloatTensor(base_in.reshape(-1, 1)).cuda()
    base_outs = []
    N = len(config['env_param'])
    train_loss, test_loss = [[] for _ in range(N)], [[] for _ in range(N)]
    meta_test_train_loss, meta_test_test_loss = [[] for _ in range(N)], [[] for _ in range(N)]
    predictions = [[] for _ in range(N)]

    train_in, train_out = [], []
    adapt_in, adapt_out = [], []
    for (f, phi) in config['env_param']:
        func = param_to_sin(f, phi)
        base_outs.append(func(base_in))
        x = np.random.uniform(0, 2 * pi, (config['meta_train_size'], 1))
        train_in.append(np.copy(x))
        train_out.append(func(x))
        x = np.random.uniform(0, 2 * pi, (config['adapt_size'], 1))
        adapt_in.append(np.copy(x))
        adapt_out.append(func(x))

    model = init_model(tasks_in=train_in, config=config)
    """ Before meta-training """
    models = [copy.deepcopy(model) for _ in range(N)]
    for task_id, m in enumerate(models):
        m.fix_task(task_id)
    for i in range(N):
        adapted_model, loss = train_model(model=models[i], train_in=adapt_in[i], train_out=adapt_out[i], task_id=i, config=config)
        train_loss[i] = np.copy(loss)
        predictions[i].append(adapted_model.predict_tensor(torch_base_in).cpu().detach().numpy())
        test_loss[i] = np.mean((predictions[i][-1][:, 0] - base_outs[i]) ** 2)

    """ Meta-training """
    trained_model, meta_train_test_loss, _, _ = train_meta(model, train_in, train_out, config, order=config['order'])
    plt.subplots(figsize=(16, 9))
    for i in range(N):
        plt.plot(meta_train_test_loss[i], label=str(i))
    plt.yscale('log')
    plt.legend()
    plt.xlabel("outer-loop iterations")
    plt.ylabel('loss')
    plt.title("Meta-training testing loss")
    plt.savefig(run_path+"/training_testing.svg")
    plt.close()
    """ Meta-adaptation """
    if config['meta_iter'] > 0:
        models = [copy.deepcopy(trained_model) for _ in range(N)]
        for i, m in enumerate(models):
            m.fix_task(i)
            adapted_model, loss = train_model(model=m, train_in=adapt_in[i], train_out=adapt_out[i], task_id=i,
                                        config=config)
            meta_test_train_loss[i] = np.copy(loss)
            predictions[i].append(adapted_model.predict_tensor(torch_base_in).cpu().detach().numpy())
            meta_test_test_loss[i] = np.mean((predictions[i][-1][:, 0] - base_outs[i]) ** 2)

    """ Plot """
    plt.subplots(figsize=(16, 9))
    for i in range(4):
        plt.plot(train_loss[i], colors[i], lw=3, ls=":", label="before meta-learning "+str(i), alpha=0.5)
        plt.plot(meta_test_train_loss[i], colors[i], label="after meta-learning", lw=3, alpha=0.5)
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
        plt.plot(base_in, predictions[i][0], ls=":", c=colors[i], label="without meta-learning")
        if config['meta_iter'] > 0:
            plt.plot(base_in, predictions[i][1], ls="--", lw=3, c=colors[i], label="with meta-learning")

    plt.legend()
    plt.xlabel("input")
    plt.ylabel('output')
    plt.title('Regression after adaptation with and without meta-learning')
    plt.savefig(run_path+"/regression.svg")
    plt.close()
    """ Saving logs """
    log = dict()
    log['base_in'] = base_in
    log['base_out'] = base_out
    log['adapt_in'] = adapt_in
    log['adapt_out'] = adapt_out
    log['predictions'] = predictions
    log['train_loss'] = train_loss
    log['meta_test_train_loss'] = meta_test_train_loss
    log['meta_test_test_loss'] = meta_test_test_loss
    log['meta_train_test_loss'] = meta_train_test_loss

    with open(run_path + '/logs.pk', 'wb') as f:
        pickle.dump(log, f)
    trained_model.save(file_path=run_path + "/meta_model.pt")
