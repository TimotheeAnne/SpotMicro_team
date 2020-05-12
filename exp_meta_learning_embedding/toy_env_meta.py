import torch
import numpy as np
import copy
import pickle
import os
from os import path
from tqdm import trange
import os, inspect
import json
import matplotlib.pyplot as plt

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


def param_to_sin(f, phi):
    return lambda x: sin(f * x + phi)


config = {
    # experience param
    "env_param": [(1, 0), (1, pi), (2, 0), (2, pi)],
    "train_size": 100,
    "adapt_size": 100,

    # model
    "dim_in": 1,
    "dim_out": 1,
    "hidden_layers": [128, 128],
    "embedding_size": 0,
    "cuda": True,
    "output_limit": None,
    "hidden_activation": "relu",

    # meta training
    "meta_iter": 500,  # 5000,
    "meta_step": 0.3,
    "inner_iter": 10,  # 10,
    "inner_step": 0.0001,
    "meta_batch_size": 32,
    "inner_sample_size": 500,

    # meta_adaptation
    "epoch": 20,
    "learning_rate": 1e-4,
    "minibatch_size": 32,
}

base_in = np.arange(0, 2 * pi, 0.1)
torch_base_in = torch.FloatTensor(base_in.reshape(-1, 1)).cuda()
base_outs = []

train_in = []
train_out = []

""" Meta-training """
for (f, phi) in config['env_param']:
    func = param_to_sin(f, phi)
    base_outs.append(func(base_in))
    x = np.random.uniform(0, 2*pi, (config['train_size'], 1))
    train_in.append(np.copy(x))
    train_out.append(func(x))

trained_model, _, _, _ = train_meta(train_in, train_out, config)
test_out = trained_model.predict_tensor(torch_base_in).cpu().detach().numpy()

""" Meta-adaptation """
adapt_out = []
for i, (f, phi) in enumerate(config['env_param']):
    func = param_to_sin(f, phi)
    x = np.random.uniform(0, 2*pi, (config['adapt_size'], 1))
    y = func(x)
    task_index = i
    adapted_model = train_model(model=copy.deepcopy(trained_model), train_in=x, train_out=y, task_id=task_index, config=config)
    adapt_out.append(adapted_model.predict_tensor(torch_base_in).cpu().detach().numpy())

""" Plot """
plt.plot(base_in, test_out, ls="-", label="before adaptation")
for i, base_out in enumerate(base_outs):
    plt.plot(base_in, base_out, colors[i], label="base "+str(i), lw=3)
    plt.scatter(train_in[i], train_out[i], marker="o", s=100, c=colors[i], label="training")
    plt.plot(base_in, adapt_out[i], ls=":", lw=3, c=colors[i], label="after adaptation")

plt.legend()
plt.show()
