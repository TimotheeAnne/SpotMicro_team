# Implements this paper: https://openai.com/blog/reptile/
import os
import torch
from torch import nn, optim
import numpy as np
from copy import deepcopy
# from pyprind import ProgBar
# from utils import ProgBar
from tqdm import tqdm


class Embedding_NN(nn.Module):
    """
    simple Embedding_NN with dropout regularization
    """

    def __init__(self, dim_in, hidden, dim_out, embedding_dim, num_tasks, CUDA=False, SEED=None, output_limit=None,
                 dropout=0, hidden_activation="tanh"):

        super(Embedding_NN, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.output_limit = output_limit
        self.hidden = hidden
        self.hidden_activation = hidden_activation
        self.dropout_p = dropout
        self.embedding_dim = embedding_dim
        self.num_tasks = num_tasks

        self.activation = nn.ReLU() if hidden_activation == "relu" else nn.Tanh()
        self.Layers = nn.ModuleList()
        self.embeddings = nn.Embedding(self.num_tasks, self.embedding_dim)
        self.Layers.append(nn.Linear(dim_in + self.embedding_dim, hidden[0]))
        for i in range(0, len(hidden) - 1):
            self.Layers.append(nn.Linear(hidden[i], hidden[i + 1]))
        self.fcout = nn.Linear(hidden[-1], dim_out)

        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.seed = SEED
        if not SEED == None:
            torch.manual_seed(SEED)
            if CUDA:
                torch.cuda.manual_seed(SEED)

        self.cuda_enabled = CUDA
        if CUDA:
            self.cuda()

        self.data_mean_input = None  # torch.FloatTensor(np.mean(training_inputs, axis=0)).cuda() if self.CUDA else torch.FloatTensor(np.mean(training_inputs, axis=0))
        self.data_std_input = None  # torch.FloatTensor(np.std(training_inputs, axis=0)).cuda() if self.CUDA else torch.FloatTensor(np.std(training_inputs, axis=0))
        self.data_mean_output = None  # torch.FloatTensor(np.mean(training_targets, axis=0)).cuda() if self.CUDA else torch.FloatTensor(np.mean(training_targets, axis=0))
        self.data_std_output = None  # torch.FloatTensor(np.std(training_targets, axis=0)).cuda() if self.CUDA else torch.FloatTensor(np.std(training_targets, axis=0))
        self._fixed_task_id = None

    def forward(self, x, task_ids=None):
        outputs = []
        embed = self.embeddings(task_ids).reshape(-1, self.embedding_dim)
        x_embed = torch.cat((x, embed), 1)
        outputs.append(self.activation(self.Layers[0](x_embed)))

        if self.training:
            for i in range(1, len(self.hidden)):
                outputs.append(self.activation(self.dropout(self.Layers[i](outputs[i - 1]))))
        else:
            for i in range(1, len(self.hidden)):
                outputs.append(self.activation(self.Layers[i](outputs[i - 1])))

        if self.output_limit is None:
            return self.fcout(outputs[-1])
        else:
            return self.Tanh(self.fcout(outputs[-1])) * self.output_limit

    def predict(self, x, task_ids=None):
        '''Use predict methods when not optimizing the model'''
        # Normalize the input
        x_tensor = (torch.Tensor(x).cuda() - self.data_mean_input) / self.data_std_input if self.cuda_enabled else (
                                                                                                                               torch.Tensor(
                                                                                                                                   x) - self.data_mean_input) / self.data_std_input

        if task_ids is not None:
            task_ids_tensor = torch.LongTensor(task_ids).cuda() if self.cuda_enabled else torch.LongTensor(task_ids)
        else:
            tensor_shape = (x.shape[0], 1)
            task_ids_tensor = torch.LongTensor(
                np.ones(tensor_shape) * self._fixed_task_id).cuda() if self.cuda_enabled else torch.LongTensor(
                np.ones(tensor_shape) * self._fixed_task_id)
        # De-normalize 
        return (self.forward(x_tensor,
                             task_ids_tensor) * self.data_std_output + self.data_mean_output).detach().cpu().numpy()

    def predict_tensor(self, x, task_ids=None):
        '''Use predict methods when not optimizing the model'''
        x_normalized = (x - self.data_mean_input) / self.data_std_input
        if task_ids is not None:
            return (self.forward(x_normalized, task_ids) * self.data_std_output + self.data_mean_output).detach()
        else:
            tensor_shape = (x.size(0), 1)
            task_ids = torch.LongTensor(
                np.ones(tensor_shape) * self._fixed_task_id).cuda() if self.cuda_enabled else torch.LongTensor(
                np.ones(tensor_shape) * self._fixed_task_id)
            return (self.forward(x_normalized, task_ids) * self.data_std_output + self.data_mean_output).detach()

    def loss_function(self, y, y_pred):
        ''' y and y-pred must be normalized'''
        SE = (y - y_pred).pow(2).sum()
        return SE

    def loss_function_numpy(self, y, y_pred):
        ''' y and y-pred is un-normalized'''
        y_normalized = (y - self.data_mean_output.cpu().numpy()) / self.data_std_output.cpu().numpy()
        y_pred_normalized = (y_pred - self.data_mean_output.cpu().numpy()) / self.data_std_output.cpu().numpy()
        SE = np.power(y_normalized - y_pred_normalized, 2).sum()
        return SE

    def fix_task(self, task_id=None):
        '''
        task_id : int
        Fix the task id for the network so that output can be predicted just sending the x alone.
        '''
        if task_id is not None:
            assert task_id >= 0 and task_id < self.num_tasks, "task_id must be a positive integer less than number of tasks "+str(self.num_tasks)
            self._fixed_task_id = task_id

    def get_embedding(self, task_ids):
        task_ids_tensor = torch.LongTensor(task_ids).cuda() if model.cuda_enabled else torch.LongTensor(task_ids)
        return self.embeddings(task_ids_tensor).reshape(-1, self.embedding_dim).detach().cpu().numpy()

    def lik_function_numpy(self, y, y_pred, var=0.001):  # TODO: Test it first
        y_pred_var = np.ones(y.shape) * var
        logLik = -0.5 * np.log(1e-12 + y_pred_var) - 0.5 * np.power(y - y_pred, 2) / (1e-12 + y_pred_var)
        return logLik.sum()

    def save(self, file_path):
        kwargs = {"dim_in": self.dim_in,
                  "hidden": self.hidden,
                  "dim_out": self.dim_out,
                  "embedding_dim": self.embedding_dim,
                  "num_tasks": self.num_tasks,
                  "CUDA": self.cuda_enabled,
                  "SEED": self.seed,
                  "output_limit": self.output_limit,
                  "dropout": self.dropout_p,
                  "hidden_activation": self.hidden_activation}
        state_dict = self.state_dict()
        others = {"data_mean_input": self.data_mean_input,
                  "data_std_input": self.data_std_input,
                  "data_mean_output": self.data_mean_output,
                  "data_std_output": self.data_std_output,
                  "fixed_task_id": self._fixed_task_id}
        torch.save({"kwargs": kwargs, "state_dict": state_dict, "others": others}, file_path)


def load_model(file_path, device=torch.device('cpu')):
    model_data = torch.load(file_path, map_location=device)
    model_data["kwargs"]["CUDA"] = True if device == torch.device('cuda') else False
    model = Embedding_NN(**model_data["kwargs"])
    model.load_state_dict(model_data["state_dict"])
    model.data_mean_input = model_data["others"]["data_mean_input"]
    model.data_std_input = model_data["others"]["data_std_input"]
    model.data_mean_output = model_data["others"]["data_mean_output"]
    model.data_std_output = model_data["others"]["data_std_output"]
    model.fix_task(model_data["others"]["fixed_task_id"])
    print("\nLoaded model on ", device)
    return model


def train_meta(model, tasks_in, tasks_out, valid_in=[], valid_out=[],
               meta_iter=1000, inner_iter=10, inner_step=1e-3, meta_step=1e-3, minibatch=32,
               inner_sample_size=None):
    """
    model: Instance of Embedding_NN,
    tasks_in: list of input data for each task
    tasks_out: list of input data for each task
    meta_iter: Outer loop (meta update) count
    inner_iter: inner loop (task update) count
    inner_step: inner loop step size
    meta_step: outer loop step size
    minibatch: inner loop minibatch
    inner_sample_size: Inner loop sampling size. Samples the data and train on that data only per meta update.
    """

    with_validation = (valid_in != [])
    all_data_in = []
    all_data_out = []
    for task_id in range(len(tasks_in)):
        for i in range(len(tasks_in[task_id])):
            all_data_in.append(tasks_in[task_id][i])
            all_data_out.append(tasks_out[task_id][i])

    model.data_mean_input = torch.Tensor(np.mean(all_data_in, axis=0)).cuda() if model.cuda_enabled else torch.Tensor(
        np.mean(all_data_in, axis=0))
    model.data_mean_output = torch.Tensor(np.mean(all_data_out, axis=0)).cuda() if model.cuda_enabled else torch.Tensor(
        np.mean(all_data_out, axis=0))
    model.data_std_input = torch.Tensor(
        np.std(all_data_in, axis=0)).cuda() + 1e-10 if model.cuda_enabled else torch.Tensor(
        np.std(all_data_in, axis=0)) + 1e-10
    model.data_std_output = torch.Tensor(
        np.std(all_data_out, axis=0)).cuda() + 1e-10 if model.cuda_enabled else torch.Tensor(
        np.std(all_data_out, axis=0)) + 1e-10

    xx = [torch.Tensor(data).cuda() if model.cuda_enabled else torch.Tensor(data) for data in tasks_in]
    yy = [torch.Tensor(data).cuda() if model.cuda_enabled else torch.Tensor(data) for data in tasks_out]

    tasks_in_tensor = [(d - model.data_mean_input) / model.data_std_input for d in xx]
    tasks_out_tensor = [(d - model.data_mean_output) / model.data_std_output for d in yy]

    if with_validation:
        valid_xx = [torch.Tensor(data).cuda() if model.cuda_enabled else torch.Tensor(data) for data in valid_in]
        valid_yy = [torch.Tensor(data).cuda() if model.cuda_enabled else torch.Tensor(data) for data in valid_out]
        valid_in_tensor = [(d - model.data_mean_input) / model.data_std_input for d in valid_xx]
        valid_out_tensor = [(d - model.data_mean_output) / model.data_std_output for d in valid_yy]

    task_losses = np.zeros(len(tasks_in))
    Task_losses = [[] for _ in range(len(tasks_in))]
    Valid_losses = [[] for _ in range(len(tasks_in))]
    saved_embeddings = torch.empty((int(meta_iter/len(tasks_in))+1, len(tasks_in), model.embedding_dim))
    saved_embeddings[0] = deepcopy(model.embeddings.weight)
    tbar = tqdm(range(meta_iter))
    for meta_count in tbar:
        weights_before = deepcopy(model.state_dict())
        task_index = int(meta_count % len(tasks_in))
        batch_size = len(tasks_in[task_index]) if minibatch > len(tasks_in[task_index]) else minibatch
        tasks_tensor = torch.LongTensor(
            [[task_index] for _ in range(batch_size)]).cuda() if model.cuda_enabled else torch.LongTensor(
            [[task_index] for _ in range(batch_size)])
        final_loss = []
        for _ in range(inner_iter):
            permutation = np.random.permutation(
                len(tasks_in[task_index])) if inner_sample_size is None else np.random.permutation(
                len(tasks_in[task_index]))[0: min(inner_sample_size, len(tasks_in[task_index]))]
            x = tasks_in_tensor[task_index][permutation]
            y = tasks_out_tensor[task_index][permutation]
            model.train(mode=True)

            final_loss = []
            for i in range(0, x.size(0) - batch_size + 1, batch_size):
                model.zero_grad()
                pred = model(x[i:i + batch_size], tasks_tensor)
                loss = model.loss_function(pred, y[i:i + batch_size])
                loss.backward()
                for param in model.parameters():
                    param.data -= inner_step * param.grad.data
                final_loss.append(loss.item() / batch_size)
        if meta_count % 100 <= 1 and with_validation:
            valid_x = valid_in_tensor[task_index]
            valid_y = valid_out_tensor[task_index]
            valid_loss = []
            for i in range(0, valid_x.size(0) - batch_size + 1, batch_size):
                pred = model(valid_x[i:i + batch_size], tasks_tensor).detach()
                loss = model.loss_function(pred, valid_y[i:i + batch_size])
                valid_loss.append(loss.item() / batch_size)
            Valid_losses[task_index].append(np.mean(valid_loss))

        task_losses[task_index] = np.mean(final_loss)
        Task_losses[task_index].append(np.mean(final_loss))

        tbar.set_description("training " + str(task_losses))

        model.train(mode=False)
        weights_after = model.state_dict()
        stepsize = meta_step * (1 - meta_count / meta_iter)  # linear schedule
        model.load_state_dict(
            {name: weights_before[name] + (weights_after[name] - weights_before[name]) * stepsize for name in
             weights_before})
        if (meta_count+1) % len(tasks_in) == 0:
            saved_embeddings[int((meta_count+1)/len(tasks_in))] = deepcopy(model.embeddings.weight)
    return Task_losses, Valid_losses, saved_embeddings.detach().cpu().numpy()


def train(model, data_in, data_out, task_id, inner_iter=100, inner_lr=1e-3, minibatch=32, optimizer=None):
    """Train the NN model with given data"""

    if model.data_mean_input is None or model.data_std_input is None:
        model.data_mean_input = torch.Tensor(np.mean(data_in, axis=0)).cuda() if model.cuda_enabled else torch.Tensor(
            np.mean(data_in, axis=0))
        model.data_std_input = torch.Tensor(
            np.std(data_in, axis=0)).cuda() + 1e-10 if model.cuda_enabled else torch.Tensor(
            np.std(data_in, axis=0)) + 1e-10
        model.data_mean_output = torch.Tensor(np.mean(data_out, axis=0)).cuda() if model.cuda_enabled else torch.Tensor(
            np.mean(data_out, axis=0))
        model.data_std_output = torch.Tensor(
            np.std(data_out, axis=0)).cuda() + 1e-10 if model.cuda_enabled else torch.Tensor(
            np.std(data_out, axis=0)) + 1e-10

    data_in_tensor = (torch.Tensor(
        data_in).cuda() - model.data_mean_input) / model.data_std_input if model.cuda_enabled else (torch.Tensor(
        data_in) - model.data_mean_input) / model.data_std_input
    data_out_tensor = (torch.Tensor(
        data_out).cuda() - model.data_mean_output) / model.data_std_output if model.cuda_enabled else (torch.Tensor(
        data_out) - model.data_mean_output) / model.data_std_output

    batch_size = len(data_in) if minibatch > len(data_in) else minibatch
    tasks_tensor = torch.LongTensor(
        [[task_id] for _ in range(batch_size)]).cuda() if model.cuda_enabled else torch.LongTensor(
        [[task_id] for _ in range(batch_size)])

    optimizer = optim.Adam(model.parameters(), lr=inner_lr) if optimizer is None else optimizer
    model.train(mode=True)

    for inner_count in range(inner_iter):
        permutation = np.random.permutation(len(data_in))
        x = data_in_tensor[permutation]
        y = data_out_tensor[permutation]
        model.train(mode=True)
        losses = []
        for i in range(0, x.size(0) - batch_size + 1, batch_size):
            optimizer.zero_grad()
            pred = model(x[i:i + batch_size], tasks_tensor)
            loss = model.loss_function(pred, y[i:i + batch_size])
            loss.backward()
            optimizer.step()
            losses.append(loss.item() / batch_size)

        # ~ bar.update(item_id= "Iter " + str(inner_count) + " | Loss: "+str(np.mean(losses)))

    model.train(mode=False)


def gen_test_task(task_id):
    phase = np.random.uniform(low=-0.1 * np.pi, high=0.1 * np.pi)
    ampl = np.random.uniform(0.3, 1.0)
    # if np.random.rand() > 0.5:
    if task_id % 2 == 0:
        f_randomsine = lambda x: np.sin(x) * ampl
    else:
        f_randomsine = lambda x: np.sin(x + np.pi) * ampl
    return f_randomsine


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from copy import deepcopy

    plt.figure(0)

    # Parameters
    num_tasks = 5
    meta_iter = 1000
    m_step = 0.1
    inner_iter = 100
    n_step = 0.001
    n_training = 2

    # Generate the meta training tasks or the data
    tasks_in = []
    tasks_out = []
    for i in range(num_tasks):
        x = np.linspace(-6, 6, np.random.randint(90, 100)).reshape(-1, 1)
        f = gen_test_task(i)
        y = f(x)
        tasks_in.append(x)
        tasks_out.append(y)

    test_in = tasks_in[0]
    test_out = tasks_out[0]

    # Training data
    indices = np.random.permutation(len(test_in))[0:n_training]
    train_in, train_out = test_in[indices], test_out[indices]

    '''-----------Meta learning------------'''
    model = Embedding_NN(dim_in=1, hidden=[20, 20, 20], dim_out=1, embedding_dim=5, num_tasks=len(tasks_in), CUDA=True,
                         SEED=None, output_limit=None, dropout=0.0)
    train_meta(model, tasks_in, tasks_out, meta_iter=meta_iter, inner_iter=inner_iter, inner_step=n_step,
               meta_step=m_step, minibatch=128)
    model.save("model.pt")
    # model = load_model("model.pt", torch.device('cuda'))
    # Model before training with data
    tasks_tensor = torch.LongTensor(
        [[0] for _ in range(len(test_in))]).cuda() if model.cuda_enabled else torch.LongTensor(
        [[0] for _ in range(len(test_in))])
    predict_before = model.predict_tensor(torch.Tensor(test_in).cuda(), tasks_tensor).data.cpu().numpy()
    plt.plot(test_in, predict_before, '--b', alpha=0.5, label="Before training")

    '''----------Train model with optimized init params-----------'''
    train(model, train_in, train_out, task_id=0, inner_iter=1000, inner_lr=1e-3, minibatch=32)

    # Model after training with data
    predict_after = model.predict_tensor(torch.Tensor(test_in).cuda(), tasks_tensor).data.cpu().numpy()
    plt.plot(test_in, predict_after, '-b', label="After training(meta)")
    plt.plot(test_in, test_out, '-r', label="Test task")
    plt.plot(train_in, train_out, 'xk')
    for i in range(len(tasks_in)):
        plt.plot(tasks_in[i], tasks_out[i], '-g', alpha=0.1)

    '''--------------No meta leaning-----------------------'''
    model2 = Embedding_NN(dim_in=1, hidden=[20, 20, 20], dim_out=1, embedding_dim=5, num_tasks=len(tasks_in),
                          CUDA=False, SEED=None, output_limit=None, dropout=0.0)
    train(model2, train_in, train_out, task_id=0, inner_iter=1000, inner_lr=1e-3, minibatch=32)
    tasks_tensor = torch.LongTensor(
        [[0] for _ in range(len(test_in))]).cuda() if model2.cuda_enabled else torch.LongTensor(
        [[0] for _ in range(len(test_in))])
    predict_after = model2.predict_tensor(torch.Tensor(test_in), tasks_tensor).data.numpy()
    plt.plot(test_in, predict_after, '-k', label="After training(no meta)")

    ''' ----Check embeddings----'''
    # x = np.arange(0, num_tasks, 1)
    # embeddings = model.get_embedding(x)
    # print(embeddings)
    # from sklearn.manifold import TSNE
    # X_embedded = TSNE(n_components=2).fit_transform(embeddings)
    # print(X_embedded)
    # plt.figure(1)
    # for t in range(num_tasks):
    #     if t%2 == 0:
    #         plt.plot(X_embedded[t][0], X_embedded[t][1], 'or')
    #     else:
    #         plt.plot(X_embedded[t][0], X_embedded[t][1], 'ob')

    # plt.plot(X_embedded[0][0], X_embedded[0][1], 'k')
    # plt.legend()
    plt.show()
