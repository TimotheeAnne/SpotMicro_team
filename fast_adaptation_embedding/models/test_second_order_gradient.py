import torch
from torch import optim
from torch import Tensor
from torch.autograd import Variable
from torch.autograd import grad
from torch import nn
from copy import deepcopy

# https://discuss.pytorch.org/t/how-to-print-gradient-graph/67245/5
# https://github.com/dragen1860/MAML-Pytorch/blob/master/meta.py


def f(x, y):
    return (x + y)**2

# some toy data
x = Variable(Tensor([1., 2.]), requires_grad=False)
y = Variable(Tensor([1.]), requires_grad=False)

x2 = Variable(Tensor([1., 0.]), requires_grad=False)
y2 = Variable(Tensor([1.]), requires_grad=False)

# linear model and squared difference loss
model = nn.Linear(2, 1)
model.weight.data.fill_(1)
model.bias.data.fill_(1)
learning_rate = 1e-2


saved_init_param = []

learner = deepcopy(model)

print("Initialization")
for (param, learner_param) in zip(model.parameters(), learner.parameters()):
    learner_param = param.clone()


loss = torch.sum((y - learner(x))**2)
loss.backward(retain_graph=True, create_graph=True)

for i, param in enumerate(learner.parameters()):
    param.data -= learning_rate * param.grad.data

print("\nFirst step")
for i, (name, param) in enumerate(model.named_parameters()):
    print(name, param)

loss2 = torch.sum((y2 - learner(x2))**2)
dd = grad(loss2, model.parameters(), create_graph=True)

print("\nSecond step")
for i, (name, param) in enumerate(model.named_parameters()):
    print(name, param)
