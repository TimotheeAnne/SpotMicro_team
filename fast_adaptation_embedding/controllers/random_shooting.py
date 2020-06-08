
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
# import scipy.stats as stats
import numpy as np

class RS_opt(object):
    def __init__(self, config):
        self.max_iters = config["max_iters"]  # 20
        self.lb, self.ub = config["lb"], config["ub"]  # -1, 1
        self.popsize = config["popsize"]  # 200
        self.sol_dim = config["sol_dim"]  # 2*10 #action dim*horizon
        self.action_dim = config["action_dim"]
        self.cost_function = config["cost_fn"]
        self.max_vel = config["max_action_velocity"]
        self.max_acc = config["max_action_acceleration"]
        self.hard_smoothing = config['hard_smoothing']
        self.soft_smoothing = config['soft_smoothing']

    def obtain_solution(self, init_mean=None, init_var=None, acs=None):
        """Optimizes the cost function using the provided initial candidate distribution
        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        if self.hard_smoothing:
            assert acs is not None, "if hard smoothing is True, the 2 last actions must be given"
            samples = np.zeros((self.max_iters*self.popsize, self.sol_dim+3*self.action_dim))
            samples[:, :self.action_dim*3] = np.array([acs]*self.max_iters*self.popsize)
            best = self.cost_function(samples)
            return best[3*self.action_dim:]
        elif self.soft_smoothing:
            assert acs is not None, "if soft smoothing is True, the 2 last actions must be given"
            samples = np.random.uniform(self.lb, self.ub, (self.max_iters*self.popsize, self.sol_dim+3*self.action_dim))
            samples[:, :self.action_dim*3] = np.array([acs]*self.max_iters*self.popsize)
            costs = self.cost_function(samples)
            return samples[np.argmin(costs)][3*self.action_dim:]
        else:
            if init_mean is None or init_var is None:
                samples = np.random.uniform(self.lb, self.ub, size=(self.max_iters*self.popsize, self.sol_dim))
                best = self.cost_function(samples)
                return best
            else:
                assert init_mean is not None and init_var is not None, "init mean and var must be provided"
                samples = np.random.normal(init_mean, init_var, size=(self.max_iters * self.popsize, self.sol_dim))
                samples = np.clip(samples, self.lb, self.ub)
                costs = self.cost_function(samples)
                return samples[np.argmin(costs)]
