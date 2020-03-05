
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import scipy.stats as stats
import numpy as np

class CEM_opt(object):
    def __init__(self, config):
        self.max_iters = config["max_iters"]#20
        self.epsilon  = config["epsilon"] #0.001
        self.lb, self.ub = config["lb"], config["ub"]#-1, 1
        self.popsize = config["popsize"] #200
        self.sol_dim = config["sol_dim"] #2*10 #action dim*horizon
        self.num_elites = config["num_elites"] #50
        self.cost_function = config["cost_fn"]
        self.alpha = config["alpha"] #0.1

    def obtain_solution(self, init_mean, init_var):
        """Optimizes the cost function using the provided initial candidate distribution
        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """
        mean, var, t = init_mean, init_var, 0
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(mean))
        elites = []
        while (t < self.max_iters) and np.max(var) > self.epsilon:
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = var # Works better #np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var)

            samples = X.rvs(size=[self.popsize, self.sol_dim]) * np.sqrt(constrained_var) + mean
            samples = np.clip(samples, -1.0, 1.0)
            costs = self.cost_function(samples)
            elites = samples[np.argsort(costs)][:self.num_elites]

            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            # var = self.alpha * var + (1 - self.alpha) * new_var
            var = new_var # Works better
            t += 1
        sol, solvar = mean, var
        return elites[0] # instead of sol works better

if __name__ == '__main__':
    from test_env import Point

    horizon = 20
    action_dim = 2
    goal = [-7, 14]
    env = Point(goal)
    env.reset()
    dummy_env = Point(goal)
    dummy_env.reset()

    def cost_fn(samples):
        global dummy_env
        current_state = env.state()
        costs = []
        for s in samples:
            dummy_env.reset(current_state)
            total_cost = 0
            for i in range(horizon):
                a = s[2*i:2*i+2]
                state, cost, _, _ = dummy_env.step(a)
                total_cost += cost

            costs.append(total_cost)
        return costs

    config = {
                "max_iters": 20, 
                "epsilon": 0.01, 
                "lb": -1, 
                "ub": 1,
                "popsize": 200,
                "sol_dim": action_dim*horizon, 
                "num_elites": 50,
                "cost_fn": cost_fn, 
                "alpha": 0.01
    }

    init_mean, init_var = np.zeros(config["sol_dim"]), np.ones(config["sol_dim"])* 0.5
    opt = CEM_opt(config)
    for i in range(100):
        sol = opt.obtain_solution(init_mean, init_var)
        init_mean, init_var = np.zeros(config["sol_dim"]) , np.ones(config["sol_dim"])* 0.5 
        a = sol[0:2]
        _ , _, _, _ = env.step(a)
        env.render()