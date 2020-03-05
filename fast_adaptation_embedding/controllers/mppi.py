
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import scipy.stats as stats
import numpy as np

class MPPI(object):
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
            samples = X.rvs(size=[self.popsize, self.sol_dim]) * np.sqrt(init_var) + mean
            samples = np.clip(samples, -1.0, 1.0)
            costs = self.cost_function(samples)
            # print(min_cost_indexes)
            min_cost_indexes = np.argsort(costs)
            reward = -costs[min_cost_indexes][:self.num_elites]
            # mini, maxi = reward[-1], reward[0]
            maxi = np.amax(np.absolute(reward)) + 1.0e-10
            # reward = np.exp((reward-mini) /(maxi-mini))
            # print("rewards: ", reward)
            reward = np.exp(10*reward/maxi)
            reward = reward / np.sum(reward)
            elites = samples[min_cost_indexes][:self.num_elites]

            weighted_elites = elites * reward.reshape(-1, 1)
            mean = np.sum(weighted_elites, axis=0)
            t += 1

        return elites[0] # instead of sol works better

if __name__ == '__main__':
    from test_env import Point

    horizon = 50
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
        return np.array(costs)

    config = {
                "max_iters": 10, 
                "epsilon": 0.01, 
                "lb": -1, 
                "ub": 1,
                "popsize": 100,
                "sol_dim": action_dim*horizon, 
                "num_elites": 10,
                "cost_fn": cost_fn, 
                "alpha": 0.01
    }

    init_mean, init_var = np.zeros(config["sol_dim"]), np.ones(config["sol_dim"])* 0.05
    opt = MPPI(config)
    for i in range(100):
        sol = opt.obtain_solution(init_mean, init_var)
        init_mean = np.zeros(len(sol))
        init_mean[0:-action_dim] = sol[action_dim::]
        a = sol[0:2]
        _ , _, _, _ = env.step(a)
        env.render()