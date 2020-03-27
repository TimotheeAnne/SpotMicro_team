import numpy as np
import pickle

files = ['random_0_traj_9999.pk', 'random_1_traj_9999.pk', 'random_2_traj_19999.pk', 'random_3_traj_9999.pk']
Res = []
for file in files:
    with open('data/'+file, 'rb') as f:
        [O, A] = pickle.load(f)
    res = []
    for obs in O:
        obs = np.array(obs)
        res.append(np.sum(obs[:, 30:32])*0.02)
    Res.append(res)

with open("compressed.pk", 'wb') as f:
    pickle.dump(Res, f)
