import numpy as np
import pickle
import os
import shutil

path = 'exp/results/spot_micro_04/'
exps = ['26_02_2020_13_55_07_full_control3_copy']

copy = 1
copy_trajectories = False
copy_videos = False
copy_logs = True
copy_test_logs = False

files_to_copy = ['config.json', 'mismatches.npy', 'test_mismatches.npy']
dir_to_copy = ['models']
if copy_trajectories:
    files_to_copy.append('trajectories.npy')
if copy_logs:
    files_to_copy.append('logs.pk')
if copy_videos:
    dir_to_copy.append('videos')
if copy_test_logs:
    files_to_copy.append('test_logs.pk')


def copytree(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    if os.path.isdir(src):
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dst, item)
            if os.path.isdir(s):
                copytree(s, d, symlinks, ignore)
            else:
                if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                    shutil.copy2(s, d)
    else:
        shutil.copy2(src, dst)


Res = []
for exp in exps:
    exp_copy = path + exp + "_copy"
    if copy:
        os.makedirs(exp_copy)
    runs = os.listdir(path + exp)
    for run in runs:
        if run == 'config.txt':
            shutil.copy2(path + exp + "/" + run, exp_copy + "/" + run)
        else:
            run_copy = exp_copy + "/" + run
            if copy:
                os.makedirs(run_copy)
            files = os.listdir(path+exp+"/"+run)
            for file in files:
                if copy:
                    if file in files_to_copy:
                        shutil.copy2(path+exp+"/"+run+"/"+file, run_copy+"/"+file)
                    if file in dir_to_copy:
                        copytree(path+exp+"/"+run+"/"+file, run_copy+"/"+file)
                if file in ['logs.pk']:
                    compressed_data = {}
                    with open(path+exp+"/"+run+"/"+file, 'rb') as f:
                        data = pickle.load(f)
                    Xs, Ls = [], []
                    for obs in data['observations']:
                        X = [np.sum(np.array(x)[:, 30]) * 0.02 for x in obs]
                        L = [len(x) * 0.02 for x in obs]
                        Xs.append(np.copy(X))
                        Ls.append(np.copy(L))
                    compressed_data['X'] = np.copy(Xs)
                    compressed_data['L'] = np.copy(Ls)
                    with open(run_copy+"/"+file, 'wb') as f:
                        pickle.dump(compressed_data, f)
                elif file in ['test_logs.pk']:
                    compressed_data = {}
                    with open(path+exp+"/"+run+"/"+file, 'rb') as f:
                        data = pickle.load(f)
                    Xs, Ls = [], []
                    for model in data['observations']:
                        Xs_model, Ls_model = [], []
                        for env in model:
                            X = [np.sum(np.array(x)[:, 30]) * 0.02 for x in env]
                            L = [len(x) * 0.02 for x in env]
                            Xs_model.append(np.copy(X))
                            Ls_model.append(np.copy(L))
                        Xs.append(np.copy(Xs_model))
                        Ls.append(np.copy(Ls_model))
                    compressed_data['X'] = np.copy(Xs)
                    compressed_data['L'] = np.copy(Ls)
                    with open(run_copy+"/"+file, 'wb') as f:
                        pickle.dump(compressed_data, f)

