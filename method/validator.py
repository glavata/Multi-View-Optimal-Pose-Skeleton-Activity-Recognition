import os
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from dtaidistance import dtw_ndim
import scipy.stats as stats
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from pathlib import Path

from utils.multi_view_util import multi_view_fuse, framewise_rotated_sequences_opt_pose
import pickle

def find_params_generator(generator, type):


    #os.makedirs(folder_name, exist_ok=True)
    #filename = folder_name + "//" + str(j)

    counter = 0
    #dists =np.load("dists.npy")
    dists = None

    # for node in generator:

    #     if(type == "mv_seq_eq"):
    #         X_act_t, _, _ = node
    #         full_poses_comb = framewise_rotated_sequences_opt_pose(X_act_t)
    #         full_poses_comb = full_poses_comb.reshape((full_poses_comb.shape[0], full_poses_comb.shape[1], -1))
    #         dists_tmp = full_poses_comb[1:]  - full_poses_comb[0]

    #         if(dists is None):
    #             dists = dists_tmp
    #         else:
    #             dists = np.concatenate([dists, dists_tmp], axis=1)
            
        
    #     counter+=1
  
    #dists = dists.reshape(dists.shape[0] * dists.shape[1], dists.shape[2])

    with open('dists.npy', 'rb') as f:
        dists = np.load(f)

    var_cls = np.sum((dists - np.mean(dists, axis=0))** 2, axis=0) /  dists.shape[0]	

    mu = np.mean(dists, axis=0)[64]
    variance = var_cls[64]

    sigma = np.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.hist(dists[:,64], density=True,bins=10000, color='b')
    plt.plot(x, stats.norm.pdf(x, mu, sigma),  color='r')
    plt.show()

    print(1)


def find_opt_kalman_params(generator):

    params_kalman = 1


def dtw_validation(dataset_tr):

    X_act_tr, seq_lens_sampl_tr, y_act_tr = dataset_tr[0], dataset_tr[1], dataset_tr[2]

    def flip_skel(skel):
        pass

    def dtw_check(a, b):
        dist, _ = dtw_ndim.warping_paths_fast(a,b)
        return a,b

    matrix = Parallel(n_jobs=4, backend="loky")(delayed(dtw_check)(y_act_tr[i], y_act_tr[j]) 
                                                for i in len(y_act_tr) for j in range(i))
    
    pass

def skel_transf_regressor(generator):


    X_act_t_stack = None
    X_act_t_rot_stack = None

    my_file = Path("skel_transf_data.npy")
    if my_file.is_file():
        with open('skel_transf_data.npy', 'rb') as f:
            X_act_t_rot_stack, X_act_t_stack = np.load(f)
    else:
        for node in generator:
            view_map, mat_all = node

            view_map_f = view_map[0]

            ind_diff = np.diff(view_map != view_map_f) != 0
            if(np.any(ind_diff)):
                view_map_s = view_map[1:][ind_diff][0]
                arg_inds = np.argwhere(ind_diff)[:, 0]

                arg_ind_f = arg_inds[0] + 1
                #arg_ind_s = arg_inds[1] + 1
                indices_sel = [0, arg_ind_f, int(arg_ind_f / 2)]
                for ind in indices_sel:
                    X_act_t, X_act_t_rot = mat_all[view_map_f, ind].reshape(75), mat_all[view_map_s, ind].reshape(75)

                    if(X_act_t_stack is None):
                        X_act_t_stack = X_act_t[None, :]
                        X_act_t_rot_stack = X_act_t_rot[None, :]
                    else:
                        X_act_t_stack = np.concatenate([X_act_t_stack, X_act_t[None, :]], axis=0)  
                        X_act_t_rot_stack = np.concatenate([X_act_t_rot_stack, X_act_t_rot[None, :]], axis=0)

    if(X_act_t_stack is not None):

        # regr = Pipeline([('poly', PolynomialFeatures(degree=2)),
        #           ('linear', MultiOutputRegressor(LinearRegression()))]) \
        #         .fit(X_act_t_rot_stack, X_act_t_stack)
        
        regr = MultiOutputRegressor(LinearRegression()).fit(X_act_t_rot_stack, X_act_t_stack)
        #regr = MLPRegressor(hidden_layer_sizes=(1024,), max_iter=50000, tol=0.1, early_stopping=True, activation='identity').fit(X_act_t_rot_stack, X_act_t_stack)
        #res = regr.predict(X_act_t_rot[None,0])
        with open('model.pkl','wb') as f:
            pickle.dump(regr,f)
    pass