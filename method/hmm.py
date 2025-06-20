
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import pomegranate
from pomegranate import IndependentComponentsDistribution, NormalDistribution, HiddenMarkovModel
#from torch.masked import masked_tensor, as_masked_tensor
#from pomegranate.hmm import DenseHMM, SparseHMM
#from pomegranate.distributions import IndependentComponents, Normal, Categorical, Uniform

import cv2 as cv2
from pathlib import Path
import time
import pickle
from utils import logger
from utils.skel_visualization import SkeletonVisualizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC, SVC
from sklearn import decomposition
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans


LINKS_KINECT_V2 = np.array([
    (1, 0), (20, 1), (2, 20), (3, 2), 
    (4, 20), (5, 4), (6, 5), (7, 6),
    (8, 20), (9, 8), (10, 9), (11, 10),
    (12, 0), (13, 12), (14, 13), (15, 14),
    (16, 0), (17, 16), (18, 17), (19, 18),
    (21, 7), (22, 7), (23, 11), (24, 11)
])
child_indices = LINKS_KINECT_V2[:,0]
parents_indices = LINKS_KINECT_V2[:,1]

class HMM():

    def __init__(self, dataset_tr, dataset_ts, n_comp=5, train_acc=False):
        self.d_tr = dataset_tr
        self.d_ts = dataset_ts
        self.n_comp = n_comp
        self.train_acc = train_acc


    def __hmm_train_new(self, X_act_tr, y_act_tr, seq_lens_sampl_tr, cls_all, bic_measure=False):

        models_hmm = []
        bic_scores = []

        act_lens_cls = seq_lens_sampl_tr[:, 0]

        X_tmp = [X_act_tr[i, :act_lens_cls[i], 3:] for i in range(X_act_tr.shape[0])]

        model_f = HiddenMarkovModel.from_samples(distribution=NormalDistribution, stop_threshold=0.1,
                                                n_components=self.n_comp, max_iterations=300,
                                                X=X_tmp, verbose=True)

        full_dense_matrix = model_f.dense_transition_matrix()
        #full_dense_matrix = full_dense_matrix[:-2, :-2]

        emission_dists = []
        for s in model_f.states:
            distr = s.distribution
            if distr is None:
                continue
            cur_dists = []
            for d in distr:
                cur_dists.append(NormalDistribution(d.parameters[0],d.parameters[1]))
            emission_dists.append(IndependentComponentsDistribution(cur_dists))

        for k in range(cls_all.shape[0]):
            
            c =  cls_all[k]
            print(" Creating HMM for class {0} from ready params...".format(c))

            ind_query_cls = np.argwhere(y_act_tr[:, 0] == c)[:, 0]

            act_cls_cur = X_act_tr[ind_query_cls, :, :]

            act_lens_cls = seq_lens_sampl_tr[ind_query_cls, 0]

            list_states = []
            X_tmp = [act_cls_cur[i, :act_lens_cls[i], 3:] for i in range(act_cls_cur.shape[0])]
            for x in X_tmp:
                logp, vmap = model_f.viterbi(x)
                list_states.extend([v[0] for v in vmap])

            unique_states = np.unique(np.array(list_states), return_counts=True)
            unique_states = np.argsort(unique_states[1])[::-1][:20]
            zero_state_map = np.ones(self.n_comp + 2, dtype=bool)
            zero_state_map[unique_states] = False
            full_dense_matrix_cls = np.copy(full_dense_matrix)
            full_dense_matrix_cls[zero_state_map, :] = 0.0
            full_dense_matrix_cls[:, zero_state_map] = 0.0

            starts = full_dense_matrix_cls[-2]
            ends = full_dense_matrix_cls[-1]
            if(np.all(starts == 0.0)):
                starts = np.ones(self.n_comp + 2) / self.n_comp
                starts[zero_state_map] = 0.0
            if(np.all(ends == 0.0)):
                ends = np.ones(self.n_comp + 2) / self.n_comp
                ends[zero_state_map] = 0.0

            model_c = HiddenMarkovModel.from_matrix(full_dense_matrix_cls[:-2, :-2], 
                                                    emission_dists, starts[:-2], ends[:-2])
            models_hmm.append(model_c)

            if(bic_measure):
                 bic_tmp = self.bic_score(X_tmp, model_c, self.n_comp)
                 bic_scores.append(bic_tmp)

        return models_hmm, bic_scores
    
    def __hmm_train(self, X_act_tr, y_act_tr, seq_lens_sampl_tr, cls_all, bic_measure=False):

        models_hmm = []
        bic_scores = []

        for k in range(cls_all.shape[0]):
            
            c =  cls_all[k]
            print(" Training HMM for class {0} ...".format(c))

            ind_query_cls = np.argwhere(y_act_tr[:, 0] == c)[:, 0]


            # np.random.shuffle(ind_query_cls)
            # if(k == 30):
            #     self.n_comp = 19
            #     #ind_query_cls = ind_query_cls[:-10]
            # else:
            #     self.n_comp = 20

            act_cls_cur = X_act_tr[ind_query_cls, :, :]

            act_lens_cls = seq_lens_sampl_tr[ind_query_cls, 0]

            X_tmp = [act_cls_cur[i, :act_lens_cls[i], 3:] for i in range(act_cls_cur.shape[0])]

            model_c = HiddenMarkovModel.from_samples(distribution=NormalDistribution, stop_threshold=0.1,
                                                    n_components=self.n_comp, max_iterations=300,
                                                    X=X_tmp, verbose=True)
            
            res_tmp = model_c.log_probability(act_cls_cur[0, :act_lens_cls[0], 3:])
            if(np.isnan(res_tmp)):
                raise "Error in training!"
            
            models_hmm.append(model_c)

            if(bic_measure):
                print("Calculating BIC score...")
                bic_tmp = self.bic_score(X_tmp, model_c, self.n_comp)
                bic_scores.append(bic_tmp)

        return models_hmm, bic_scores
    
    def __hmm_test(self, models_hmm, X_act_ts, seq_lens_sampl_ts, cls_all):

        pred_all = np.zeros((X_act_ts.shape[0], cls_all.shape[0]))

        for i in range(X_act_ts.shape[0]):

            act_cls_cur = X_act_ts[i]
            act_lens_cls = seq_lens_sampl_ts[i, 0]

            for m in range(len(models_hmm)):
                pred_all[i, m] =  models_hmm[m].log_probability(act_cls_cur[:act_lens_cls, 3:])

        return pred_all

    #takes already formatted X_tmp
    def bic_score(self, X_tmp, model, n_comp):
        #param_p = (n_comp - 1) + n_comp * (n_comp - 1) + 2 * (n_comp * X_tmp[0].shape[-1]) 
        param_p = n_comp
        param_c = len(X_tmp)

        total_score = 0
        for i in range(len(X_tmp)):
            total_score += model.log_probability(X_tmp[i])
            
        total_score = -2 * total_score + param_p * np.log(param_c)
        return total_score

    def hmm_method_pome(self, X_act_tr, X_act_ts, y_act_tr, y_act_ts, seq_lens_sampl_tr, seq_lens_sampl_ts):
        cls_all = np.unique(y_act_tr)

        start = time.time()  
        #training
        models_hmm, bic_scores = self.__hmm_train(X_act_tr, y_act_tr, seq_lens_sampl_tr, cls_all)
        #models_hmm, bic_scores = self.__hmm_train_new(X_act_tr, y_act_tr, seq_lens_sampl_tr, cls_all)

        end = time.time()
        train_time = end - start
        print(" Training time for n_comp={0} : {1} seconds".format(self.n_comp, train_time))


        start = time.time()

        print("Testing data...")
        pred_all_test = self.__hmm_test(models_hmm, X_act_ts, seq_lens_sampl_ts, cls_all)
        y_pred_test = np.argmax(pred_all_test, axis=1)
        y_pred_test = cls_all[y_pred_test]

        end = time.time()
        test_time = end - start
        print(" Testing time for n_comp={0} : {1} seconds".format(self.n_comp, test_time))
        
        #train accuracy too?
        if(self.train_acc):
            print("Testing training data...")
            pred_all_train = self.__hmm_test(models_hmm, X_act_tr, seq_lens_sampl_tr, cls_all)
            y_pred_train = np.argmax(pred_all_train, axis=1)
            y_pred_train = cls_all[y_pred_train]
            acc_train = accuracy_score(y_act_tr[:, 0], y_pred_train)
        else:
            acc_train = None

        acc_test = accuracy_score(y_act_ts[:, 0], y_pred_test)
        conf_mat_test = confusion_matrix(y_act_ts[:, 0], y_pred_test)
        print(" Test accuracy for n_comp={0}".format(acc_test))

        return acc_test, conf_mat_test, acc_train, models_hmm, bic_scores, train_time, test_time

def save_hmm_model(filename, hmm):
    python_obj = [v.to_json() for k, v in enumerate(hmm)]

    with open("results//hmm_models//{0}.pkl".format(filename), 'wb') as outp:
        pickle.dump(python_obj, outp, pickle.HIGHEST_PROTOCOL)

def load_hmm_model(filename):
    with open("results//hmm_models//{0}.pkl".format(filename), 'rb') as pickle_file:
        hmm_obj_json = pickle.load(pickle_file)
    hmm_models = [HiddenMarkovModel.from_json(o) for o in hmm_obj_json]
    return hmm_models



def train_hmm(dataset_tr, dataset_ts, fused_str, n_comp_arr=[5], train_acc=False):
    
    if(not isinstance(n_comp_arr, list)):
        raise Exception("n_comp_arr parameter must be a list")

    X_act_tr, seq_lens_sampl_tr, y_act_tr = dataset_tr[0], dataset_tr[1], dataset_tr[2]
    X_act_ts, seq_lens_sampl_ts, y_act_ts = dataset_ts[0], dataset_ts[1], dataset_ts[2]

    acc_list = []
    #models_hmm_list = []
    #conf_mats = []
    bic_scores_list = []
    acc_tr_list = []
    train_times = []
    test_times = []

    for n in n_comp_arr:
        hmm = HMM(dataset_tr, dataset_ts, n_comp=n, train_acc=train_acc)

        print("Starting {0} HMM test with n_comp={1}".format(fused_str, n))

        acc_ts, conf_mat, acc_tr, models_hmm, bic_scores, train_time, test_time = \
        hmm.hmm_method_pome(X_act_tr, X_act_ts, y_act_tr, y_act_ts, seq_lens_sampl_tr, seq_lens_sampl_ts)
        
        save_hmm_model("{0}_HMM_model_{1}_states".format(fused_str, n), models_hmm)

        logger.save_conf_mat(conf_mat, "Conf_mat_{0}_n_comp_{1}".format(fused_str, n))
        logger.save_hmm_model_data("{0}_hmm_model_data_{1}_states".format(fused_str, n), models_hmm)

        print(" Accuracy of n_comp={0} {1}".format(n, acc_ts))
        logger.save_acc_prec_rec("{0}_metrics_{1}_states".format(fused_str, n), acc_ts, conf_mat, acc_tr)

        acc_list.append(acc_ts)
        #conf_mats.append(conf_mat)
        #models_hmm_list.append(models_hmm)
        bic_scores_list.append(bic_scores)
        acc_tr_list.append(acc_tr)
        train_times.append(train_time)
        test_times.append(test_time)

    if(len(n_comp_arr) > 1):
        logger.save_acc_plot("{0}_accuracies_{1}_states_list".format(fused_str, str(n_comp_arr)), n_comp_arr, acc_list, acc_tr_list)
        logger.save_bic_scores_plot("{0}_bic_scores_{1}_states_list".format(fused_str, str(n_comp_arr)), n_comp_arr, bic_scores_list)
    print(train_times)
    print(test_times)
    print("Process done!")
    # ind_sel = np.argmax(acc_list)
    # acc_max = acc_list[ind_sel]
    # model_interest = models_hmm_list[ind_sel]
    # n_comp_final = n_comp_arr[ind_sel]
    # conf_mat_sel = conf_mats[ind_sel]
    #n_comp_final = 20

    # print("{0} data: Best n_comp {1} with an accuracy of {2}".format(fused_str, n_comp_final, acc_max))
    # logger.save_conf_mat(conf_mat_sel, "Conf_mat_{0}_n_comp_{1}".format(fused_str, n_comp_final))
    # save_hmm_model("{0}_hmm_model_{1}_states".format(fused_str, n_comp_final), model_interest)
    # #model_interest = self.load_hmm_model("Non-fused_hmm_model_{0}_states".format(n_comp_final))
    # logger.save_hmm_model_data("{0}_hmm_model_data_{1}_states".format(fused_str, n_comp_final), model_interest)














