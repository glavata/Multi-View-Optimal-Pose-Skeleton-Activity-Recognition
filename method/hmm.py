
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from pomegranate import NormalDistribution, HiddenMarkovModel
import cv2 as cv2
from pathlib import Path
import time
import pickle
from utils import logger
from utils.skel_visualization import SkeletonVisualizer

class HMM():

    def __init__(self, dataset_tr, dataset_ts, joint_c = 25, n_comp=5):
        self.d_tr = dataset_tr
        self.d_ts = dataset_ts
        self.joint_c = joint_c
        self.n_comp = n_comp



    def hmm_method_pome(self, X_act_tr, X_act_ts, y_act_tr, y_act_ts, seq_lens_sampl_tr, seq_lens_sampl_ts):
        cls_all = np.unique(y_act_tr)
       
        models_hmm = []
        class_counts = np.zeros((cls_all.shape[0]))

        start = time.time()  
        #training
        for k in range(cls_all.shape[0]):
            #print(" Training for class {0} ...".format(k))
            c =  cls_all[k]
            
            ind_query_cls = np.argwhere(y_act_tr[:, 0] == c)[:, 0]

            act_cls_cur = X_act_tr[ind_query_cls, :, :]
            class_counts[k] = act_cls_cur.shape[0]

            act_lens_cls = seq_lens_sampl_tr[ind_query_cls, 0]

            X_tmp = [act_cls_cur[i, :act_lens_cls[i], 3:] for i in range(act_cls_cur.shape[0])]

            #15 - 0.75
            #20 - 0.78
            #25 - 0.00/ nan
            model_c = HiddenMarkovModel.from_samples(distribution=NormalDistribution, 
                                                    n_components=self.n_comp,
                                                    X=X_tmp)
            models_hmm.append(model_c)

        end = time.time()
        print(" Training time for n_comp={0} : {1} seconds".format(self.n_comp, end - start))

        class_counts = class_counts / np.sum(class_counts)
        pred_all = np.zeros((X_act_ts.shape[0], cls_all.shape[0]))

        start = time.time()

        for i in range(X_act_ts.shape[0]):
            #print(" Testing for class {0} ...".format(i))
            act_cls_cur = X_act_ts[i]
            #class_c.append(act_cls_cur.shape[0])
            act_lens_cls = seq_lens_sampl_ts[i, 0]
            #cur_class_prob = class_counts[np.where(cls_all == y_act_ts[i, 0])[0]]

            for m in range(len(models_hmm)):
                pred_all[i, m] =  models_hmm[m].log_probability(act_cls_cur[:act_lens_cls, 3:])# * (1 - cur_class_prob) 
                
            
        y_pred = np.argmax(pred_all, axis=1)
        y_pred = cls_all[y_pred]
        end = time.time()

        print(" Testing time for n_comp={0} : {1} seconds".format(self.n_comp, end - start))

        acc = accuracy_score(y_act_ts[:, 0], y_pred)
        conf_mat = confusion_matrix(y_act_ts[:, 0], y_pred)

        return acc, conf_mat, models_hmm


def save_hmm_model(filename, hmm):
    python_obj = [v.to_json() for k, v in enumerate(hmm)]

    with open("results/{0}.pkl".format(filename), 'wb') as outp:
        pickle.dump(python_obj, outp, pickle.HIGHEST_PROTOCOL)

def load_hmm_model(filename):
    with open("results/{0}.pkl".format(filename), 'rb') as pickle_file:
        hmm_obj_json = pickle.load(pickle_file)
    hmm_models = [HiddenMarkovModel.from_json(o) for o in hmm_obj_json]
    return hmm_models

def train_hmm(dataset_tr, dataset_ts, joint_c, merged_str, n_comp_arr=[5]):
    
    if(not isinstance(n_comp_arr, list)):
        raise Exception("n_comp_arr parameter must be a list")

    X_act_tr, seq_lens_sampl_tr, y_act_tr = dataset_tr[0], dataset_tr[1], dataset_tr[2]
    X_act_ts, seq_lens_sampl_ts, y_act_ts = dataset_ts[0], dataset_ts[1], dataset_ts[2]

    acc_list = []
    models_hmm_list = []
    conf_mats = []

    for n in n_comp_arr:
        hmm = HMM(dataset_tr, dataset_ts,  joint_c, n_comp=n)

        print("Starting {0} HMM test with n_comp={1}".format(merged_str, n))

        acc_hmm_tr, conf_mat, models_hmm = hmm.hmm_method_pome(X_act_tr, X_act_ts, y_act_tr, y_act_ts, seq_lens_sampl_tr, seq_lens_sampl_ts)

        save_hmm_model("{0}_HMM_model_{1}_states".format(merged_str, n), models_hmm)

        logger.save_conf_mat(conf_mat, "Conf_mat_{0}_n_comp_{1}".format(merged_str, n))
        logger.save_hmm_model_data("{0}_hmm_model_data_{1}_states".format(merged_str, n), models_hmm)

        print(" Accuracy of n_comp={0} {1}".format(n, acc_hmm_tr))
        logger.save_acc_prec_rec("{0}_metrics_{1}_states".format(merged_str, n), acc_hmm_tr, conf_mat)

        acc_list.append(acc_hmm_tr)
        conf_mats.append(conf_mat)
        models_hmm_list.append(models_hmm)
        

    ind_sel = np.argmax(acc_list)
    acc_max = acc_list[ind_sel]
    model_interest = models_hmm_list[ind_sel]
    n_comp_final = n_comp_arr[ind_sel]
    conf_mat_sel = conf_mats[ind_sel]
    #n_comp_final = 20

    print("{0} data: Best n_comp {1} with an accuracy of {2}".format(merged_str, n_comp_final, acc_max))
    logger.save_conf_mat(conf_mat_sel, "Conf_mat_{0}_n_comp_{1}".format(merged_str, n_comp_final))
    save_hmm_model("{0}_hmm_model_{1}_states".format(merged_str, n_comp_final), model_interest)
    #model_interest = self.load_hmm_model("Non-merged_hmm_model_{0}_states".format(n_comp_final))
    logger.save_hmm_model_data("{0}_hmm_model_data_{1}_states".format(merged_str, n_comp_final), model_interest)














