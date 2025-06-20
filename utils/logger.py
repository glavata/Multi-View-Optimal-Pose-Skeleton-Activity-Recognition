import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
import cv2 as cv2

def save_conf_mat(conf_mat, file_name):
    labels_tick = list(range(1,conf_mat.shape[0] + 1))
    plt.figure(figsize=(20, 10)) 
    ax = plt.subplot()
    sns.heatmap(conf_mat, annot=True, cmap='Blues',  fmt='g', xticklabels=True, yticklabels=True, cbar=False,
                 annot_kws={'fontsize': 10}) 
    ax.set_xlabel('Predicted labels',fontsize=16, labelpad=20)
    ax.set_ylabel('True labels',fontsize=16, labelpad=20)
    #ax.set_title('Confusion Matrix for PKU-MMD (pre-merge)',fontsize=20)
    ax.xaxis.set_ticklabels(labels_tick)
    ax.yaxis.set_ticklabels(labels_tick)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    ax.set_aspect(aspect=0.7)
    plt.tight_layout()
    plt.savefig('results//metrics//{0}.png'.format(file_name), bbox_inches='tight', dpi=300, orientation="landscape")

def save_hmm_model_data(filename, hmm):
    
    list_hmm = []
    c = 0
    np.seterr(under='warn')
    try:
        for cls_m in hmm:
            list_hmm.append({"dtr":None, "states":[]})
            for s in cls_m.states:
                distr = s.distribution
                if distr is None:
                    distr = "None"
                    list_hmm[c]["states"].append("None")
                else:
                    list_hmm[c]["states"].append([])
                    for param in distr.parameters[0]:
                        list_hmm[c]["states"][-1].append(param.parameters)
            list_hmm[c]["dtr"] = [list(m) for m in cls_m.dense_transition_matrix()]
            c+=1
    except:
        print(" Error calculating HMM parameters.")

    json_object = json.dumps(list_hmm, indent=4)
    np.seterr(all='raise')
    # Writing to sample.json
    with open("results//hmm_models//{0}.json".format(filename), "w") as outfile:
        outfile.write(json_object)

def save_hmm_model_data_new(filename, hmm):
    
    list_hmm = []
    c = 0
    try:
        for cls_m in hmm:
            list_hmm.append({"dtr":None, "states":[]})
            for s in cls_m.distributions:
                if s is None:
                    list_hmm[c]["states"].append("None")
                else:
                    list_hmm[c]["states"].append([])
                    for param in s.distributions:
                        list_hmm[c]["states"][-1].append(float(param.means))
                        list_hmm[c]["states"][-1].append(float(param.covs))
            list_hmm[c]["dtr"] = [list(e.numpy()) for e in cls_m.edges]
            c+=1
    except:
        print(" Error calculating HMM parameters.")
    
    json_object = json.dumps(list_hmm, indent=4)

    # Writing to sample.json
    with open("results//hmm_models//{0}.json".format(filename), "w") as outfile:
        outfile.write(json_object)  

def save_acc_prec_rec(filename,  acc, conf_mat, acc_tr=None):
    recall_cls, precision_cls, recall_mean, precision_mean = calc_prec_rec(conf_mat)

    dictionary = {
        "acc": float(acc),
        "acc_tr": None,
        "recall_cls": list(recall_cls),
        "precision_cls": list(precision_cls),
        "recall_mean": float(recall_mean),
        "precision_mean": float(precision_mean)
    }
    if(acc_tr is not None):
        dictionary["acc_tr"] = float(acc_tr)

    json_object = json.dumps(dictionary, indent=4)

    # Writing to sample.json
    with open("results//metrics//{0}.json".format(filename), "w") as outfile:
        outfile.write(json_object)

def save_acc_plot(filename, n_comp_arr, acc_list, acc_tr_list):
    dictionary = {
        "n_components": n_comp_arr,
        "acc_train": acc_tr_list,
        "acc_test": acc_list
    }
    json_object = json.dumps(dictionary, indent=4)

    full_dir_json = "results//metrics//{0}.json".format(filename)
    full_dir_img = "results//metrics//{0}.png".format(filename)

    with open(full_dir_json, "w") as outfile:
        outfile.write(json_object)

    fig = plt.figure()
    plt.plot(n_comp_arr, acc_tr_list, "b", label="Training accuracy", marker = 'o')
    plt.plot(n_comp_arr, acc_list, "r", label="Testing accuracy", marker = 'o')
    plt.legend(['train', 'test'], loc='upper left')
    plt.title("Accuracy over number of states")
    plt.xlabel("Number of states")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    fig.canvas.draw()
    img_plot = np.array(fig.canvas.renderer.buffer_rgba())
    img_bgr = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(full_dir_img, img_bgr)
    plt.close()

def save_bic_scores_plot(filename, n_comp_arr, bic_scores_list):
    dictionary = {
        "n_components": n_comp_arr,
        "bic_scores": bic_scores_list
    }
    json_object = json.dumps(dictionary, indent=4)

    full_dir_json = "results//metrics//{0}.json".format(filename)
    full_dir_img = "results//metrics//{0}.png".format(filename)

    with open(full_dir_json, "w") as outfile:
        outfile.write(json_object)

    bic_scores_list_np = np.array(bic_scores_list)
    fig = plt.figure()
    plt.plot(n_comp_arr, np.mean(bic_scores_list_np, axis=1), "b", marker = 'o')
    plt.title("Mean BIC score over number of states")
    plt.xlabel("Number of states")
    plt.ylabel("Mean BIC score")
    plt.tight_layout()

    fig.canvas.draw()
    img_plot = np.array(fig.canvas.renderer.buffer_rgba())
    img_bgr = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(full_dir_img, img_bgr)

    plt.close()

def calc_prec_rec(conf_mat):

    recall_cls = 0
    precision_cls = 0
    recall_mean = 0
    precision_mean = 0


    recall_sum = np.sum(np.array(conf_mat), axis = 1)
    precision_sum = np.sum(np.array(conf_mat), axis = 0)

    recall_sum[recall_sum == 0] = 1
    precision_sum[precision_sum == 0] = 1

    recall_cls = np.diag(conf_mat) / recall_sum
    precision_cls = np.diag(conf_mat) / precision_sum
    recall_mean = np.mean(recall_cls)
    precision_mean = np.mean(precision_cls)


    return recall_cls, precision_cls, recall_mean, precision_mean



