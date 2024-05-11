import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

def save_conf_mat(conf_mat, file_name):
    labels_tick = list(range(1,44))
    plt.figure(figsize=(20, 10)) 
    ax = plt.subplot()
    sns.heatmap(conf_mat, annot=True, cmap='Blues',  fmt='g', xticklabels=True, yticklabels=True, cbar=False, annot_kws={'fontsize': 10}) 
    ax.set_xlabel('Predicted labels',fontsize=16, labelpad=20)
    ax.set_ylabel('True labels',fontsize=16, labelpad=20)
    #ax.set_title('Confusion Matrix for PKU-MMD (pre-merge)',fontsize=20)
    ax.xaxis.set_ticklabels(labels_tick)
    ax.yaxis.set_ticklabels(labels_tick)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    ax.set_aspect(aspect=0.7)
    plt.tight_layout()
    plt.savefig('results/{0}.png'.format(file_name), bbox_inches='tight', dpi=300, orientation="landscape")

def save_hmm_model_data(filename, hmm):
    
    list_hmm = []
    c = 0
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
            list_hmm[c]["dtr"] = cls_m.dense_transition_matrix().tolist()
            c+=1
    except:
        print(" Error calculating HMM parameters.")

    json_object = json.dumps(list_hmm, indent=4)

    # Writing to sample.json
    with open("results/{0}.json".format(filename), "w") as outfile:
        outfile.write(json_object)

def save_acc_prec_rec(filename,  acc, conf_mat):
    recall_cls, precision_cls, recall_mean, precision_mean = calc_prec_rec(conf_mat)

    dictionary = {
        "acc": acc,
        "recall_cls": list(recall_cls),
        "precision_cls": list(precision_cls),
        "recall_mean": recall_mean,
        "precision_mean":precision_mean
    }
    json_object = json.dumps(dictionary, indent=4)

    # Writing to sample.json
    with open("results/{0}.json".format(filename), "w") as outfile:
        outfile.write(json_object)

def calc_prec_rec(conf_mat):

    recall_cls = 0
    precision_cls = 0
    recall_mean = 0
    precision_mean = 0


    recall_sum = np.sum(conf_mat, axis = 1)
    precision_sum = np.sum(conf_mat, axis = 0)

    recall_sum[recall_sum == 0] = 1
    precision_sum[precision_sum == 0] = 1

    recall_cls = np.diag(conf_mat) / recall_sum
    precision_cls = np.diag(conf_mat) / precision_sum
    recall_mean = np.mean(recall_cls)
    precision_mean = np.mean(precision_cls)


    return recall_cls, precision_cls, recall_mean, precision_mean



