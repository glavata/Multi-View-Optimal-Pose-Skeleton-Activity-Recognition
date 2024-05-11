import numpy as np
from pathlib import Path
import pickle
import tables
from method.hmm import train_hmm
from method.visualizer import draw_dataset_generic
from generator.seq_gen import SkeletonSeqGenerator, Dataset, NormType, RepType, RotType
from generator.dataset_gen import local_gen_pku, local_gen_ntu
from utils.compiler import compile_kalman

np.seterr(all='raise')

NTU_SUBJ_TRAIN = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
NTU_SUBJ_TEST = [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 32, 33, 36, 37, 39, 40]
NTU_CAMERA_TRAIN = [2, 3]
NTU_CAMERA_TEST = [1]

#TODO ^^ move PKU classes here too?

def get_params(params_d):
    norm_type_param = NormType.NORM_SKEL_REF
    rep_type_param = RepType.SEQUENCE
    rot_type_param = RotType.ROT_SEQ
    mv_merge_param = False

    if 'mv_merged' in params_d.keys():
        mv_merge_param = params_d['mv_merged']
    if 'norm_type' in params_d.keys():
        norm_type_param = params_d['norm_type']
    if 'rep_type' in params_d.keys():
        rep_type_param = params_d['rep_type']
    if 'rot_type' in params_d.keys():
        rot_type_param = params_d['rot_type']

    return mv_merge_param, norm_type_param, rep_type_param, rot_type_param

def check_file_exist(type_d, mv_merge_param, norm_type_param, rep_type_param, rot_type_param, data_preprocess, classes=None):
    param_d_str =  "{0}-{1}-{2}-{3}".format(int(mv_merge_param), norm_type_param.value, rep_type_param.value, rot_type_param.value)
    
    file_name = 'processed_data//' + type_d 
    if(classes is None):
        file_name = file_name + '_param_' + param_d_str + ".h5"
    else:
        file_name = file_name + "_cls_" + classes[:3] + '_param_' + param_d_str + ".h5"
    
    if(data_preprocess == False):
        my_file = Path(file_name)
        if my_file.is_file():
            try:
                f = tables.open_file(file_name, mode='r+')
                data_loaded = [col for col in f.root]
            except Exception as e:
                print("Exception {0} opening file, starting from scratch".format(e))
                Path.unlink(file_name) #TODO error when file not existing?
                return False, file_name

            return True, data_loaded, f

    return False, file_name

def get_dataset_pku(type_d='pku_cv_train', data_preprocess = False, params_d = {}):
    mv_merge_param, norm_type_param, rep_type_param, rot_type_param = get_params(params_d)
    classes = params_d['classes']
    #subsample = params_d['subsample']
    plot_mv = params_d['plot_mv']
    kalman_params = params_d['kalman_f']

    #subtype = params_d["subtype"] #full video sequences or separated action sequences only?
    #type_d = type_d + "_" + subtype

    res = check_file_exist(type_d, mv_merge_param, norm_type_param, rep_type_param, rot_type_param, data_preprocess)
    if(res[0]):
        return res[1], res[2], Path(res[2].filename).name
    file_name = res[1]

    print("Preparing PKU_MMD data / type {0}... \n".format(type_d))

    dataset_conf = {'type': Dataset.PKUMMD,
                'norm_type' : norm_type_param,
                'rep_type' : rep_type_param,
                'rot_type' : rot_type_param,
                'kalman_f' : kalman_params,
                'param' : { 'classes' : classes,
                            'datatype' : type_d
                },
                'max_clust' : None}

    batch_size = 3
    train_generator = SkeletonSeqGenerator(dataset_conf, batch_size, shuffle=False)

    return prepare_dataset_gen(dataset_conf['type'], train_generator, file_name, kalman_params, mv_merge_param, plot_mv, coords=75)


def get_dataset_utd(type_d='utd_utd_train', data_preprocess = False, params_d = {}):
    
    _, norm_type_param, rep_type_param, rot_type_param = get_params(params_d)
    res = check_file_exist(type_d, norm_type_param, rep_type_param, rot_type_param, data_preprocess)
    if(res[0]):
        return res[1]
    file_name = res[1]

    if(type_d=="utd_utd_train"):
        tgt_subj = [1, 3, 5, 7]
    elif(type_d=="utd_utd_test"):
        tgt_subj = [2, 4, 6, 8]
    else:
        raise "Error"

    print("Preparing UTD-MHAD data / type {0}... \n".format(type_d))

    dataset = {'type': Dataset.UTD_MHAD,
                'norm_type' : norm_type_param,
                'rep_type' : rep_type_param,
                'rot_type' : rot_type_param,
                'param' : { 'subject': tgt_subj },
                'max_clust' : None}

    batch_size = 128
    train_generator = SkeletonSeqGenerator(dataset, batch_size, shuffle=False)

    return prepare_dataset_gen(train_generator, file_name, coords=60)

def get_dataset_ntu(type_d='ntu_cs_train', data_preprocess = False, params_d = {}):

    classes = params_d['classes']
    kalman_params = params_d['kalman_f']
    plot_mv = params_d['plot_mv']
    mv_merge_param, norm_type_param, rep_type_param, rot_type_param = get_params(params_d)
    res = check_file_exist(type_d, mv_merge_param, norm_type_param, rep_type_param, rot_type_param, data_preprocess, classes)
    
    if(res[0]):
        return res[1], res[2], Path(res[2].filename).name
    file_name = res[1]

    tgt_subj, tgt_cls, tgt_cam = None, None, None

    if(type_d=="cs_train"):
        tgt_subj = NTU_SUBJ_TRAIN
        tgt_cam = NTU_CAMERA_TRAIN + NTU_CAMERA_TEST
    elif(type_d=="cs_test"):
        tgt_subj = NTU_SUBJ_TEST
        tgt_cam = NTU_CAMERA_TRAIN + NTU_CAMERA_TEST
    elif(type_d == "cv_train"):
        tgt_subj = NTU_SUBJ_TRAIN + NTU_SUBJ_TEST
        tgt_cam = NTU_CAMERA_TRAIN
    elif (type_d == "cv_test"):
        tgt_subj = NTU_SUBJ_TRAIN + NTU_SUBJ_TEST
        tgt_cam = NTU_CAMERA_TEST
    else:
        raise "Error"

    if(classes=="single"):
        tgt_cls = list(range(1,50)) #50
    elif(classes=="all"):
        tgt_cls = list(range(1,61))
    else:
        raise "Error"

    print("Preparing NTU-RGB data / type {0}... \n".format(type_d))

    dataset_conf = {'type': Dataset.NTU_RGB, 
                    'norm_type' : norm_type_param,
                    'rep_type' : rep_type_param,
                    'rot_type' : rot_type_param,
                    'kalman_f' : kalman_params,
                    'param' : { 'subject': tgt_subj,
                                'classes' : tgt_cls,
                                'camera' : tgt_cam
                    },
                    'max_clust' : None}

    batch_size = 128
    train_generator = SkeletonSeqGenerator(dataset_conf, batch_size, shuffle=False)

    return prepare_dataset_gen(dataset_conf['type'], train_generator, file_name, kalman_params, mv_merge_param, plot_mv, coords=75)

#@jit
def prepare_dataset_gen_old(train_generator, file_name, coords=75):
    
    cur_max_len = 15
    cur_max_len_ind = 10
    
    X_act = np.zeros((5000,cur_max_len,coords))
    seq_lens_sampl = np.empty((0), dtype=np.int32)
    y_act = np.empty((0), dtype=np.int32)
    ind_splits_total = np.empty((0,cur_max_len_ind), dtype=np.int32)
    sample_counter = 0

    for X, y, seq_len, ind_splits in train_generator:
        for b in range(X.shape[0]):
            X_tmp = X[b]
            y_tmp = y[b]

            if(ind_splits is not None):
                ind_split = ind_splits[b]
                if(ind_split.shape[1] > cur_max_len_ind):
                    new_pad = ind_split.shape[1] - cur_max_len_ind
                    ind_splits_total = np.pad(ind_splits_total, ((0,0), (0,new_pad)))
                    cur_max_len_ind = ind_split.shape[1]

                ind_splits_total = np.concatenate([ind_splits_total, ind_split[np.newaxis, :]])

            if(sample_counter == X_act.shape[0]):
                X_act = np.concatenate([X_act, np.zeros((5000, cur_max_len, coords))])

            #if(X_act.shape[1] != X.shape[1]):
            #    x_tmp = np.zeros((1, X_act.shape[1], X_act.shape[2]))
            #    x_tmp[0, :X.shape[1], :] = X[b]
            #    X_act[sample_counter] = x_tmp
            #else:
            if(X_tmp.shape[0] > cur_max_len):
                new_pad = X_tmp.shape[0] - cur_max_len
                X_act = np.pad(X_act, ((0,0), (0,new_pad), (0,0)))
                cur_max_len = X_tmp.shape[0]

            X_act[sample_counter] = X_tmp

            y_act = np.concatenate([y_act, np.expand_dims(y_tmp,axis=0)])
            seq_lens_sampl = np.concatenate([seq_lens_sampl, np.expand_dims(seq_len[b],axis=0)])

            print("Sample " + str(sample_counter) + "\r")
            sample_counter+=1

    X_act = X_act[:sample_counter]

    f = open(file_name,"r+")
    pickle.dump([X_act, y_act, seq_lens_sampl], f, protocol=4)
    f.close()

    return X_act, y_act, seq_lens_sampl

def prepare_dataset_gen_pku_OLD(train_generator, file_name, coords=75, window_size=15, subsample=True):
    cur_max_len = 15
    
    seq_lens_sampl = np.zeros((5000), dtype=np.int32)
    X_act = np.zeros((5000,cur_max_len,coords))
    y_act = np.zeros((5000), dtype=np.int32)
    seq_lens_act = np.zeros((5000), dtype=np.int32)
    #y_mview = np.empty((0), dtype=np.int32)
    sample_counter_seq = 0

    X_window_seg_vec = np.zeros((5000, 75))
    y_window_seg = np.zeros((5000), dtype=np.int32)
    y_window_pose = np.zeros((5000), dtype=np.int32)
    y_sample_num = np.zeros((5000), dtype=np.int32)
    sample_counter_window = 0

    sample_counter = 0

    for X, _, seq_len, ind_splits in train_generator:
        for b in range(X.shape[0]):
            #sample_counter +=1
            #continue
            X_tmp = X[b]
            #y_tmp = y[b]
            seq_len_tmp = seq_len[b]
            i_tmp = ind_splits[b]
            i_tmp = i_tmp[i_tmp != -1]

            #y_mview = np.concatenate([y_mview, np.expand_dims(y_tmp, axis=0)]) #used for indexing multi-view files

            for i in range(len(i_tmp) - 1):
                st, end = i_tmp[i], i_tmp[i+1]

                if(st == end):
                    continue
                
                seg_class = X_tmp[st, 150]
                if(seg_class == 0):
                    continue
                else:
                    index_split = np.unique(X_tmp[st:end, 151], return_index=True)[1][1:]
                    res_new = np.split(X_tmp[st:end, 0:75], index_split, axis=0)
                    key_poses = np.concatenate([np.median(np.expand_dims(x, axis=0), axis=1) for x in res_new])

                    if(key_poses.shape[0] > cur_max_len):
                        new_pad = key_poses.shape[0] - cur_max_len
                        X_act = np.pad(X_act, ((0,0), (0,new_pad), (0,0)))
                        cur_max_len = key_poses.shape[0]

                    key_poses_tmp = np.zeros((1,cur_max_len,coords), dtype=X_act.dtype)
                    key_poses_tmp[0, :key_poses.shape[0], :] = key_poses

                    if(sample_counter_seq == X_act.shape[0]):
                        X_act = np.concatenate([X_act, np.zeros((5000,cur_max_len,coords))])
                        y_act = np.concatenate([y_act, np.zeros((5000))])
                        seq_lens_sampl = np.concatenate([seq_lens_sampl, np.zeros((5000))])
                        seq_lens_act = np.concatenate([seq_lens_act, np.zeros((5000))])

                    X_act[sample_counter_seq] = key_poses_tmp
                    y_act[sample_counter_seq] = seg_class
                    seq_lens_sampl[sample_counter_seq] = seq_len_tmp
                    seq_lens_act[sample_counter_seq] = key_poses.shape[0]

                    sample_counter_seq += 1

            last_label_seg_change = X_tmp[0,150]
            last_label_pose_change_f = X_tmp[0,151]
            last_label_pose_change_s = X_tmp[0,152]

            for c in range(0, X_tmp.shape[0] - 1):
                start = max(0, c - window_size)
                cur_window = X_tmp[start:(c + 1), 0:75]
                cur_window_vec = np.zeros((1, 75))

                seg_change_label = 0 #no change
                pose_change_label = 0 #no change
                if(c > 0):
                    if(X_tmp[c,150] != last_label_seg_change):          
                        if(last_label_seg_change == 0):
                            seg_change_label = 1 #from no activity to activity
                        else:
                            seg_change_label = 2 #from activity to no activity
                    elif(X_tmp[c,151] != last_label_pose_change_f):
                        pose_change_label = 1 #change of pose
                    elif(last_label_pose_change_f != -1):
                        pose_change_label = 2 #in activity but no change of pose

                last_label_seg_change = X_tmp[c,150]
                last_label_pose_change_f = X_tmp[c,151]

                if(subsample and seg_change_label == 0 and pose_change_label == 0 and c % 3 == 0):
                    continue

                for j in range(cur_window.shape[0] - 1):
                    cur_window_vec += (cur_window[j + 1] - cur_window[j])
                
                if(sample_counter_window == X_window_seg_vec.shape[0]):
                    X_window_seg_vec = np.concatenate([X_window_seg_vec, np.zeros((5000, 75))])
                    y_window_seg = np.concatenate([y_window_seg, np.zeros((5000), dtype=np.int32)])
                    y_window_pose = np.concatenate([y_window_pose, np.zeros((5000), dtype=np.int32)])
                    y_sample_num = np.concatenate([y_sample_num, np.zeros((5000), dtype=np.int32)])

                X_window_seg_vec[sample_counter_window] = cur_window_vec
                y_window_seg[sample_counter_window] = seg_change_label
                y_window_pose[sample_counter_window] = pose_change_label
                y_sample_num[sample_counter_window] = sample_counter

                sample_counter_window +=1

            print("Sample " + str(sample_counter) + "\r")
            sample_counter +=1
    
    X_act = X_act[:sample_counter_seq]
    y_act = y_act[:sample_counter_seq]
    seq_lens_sampl = seq_lens_sampl[:sample_counter_seq]
    seq_lens_act = seq_lens_act[:sample_counter_seq]

    X_window_seg_vec = X_window_seg_vec[:sample_counter_window]
    y_window_seg = y_window_seg[:sample_counter_window]
    y_window_pose = y_window_pose[:sample_counter_window]
    y_sample_num = y_sample_num[:sample_counter_window]

    f = open(file_name,"wb")
    pickle.dump([X_act, y_act, seq_lens_sampl, seq_lens_act, X_window_seg_vec, y_window_seg, y_window_pose, y_sample_num], f, protocol=4)
    f.close()

    return X_act, y_act, seq_lens_sampl, seq_lens_act, X_window_seg_vec, y_window_seg, y_window_pose, y_sample_num


def prepare_dataset_gen(dataset_type, train_generator, file_name, kalman_params, merge = False, plot_multiview = False, coords=75):
    cur_max_len_act = 1000

    X_act = np.zeros((1,cur_max_len_act,coords))
    y_act = np.zeros((1), dtype=np.int32)
    seq_lens_act = np.zeros((1), dtype=np.int32)

    local_gen_func = local_gen_pku

    if(dataset_type == Dataset.NTU_RGB):
        local_gen_func = local_gen_ntu

    f = tables.open_file(file_name, mode='w')

    array_X_act = f.create_earray(f.root, 'X_act', tables.Float64Atom(), (0, 1000, 75))
    array_y_act = f.create_earray(f.root, 'y_act', tables.Int32Atom(), (0, 1))
    array_seq_lens_act = f.create_earray(f.root, 'seq_lens_act', tables.Int32Atom(), (0, 1))

    for node in local_gen_func(train_generator, cur_max_len_act, coords, merge, kalman_params, plot_multiview):
        X_act_t, y_act_t, seq_lens_act_t = node

        array_X_act.append(X_act_t)
        array_y_act.append(y_act_t)
        array_seq_lens_act.append(seq_lens_act_t)

    f.close()

    f = tables.open_file(file_name, mode='r')
    X_act, y_act, seq_lens_act = f.root
    
    return [X_act, y_act, seq_lens_act], f, Path(file_name).name


def load_params_default(params_d):
    params_default={'kalman_f' : {'dt' : 1/30, 'P_mul': 3, 'R_mul':  0.05, 'Q_mul': 100},
                    'norm_type':NormType.NORM_SKEL_REF, 
                    'rep_type':RepType.SEQUENCE,
                    'rot_type':RotType.NO_ROT}

    for key in params_default.keys():
        if(key in params_d):
            params_default[key] = params_d[key]

    params_d = {**params_d, **params_default}

    return params_d

def training_detection(dataset, bm_train_type="cv", mv_merge_type="none", params_d={}):

    params_d = load_params_default(params_d)


def training_classification(dataset, bm_train_type="cv", mv_merge_type="none", method='hmm', params_d={}):

    params_d = load_params_default(params_d)

    joint_c = 25
    dataset_func = None
    if(dataset == 'ntu'):
        dataset_func = get_dataset_ntu
    elif(dataset=='utd'):
        joint_c = 20
        dataset_func = get_dataset_utd
    elif(dataset=='pku'):
        dataset_func = get_dataset_pku
    else:
        raise "Non-existant dataset!"

    params_train_merged, params_test_merged = False, False
    merged_str = "Non_Merged"
    if(mv_merge_type == "train_only"):
        params_train_merged = True
        merged_str = "Train_Merged"
    elif(mv_merge_type == "both"):
        params_train_merged = True
        params_test_merged = True
        merged_str = "Fully_Merged"

    params_d['mv_merged'] = params_train_merged

    dataset_tr, f_tr, filename_tr = dataset_func(dataset + "_" + bm_train_type + "_train", 
                                                data_preprocess = False,
                                                params_d = params_d)

    params_d['mv_merged'] = params_test_merged

    dataset_ts, f_ts, filename_ts = dataset_func(dataset + "_" + bm_train_type + "_test",
                                                data_preprocess = False,
                                                params_d = params_d)

    if(method == 'hmm'):
        hmm_n_comp = []
        if("hmm_n_comp" in params_d):
            hmm_n_comp = params_d['hmm_n_comp']

        train_hmm(dataset_tr, dataset_ts, joint_c, merged_str, hmm_n_comp)
    elif(method == 'draw'):
        if("dataset_draw" not in params_d or params_d["dataset_draw"] == "train"):
            dataset_draw = dataset_tr
            filename_draw = filename_tr
        else:
            dataset_draw = dataset_ts
            filename_draw = filename_ts

        if("draw_type" not in params_d):
            draw_type = 'gif'
        else:
            draw_type = params_d["draw_type"]

        draw_dataset_generic(dataset_draw, Path(filename_draw).stem, draw_type)


    f_tr.close()
    f_ts.close()

tables.file._open_files.close_all()


#TODO: console input and validate parameters

#kalman after merging data? YES, makes sense

#compile_kalman()

training_classification('pku', 'cs', "both", method='hmm',
                        params_d={
                            "subtype":"full",
                            "plot_mv":False,
                            "classes": "single",
                            #"kalman_f":None,
                            "dataset_draw" : "train",
                            "hmm_n_comp" : [20]})

# training_classification('pku', 'cs', "none", method='hmm',
#                         params_d={
#                             "subtype":"full",
#                             "plot_mv":False,
#                             "classes": "single",
#                             "hmm_n_comp" : [25]})






#training_detection('pku_2', bm_train_type="cv", mv_merge_type="none", method='hmm', params_d={})


#LAST
# training_generic('utd', 'utd', method='dtw_pose', 
#                         params_d={
#                            "rep_type":RepType.SEQUENCE, 
#                            "rot_type":RotType.NO_ROT})

#training_generic('ntu', 'cs', "none", method='hmm',
#                    params_d={"classes":"single", 
#                            "rep_type":RepType.KEY_POSES, 
#                            "rot_type":RotType.NO_ROT})


#TODO: set default params_d
#training_generic('pku','cs', method='seg',
#                    params_d={"subtype":"full",
#                            "subsample" : False,
#                            "classes":"single", 
#                            "rep_type":RepType.SEQUENCE, 
#                            "rot_type":RotType.NO_ROT})





