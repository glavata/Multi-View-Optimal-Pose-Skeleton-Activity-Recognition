import numpy as np
from pathlib import Path
import tables
from method.hmm import train_hmm
from method.st_gcn import train_gcn
from method.hcn_train import train_hcn
from method.visualizer import draw_dataset_generic, draw_dataset_generator, draw_dataset_hidden_states
from method.validator import find_params_generator, dtw_validation, skel_transf_regressor
from generator.seq_gen import SkeletonSeqGenerator, RepType
from generator.dataset_gen import local_gen_pku, local_gen_ntu, local_gen_ntu_simple
#from utils.compiler import compile_kalman
from utils.multi_view_util import FuseType, Dataset, NormType, RotType


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
    mv_fuse_param = FuseType.NONE

    if 'mv_fuse_type' in params_d.keys():
        mv_fuse_param = params_d['mv_fuse_type']
    if 'norm_type' in params_d.keys():
        norm_type_param = params_d['norm_type']
    if 'rep_type' in params_d.keys():
        rep_type_param = params_d['rep_type']
    if 'rot_type' in params_d.keys():
        rot_type_param = params_d['rot_type']

    return mv_fuse_param, norm_type_param, rep_type_param, rot_type_param

def check_file_exist(type_d, type_params, data_preprocess, classes=None):
    param_d_str =  "{0}-{1}-{2}-{3}".format(*type_params)
    
    file_name = 'processed_data//' + type_d 
    if(classes is None):
        file_name = f"{file_name}_param_{param_d_str}.h5"
    else:
        file_name = f"{file_name}_cls_{classes[:3]}_param_{param_d_str}.h5"
    
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

def get_dataset_pku(type_d='pku_cv_train', data_preprocess = False, return_gen=False, params_d = {}):
    mv_fuse_param, norm_type_param, rep_type_param, rot_type_param = get_params(params_d)
    classes = params_d['classes']
    #subsample = params_d['subsample']
    kalman_params = params_d['kalman_f']
    
    #subtype = params_d["subtype"] #full video sequences or separated action sequences only?
    #type_d = type_d + "_" + subtype
    type_params = [mv_fuse_param.value, norm_type_param.value, rep_type_param.value, rot_type_param.value]
    res = check_file_exist(type_d, type_params, data_preprocess)
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

    return prepare_dataset_gen(dataset_conf['type'], train_generator, file_name, return_gen, params_d, coords=75)

def get_dataset_ntu(type_d='ntu_cs_train', data_preprocess = False, return_gen=False, params_d = {}):

    classes = params_d['classes']
    kalman_params = params_d['kalman_f']

    mv_fuse_param, norm_type_param, rep_type_param, rot_type_param = get_params(params_d)
    type_params = [mv_fuse_param.value, norm_type_param.value, rep_type_param.value, rot_type_param.value]

    res = check_file_exist(type_d, type_params, data_preprocess, classes)
    
    if(res[0]):
        return res[1], res[2], res[2].filename
    file_name = res[1]

    tgt_subj, tgt_cls, tgt_cam = None, None, None

    #TODO: Move to seq_gen?
    if(type_d=="ntu_cs_train"):
        tgt_subj = NTU_SUBJ_TRAIN
        tgt_cam = NTU_CAMERA_TRAIN + NTU_CAMERA_TEST
    elif(type_d=="ntu_cs_test"):
        tgt_subj = NTU_SUBJ_TEST
        tgt_cam = NTU_CAMERA_TRAIN + NTU_CAMERA_TEST
    elif(type_d == "ntu_cv_train"):
        tgt_subj = NTU_SUBJ_TRAIN + NTU_SUBJ_TEST
        tgt_cam = NTU_CAMERA_TRAIN
    elif (type_d == "ntu_cv_test"):
        tgt_subj = NTU_SUBJ_TRAIN + NTU_SUBJ_TEST
        tgt_cam = NTU_CAMERA_TEST
    elif(type_d == "ntu_all_test" or type_d == "ntu_all_train"):
        tgt_subj = NTU_SUBJ_TRAIN + NTU_SUBJ_TEST
        tgt_cam = NTU_CAMERA_TRAIN + NTU_CAMERA_TEST
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

    return prepare_dataset_gen(dataset_conf['type'], train_generator, file_name, return_gen, params_d, coords=75)

def prepare_dataset_gen(dataset_type, train_generator, file_name, return_gen, params_d, coords=75):
    cur_max_len_act = 1000

    X_act = np.zeros((1,cur_max_len_act,coords))
    y_act = np.zeros((1), dtype=np.int32)
    seq_lens_act = np.zeros((1), dtype=np.int32)

    kalman_params, fuse_type, yield_non_full = params_d['kalman_f'],  params_d['mv_fuse_type'],  params_d['yield_non_full']

    return_type = params_d['draw_type']
    if(return_type == 'none'):
        return_type = params_d['val_type']

    local_gen_func = local_gen_pku

    if(dataset_type == Dataset.NTU_RGB):
        local_gen_func = local_gen_ntu#_simple
        cur_max_len_act = 300


    local_gen_func_iter = local_gen_func(train_generator, cur_max_len_act, coords, fuse_type, kalman_params, return_type, yield_non_full)
    if(return_gen):
        return local_gen_func_iter, file_name

    f = tables.open_file(file_name, mode='w')

    array_X_act = f.create_earray(f.root, 'X_act', tables.Float64Atom(), (0, cur_max_len_act, 75))
    array_y_act = f.create_earray(f.root, 'y_act', tables.Int32Atom(), (0, 1))
    array_seq_lens_act = f.create_earray(f.root, 'seq_lens_act', tables.Int32Atom(), (0, 1))

    for node in local_gen_func_iter:
    #for node in local_gen_func(train_generator):
        X_act_t, y_act_t, seq_lens_act_t= node

        array_X_act.append(X_act_t)
        array_y_act.append(y_act_t)
        array_seq_lens_act.append(seq_lens_act_t)

    f.close()

    f = tables.open_file(file_name, mode='r')
    X_act, y_act, seq_lens_act = f.root
    
    return [X_act, y_act, seq_lens_act], f, file_name


def load_params_default(params_d):
    params_default={'kalman_f' : {'dt' : 1/30, 'P_mul': 3, 'R_mul':  0.05, 'Q_mul': 100},
                    'norm_type':NormType.NORM_SKEL_REF,
                    'rep_type':RepType.SEQUENCE,
                    'rot_type':RotType.NO_ROT,
                    'classes': 'single',
                    'draw_type':'none',
                    "val_type":'none',
                    "yield_non_full" : False
                    }

    for key in params_default.keys():
        if(key in params_d):
            params_default[key] = params_d[key]

    params_d = {**params_d, **params_default}

    return params_d





def process_common(dataset, bm_train_type="cv", mv_fuse_data="none", 
                            method='hmm', method_type='classification', params_d={}):

    params_d = load_params_default(params_d)

    dataset_func = None

    if(dataset == 'ntu'):
        dataset_func = get_dataset_ntu
    elif(dataset=='pku'):
        dataset_func = get_dataset_pku
    else:
        raise "Non-existent dataset!"

    params_train_fused_type, params_test_fused_type = FuseType.NONE, FuseType.NONE
    fused_str = "Non_fused"

    if(mv_fuse_data == "train_only"):
        params_train_fused_type = params_d["mv_fuse_type"]
        fused_str = "TrainOnly_fused_" + str(params_d["mv_fuse_type"].name)
    elif(mv_fuse_data == "both"):
        params_train_fused_type = params_d["mv_fuse_type"]
        params_test_fused_type = params_d["mv_fuse_type"]
        fused_str = "Both_fused_" + str(params_d["mv_fuse_type"].name)
    elif(mv_fuse_data == "none"):
        params_d["mv_fuse_type"] = FuseType.NONE
    else:
        raise "Non-existent fusion mode!"

    return_gen = False
    data_preprocess = False
    if(method_type == "visualization" or method_type == "validation"):
        if params_d["draw_type"] == "none" and method_type == "visualization":
            raise "Draw type needs to be set for method type visualization!"
        if params_d["val_type"] == "none" and method_type == "validation":
            raise "Validation type needs to be set for method type validation!"
        return_gen = True
        data_preprocess = True
    
    params_d['mv_fuse_type'] = params_train_fused_type

    train_generic_dataset = dataset_func(dataset + "_" + bm_train_type + "_train", 
                                        data_preprocess, return_gen,
                                        params_d = params_d)

    params_d['mv_fuse_type'] = params_test_fused_type

    if(bm_train_type == 'all' or method_type=='validation'):
        test_generic_dataset = None, None, None
    else:
        test_generic_dataset = dataset_func(dataset + "_" + bm_train_type + "_test",
                                            data_preprocess, return_gen,
                                            params_d = params_d)
    
    f_tr, f_ts = None, None
    if(method_type == "classification"):
        dataset_tr, f_tr, filename_tr = train_generic_dataset
        dataset_ts, f_ts, filename_ts = test_generic_dataset

        if(method == 'hmm'):
            hmm_n_comp = []
            if("hmm_n_comp" in params_d):
                hmm_n_comp = params_d['hmm_n_comp']

            train_hmm(dataset_tr, dataset_ts, fused_str, hmm_n_comp)
        elif(method == 'hcn'):
            #epochs 10/30/50
            #TODO: params
            num_classes = np.max(dataset_tr[2]) + 1

            f_tr.close()
            f_ts.close()

            #args = {'epochs':40, 'batch_size':64, 'base_lr':1e-1, 'steps':[10, 40], 'save_freq':10}
            args = {'epochs':100, 'batch_size':64, 'base_lr':0.001, 'steps':[30,90,150], 'save_freq':50}
            train_hcn(filename_tr, filename_ts, num_classes, fused_str, args)
        elif(method == 'stgcn'):
            num_classes = np.max(dataset_tr[2]) + 1

            f_tr.close()
            f_ts.close()

            args = {'epochs':30, 'batch_size':64, 'base_lr':0.1, 'save_freq':4}
            train_gcn(filename_tr, filename_ts, num_classes, fused_str, args)
        else:
            raise "Non-existent method!"
    elif(method_type == "regression"):
        dataset_tr, f_tr, filename_tr = train_generic_dataset
        dataset_ts, f_ts, filename_ts = test_generic_dataset

        pass
    elif(method_type == "visualization"):

        if(params_d['draw_type'] == 'h_states'):
            draw_dataset_hidden_states()
        else:
            train_dataset_local_gen, filename_tr = train_generic_dataset
            test_dataset_local_gen, filename_ts = test_generic_dataset

            #TODO:Make it for both datasets??
            if("dataset_draw" not in params_d or params_d["dataset_draw"] == "train"):
                dataset_draw_gen = train_dataset_local_gen
                filename_draw = Path(filename_tr).stem
            else:
                dataset_draw_gen = test_dataset_local_gen
                filename_draw = Path(filename_ts).stem

            draw_dataset_generator(dataset_draw_gen, filename_draw, params_d['draw_type'], 300, 75) #TODO: change 300
    elif(method_type == "validation_param"):
        train_dataset_local_gen, _ = train_generic_dataset

        find_params_generator(train_dataset_local_gen, params_d['val_type'])
    elif(method_type == "validation"):
        train_dataset_local_gen, _ = train_generic_dataset

        skel_transf_regressor(train_dataset_local_gen)

    if(f_tr is not None):
        f_tr.close()
    if(f_ts is not None):
        f_ts.close()


if __name__ == '__main__':
    tables.file._open_files.close_all()


    #TODO: console input and validate parameters

    # process_common('ntu', 'cs', "none", method='odcnn',
    #                         params_d={
    #                             'norm_type':NormType.NORM_NECK_TORSO,
    #                             "classes": "single",
    #                             "yield_non_full":True})


    # process_common('ntu', 'cs', "both", method_type='validation',
    #                         params_d={
    #                             'norm_type':NormType.NORM_BONE_UNIT_VEC,
    #                             "mv_fuse_type":FuseType.OPT_POSE,
    #                             #"val_type":"mv_seq_eq",
    #                             "val_type":"rot_pose_regr",
    #                             "classes": "single"})

    process_common('ntu', 'cs', "both", method_type='visualization',
                            params_d={
                                'norm_type':NormType.NORM_BONE_UNIT_VEC,
                                "mv_fuse_type":FuseType.OPT_POSE,
                                "draw_type":"gif_single",
                                "classes": "single"})

    # process_common('pku', 'cs', "both", method='hmm',
    #                         params_d={
    #                             "subtype":"full",
    #                             "plot_mv":False,
    #                             "classes": "single",
    #                             'norm_type':NormType.NORM_BONE_UNIT_VEC,
    #                             "mv_fuse_type":FuseType.OPT_POSE,
    #                             "rot_type":RotType.NO_ROT,
    #                             "hmm_n_comp" : [10]})

    #TOVA POSLEDNO
    # process_common('ntu', 'cs', "both", method='hcn',
    #                         params_d={
    #                             'norm_type':NormType.NORM_BONE_UNIT_VEC,
    #                             "subtype": "full",
    #                             #"kalman_f":None,
    #                             "mv_fuse_type": FuseType.OPT_POSE,
    #                             "plot_mv": False,
    #                             "classes": "single"})



#TODO VISUALIZATION
    # process_common('ntu', 'cs', "both", method_type='visualization',
    #                         params_d={
    #                             'norm_type':NormType.NORM_BONE_UNIT_VEC,
    #                             "mv_fuse_type":FuseType.OPT_POSE,
    #                             "draw_type":"mv_seq_uneq_dtw", 
    #                             "classes": "single"})

