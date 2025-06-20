import numpy as np
from utils.multi_view_util import multi_view_fuse, equalize_samples, dtw_filter_extra_values, get_angl_multi_wrap, get_cs_multi_wrap
from utils.skel_visualization import SkeletonVisualizer
from generator.kalman import Kalman
import pickle
from pathlib import Path

#TODO: move to multi_view_util??
zero_check_skel_joint_coord = np.r_[np.r_[(1*3):(1*3+3)], np.r_[(12*3):(12*3+3)], np.r_[(16*3):(16*3+3)]]

def local_gen_ntu_simple(train_generator):
   
    sample_counter = 0
   
    for X, y, seq_len, unique_view_ids in train_generator:
        for b in range(X.shape[0]):
            X_tmp, y_tmp, seq_len_tmp = X[b], y[b], seq_len[b]

            yield X_tmp[np.newaxis, :], np.array([y_tmp])[np.newaxis, :], np.array([seq_len_tmp])[np.newaxis, :]
            
            print("Sample {}\r".format(sample_counter))
            sample_counter+=1


def local_gen_ntu(train_generator, cur_max_len_act, coords, fuse_type, kalman_f, yield_sec_type="none", non_fuse_yield_full = False):
    sample_counter = 0

    X_act_local_buffer = np.zeros((3,cur_max_len_act,coords), dtype=np.float64)
    seq_len_local_buffer = np.zeros((3), dtype=np.int32)

    last_label = -1
    cur_view_counter = 0
    last_view_id = -1

    fuse = bool(fuse_type.value)

    #for non-fuse mode yield every sample regardless of presence in incomplete views (2 instead of 3)
    non_fuse_yield_full = non_fuse_yield_full and (not fuse)
        
    kf = None
    if(kalman_f is not None):
        kf = Kalman(kalman_f['dt'], kalman_f['P_mul'], kalman_f['R_mul'], kalman_f['Q_mul'])

    my_file = Path("model.npklpy")
    if my_file.is_file():
        with open('model.pkl', 'rb') as f:
            lin_transf_model = pickle.load(f)
    else:
        lin_transf_model = None


    for X, y, seq_len, unique_view_ids in train_generator:
        for b in range(X.shape[0]):

            X_tmp, y_tmp, seq_len_tmp = X[b], y[b], seq_len[b]
            un_view_id = unique_view_ids[b]

            #New triplet begins, try to yield the last one
            if(last_view_id != -1 and un_view_id != last_view_id): 
                
                if(cur_view_counter < 3 and not non_fuse_yield_full):
                    print("     Less than 3 views at multi-view sequence {0}, continue".format(un_view_id))                        
                else:
                    if(fuse):

                        X_seq_eq, dtw_paths = equalize_samples(X_act_local_buffer, seq_len_local_buffer)
                        if(yield_sec_type == "mv_seq_uneq_dtw"):
                            #TODO: dtw_paths None
                            yield X_act_local_buffer, last_label, seq_len_local_buffer, dtw_paths
                        elif(yield_sec_type == "mv_seq_eq"):
                            yield X_seq_eq, last_label, seq_len_local_buffer
                        else:
                            view_map_smooth = True
                            if(yield_sec_type == "mv_seq_eq_full" or yield_sec_type=="gif_triple"):
                                view_map_smooth = False

                            fused_skel_mat_tmp, view_map, coord_sys, angl_multi, mat_all = multi_view_fuse(X_seq_eq, fuse_type, kf, view_map_smooth, lin_transf_model)
                            seq_len_ret = fused_skel_mat_tmp.shape[0]

                            if(yield_sec_type=="gif_triple"):
                                yield X_seq_eq, last_label, coord_sys, view_map
                            elif(yield_sec_type=="gif_single"):
                                yield fused_skel_mat_tmp, last_label, None
                            elif(yield_sec_type == "mv_seq_eq_full"):
                                X_seq_eq_skip = X_seq_eq[:, ::5, :]
                                seq_len_skip = int(seq_len_ret / 5)
                                view_map_skip = view_map[::5]
                                coord_sys_skip = coord_sys[:, ::5, :, :]
                                angl_multi_res = np.reshape(angl_multi, (X_seq_eq.shape[0], X_seq_eq.shape[1], angl_multi.shape[-1]), order='F')
                                angl_multi_res_skip = angl_multi_res[:, ::5, :]
                                
                                yield X_seq_eq_skip, last_label, seq_len_skip, view_map_skip, coord_sys_skip, angl_multi_res_skip
                            elif(yield_sec_type == "rot_pose_regr"):
                                yield view_map, mat_all
                            else:
                                X_ret = np.pad(fused_skel_mat_tmp, ((0, cur_max_len_act - seq_len_ret), (0,0)))

                                yield X_ret[np.newaxis, :], np.array([last_label])[np.newaxis, :], np.array([seq_len_ret])[np.newaxis, :]
                    else:
                        for v in range(cur_view_counter):
                            X_ret, seq_len_ret = X_act_local_buffer[v], seq_len_local_buffer[v]
                            if(yield_sec_type=="gif_single"):
                                cs_all = get_cs_multi_wrap(X_ret[:seq_len_ret])
                                angl_multi = get_angl_multi_wrap(cs_all)
                                yield X_ret[:seq_len_ret], last_label, angl_multi
                            else:    
                                yield X_ret[np.newaxis, :], np.array([last_label])[np.newaxis, :], np.array([seq_len_ret])[np.newaxis, :]

                cur_view_counter = 0

            X_act_local_buffer[cur_view_counter] = X_tmp
            seq_len_local_buffer[cur_view_counter] = seq_len_tmp

            last_view_id = un_view_id
            last_label = y_tmp

            print("Sample {}, triplet {}".format(sample_counter, un_view_id), end='\r')
            cur_view_counter +=1
            sample_counter+=1


def local_gen_pku(train_generator, cur_max_len_act, coords, fuse_type, kalman_f, yield_sec_type="none", non_fuse_yield_full = False):
    sample_counter = 0
    cur_view_c = -1
    last_un_file = -1

    X_act_local_buffer = np.zeros((3,50,cur_max_len_act,coords))
    seq_len_local_buffer = np.zeros((3,50))
    seg_class_local_buffer = np.zeros((3,50))
    cur_max_len_tmp = 0
    cur_max_len_arr = []
    last_file_view_c = -1

    fuse = bool(fuse_type.value)

    non_fuse_yield_full = non_fuse_yield_full and fuse

    if(kalman_f is not None):
        kf = Kalman(75, kalman_f['dt'], kalman_f['P_mul'], kalman_f['R_mul'], kalman_f['Q_mul'])

    def yield_activities_pku(X_act_buffer, seq_lens_buffer, seq_class_buffer, seq_count, fuse):

        for l in range(seq_count):
            X_seq_tmp_to_fuse = X_act_buffer[:, l, :, 0:coords]

            if(fuse):
                X_seq_eq, dtw_paths = equalize_samples(X_seq_tmp_to_fuse, seq_lens_buffer[:, l].astype(int))

                seg_class_tmp = seq_class_buffer[0, l].astype(int)

                if(yield_sec_type == "mv_seq_uneq_dtw"):
                    X_seq_tmp_to_fuse_c = X_seq_tmp_to_fuse[:,::5]
                    seq_lens_new_c = (seq_lens_buffer[:, l].astype(int) / 5).astype(int)
                    _, dtw_paths_c = equalize_samples(X_seq_tmp_to_fuse_c, seq_lens_new_c)

                    yield X_seq_tmp_to_fuse_c, seg_class_tmp, seq_lens_new_c, dtw_paths_c
                elif(yield_sec_type == "mv_seq_eq"):
                    yield X_seq_eq, seg_class_tmp, seq_lens_buffer[:, l].astype(int)
                else:
                    view_map_smooth = True
                    if(yield_sec_type == "mv_seq_eq_full"):
                        view_map_smooth = False

                    fused_skel_mat_tmp, view_map, coord_sys, angl_multi = multi_view_fuse(X_seq_eq, fuse_type, kf, view_map_smooth)
                    if(yield_sec_type == "mv_seq_eq_full"):
                        X_seq_eq_skip = X_seq_eq[:, ::5, :]
                        seq_len_skip = int(fused_skel_mat_tmp.shape[0] / 5)
                        view_map_skip = view_map[::5]
                        coord_sys_skip = coord_sys[:, ::5, :, :]
                        angl_multi_res = np.reshape(angl_multi, (X_seq_eq.shape[0], X_seq_eq.shape[1], angl_multi.shape[-1]), order='F')
                        angl_multi_res_skip = angl_multi_res[:, ::5, :]

                        yield X_seq_eq_skip, seg_class_tmp, seq_len_skip, view_map_skip, coord_sys_skip, angl_multi_res_skip
                    else:
                        X_seq_tmp = np.zeros((1,cur_max_len_act,coords), dtype=X_act_buffer.dtype)
                        X_seq_tmp[0, :fused_skel_mat_tmp.shape[0], :] = fused_skel_mat_tmp

                        yield X_seq_tmp, np.array([seg_class_tmp])[np.newaxis, :], np.array([fused_skel_mat_tmp.shape[0]])[np.newaxis, :]
            else:
                for v in range(X_seq_tmp_to_fuse.shape[0]):
                    seq_lens_tmp = seq_lens_buffer[v, l].astype(int)
                    seg_class_tmp = seq_class_buffer[0, l].astype(int)
                    X_seq_no_fuse = X_seq_tmp_to_fuse[v, :seq_lens_tmp]

                    X_seq_tmp = np.zeros((1,cur_max_len_act,coords), dtype=X_act_buffer.dtype)
                    X_seq_tmp[0, :seq_lens_tmp, :] = X_seq_no_fuse

                    yield X_seq_tmp, np.array([seg_class_tmp])[np.newaxis, :], np.array([seq_lens_tmp])[np.newaxis, :]


    for X, y, seq_len, ind_splits in train_generator:
        for b in range(X.shape[0]):

            X_tmp = X[b]

            if(~np.all(X_tmp[:,75:150] == 0)):
                continue

            y_tmp = y[b] #unique file identifier (for PKUMMD only)
            i_tmp = ind_splits[b]
            i_tmp = i_tmp[i_tmp != -1]

            if(last_un_file != y_tmp):
                cur_view_c = 0
            else:
                cur_view_c += 1

            last_un_file = y_tmp
            
            sample_counter_in = 0
            
            #fuse split sequences in X_act_local_buffer
            if(cur_view_c == 0 and cur_max_len_tmp > 0):
                
                cur_max_len_arr_np = np.array(cur_max_len_arr)
                last_file_view_c = cur_max_len_arr_np.shape[0]
                cur_max_len_tmp = 0
                cur_max_len_arr = []

                if(last_file_view_c < 3 and not non_fuse_yield_full):
                    print("{0} views only for file {1}, skipping...".format(last_file_view_c, sample_counter_in))
                else:

                    cur_max_len_un_len = np.unique(cur_max_len_arr_np)
                    min_len_seq = np.min(cur_max_len_arr_np)

                    wrong_labels = np.any(np.any(seg_class_local_buffer[:, :min_len_seq] 
                                                 != seg_class_local_buffer[0, :min_len_seq], axis=0))

                    if(cur_max_len_un_len.shape[0] > 1 or wrong_labels):
                        

                        #np.applyalongaxis ????
                        #remove duplicates from each view
                        for v in range(seg_class_local_buffer.shape[0]):
                            seg_class_local_buffer[v, cur_max_len_arr_np[v]:] = 0
                            un_cl_v, v_c = np.unique(seg_class_local_buffer[v, :cur_max_len_arr_np[v]], return_counts=True)
                            ind_dupl_v = np.argwhere(v_c > 1)
                            if(len(ind_dupl_v) > 0):
                                ind_dupl_v = ind_dupl_v[:,0]
                                target_cls_v = un_cl_v[ind_dupl_v]
                                non_zeros = seg_class_local_buffer[v] != 0
                                #non_zeros[cur_max_len_arr_np[v]:] = False
                                non_dupl_mask = ~np.in1d(seg_class_local_buffer[v], target_cls_v) & non_zeros
                                new_len_tmp = np.sum(non_dupl_mask)

                                seg_class_local_buffer[v, :new_len_tmp] = seg_class_local_buffer[v, non_dupl_mask]
                                X_act_local_buffer[v, :new_len_tmp] = X_act_local_buffer[v, non_dupl_mask]
                                seq_len_local_buffer[v, :new_len_tmp] = seq_len_local_buffer[v, non_dupl_mask]
                                seg_class_local_buffer[v, new_len_tmp:] = 0
                                X_act_local_buffer[v, new_len_tmp:] = 0
                                seq_len_local_buffer[v, new_len_tmp:] = 0
                                cur_max_len_arr_np[v] = new_len_tmp

                        #select classes that are present in every view
                        flatten_classes = seg_class_local_buffer.flatten()
                        flatten_classes_non_zero = flatten_classes[flatten_classes != 0]
                        view_class_duplicates = np.unique(flatten_classes_non_zero, return_counts=True)
                        mask_non_dupl = view_class_duplicates[1] == 3
                        target_classes = view_class_duplicates[0][mask_non_dupl]
                        new_seq_count = target_classes.shape[0]

                        #order classes in each view
                        for v in range(seg_class_local_buffer.shape[0]):
                            cur_seq_class_ind = seg_class_local_buffer[v, :cur_max_len_arr_np[v]]

                            sort_idx = np.argsort(cur_seq_class_ind)
                            v_classes_index = np.searchsorted(cur_seq_class_ind[sort_idx], target_classes)
                            v_classes_index = sort_idx[v_classes_index]

                            X_act_local_buffer[v, :new_seq_count, :, :] = X_act_local_buffer[v, v_classes_index, :, :]
                            seg_class_local_buffer[v, :new_seq_count] = seg_class_local_buffer[v, v_classes_index]
                            seq_len_local_buffer[v, :new_seq_count] = seq_len_local_buffer[v, v_classes_index]
                    else:
                        new_seq_count = min_len_seq

                    assert(np.unique(seg_class_local_buffer[:,:new_seq_count], axis=0).shape[0] == 1)

                    #Finally, yield accumulated views
                    yield from yield_activities_pku(X_act_local_buffer, seq_len_local_buffer, seg_class_local_buffer, new_seq_count, fuse)


            cur_max_len_tmp = 0  #count subsequences in each file

            for i in range(len(i_tmp) - 1):
                st, end = i_tmp[i], i_tmp[i+1]                 
                seg_class = int(X_tmp[st, 150]) #class label
                
                if(st >= end or seg_class in [0, 13, 15, 17, 19, 22, 25, 27, 28]):
                    continue

                len_cur = end - st
                
                X_seq_tmp = np.zeros((1,cur_max_len_act,coords), dtype=X_tmp.dtype)
                X_seq_tmp[0, :(end-st), :] = X_tmp[st:end, 0:coords]
  
                X_act_local_buffer[cur_view_c, sample_counter_in, :, 0:coords] = X_seq_tmp
                seq_len_local_buffer[cur_view_c, sample_counter_in] = len_cur
                seg_class_local_buffer[cur_view_c, sample_counter_in] = seg_class
                cur_max_len_tmp += 1

                sample_counter_in += 1

            cur_max_len_arr.append(cur_max_len_tmp)

            print("Sample " + str(sample_counter), end='\r')
            sample_counter += 1    


