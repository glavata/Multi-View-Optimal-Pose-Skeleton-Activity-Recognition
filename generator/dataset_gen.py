import numpy as np
from utils.multi_view_util import multi_view_test, equalize_samples, dtw_filter_extra_values 
from utils.skel_visualization import SkeletonVisualizer
from generator.kalman import Kalman

def local_gen_ntu(train_generator):
    sample_counter = 0

    for X, y, seq_len, ind_splits in train_generator:
        for b in range(X.shape[0]):
            X_tmp = X[b]
            y_tmp = y[b]


            y_act = np.concatenate([y_act, np.expand_dims(y_tmp,axis=0)])
            seq_lens_sampl = np.concatenate([seq_lens_sampl, np.expand_dims(seq_len[b],axis=0)])

            print("Sample " + str(sample_counter) + "\r")
            sample_counter+=1


def local_gen_pku(train_generator, cur_max_len_act, coords, merge, kalman_f, plot_multiview):
    sample_counter = 0
    cur_view_c = -1
    last_un_file = -1

    X_act_local_buffer = np.zeros((3,50,cur_max_len_act,coords))
    seq_len_local_buffer = np.zeros((3,50))
    seg_class_local_buffer = np.zeros((3,50))
    cur_max_len_tmp = 0
    cur_max_len_arr = []
    last_file_view_c = -1

    if(plot_multiview):
        skv = SkeletonVisualizer()
    #default - 75, 1/30,  3, 0.05, 100
    if(kalman_f is not None):
        kf = Kalman(75, kalman_f['dt'], kalman_f['P_mul'], kalman_f['R_mul'], kalman_f['Q_mul'])

    def merge_activities_pku(X_act_buffer, seq_lens_buffer, seq_class_buffer, seq_count):

        for l in range(seq_count):
            X_seq_tmp_to_merge = X_act_buffer[:, l, :, 0:coords]

            # if(np.any(seq_lens_buffer[:,l] == 0)):
            #     print("Zero length at subsequence {0}, continue".format(l))
            #     continue

            #Check for zero vectors
            X_act_local_long = np.reshape(X_seq_tmp_to_merge, (3000, 75))
            indices = np.arange(1000)
            mask_local = (indices < seq_lens_buffer[:,l][:,None]).reshape((3000))
            zero_act_ = np.any(np.all(X_act_local_long[mask_local] == 0.0, axis=1))

            if(zero_act_ == True):
                print("Zero skeletons at subsequence {0}, continue".format(l))
                continue

            X_seq_eq = equalize_samples(X_seq_tmp_to_merge, seq_lens_buffer[:, l].astype(int))
            merged_skel_mat_tmp, view_map, coord_sys = multi_view_test(X_seq_eq)
            #merged_skel_mat_tmp = kf.filter_fast(merged_skel_mat_tmp) #filtered

            filename_merged = "tmp//mv_opt_file_{0}_seq_{1}".format(sample_counter, l)

            if(plot_multiview):
                skv.plot_anim_multiview(X_seq_eq, view_map, coord_sys, filename_merged)

            seg_class_tmp = seq_class_buffer[0, l].astype(int)

            X_seq_tmp = np.zeros((1,cur_max_len_act,coords), dtype=X_act_buffer.dtype)

            if(kalman_f is not None):
                merged_skel_mat_tmp = kf.filter_c(merged_skel_mat_tmp)

            X_seq_tmp[0, :merged_skel_mat_tmp.shape[0], :] = merged_skel_mat_tmp

            #skv.plot_anim(X_seq_tmp[0, :merged_skel_mat_tmp.shape[0], :])

            yield X_seq_tmp, np.array([seg_class_tmp])[np.newaxis, :], np.array([merged_skel_mat_tmp.shape[0]])[np.newaxis, :]


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
            
            #merge split sequences in X_act_local_buffer
            if(merge and cur_view_c == 0 and cur_max_len_tmp > 0):
                
                cur_max_len_arr_np = np.array(cur_max_len_arr)
                last_file_view_c = cur_max_len_arr_np.shape[0]
                cur_max_len_tmp = 0
                cur_max_len_arr = []

                if(last_file_view_c < 3):
                    print("{0} views only for file {1}, skipping...".format(last_file_view_c, sample_counter_in))
                else:
                    #Merge accumulated views
                    cur_max_len_un_len = np.unique(cur_max_len_arr_np)
                    min_len_seq = np.min(cur_max_len_arr_np)

                    wrong_labels = np.any(np.any(seg_class_local_buffer[:, :min_len_seq] != seg_class_local_buffer[0, :min_len_seq], axis=0))

                    if(cur_max_len_un_len.shape[0] > 2):
                        print("Sequence label mismatch at file {0}, continue".format(sample_counter_in))
                        continue
                    elif(cur_max_len_un_len.shape[0] == 2 or wrong_labels):

                        min_seq_ind = np.argwhere(cur_max_len_arr_np == min_len_seq)[0,0]

                        for v in range(seg_class_local_buffer.shape[0]):
                            if(v != min_seq_ind):
                                
                                seq_target = seg_class_local_buffer[min_seq_ind, :min_len_seq][:, np.newaxis]
                                seq_to_filter = seg_class_local_buffer[v, :cur_max_len_arr_np[v]][:, np.newaxis]
                                mask_filter = dtw_filter_extra_values(seq_target, seq_to_filter)

                                assert(mask_filter.shape[0] == min_len_seq)                            

                                X_act_local_buffer[v, :min_len_seq, :, :] = X_act_local_buffer[v, mask_filter, :, :]
                                seg_class_local_buffer[v, :min_len_seq] = seg_class_local_buffer[v, mask_filter]
                                seq_len_local_buffer[v, :min_len_seq] = seq_len_local_buffer[v, mask_filter]

                    # #Final check
                    wrong_labels = np.any(seg_class_local_buffer[:, :min_len_seq] != seg_class_local_buffer[0, :min_len_seq], axis=0)
                    wrong_labels_count = np.sum(wrong_labels)
                    if(wrong_labels_count > 2):
                        print("Wrong labelling at file {0}, continue".format(sample_counter_in))
                        continue
                    elif(wrong_labels_count >= 1):
                        mask_filter_repeat = np.pad(~wrong_labels, (0, 50 - min_len_seq))
                        min_len_seq -= wrong_labels_count               
                        X_act_local_buffer[:, :min_len_seq, :, :] = X_act_local_buffer[:, mask_filter_repeat, :, :]
                        seg_class_local_buffer[:, :min_len_seq] = seg_class_local_buffer[:, mask_filter_repeat]
                        seq_len_local_buffer[:, :min_len_seq] = seq_len_local_buffer[:, mask_filter_repeat]
                        

                    yield from merge_activities_pku(X_act_local_buffer, seq_len_local_buffer, seg_class_local_buffer, min_len_seq)
            
            
            cur_max_len_tmp = 0   

            for i in range(len(i_tmp) - 1):
                st, end = i_tmp[i], i_tmp[i+1]                 
                seg_class = int(X_tmp[st, 150]) #class label
                
                if(st >= end or seg_class in [0, 13, 15, 17, 19, 22, 25, 27, 28]):
                    continue

                if(end - st < 18):
                    miss_len = 18 - (end - st)
                    if(st >= miss_len):
                        st -= miss_len
                    else:
                        end += miss_len

                len_cur = end - st
                
                X_seq_tmp = np.zeros((1,cur_max_len_act,coords), dtype=X_tmp.dtype)
                X_seq_tmp[0, :(end-st), :] = X_tmp[st:end, 0:coords]

                #mv_sample_counter = y_tmp * 50 + sample_counter_in #to index same subsequnces in different views
                
                if(merge == False):
                    
                    zero_act_ = np.any(np.all(X_tmp[st:end, 0:coords] == 0.0, axis=1))
                    if(zero_act_ == True):
                        print("Zero skeletons at subsequence {0}, continue".format(i))
                        continue
                    
                    if(kalman_f is not None):
                        X_seq_tmp = kf.filter_c(X_seq_tmp)

                    yield X_seq_tmp, np.array([seg_class])[np.newaxis, :], np.array([len_cur])[np.newaxis, :] #,np.array([y_tmp])[np.newaxis, :]
                else:
                    X_act_local_buffer[cur_view_c, sample_counter_in, :, 0:coords] = X_seq_tmp
                    seq_len_local_buffer[cur_view_c, sample_counter_in] = len_cur
                    seg_class_local_buffer[cur_view_c, sample_counter_in] = seg_class
                    cur_max_len_tmp += 1

                sample_counter_in += 1

            cur_max_len_arr.append(cur_max_len_tmp)


            print("Sample " + str(sample_counter) + "\r")
            sample_counter += 1    


