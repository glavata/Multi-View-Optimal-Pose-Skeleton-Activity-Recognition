import numpy as np
from scipy.spatial.transform import Rotation
from dtaidistance import dtw_ndim
from dtaidistance.dtw import best_path
from scipy.stats import mode
from numpy.lib.stride_tricks import sliding_window_view
from scipy.spatial.transform import Rotation
import pandas as pd
#from generator.data_format import norm_frames_bone_unit_vec
from generator.data_format import get_cs_multi, get_angl_multi
from utils.common_param import *
from sklearn.cluster import BisectingKMeans, KMeans



def __gram_schmidt_multi(X):
    
    X_new = np.copy(X)

    def qr_vec(i):    
        Q_tmp, _ = np.linalg.qr(X[i])
        X_new[i, :, :] = Q_tmp
        
    vfunc = np.vectorize(qr_vec)
    vfunc(range(X.shape[0]))
    X_new *= -1

    return X_new


def __get_cs(sk_mat):

    coord_x = np.copy(sk_mat[(16*3):(16*3+3)]) - sk_mat[(12*3):(12*3+3)]
    coord_y = np.copy(sk_mat[(1*3):(1*3 + 3)])

    coord_x = coord_x / max(np.linalg.norm(coord_x), 0.00000001)
    coord_y = coord_y / max(np.linalg.norm(coord_y), 0.00000001)

    rot_mat_fin = Rotation.from_rotvec(np.pi/2 * coord_y)
    coord_z = rot_mat_fin.as_matrix() @ coord_x
    z_norm = max(np.linalg.norm(coord_z), 0.00000001)
    coord_z = coord_z / z_norm

    coord_frame = np.vstack([coord_x, coord_y, coord_z]).T
    coord_system = gram_schmidt(coord_frame, False, False)

    return coord_system

#vectorized
def __get_cs_multi_old(sk_mat, upper_body=False):

    #sk_mat = N skels X K features
    if(upper_body):
        coord_x = sk_mat[:, (8*3):(8*3+3)] - sk_mat[:, (4*3):(4*3+3)]
        coord_y = sk_mat[:, (3*3):(3*3 + 3)] - sk_mat[:, (20*3):(20*3+3)]
    else:
        coord_x = sk_mat[:, (16*3):(16*3+3)] - sk_mat[:, (12*3):(12*3+3)]
        coord_y = sk_mat[:, (1*3):(1*3 + 3)]

    coord_x = coord_x / np.maximum(np.linalg.norm(coord_x, axis=1), 0.00000001)[:, np.newaxis]
    coord_y = coord_y / np.maximum(np.linalg.norm(coord_y, axis=1), 0.00000001)[:, np.newaxis]

    rot_mat_fin = Rotation.from_rotvec(np.pi/2 * coord_y)
    coord_z = np.einsum("ijk, ik -> ij", rot_mat_fin.as_matrix(), coord_x)
    z_norm = np.maximum(np.linalg.norm(coord_z, axis=1), 0.00000001)[:, np.newaxis]
    coord_z = coord_z / z_norm

    coord_frame = np.stack([coord_x, coord_y, coord_z], axis=1)
    #coord_frame = coord_frame.reshape(coord_frame.shape[0], 3, 3)
    coord_frame = np.transpose(coord_frame, (0, 2, 1))


    #coord_system = self.gram_schmidt_multi(coord_frame)
    coord_system = np.copy(coord_frame)

    for a in range(coord_frame.shape[0]):
        coord_system[a] = __gram_schmidt(coord_frame[a], False, False)
    

    return coord_system     


def __get_angl(cs1, cs2):
    cs1_v1, cs2_v1 = cs1[0,0::2], cs2[0,0::2] #X_axis only
    cs1_v2, cs2_v2 = cs1[1,0:2], cs2[1,0:2] #Y_axis only
    cs1_v3, cs2_v3 = cs1[2,1:3], cs2[0,1:3] #Z_axis only

    ang_X = np.arccos(np.dot(cs1_v1, cs2_v1))
    ang_Y = np.arccos(np.dot(cs1_v2, cs2_v2))
    ang_Z = np.arccos(np.dot(cs1_v3, cs2_v3))

    return np.array([ang_X, ang_Y, ang_Z])

def get_cs_multi_wrap(skel):
    return get_cs_multi(skel)

def get_angl_multi_wrap(cs):
    return get_angl_multi(cs)

def multi_view_fuse(mat_all, m_type, kalman=None, view_map_smooth=True, lin_transf_model = None):

    if(m_type == FuseType.OPT_POSE):
        return __opt_pose_fuse(mat_all, kalman, view_map_smooth = view_map_smooth, lin_transf_model = lin_transf_model)
    elif(m_type == FuseType.OPT_POSE_KALMAN):
        return __opt_pose_fuse_kalman(mat_all, kalman, view_map_smooth = view_map_smooth)
    elif(m_type == FuseType.MID_VIEW_ONLY):
        ind_mid = int(mat_all.shape[0] / 2)
        return mat_all[2], None, None, None, None

def rotate_sequences_coord_basis(frames): #TODO: REMOVE?
    frames_res = frames.reshape(frames.shape[0], frames.shape[1] * frames.shape[2])
    cs_all = get_cs_multi(frames_res)
    
    #rot_mats = np.einsum('ij, njk -> nki', CENTER_COORD_SYS, cs_all)
    rot_mat_first = CENTER_COORD_SYS @ cs_all[0]
    #frames_res_rot = np.einsum("njk, nkr -> njr", frames, rot_mats)
    frames_res_rot = np.einsum("njk, kr -> njr", frames, rot_mat_first.T)

    # rot_mats_new = np.copy(rot_mats)
    # rot_mats_new_T = np.copy(rot_mats)
    # for c in range(rot_mats_new.shape[0]):

    #     rot = Rotation.align_vectors(CENTER_COORD_SYS, cs_all[c])
    #     rot_mats_new[c] = rot[0].as_matrix().T
    #     rot_mats_new_T[c] = rot[0].as_matrix()

    return frames_res_rot

def framewise_rotated_sequences_opt_pose(mat_all):
    mat_all_res_sk, cs_all, cs_dists_res, view_map, angl_multi = __opt_pose_common(mat_all, True)

    opt_poses, non_opt_poses_rot_ALIGN, angl_multi_comb = __rotate_sequences_opt_pose(mat_all, mat_all_res_sk, cs_dists_res, view_map, angl_multi)

    full_poses_comb = np.zeros((3, mat_all.shape[1], 25, 3))
    full_poses_comb[0] = opt_poses[:]
    full_poses_comb[1:] = non_opt_poses_rot_ALIGN[:]

    return full_poses_comb

def __rotate_sequences_opt_pose(mat_all, mat_all_res_sk, cs_dists_res, view_map, angl_multi):

    #cs_dists_res[cs_dists_res == 0.0] = 0.01

    ind_view_all = np.arange(cs_dists_res.shape[0])
    mask_sel_flat = (ind_view_all == view_map[:,None]).flatten()
    mask_sel_flat_inv = ~mask_sel_flat

    new_shape = np.concatenate([[2], mat_all.shape[1:]])
    non_opt_poses = mat_all_res_sk[mask_sel_flat_inv].reshape(new_shape[0], new_shape[1], 25, 3)
    opt_poses = mat_all_res_sk[mask_sel_flat].reshape(new_shape[1], 25, 3) #117 x 25 x 3

    non_opt_poses_rot_ALIGN = np.zeros((new_shape[0],new_shape[1], 25, 3))
    angl_multi_comb = np.zeros((mat_all.shape[0], mat_all.shape[1], 3))
    angl_multi_comb[0] = angl_multi[mask_sel_flat]
    angl_multi_comb[1:] = angl_multi[~mask_sel_flat].reshape((mat_all.shape[0] - 1, mat_all.shape[1], 3), order='F')

    #tgt_R = np.ones((opt_poses.shape[0])) * 0.015
    for c in range(new_shape[1]):
        for v in range(2):
            # rot_f = Rotation.from_euler("XYZ", angl_multi_comb[v, c])
            # tmp_rot_pose = non_opt_poses[v, c] @ (CENTER_COORD_SYS @ rot_f.as_matrix()).T

            rot = Rotation.align_vectors(opt_poses[c], non_opt_poses[v, c])
            non_opt_poses_rot_ALIGN[v, c] = non_opt_poses[v, c] @ rot[0].as_matrix().T

            # dist_vec = np.linalg.norm(non_opt_poses_rot_ALIGN[v, c].flatten() - opt_poses[c].flatten())
            # if(dist_vec > 0.94):
            #     tgt_R[c] = (0.06 - 0.015) * ((dist_vec - 0.94) / (3.723 - 0.94)) + 0.015


    return opt_poses, non_opt_poses_rot_ALIGN, angl_multi_comb

def __opt_pose_common(mat_all, smooth_map = True):

    #check for same joints
    for v in range(mat_all.shape[0]):
        mat_all_v = mat_all[v]
        same_j_mask = np.any(mat_all_v[:, 16*3:16*3+3] - mat_all_v[:, 12*3:12*3+3] == 0, axis=1)
        cond_1 = np.any(same_j_mask)

        if(cond_1):
            mat_all_v[same_j_mask, 16*3:16*3+3] = np.nan
            pd_frames = pd.DataFrame(mat_all_v)
            mat_all[v] = pd_frames.interpolate(method="spline", order=3, axis=0).to_numpy()
            mat_all[v] = pd_frames.bfill(axis=0).to_numpy()

    mat_all_res_sk = mat_all.reshape(mat_all.shape[0] * mat_all.shape[1], mat_all.shape[2], order='F')

    cs_all = get_cs_multi(mat_all_res_sk)
    angl_multi = get_angl_multi(cs_all)
    #angl_multi_sec = get_angl_multi_in(cs_all)

    bend_threshold_roll = 1
    bend_threshold_pitch = 0.3
 
    angl_multi_res = angl_multi.reshape(mat_all.shape[0], mat_all.shape[1], 3, order='F')
    #angl_multi_res[:, 2:-2, :] = np.median(sliding_window_view(np.abs(angl_multi_res), 5, axis=1), axis=-1)
    angl_multi_res = np.abs(angl_multi_res)

    #view_map = np.argmin(np.linalg.norm(angl_multi_res[:,:,:], axis=-1), axis=0).astype(np.int32)
    view_map = np.argmin(angl_multi_res[:,:,-1], axis=0).astype(np.int32)

    view_map[:] = np.mean(view_map[0:10]).astype(np.int32)
    
                                    #| (angl_multi_res[:, :, 0] > bend_threshold_pitch)
    view_map_bend_any_map = np.any(((angl_multi_res[:, :, 2] > bend_threshold_roll))
                                   .reshape(mat_all.shape[0], mat_all.shape[1], order='F'),
                                    axis=0)
    
    if(np.any(view_map_bend_any_map)):

        # view_map_f = view_map[0]
        
        # #k-means
        # init_center_bend = angl_multi_res[view_map_f, view_map_bend_any_map][int(view_map_bend_any_map.shape[0] / 2)]
        # init_center_non_bend = angl_multi_res[view_map_f, ~view_map_bend_any_map][0]

        # centers_kmeans_init = np.concatenate([init_center_non_bend[np.newaxis,:], init_center_bend[np.newaxis,:]])

        # common_params = {
        #     "n_init": "auto",
        #     "init" :centers_kmeans_init
        #     #"random_state": 42,
        # }

        # view_map_kmeans = KMeans(n_clusters=2, **common_params).fit_predict(angl_multi_res[view_map_f])
        # view_map_kmeans[2:-2] = mode(sliding_window_view(view_map_kmeans, 5),axis=-1)[0].astype(bool)

        angl_view_bend_val = angl_multi_res[:, view_map_bend_any_map, 2] - \
                            angl_multi_res[:, view_map_bend_any_map, 0] - \
                            angl_multi_res[:, view_map_bend_any_map, 1]
        view_map[view_map_bend_any_map] = np.argmax(angl_view_bend_val, axis=0)  

    view_map[2:-2] = mode(sliding_window_view(view_map, 5),axis=-1)[0]

    return mat_all_res_sk, cs_all, angl_multi_res[:, :, 1], view_map, angl_multi

def __opt_pose_fuse(mat_all, kalman_f, view_map_smooth=True, lin_transf_model = None):

    mat_all_res_sk, cs_all, cs_dists_res, view_map, angl_multi = __opt_pose_common(mat_all, view_map_smooth)

    ind_view_all = np.arange(cs_dists_res.shape[0])
    mask_sel_flat = (ind_view_all == view_map[:,None]).flatten()

    # view_map_lower = np.argmax(cs_dists_res, axis=0).astype(np.int32)
    # mask_sel_flat_lower = (ind_view_all == view_map_lower[:,None]).flatten()

    res_vec_sel = mat_all_res_sk[mask_sel_flat]

    cs_all_res = cs_all.reshape(mat_all.shape[0], mat_all.shape[1], cs_all.shape[1], cs_all.shape[2], order='F')

    res_vec_sel_res = res_vec_sel.reshape(res_vec_sel.shape[0], 25, 3)
    res_vec_sel_res_new = np.copy(res_vec_sel_res)

    diff_map = np.diff(view_map != view_map[0]) != 0
    diff_map = np.insert(diff_map, 0, False, axis=0)
    if(np.any(diff_map)):
        ind_where_all = np.argwhere(diff_map)[:,0]
        ind_where_starts = ind_where_all[::2]

        if(ind_where_all.shape[0] > 1):
            if(ind_where_all.shape[0] % 2 != 0):
                ind_where_all = np.insert(ind_where_all, ind_where_all.shape[0], view_map.shape[0])

            seg_sizes = np.diff(ind_where_all)[::2]
        else:
            seg_sizes = np.array([view_map.shape[0] - ind_where_all[0]])

        # if(ind_where_all.shape[0] > 1):
        #     ind_where_ends = ind_where_all[1::2,0]
        # else:
        #     ind_where_ends = diff_map.shape[0]

        #single angle
        mean_vecs_to = np.atleast_2d(np.mean(res_vec_sel_res_new[ind_where_starts - 1][:, JOINTS_CMP_UPPER_BASIC], axis=1))
        mean_vecs_from = np.atleast_2d(np.mean(res_vec_sel_res_new[ind_where_starts][:, JOINTS_CMP_UPPER_BASIC], axis=1))

        mean_vecs_to[:, 1] = 0
        mean_vecs_from[:, 1] = 0

        mean_vecs_to = mean_vecs_to / np.linalg.norm(mean_vecs_to, axis=1)[:,np.newaxis]
        mean_vecs_from = mean_vecs_from / np.linalg.norm(mean_vecs_from, axis=1)[:,np.newaxis]

        angles_yaw = np.arctan2(mean_vecs_from[:, 2], mean_vecs_from[:, 0]) - np.arctan2(mean_vecs_to[:, 2], mean_vecs_to[:, 0])
        rot_objs_cur = Rotation.from_euler('y', angles_yaw)
        rot_objs_cur_mat = rot_objs_cur.as_matrix()

        assert(seg_sizes.shape[0] == rot_objs_cur_mat.shape[0])

        res_vec_sel_res_new[view_map != view_map[0]] = np.einsum('bmk, bjk -> bjm', np.repeat(rot_objs_cur_mat,seg_sizes, axis=0),
                                                        res_vec_sel_res_new[view_map != view_map[0]])


    #cs_first_frame = get_cs_multi(res_vec_sel_res_new[0].reshape(1,75))[0]
    #rot_whole = CENTER_COORD_SYS @ cs_first_frame
    ## equal to mat @ rot_whole.T
    #res_vec_sel_res_new = np.einsum('mk, ijk -> ijm', rot_whole, res_vec_sel_res_new)

    #Rotate whole sequence from first vector
    cs_first_frame = get_cs_multi(res_vec_sel_res_new[0].reshape(1,75))[0]
    first_frame_z_vec = cs_first_frame[:, 2]
    first_frame_z_vec[1] = 0
    first_frame_z_vec = first_frame_z_vec / np.linalg.norm(first_frame_z_vec)
    z_vec_to = CENTER_COORD_SYS[:, 2]

    angle_yaw_first = np.arctan2(first_frame_z_vec[2], first_frame_z_vec[0]) - np.arctan2(z_vec_to[2], z_vec_to[0])
    rot_whole = Rotation.from_euler('y', angle_yaw_first)
    rot_whole_mat = rot_whole.as_matrix()
    res_vec_sel_res_new = np.einsum('mk, ijk -> ijm', rot_whole_mat, res_vec_sel_res_new)

    #Transform tilted skeletons with linear model
    res_vec_sel = res_vec_sel_res_new.reshape(res_vec_sel.shape[0], 75)
    if(lin_transf_model is not None):
        view_map_transf = view_map != view_map[0]
        if(np.any(view_map_transf)):
            to_transf = res_vec_sel[view_map_transf]
            res_tmp = lin_transf_model.predict(to_transf)
            res_vec_sel[view_map_transf] = res_tmp


    res_vec_sel[:, 3:] = kalman_f.filter_old(res_vec_sel[:, 3:])

    return res_vec_sel, view_map, cs_all_res, angl_multi, mat_all

def __opt_pose_fuse_kalman(mat_all, kalman_f, view_map_smooth=True):
    mat_all_res_sk, cs_all, cs_dists_res, view_map, angl_multi = __opt_pose_common(mat_all, view_map_smooth)

    opt_poses, non_opt_poses_rot_ALIGN, _ = __rotate_sequences_opt_pose(mat_all, mat_all_res_sk, cs_dists_res, view_map, angl_multi)

    full_poses_comb = np.zeros((3, mat_all.shape[1], 25, 3))
    full_poses_comb[0] = opt_poses[:]
    full_poses_comb[1:] = non_opt_poses_rot_ALIGN[:]

    #update_map = np.ones(full_poses_comb.shape[1], dtype=bool)
    #mask_tmp = np.diff(view_map) != 0
    #update_map[1:][mask_tmp] = False

    #opt_poses.reshape(opt_poses.shape[0], 75)[:,3:]
    fused_skel_mat_tmp = np.zeros_like(opt_poses.reshape(opt_poses.shape[0], 75))
    full_poses_comb = full_poses_comb.reshape(full_poses_comb.shape[0], full_poses_comb.shape[1], 75)
    
    #res_vec_sel_res = full_poses_comb[0].reshape(full_poses_comb[0].shape[0], 25, 3)
    
    #cur_rot_mat = None

    # for c in range(1, res_vec_sel_res.shape[0]):
    #     if(view_map[c] != view_map[c-1]):
    #         rot_tmp = Rotation.align_vectors(res_vec_sel_res[c-1, JOINTS_CMP], res_vec_sel_res[c, JOINTS_CMP])
    #         cur_rot_mat = rot_tmp[0].as_matrix().T 
    #     if(cur_rot_mat is not None):
    #         res_vec_sel_res[c] = res_vec_sel_res[c] @ cur_rot_mat


    # res_vec_sel_res = full_poses_comb[0].reshape(full_poses_comb[0].shape[0], 25, 3)

    # for c in range(0, res_vec_sel_res.shape[0]):
    #     rot_tmp = Rotation.align_vectors(KINECT_SKEL_REF[JOINTS_CMP], res_vec_sel_res[c, JOINTS_CMP])
    #     res_vec_sel_res[c] = res_vec_sel_res[c] @ rot_tmp[0].as_matrix().T 

    # full_poses_comb[0]  = res_vec_sel_res.reshape(res_vec_sel_res.shape[0], -1)

    R_mat_sec = np.concatenate([NON_OPT_POSE_R_MAT[None, 3:, 3:], NON_OPT_POSE_R_MAT[None, 3:, 3:]])
    #fused_skel_mat_tmp[:,3:] = kalman_f.uk_filter(full_poses_comb[:, :, 3:])
    fused_skel_mat_tmp[:, 3:] = kalman_f.filter_old(full_poses_comb[:, :, 3:], R_mat_sec)
    #fused_skel_mat_tmp[:,3:] = kalman_f.uk_filter(full_poses_comb[:, :, 3:], R_mat_sec)



    # res_vec_sel_res = fused_skel_mat_tmp.reshape(fused_skel_mat_tmp.shape[0], 25, 3)

    # cur_rot_mat = None

    # for c in range(1, res_vec_sel_res.shape[0]):
    #     if(view_map[c] != view_map[c-1]):
    #         rot_tmp = Rotation.align_vectors(res_vec_sel_res[c-1, JOINTS_CMP], res_vec_sel_res[c, JOINTS_CMP])
    #         cur_rot_mat = rot_tmp[0].as_matrix().T 
    #     if(cur_rot_mat is not None):
    #         res_vec_sel_res[c] = res_vec_sel_res[c] @ cur_rot_mat

    # fused_skel_mat_tmp  = res_vec_sel_res.reshape(res_vec_sel_res.shape[0], -1)

    cs_all_res = cs_all.reshape(mat_all.shape[0], mat_all.shape[1], cs_all.shape[1], cs_all.shape[2], order='F')

    return fused_skel_mat_tmp, view_map, cs_all_res, angl_multi, mat_all



def equalize_samples(X_act, seq_lens):
    min_len_seq_ind = np.argmin((seq_lens))
    min_len_seq = np.min(seq_lens)
    X_act_min = X_act[min_len_seq_ind,:seq_lens[min_len_seq_ind]]

    min_sample_period = min(min_len_seq, 30)
    offsets = np.zeros((len(seq_lens)), dtype=int)

    for f_view_ind in range(len(seq_lens)):
        if(f_view_ind != min_len_seq_ind):

            d, tmp_paths = dtw_ndim.warping_paths_fast(X_act_min[:min_sample_period], X_act[f_view_ind, :min_sample_period])
            best_path_r = best_path(tmp_paths)
            bp = np.array(best_path_r)
            avg_offset = np.round(np.mean(bp[:,1] - bp[:,0])).astype(int)

            offsets[f_view_ind] = avg_offset

    min_offset = np.min(offsets)
    if(min_offset < 0):
        offsets += min_offset * -1

    min_len_new = np.min(seq_lens - offsets).astype(int)
    X_act_new = np.zeros((3, min_len_new, X_act.shape[-1]))

    for f_view_ind in range(len(seq_lens)):
        X_act_new[f_view_ind, :min_len_new] = X_act[f_view_ind, offsets[f_view_ind]:min_len_new+offsets[f_view_ind]]

    return X_act_new, None

def equalize_samples_old(X_act, seq_lens):
    
    
    min_len_seq_ind = np.argmin((seq_lens))
    X_act_min = X_act[min_len_seq_ind,:seq_lens[min_len_seq_ind]]
    X_act_new = np.zeros((3,seq_lens[min_len_seq_ind], X_act.shape[-1]))
    paths = [None]* 3

    #EQUALIZE SEQUENCES
    for f_view_ind in range(len(seq_lens)):
        if(f_view_ind == min_len_seq_ind):
            X_act_new[f_view_ind] = X_act[f_view_ind, :seq_lens[f_view_ind]]
            continue
        
        mask_sel, best_path = dtw_filter_extra_values(X_act_min, X_act[f_view_ind, :seq_lens[f_view_ind]])
        paths[f_view_ind] = best_path
        X_act_new[f_view_ind] = X_act[f_view_ind][mask_sel, :]

    return X_act_new, paths


def dtw_filter_extra_values(seq_arr_f, seq_arr_s):
    d, tmp_paths = dtw_ndim.warping_paths_fast(seq_arr_f, seq_arr_s)
    best_path_r = best_path(tmp_paths)

    bp = np.array(best_path_r)

    indices_sel = np.unique(bp[:,0], return_index=True)[1]
    if(np.all(bp[indices_sel[0] : indices_sel[1], 0] == bp[indices_sel[0], 0])):
        indices_sel[0] = indices_sel[1] - 1
    
    return bp[indices_sel, 1], bp

    # un_val = np.unique(bp[:,0], return_counts=True)
    # dupl = np.argwhere(un_val[1] > 1)

    # mask_keep = np.ones_like(bp[:,0], dtype=bool)
    # if(len(dupl) > 0):
        
    #     for d in range(len(dupl)):
    #         tgt_val = dupl[d][0]
    #         mask_to_join = np.in1d(bp[:,0], tgt_val)
    #         ind_to_join = np.argwhere(mask_to_join)[:, 0]

    #         if(tgt_val == 0):
    #             tgt_ind = ind_to_join[-1]
    #         elif(tgt_val == bp[-1,0]):
    #             tgt_ind = ind_to_join[0]
    #         else:
    #             tgt_ind = ind_to_join[int(ind_to_join.shape[0] / 2)] 

    #         mask_keep[mask_to_join] = False
    #         mask_keep[tgt_ind] = True

    # #check for duplicates
    # assert np.all(np.unique(bp[mask_keep, 0], return_counts=True)[1] == 1)
    # #check for resulting shape
    # assert seq_arr_f.shape[0] == np.sum(mask_keep)

    # return bp[mask_keep, 1], bp





   

