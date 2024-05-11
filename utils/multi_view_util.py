import numpy as np
from scipy.spatial.transform import Rotation
from dtaidistance import dtw_ndim
from dtaidistance.dtw import best_path


def gram_schmidt(X, row_vecs=True, norm = True):
    if not row_vecs:
        X = X.T

    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        norm_tmp = np.linalg.norm(Y,axis=1)
        try:
            proj = np.diag((X[i,:].dot(Y.T)/norm_tmp**2).flat).dot(Y)
        except:
            print(1)

        Y = np.vstack((Y, X[i,:] - proj.sum(0)))

    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T

def gram_schmidt_multi(X):
    
    X_new = np.copy(X)

    def qr_vec(i):    
        Q_tmp, _ = np.linalg.qr(X[i])
        X_new[i, :, :] = Q_tmp
        
    vfunc = np.vectorize(qr_vec)
    vfunc(range(X.shape[0]))
    X_new *= -1

    return X_new


def get_cs(sk_mat):

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
def get_cs_multi(sk_mat):

    #sk_mat = N skels X K features
    coord_x = np.copy(sk_mat[:, (16*3):(16*3+3)]) - sk_mat[:, (12*3):(12*3+3)]
    coord_y = np.copy(sk_mat[:, (1*3):(1*3 + 3)])

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
        coord_system[a] = gram_schmidt(coord_frame[a], False, False)
    

    return coord_system     

def get_angl(cs1, cs2):
    cs1_v1, cs2_v1 = cs1[0,0::2], cs2[0,0::2] #X_axis only
    cs1_v2, cs2_v2 = cs1[1,0:2], cs2[1,0:2] #Y_axis only
    cs1_v3, cs2_v3 = cs1[2,1:3], cs2[0,1:3] #Z_axis only

    ang_X = np.arccos(np.dot(cs1_v1, cs2_v1))
    ang_Y = np.arccos(np.dot(cs1_v2, cs2_v2))
    ang_Z = np.arccos(np.dot(cs1_v3, cs2_v3))

    return np.array([ang_X, ang_Y, ang_Z])

def get_angl_multi(cs, ref_cs):
    cs_v1, ref_cs_v1 = cs[:,0,0::2], ref_cs[0,0::2] #X_axis only
    cs_v2, ref_cs_v2 = cs[:,1,0:2], ref_cs[1,0:2] #Y_axis only
    cs_v3, ref_cs_v3 = cs[:,2,1:3], ref_cs[0,1:3] #Z_axis only

    #dot_1 = np.einsum("ik, ik -> i", cs_v1, ref_cs_v1)
    #dot_2 = np.einsum("ik, ik -> i", cs_v2, ref_cs_v2)
    #dot_3 = np.einsum("ik, ik -> i", cs_v3, ref_cs_v3)

    ang_X = np.arccos(cs_v1 @ ref_cs_v1)
    ang_Y = np.arccos(cs_v2 @ ref_cs_v2)
    ang_Z = np.arccos(cs_v3 @ ref_cs_v3)

    return np.stack([ang_X, ang_Y, ang_Z], axis=1)

def multi_view_test(mat_all):

    check_all_zero_vec = np.any(np.all(mat_all == 0, axis=2) == True)
    if(check_all_zero_vec):
        return False, False
        #raise "All zero vector!"

    vecs_comp = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, -1]])

    mat_all_res_sk = mat_all.reshape(mat_all.shape[0] * mat_all.shape[1], mat_all.shape[2], order='F')

    #test reshape
    #mat_tester = [[[3, 4, 5, 6, 7], [6.24, 3.01, 6.91, 2.99, 1], [7, 6, 5.5, 1.2, 8.8]], 
    #[[7.3, 1.23, 28.6, 0.44, 4.12], [78.2, 22.3, 12.1, 90, 45], [7.33, 5.2, 42.1, 8.6, 7.2]]]

    cs_all = get_cs_multi(mat_all_res_sk)

    cs_dists = np.sum(get_angl_multi(cs_all, vecs_comp) ** 2, axis=1)
    cs_dists_res = cs_dists.reshape(mat_all.shape[0], mat_all.shape[1])

    view_map = np.argmin(cs_dists_res, axis=0).astype(np.int32)

    ind_view_all = np.arange(cs_dists_res.shape[0])
    mask_sel_flat = (ind_view_all == view_map[:,None]).flatten()
    res_vec_sel = mat_all_res_sk[mask_sel_flat]

    cs_all_res = cs_all.reshape(mat_all.shape[0], mat_all.shape[1], cs_all.shape[1], cs_all.shape[2])
    # for i in range(mat_all.shape[1]):
    #     skv = SkeletonVisualizer()

    #     sk1_res = mat_all[0,i].reshape(25,3)
    #     sk2_res = mat_all[1,i].reshape(25,3)
    #     sk3_res = mat_all[2,i].reshape(25,3)

    #     cs1 = cs_all_res[0,i]
    #     cs2 = cs_all_res[1,i]
    #     cs3 = cs_all_res[2,i]

    #     colors = ['b', 'b', 'b']
    #     colors[view_map[i]] = 'g'

    #     skv.plot_skeleton_multi([sk1_res, sk2_res, sk3_res], [cs1, cs2, cs3], colors)

    #     plt.savefig('img_test/foo' + str(i) + '.png')
    #     plt.close()            

    return res_vec_sel, view_map, cs_all_res


def equalize_samples(X_act, seq_lens):
    
    
    min_len_seq_ind = np.argmin((seq_lens))
    X_act_min = X_act[min_len_seq_ind,:seq_lens[min_len_seq_ind]]
    X_act_new = np.zeros((3,seq_lens[min_len_seq_ind], X_act.shape[-1]))

    #EQUALIZE SEQUENCES
    for f_view_ind in range(len(seq_lens)):
        if(f_view_ind == min_len_seq_ind):
            X_act_new[f_view_ind] = X_act[f_view_ind, :seq_lens[f_view_ind]]
            continue
        
        mask_sel = dtw_filter_extra_values(X_act_min, X_act[f_view_ind, :seq_lens[f_view_ind]])
        X_act_new[f_view_ind] = X_act[f_view_ind][mask_sel, :75]

    return X_act_new


def dtw_filter_extra_values(seq_arr_f, seq_arr_s):
    d, tmp_paths = dtw_ndim.warping_paths_fast(seq_arr_f, seq_arr_s)
    best_path_r = best_path(tmp_paths)

    bp = np.array(best_path_r)
    indices_sel = np.unique(bp[:,0], return_index=True)[1]

    return bp[indices_sel, 1]

