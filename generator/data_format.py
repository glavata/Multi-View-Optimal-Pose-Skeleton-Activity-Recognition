
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
#from utils.multi_view_util import rotate_sequences_coord_basis
from utils.common_param import *



def __gram_schmidt(X, row_vecs=True, norm = True):
    if not row_vecs:
        X = X.T

    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        norm_tmp = np.linalg.norm(Y,axis=1)
        try:
            proj = np.diag((X[i,:].dot(Y.T)/norm_tmp**2).flat).dot(Y)
        except Exception as e: #most likely zero-coords due to zero vectors
            raise(e)

        Y = np.vstack((Y, X[i,:] - proj.sum(0)))

    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T

def get_cs_multi(sk_mat, upper_body=False):

    #sk_mat = N skels X K features
    if(upper_body):
        coord_x = sk_mat[:, (8*3):(8*3+3)] - sk_mat[:, (4*3):(4*3+3)]
        coord_y = sk_mat[:, (3*3):(3*3 + 3)] - sk_mat[:, (20*3):(20*3+3)]
    else:
        coord_x = sk_mat[:, (16*3):(16*3+3)] - sk_mat[:, (12*3):(12*3+3)]
        coord_y = sk_mat[:, (1*3):(1*3 + 3)]

    coord_x = coord_x / np.maximum(np.linalg.norm(coord_x, axis=1), 0.00000001)[:, np.newaxis]
    coord_y = coord_y / np.maximum(np.linalg.norm(coord_y, axis=1), 0.00000001)[:, np.newaxis]


    # norms_tmp = np.linalg.norm(coord_y,axis=-1)
    # proj = np.multiply((np.sum(np.multiply(coord_x, coord_y), axis=1) / norms_tmp**2)[:, None], coord_y)
    # coord_x_new = coord_x - proj

    # dot_y_inv = np.linalg.norm(np.einsum("njk, nmk -> njm", coord_y[:, :, None], coord_y[:, :, None]))
    # dot_y_inv_y =  np.einsum("nmj, nm -> nj", dot_y_inv, coord_y)

    # proj = np.repeat(np.eye(3), dot_y_inv.shape[0]).reshape(dot_y_inv.shape[0], 3, 3)  - dot_y_inv
    # coord_x_new = np.sum(np.multiply(coord_x, proj), axis=1)
    
    #orthogonalize
    vu = np.sum(np.multiply(coord_x, coord_y), axis=1)

    #orthogonalize y vector
    uu = np.sum(np.multiply(coord_x, coord_x), axis=1)
    coord_y = coord_y -  (vu / uu)[:, None] * coord_x
    coord_y = coord_y / np.linalg.norm(coord_y, axis=1)[:, np.newaxis]

    #Orthogonalize x vector instead?
    #uu = np.sum(np.multiply(coord_y, coord_y), axis=1)
    #coord_x = coord_x -  (vu / uu)[:, None] * coord_y
    #coord_x = coord_x / np.linalg.norm(coord_x, axis=1)[:, np.newaxis]
    

    #find init z-vector
    rot_mat_fin = Rotation.from_rotvec(np.pi/2 * coord_y)
    coord_z_f = np.einsum("ijk, ik -> ij", rot_mat_fin.as_matrix(), coord_x)
    z_norm = np.maximum(np.linalg.norm(coord_z_f, axis=1), 0.00000001)[:, np.newaxis]
    coord_z_f = coord_z_f / z_norm

    # #DEPRECATED ##################################

    # #create new z-vectors from joints
    # sk_mat_res = sk_mat.reshape((sk_mat.shape[0], 25, 3))
    # coord_z_dir = np.mean(sk_mat_res[:,JOINTS_CMP_BASIC,:], axis=1)
    # z_norm = np.maximum(np.linalg.norm(coord_z_dir, axis=1), 0.00000001)[:, np.newaxis]
    # coord_z_dir = coord_z_dir / z_norm

    # coord_z_dir[:, 1] = coord_z_f[:, 1]
    # z_pos_mask = coord_z_dir[:, 2] >= 0
    # if(np.any(z_pos_mask)):
    #     coord_z_dir[z_pos_mask, 2] = coord_z_f[z_pos_mask, 2]

    # #final normalization of z_vec
    # z_norm = np.maximum(np.linalg.norm(coord_z_dir, axis=1), 0.00000001)[:, np.newaxis]
    # coord_z_dir = coord_z_dir / z_norm
    
    # #reorthogonalize x_vec
    # vu_sec = np.sum(np.multiply(coord_x, coord_z_dir), axis=1)
    # uu_sec = np.sum(np.multiply(coord_z_dir, coord_z_dir), axis=1)
    # coord_x = coord_x -  (vu_sec / uu_sec)[:, None] * coord_z_dir
    # coord_x = coord_x / np.linalg.norm(coord_x, axis=1)[:, np.newaxis]

    # #rotate x_vec into y_vec
    # rot_mat_fin = Rotation.from_rotvec(-np.pi/2 * coord_z_dir)
    # coord_y = np.einsum("ijk, ik -> ij", rot_mat_fin.as_matrix(), coord_x)
    # #DEPRECATED ##################################

    #stack coordinates
    coord_frame = np.stack([coord_x, coord_y, coord_z_f], axis=1)
    coord_system = np.transpose(coord_frame, (0, 2, 1))

    # coord_system = np.copy(coord_frame)

    # for a in range(coord_frame.shape[0]):
    #     coord_system[a] = __gram_schmidt(coord_frame[a], False, False)
    

    return coord_system


#zyx (rev extrinsic) == XYZ (intrinsic)

def get_angl_multi_in(cs):
    #XYZ -> intrinsic Tait-Bryan
    
    cs_tr = np.einsum('ij, njk -> nki', CENTER_COORD_SYS, cs)

    alpha = np.arctan(-cs_tr[:, 1, 2] / cs_tr[:, 2, 2])
    beta = np.arcsin(cs_tr[:, 0, 2])
    gamma = np.arctan(-cs_tr[:, 0, 1] / cs_tr[:, 0, 0])

    return np.stack([alpha, beta, gamma], axis=1)


def get_angl_multi_ex_rev(cs):
    #zyx -> rev extrinsic Tait-Bryan
    
    cs_tr = np.einsum('ij, njk -> nki', CENTER_COORD_SYS, cs)

    angles = Rotation.from_matrix(cs_tr).as_euler('zyx')
    return angles

def get_angl_multi_ex(cs):
    #xyz -> extrinsic Tait-Bryan
    
    cs_tr = np.einsum('ij, njk -> nki', CENTER_COORD_SYS, cs)

    angles = Rotation.from_matrix(cs_tr).as_euler('xyz')
    return angles

def get_angl_multi(cs):
    #XYZ -> intrinsic Tait-Bryan
    
    cs_tr = np.einsum('ij, njk -> nki', CENTER_COORD_SYS, cs)

    angles = Rotation.from_matrix(cs_tr).as_euler('XYZ')
    return angles

def get_angl_multi_sep(cs):
    #X,Y,Z -> separate axes rotations #USELESS?
    
    cs_tr = np.einsum('ij, njk -> nki', CENTER_COORD_SYS, cs)

    angles_x = Rotation.from_matrix(cs_tr).as_euler('XYZ')[:, 0]
    angles_y = Rotation.from_matrix(cs_tr).as_euler('YXZ')[:, 0]
    angles_z = Rotation.from_matrix(cs_tr).as_euler('ZYX')[:, 0]

    return np.stack([angles_x, angles_y, angles_z], axis=1)


def get_angl_multi_vector_wise(cs):

    # The z-vector is our forward direction
    z_vectors = cs[:, :, 2]
    z_vectors_y = np.copy(cs[:, :, 2])
    y_vectors = np.copy(cs[:, :, 2])

    # Project the z-vector onto the XZ plane (ignore Y component)
    z_vectors[:, 1] = 0
    z_vectors = z_vectors / np.linalg.norm(z_vectors, axis=1)[:, np.newaxis]  # Normalize

    z_vectors_y[:, 0] = 0
    z_vectors_y = z_vectors_y / np.linalg.norm(z_vectors_y, axis=1)[:, np.newaxis]  # Normalize

    y_vectors[:, 2] = 0
    y_vectors = y_vectors / np.linalg.norm(y_vectors, axis=1)[:, np.newaxis]  # Normalize

    # Calculate the angle between this vector and the negative Z-axis [0, 0, -1]
    neg_z_axis = np.array([0, 0, -1])
    pos_y_axis = np.array([0, 1, 0])

    # The dot product gives us the cosine of the angle
    cos_angles_yaw = np.dot(z_vectors, neg_z_axis)
    cos_angles_pitch = np.dot(z_vectors_y, neg_z_axis)
    cos_angles_roll = np.dot(y_vectors, pos_y_axis)

    # The sign of the X component tells us the direction
    angles_yaw = np.arccos(np.clip(cos_angles_yaw, -1.0, 1.0))
    angles_yaw[z_vectors[:, 0] < 0] *= -1

    angles_pitch = np.arccos(np.clip(cos_angles_pitch, -1.0, 1.0))
    #angles_pitch[y_vectors[:, 0] < 0] *= -1
    #angles_roll = np.zeros_like(angles_pitch)
    angles_roll = np.pi - np.arccos(np.clip(cos_angles_roll, -1.0, 1.0))

    angles_stack = np.stack([angles_pitch, angles_yaw, angles_roll], axis=1)

    return angles_stack


#zyx EXTRINSIC swapped = XYZ INTRINSIC 

def normalize_skeleton_neck_torso(frames, torso_ind, neck_ind, joint_c, coord_c):
    frame_c = frames.shape[0]

    torso_coords = frames[:,torso_ind,:]
    neck_coords = frames[:,neck_ind,:]
    norms = np.linalg.norm(neck_coords - torso_coords, axis=1)
    norms[norms==0.0] = 0.00001

    avg_joints = np.mean(frames, 1)
    transl_joints = frames - avg_joints[:,np.newaxis,:]

    frames_normed = transl_joints.reshape(frame_c, joint_c * coord_c) / norms[:,None]
    frames_normed = frames_normed.reshape(frame_c, joint_c, coord_c)

    return frames_normed

def norm_frames_bone_unit_vec(frames):
    child_indices = LINKS_KINECT_V2[:,0]
    parents_indices = LINKS_KINECT_V2[:,1]

    transl_children_joints = frames[:, child_indices, :] - frames[:, parents_indices, :]
    norms_children_tmp = np.linalg.norm(transl_children_joints, axis=2)

    norms_children_tmp[norms_children_tmp==0.0] = 0.00001
    children_normed_tmp = transl_children_joints / norms_children_tmp[:, :, np.newaxis] * BONE_LENGTHS_KV2[np.newaxis, :, np.newaxis]

    frames_new = np.zeros_like(frames)
    for i in range(LINKS_KINECT_V2.shape[0]):
        j_ind_child, j_ind_parent = LINKS_KINECT_V2[i, 0], LINKS_KINECT_V2[i, 1]
        frames_new[:, j_ind_child, :] = frames_new[:, j_ind_parent, :] + children_normed_tmp[:, i, :]

    return frames_new

def norm_frames_with_distances(frames, joint_c, coord_c):
    
    frame_count = frames.shape[0]

    avg_joints = np.mean(frames, 1)

    transl_joints = frames - avg_joints[:,np.newaxis,:]
    joint_distances = np.linalg.norm(transl_joints, axis=2).reshape(frame_count, joint_c)
    norms = np.reshape(np.mean(joint_distances, axis=1), (frame_count, 1))
    norms[norms==0.0] = 0.00001

    joints_final = np.divide(np.reshape(transl_joints,(frame_count, joint_c * coord_c)), norms)
    joints_final = np.reshape(joints_final, (frame_count, joint_c, coord_c))

    return joints_final

def norm_frames_with_skel_ref(frames, joint_c, coord_c):
    frame_count = frames.shape[0]

    joint_distances = np.linalg.norm(frames, axis=2).reshape(frame_count, joint_c)
    norms = np.reshape(np.mean(joint_distances, axis=1), (frame_count, 1))
    norms[norms==0.0] = 0.00001
    norms = norms * KINECT_SKEL_REF

    joints_final = np.divide(np.reshape(frames,(frame_count, joint_c * coord_c)), norms)
    joints_final = np.reshape(joints_final, (frame_count, joint_c, coord_c))

    return joints_final

def rotate_frames_ref_old(frames, dataset_type):
    frames_new = frames.copy()

    skel_ref = None

    if(dataset_type == Dataset.NTU_RGB or dataset_type == Dataset.PKUMMD):
        skel_ref = KINECT_SKEL_REF
    #old shape is N * 25 * 3 -> new N * 75
    #frames_new = np.reshape(frames_new, (frames.shape[0], frames.shape[1] * frames.shape[2]))

    for i in range(frames.shape[0]):
        rot = Rotation.align_vectors(skel_ref, frames[i])
        frames_new[i, :, :] = frames[i] @ rot[0].as_matrix().T 

    return frames_new


def rotate_frames_coord_sys(frames):
    #old shape is N * 25 * 3 -> new N * 75
    raise NotImplementedError()

#def rotate_frames_seq_coord_sys(frames):
#    return rotate_sequences_coord_basis(frames)


def rotate_remove_roll(frames):

    frames_res = frames.reshape((frames.shape[0], 75))
    rot_mats = get_cs_multi(frames_res)

    rot_obj_euler = get_angl_multi_old(rot_mats)

    #rot_mats_obj = Rotation.from_matrix(rot_mats.transpose(0,2,1))
    #rot_obj_euler = rot_mats_obj.as_euler('ZXY')


    rot_obj_euler_new = np.copy(rot_obj_euler)
    rot_obj_euler_new[:,:2] = 0

    new_rot_euler = Rotation.from_euler('XYZ',np.repeat(rot_obj_euler_new, 25, axis=0))
    frames_new = new_rot_euler.apply(frames.reshape((frames.shape[0] *  frames.shape[1], -1)))
    frames_new = frames_new.reshape((frames.shape[0], 25, 3), order='C')

    ################################
    # frames_new = np.copy(frames)
    # for c in range(frames_new.shape[0]):

    #     rot_obj_euler_new = np.copy(rot_obj_euler[c])
    #     rot_obj_euler_new[1:] = 0

    #     new_rot_euler = Rotation.from_euler('ZYX',rot_obj_euler_new)

    #     frames_new[c] = new_rot_euler.apply(frames_new[c])


    return frames_new


def rotate_frames_ref(frames):

    for c in range(frames.shape[0]):
        rot = Rotation.align_vectors(KINECT_SKEL_REF[JOINTS_CMP_BASIC], frames[c, JOINTS_CMP_BASIC])
        frames[c] = frames[c] @ rot[0].as_matrix().T

    return frames

def rotate_frames_seq_ref(frames):
    rot = Rotation.align_vectors(KINECT_SKEL_REF[JOINTS_CMP_BASIC], frames[0, JOINTS_CMP_BASIC])
    frames_seq_rot = frames @ rot[0].as_matrix().T
    return frames_seq_rot

def interpolate_missing_data(frames):

    frames_shape = frames.shape
    empty_mask = np.zeros_like(frames, dtype=bool)
    all_zero_joint_mask = np.all(frames == 0, axis=2)
    full_zero_vec_mask = np.all(all_zero_joint_mask, axis=1)
    if(np.sum(full_zero_vec_mask) >= int(frames_shape[0] / 5)):
        return None, False
    
    #return frames, True

    #     ind_zero_vec = np.argwhere(full_zero_vec_mask)[:,0]
    #     first_der_diff = np.diff(ind_zero_vec)
    #     first_der_diff_one = first_der_diff == 1
    #     if(np.any(first_der_diff_one)):
    #         ind_split = [0]
    #         first_der_diff_more = first_der_diff == 2
    #         if(np.any(first_der_diff_more)):
    #             ind_split = np.argwhere(first_der_diff_more)[:,0]
            


    #     print("Empty zero vec segment in activity, continuing")
    #     return None, False
    empty_mask[all_zero_joint_mask] = True
    empty_mask[:, 0, :] = False
    empty_mask[np.isnan(frames)] = True
    new_frames = np.copy(frames)

    if(np.any(empty_mask)):
        print("Empty remaining values, filling ...")
        new_frames[empty_mask] = np.nan
        new_frames = new_frames.reshape(frames_shape[0], frames_shape[1] * frames_shape[2])
        empty_mask = empty_mask.reshape(frames_shape[0], frames_shape[1] * frames_shape[2])

        pd_frames = pd.DataFrame(new_frames)
        try:
            pd_frames = pd_frames.interpolate(method="spline", order=3, axis=0)
        except:
            return None, False
        
        new_frames = pd_frames.to_numpy()

        new_frames = new_frames.reshape(new_frames.shape[0], frames_shape[1], frames_shape[2])
        left_nan = np.isnan(new_frames)
        if(np.any(left_nan)):
            return None, False
        #new_frames[left_nan] = 0

    return new_frames, True
