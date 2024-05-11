from keras.utils import Sequence
import os
import re
from enum import Enum
import numpy as np
import math
from sklearn.model_selection import StratifiedKFold
from scipy.spatial.transform import Rotation
from scipy.io import loadmat
#from generator.pose_sel import PoseSelector
import pandas as pd
from .kalman import Kalman

class Dataset(Enum):
    NTU_RGB = 1
    PKUMMD = 2
    UTD_MHAD = 3

class NormType(Enum):
    NORM_NECK_TORSO = 1
    NORM_JOINT_DIFF = 2
    NORM_SKEL_REF = 3
    NO_NORM = 4

class RotType(Enum):
    ROT_POSE = 1
    ROT_SEQ = 2
    NO_ROT = 3

class RepType(Enum):
    SEQUENCE = 1
    KEY_POSES = 2
    SUB_SEQ = 3

MAX_LEN_INDICES = 75
MAX_LEN_KEY_POSES = 15
MAX_LEN_SEQUENCE = 500

#TODO: dynamic folder
DATASET_DIR = 'datasets//'

SKELETON_REF_NTU = np.array([-5.08644832e-04,  4.10482094e-02,  3.35510274e-02, -5.87525592e-02,
						  8.27482678e-01, -1.41923286e-01, -1.11551306e-01,  1.59272302e+00,
						 -3.53155487e-01, -6.26085793e-02,  1.94992901e+00, -3.10489422e-01,
						 -2.34516939e-01,  1.19504653e+00, -6.46653137e-01, -2.91774514e-01,
						  5.54426658e-01, -6.90407778e-01, -5.43704238e-02,  8.52638996e-02,
						 -6.14411486e-01,  2.54518901e-02, -9.04123777e-02, -5.48229338e-01,
						  1.50847498e-01,  1.35998732e+00,  6.02157697e-02,  1.32458550e-01,
						  7.80881183e-01,  3.63062822e-01,  2.29890842e-01,  1.27701862e-01,
						  4.49861721e-01,  2.73979814e-01, -1.02873944e-01,  4.76658572e-01,
						 -6.23304711e-02,  3.73430389e-02, -1.59946356e-01, -1.75772910e-01,
						 -9.37253198e-01, -1.92360121e-02, -3.06432846e-01, -1.82822218e+00,
						  3.08631343e-01, -1.51400666e-01, -1.93495144e+00,  3.35705389e-01,
						  7.02673504e-02,  6.03327187e-02,  4.35901708e-02,  9.59363763e-02,
						 -8.07132581e-01,  4.44554781e-01, -6.82484612e-02, -1.74878921e+00,
						  6.94970723e-01,  3.91563607e-02, -1.90270849e+00,  6.91936480e-01,
						 -9.92006881e-02,  1.40433136e+00, -2.94570816e-01,  1.11453753e-01,
						 -2.45923648e-01, -5.10284714e-01, -3.48986001e-02, -1.20252818e-01,
						 -5.73516219e-01,  2.83259661e-01, -1.93448595e-01,  4.88855111e-01,
						  2.99665512e-01, -1.04529014e-01,  4.71230140e-01]).reshape(25,3)
SKELETON_REF_NTU_WAIST_MEAN = SKELETON_REF_NTU - SKELETON_REF_NTU[0,:]

SKELETON_REF_UTD = np.array([[-0.07908871,  1.88172402, -0.01498665],
                            [-0.04616699,  1.45447169,  0.06974048],
                            [-0.01370212,  0.61279399,  0.0842412 ],
                            [-0.00407983,  0.46469441, -0.04288239],
                            [-0.42999305,  1.16276405,  0.04548949],
                            [-0.49625767,  0.53746016, -0.0532222 ],
                            [-0.4703676 , -0.00532979, -0.1783794 ],
                            [-0.44221522, -0.14001433, -0.20927558],
                            [ 0.34624329,  1.19045701,  0.07059323],
                            [ 0.50156499,  0.57929212,  0.01845539],
                            [ 0.5365657 ,  0.13138926, -0.29458758],
                            [ 0.52128139, -0.00312384, -0.38384474],
                            [-0.17254128,  0.28460238, -0.08378449],
                            [-0.19406794, -0.78393388,  0.1610923 ],
                            [-0.12727782, -1.64792821,  0.2340099 ],
                            [-0.22381985, -1.80800897,  0.19190619],
                            [ 0.17819689,  0.29709142, -0.05407636],
                            [ 0.2246719 , -0.78495798,  0.11624358],
                            [ 0.15962734, -1.63850268,  0.21460979],
                            [ 0.23142661, -1.78494083,  0.10865784]])
SKELETON_REF_UTD_WAIST_MEAN = SKELETON_REF_UTD - SKELETON_REF_UTD[3,:]

PKU_MMD_CLS_MULTI = [12, 16, 18, 21, 24, 26, 27]
PKU_MMD_CLS_SINGLE = list(x for x in range(1,51) if x not in PKU_MMD_CLS_MULTI)

def get_skeleton_norm(frame, joint_c):
    avg_joints = np.mean(frame, 1)
    transl_joints = frame - avg_joints[:,np.newaxis,:]
    joint_distances = np.linalg.norm(transl_joints, axis=2).reshape(1, joint_c)
    norms = np.reshape(np.mean(joint_distances, axis=1), (1, 1))
    norms[norms==0.0] = 0.00001

    return norms

NORM_REF_UTD = get_skeleton_norm(SKELETON_REF_UTD[np.newaxis, :, :], 20)

COORD_3D_BASIS = np.array([[1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1]]).T

#gram-schmidt
def gs(X, row_vecs=True, norm = True):
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i,:] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    if row_vecs:
        return Y
    else:
        return Y.T


class SkeletonSeqGenerator(Sequence):
  
    def __init__(self, dataset, batch_size=1, shuffle = True, max_body = 2):
        
        self.__dataset = dataset
        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.__dataset_name = str(dataset['type'].name)
        self.__max_body = max_body
        self.__kalman_active = False
        self.__load_files()
        self.cur_epoch = 0
        self.on_epoch_end()
        
 
        #shuffle is done only once
        #self.__stratified_shuffle()
        self.ind_splits = {}

    def __len__(self):
        return math.ceil(self.__dataset_length / self.__batch_size)
  
    def __getitem__(self, index):
        
        batch_start = index*self.__batch_size
        batch_end = min(self.__dataset_length, (index+1)*self.__batch_size)

        idxs = [i for i in range(batch_start,batch_end)]
        list_IDs_temp = [self.__indexes[k] for k in idxs]

        try:
            X, seq_lens, ind_split = self.__prepare_file_batch(list_IDs_temp)    
        except Warning:
            print(1)
        

        #ind_split = [self.ind_splits[id] for id in list_IDs_temp]
        
        #y = np.repeat(self.labels_dataset[list_IDs_temp], seq_len)[np.newaxis, :, np.newaxis]
        y = self.labels_dataset[list_IDs_temp]

        return X, y, seq_lens, ind_split

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            
            if i == self.__len__()-1:
                self.on_epoch_end()

    def __stratified_shuffle(self):
        splits = int(self.__dataset_length / self.__batch_size)
        skf = StratifiedKFold(n_splits=splits, shuffle=True)
        indices = list(skf.split(np.zeros(self.__dataset_length), self.labels_dataset))
        test_sets = [k for a, k in indices]
        self.__indexes = np.hstack(np.array(test_sets))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.__indexes = np.arange(self.__dataset_length)
        #np.random.shuffle(self.__indexes)
        
        self.cur_epoch +=1

        if self.__shuffle == True:

            if(self.__batch_size == 1):
                np.random.shuffle(self.__indexes)
            else:
                self.__stratified_shuffle()



    def __format_mat(self, mat):

        joint_c = mat.shape[1]
        assert(joint_c == self.joint_c)
        coord_c = mat.shape[2]
        assert(coord_c == self.coord_c)

        if(self.__dataset['norm_type'] == NormType.NORM_JOINT_DIFF):
            norm_mat = self.__norm_frames_with_distances(mat)
        elif(self.__dataset['norm_type'] == NormType.NORM_NECK_TORSO):
            #TODO - Global const for enumeration of joints and number of joints for each dataset
            norm_mat = self.__normalize_skeleton_neck_torso(mat,20,1)
        elif(self.__dataset['norm_type'] == NormType.NORM_SKEL_REF):
            norm_mat = self.__norm_frames_with_skel_ref(mat)
        elif(self.__dataset['norm_type'] == NormType.NO_NORM):
            norm_mat = mat

        if(self.__dataset['rot_type'] in [RotType.ROT_POSE, RotType.ROT_SEQ]):
            norm_mat = self.__rotate_frames_ref(norm_mat, self.__dataset['rot_type'])

        if(self.__kalman_active):
            norm_mat = norm_mat.reshape(norm_mat.shape[0], norm_mat.shape[1] * norm_mat.shape[2])
            norm_mat = self.kf.filter_c(norm_mat)
            #norm_mat = self.kf.filter_fast(norm_mat)
            #norm_mat = self.kf.filter(norm_mat)

            norm_mat = norm_mat.reshape(norm_mat.shape[0], joint_c, 3)

        return norm_mat

    def __prepare_file_batch(self,list_Id):
        
        X = np.ones((0, self.act_size_max, self.coord_total))
        seq_lens = []
        ind_splits = np.zeros((0, MAX_LEN_INDICES), dtype=int)
        pad_val = self.act_size_max
        
        for _, f_id in enumerate(list_Id):

            file_n = self.__files_dataset[f_id]

            mat = None
            norm_mat = None

            act_label = self.labels_dataset[f_id]
            sec_body_null = True

            if(self.__dataset['type'] == Dataset.NTU_RGB):
                mat = self.__ntu_rgb_read_file(file_n,act_label)
                if(len(mat) == 1):
                    mat = mat[0]
                else:
                    sec_body_null = False

            elif(self.__dataset['type'] == Dataset.UTD_MHAD):
                mat = self.__utd_read_file(file_n)
            elif(self.__dataset['type'] == Dataset.PKUMMD):
                mat, pku_split = self.__pku_read_file(file_n)

            if(self.__dataset['type'] != Dataset.PKUMMD):
                if(sec_body_null == False):
                    mat_f = mat[0].reshape(mat[0].shape[0], self.joint_c, self.coord_c)
                    mat_s = mat[1].reshape(mat[1].shape[0], self.joint_c, self.coord_c)
                else:
                    pass
                    #mat = mat.reshape(mat.shape[0], self.joint_c, self.coord_c)

                if(self.__dataset['rep_type'] == RepType.KEY_POSES):
                    if(self.__dataset['max_clust'] is not None):
                        max_clusters = self.__dataset['max_clust'][act_label]
                        max_clusters_each = int(max_clusters / 2)
                    else:
                        max_clusters, max_clusters_each = None, None

                    if(sec_body_null == False):
                        mat_f, _, _ = self.ps.select_poses(mat_f, max_clusters_each) #TODO TO CHANGE
                        shape_f = mat_f.shape[0]
                        mat_s, _, _ = self.ps.select_poses(mat_s, max_clusters_each) #TODO TO CHANGE
                        shape_s = mat_s.shape[0]

                        #match lengths of two-person activities
                        if(shape_f > shape_s):
                            mat_f, _, _ = self.ps.select_poses(mat[0], shape_s) #TODO TO CHANGE
                        elif(shape_s > shape_f):
                            mat_s, _. _ = self.ps.select_poses(mat[1], shape_f) #TODO TO CHANGE

                        #norm_mat_f = self.__format_mat(mat_f)
                        #norm_mat_s = self.__format_mat(mat_s)
                        #norm_mat = np.mean(np.array([norm_mat_f, norm_mat_s]), axis=0)
                        mat = np.mean(np.array([mat_f, mat_s]), axis=0)
                    else:
                        mat = self.ps.select_poses(mat, max_clusters)
                elif(self.__dataset['rep_type'] == RepType.SUB_SEQ):
                    _, ind_spl, _ = self.ps.select_poses(mat, None) #TODO TO CHANGE

                mat = mat.reshape(mat.shape[0], self.joint_c, self.coord_c)
                norm_mat = self.__format_mat(mat)

                norm_mat = np.reshape(norm_mat, (1, norm_mat.shape[0], norm_mat.shape[1] * norm_mat.shape[2]))
                
            else:
                #mat, pku_split
                #75 for both people, 1 for label, 1 for split for first person, 1 for split for sec person
                norm_mat = np.zeros((1, mat.shape[0], 153), dtype=mat.dtype)
                sec_person = True

                #check if all zeros at frame 500
                if(np.all(mat[500, 75:150]==0)):
                    sec_person = False

                empty_spl_mat = np.ones((1, MAX_LEN_INDICES), dtype=int) * -1
                empty_spl_mat[0, :pku_split.shape[0]] = pku_split
                ind_splits = np.concatenate([ind_splits, empty_spl_mat])

                #che.ck indices -> debug, start from 0??
                for i in range(pku_split.shape[0] - 1):
                    st, end = pku_split[i], pku_split[i + 1]
                    if(st == end):
                        continue

                    cls_label = mat[st, 150]
                    mat_pku = mat[st:end]
                    mat_pku_f = mat_pku[:, 0:75]
                    mat_pku_s = mat_pku[:, 75:150]
                    norm_mat[0, st:end, 150] = cls_label

                    mat_pku_f = mat_pku_f.reshape(mat_pku_f.shape[0], self.joint_c, self.coord_c)
                    #lbl_split_f = self.ps.select_poses(mat_pku_f, None, True)
                    norm_mat_f = self.__format_mat(mat_pku_f)
                    norm_mat_f = np.reshape(norm_mat_f, (norm_mat_f.shape[0], norm_mat_f.shape[1] * norm_mat_f.shape[2]))
                    norm_mat[0, st:end, 0:75] = norm_mat_f
                    
                    if(sec_person):
                        mat_pku_s = mat_pku_s.reshape(mat_pku_s.shape[0], self.joint_c, self.coord_c)
                        #lbl_split_s = self.ps.select_poses(mat_pku_s, None, True)
                        norm_mat_s = self.__format_mat(mat_pku_s)
                        norm_mat_s = np.reshape(norm_mat_s, (norm_mat_s.shape[0], norm_mat_s.shape[1] * norm_mat_s.shape[2]))
                        norm_mat[0, st:end, 75:150] = norm_mat_s

                    # if(cls_label != 0):
                    #     norm_mat[0, st:end, 151] = lbl_split_f
                    #     if(sec_person):
                    #         norm_mat[0, st:end, 152] = lbl_split_s
                    # else:
                    #     norm_mat[0, st:end, 150] = 0
                    #     norm_mat[0, st:end, 151:153] = -1

            seq_len = norm_mat.shape[1]

            if(seq_len > pad_val):
                pad_val = seq_len

            to_pad = pad_val - seq_len

            padded_mat = np.pad(norm_mat, ((0,0), (0,to_pad), (0,0)))

            #???? how does this work?
            new_pad = pad_val - X.shape[1]
            X = np.pad(X, ((0,0), (0,new_pad), (0,0)))
            X = np.concatenate((X, padded_mat))
            seq_lens.append(seq_len)

            if(self.__dataset['rep_type'] == RepType.SUB_SEQ):
                empty_spl_mat = np.ones((1, self.act_size_max)) * -1
                empty_spl_mat[0, :ind_spl.shape[0]] = ind_spl
                ind_splits = np.concatenate([ind_splits, empty_spl_mat])

        seq_lens = np.array(seq_lens, dtype=np.int32)
        if(self.__dataset['rep_type'] == RepType.SUB_SEQ or self.__dataset['type'] == Dataset.PKUMMD):
            return X, seq_lens, ind_splits
        else:
            return X, seq_lens, None


    def __load_files(self):
        params = self.__dataset['param']
        self.joint_c = 25 
        self.coord_c = 3

        dir_ = DATASET_DIR + self.__dataset_name + '//' 

        files = []
        classes = []

        if(self.__dataset['type'] == Dataset.NTU_RGB):
            camera_ind = params['camera']
            sub_ind = params['subject']
            class_ind = params['classes']
            
            for f in os.listdir(dir_):
                match =  re.search(r'S([0-9]+)C([0-9]+)P([0-9]+)R([0-9]+)A([0-9]+).skeleton', f)
                if(match == None):
                    continue

                cam, subj, act = int(match.group(2)), int(match.group(3)),  int(match.group(5))

                if(sub_ind != 'all' and subj not in sub_ind):
                    continue

                if(camera_ind != 'all' and cam not in camera_ind):
                    continue

                if(class_ind != 'all' and act not in class_ind):
                    continue

                files.append(dir_ + f)
                classes.append(act)

        elif(self.__dataset['type'] == Dataset.UTD_MHAD):
            self.joint_c = 20
            sub_ind = params['subject']

            for file in os.listdir(dir_):
                if(not file.endswith('.mat')):
                    continue

                vals = file.split("_")
                class_label = int(vals[0][1:])
                subj = int(vals[1][1:])

                if(sub_ind != 'all' and subj not in sub_ind):
                    continue

                files.append(dir_ + file)
                classes.append(class_label)
        elif(self.__dataset['type'] == Dataset.PKUMMD):
            
            file_split = ""
            if "_cs_" in params['datatype']:
                file_split = dir_ + "cross-subject.txt"
            elif "_cv_" in params['datatype']:
                file_split = dir_ + "cross-view.txt"
            else:
                raise "Wrong PKUMMD dataset split type!"

            with open(file_split) as f:
                file_contents = f.readlines()

            target_files_list = []
            if "_train" in params['datatype']:
                target_files_list = file_contents[1].strip(", \n").split(", ")
            if "_test" in params['datatype']:
                target_files_list = file_contents[3].strip(", \n").split(", ")

            #subtype = params['subtype'] #full or sep
            #target_files_list # sort or not??
            #target_files_list.sort()
            
            #3 views for each sample not necessary
            #assert(len(target_files_list) % 3 == 0)

            #target_files_list[0] = "0007-L"

            un_count = 0
            for f in range(len(target_files_list)):
                cur_file = target_files_list[f]

                #file blacklist (messed up labels)
                #if(cur_file[:4] in ['0099', '0119']):
                #    continue

                if(f > 0 and target_files_list[f][:4] != target_files_list[f - 1][:4]):
                    un_count += 1

                file_arr = (dir_ + 'PKU_Skeleton_Renew//' + cur_file + ".txt", dir_ + 'Train_Label_PKU_final//' + cur_file + ".txt")
                files.append(file_arr)
                classes.append(un_count)

        self.coord_total = self.joint_c * self.coord_c
        if(self.__dataset['type'] == Dataset.PKUMMD):
            self.coord_total = 153

        # if(self.__dataset['kalman_f'] is not None):
        #     self.__kalman_active = True
        #     k_param = self.__dataset['kalman_f']
        #     self.kf = Kalman(75, k_param['dt'], k_param['P_mul'], k_param['R_mul'], k_param['Q_mul'])

        self.act_size_max = MAX_LEN_SEQUENCE

        y = np.array(classes)
        v = np.unique(y)
        z = len(v)

        indices = list(range(0,z))
        dict_tmp = dict(zip(v.tolist(),indices))
        self.labels_dataset = np.array([dict_tmp.get(i, -1) for i in y], dtype=np.int32)
        #self.__labels_dataset = to_categorical(y_formatted, z)

        self.__files_dataset = files
        self.__dataset_length = self.labels_dataset.size



    def __ntu_rgb_read_file(self, file_n, act):

        file_h = open(file_n, 'r')

        frame_rows_dict = []
        cur_max = -1

        cur_max_bod = self.__max_body

        if(cur_max_bod == 2 and act not in range(49,60)):
            cur_max_bod = 1

        while True:
            line = file_h.readline()
            if not line:
                break

            framecount = int(line)
              
            for f in range(framecount):
                body_count = int(file_h.readline())

                for b in range(body_count):
                    file_h.readline()

                    if cur_max < b:
                        cur_max = b
                        frame_rows_dict.append(False)

                    joint_c = int(file_h.readline())
                    
                    frame_row = np.empty((1,75))

                    for j in range(joint_c):
                        joint_info = str(file_h.readline()).split(' ')
                        x, y, z = float(joint_info[0]), float(joint_info[1]), float(joint_info[2])
                        frame_row[0, j*3],frame_row[0, j*3 + 1],frame_row[0, j*3 + 2] = x, y, z

                    if(type(frame_rows_dict[b]) is bool):
                        frame_rows_dict[b] = frame_row
                    else:
                        frame_rows_dict[b] = np.vstack((frame_rows_dict[b],frame_row))

        bod_c = cur_max + 1
        #frame_rows_dict = np.array(frame_rows_dict)
        
        if(bod_c > 1):        
            std_s = np.array([k[:,0::3].std() + k[:,1::3].std() + k[:,2::3].std() for k in frame_rows_dict])
            indices = std_s.argsort()[::-1]
            
            if(frame_rows_dict[indices[0]].shape[0] == 1):
                tmp_ind = indices[0]
                indices[0] = indices[1]
                indices[1] = tmp_ind

            indices_sel = indices[0:cur_max_bod]

            frame_rows_dict = [frame_rows_dict[ind] for ind in indices_sel]   

        file_h.close()

        return frame_rows_dict

    def __utd_read_file(self, file_n):
        frames = loadmat(file_n)
        frames = np.array(frames['d_skel'], dtype=np.float64)
        frames_format = np.transpose(frames, (2, 0, 1))
        frames_format = np.reshape(frames_format, (frames_format.shape[0], frames_format.shape[1]*frames_format.shape[2]))

        return frames_format

    def __pku_read_file(self, file_n):

        file_skel, file_labels = file_n[0], file_n[1]
        table_frames = pd.read_csv(file_skel, delimiter=' ', index_col=False, header=None)
        table_labels = pd.read_csv(file_labels, delimiter=',', index_col=False, header=None)
        table_labels = table_labels.sort_values(by=[1])

        table_frames_n = table_frames.to_numpy() #5000x150
        table_frames_n = np.pad(table_frames_n, ((0,0), (0,1))) #5000x151

        last_seq_end = 0
        indices_split = [0]

        for _, row in table_labels.iterrows():
            
            class_label, frame_start, frame_end, confidence = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            
            if(frame_start > last_seq_end):
                table_frames_n[last_seq_end: frame_start, 150] = 0

            last_seq_end = frame_end

            if(frame_end <= frame_start):
                raise "Frame start can not be after frame end!"
            else:
                table_frames_n[frame_start: frame_end, 150] = class_label + 1
                if(frame_start != 0):
                    indices_split.append(frame_start)
                indices_split.append(frame_end)

        max_frame = table_frames_n.shape[0]
        if(max_frame > last_seq_end):
            table_frames_n[last_seq_end: max_frame, 150] = 0
            indices_split.append(max_frame)

        #indices_split = np.unique(table_frames_n[:,150], return_index=True)[1][1:]
        return table_frames_n, np.array(indices_split, dtype=int)


    def __normalize_skeleton_neck_torso_OLD(self, frames, torso_ind, neck_ind):
        torso_coords = frames[:,torso_ind:torso_ind+3]
        neck_coords = frames[:,neck_ind:neck_ind+3]
        norms = np.linalg.norm(neck_coords - torso_coords, axis=1)
        norms[norms==0.0] = 0.00001
        torso_coords_mask = np.tile(torso_coords, (1,self.joint_c))
        frames_normed = (frames - torso_coords_mask) / norms[:,None]

        return frames_normed

    def __normalize_skeleton_neck_torso(self, frames, torso_ind, neck_ind):
        frame_c = frames.shape[0]

        torso_coords = frames[:,torso_ind,:]
        neck_coords = frames[:,neck_ind,:]
        norms = np.linalg.norm(neck_coords - torso_coords, axis=1)
        norms[norms==0.0] = 0.00001

        avg_joints = np.mean(frames, 1)
        transl_joints = frames - avg_joints[:,np.newaxis,:]

        torso_coords_mask = np.tile(torso_coords, (1,self.joint_c))
        frames_normed = transl_joints.reshape(frame_c, self.joint_c * self.coord_c) / norms[:,None]
        frames_normed = frames_normed.reshape(frame_c, self.joint_c, self.coord_c)

        return frames_normed

    def __norm_frames_with_distances(self, frames):
     
        frame_count = frames.shape[0]

        avg_joints = np.mean(frames, 1)
        #avg_joints = frames[:, 0, :]

        transl_joints = frames - avg_joints[:,np.newaxis,:]
        joint_distances = np.linalg.norm(transl_joints, axis=2).reshape(frame_count, self.joint_c)
        norms = np.reshape(np.mean(joint_distances, axis=1), (frame_count, 1))
        norms[norms==0.0] = 0.00001

        joints_final = np.divide(np.reshape(transl_joints,(frame_count, self.joint_c * self.coord_c)), norms)
        joints_final = np.reshape(joints_final, (frame_count, self.joint_c, self.coord_c))

        return joints_final

    def __norm_frames_with_skel_ref(self, frames):
        frame_count = frames.shape[0]

        #avg_joints = np.mean(frames, 1)
        if(self.__dataset['type'] == Dataset.NTU_RGB or self.__dataset['type'] == Dataset.PKUMMD):
            avg_joints = frames[:, 0, :]
        elif(self.__dataset['type'] == Dataset.UTD_MHAD):
            avg_joints = frames[:, 3, :]
        else:
            raise "Non-existant dataset"

        transl_joints = frames - avg_joints[:,np.newaxis,:]
        joint_distances = np.linalg.norm(transl_joints, axis=2).reshape(frame_count, self.joint_c)
        norms = np.reshape(np.mean(joint_distances, axis=1), (frame_count, 1))
        norms[norms==0.0] = 0.00001
        norms = norms * NORM_REF_UTD

        joints_final = np.divide(np.reshape(transl_joints,(frame_count, self.joint_c * self.coord_c)), norms)
        joints_final = np.reshape(joints_final, (frame_count, self.joint_c, self.coord_c))

        return joints_final

    def __norm_frames_with_distances_OLD(self, frames):
        
        frames = frames.reshape(frames.shape[0], frames.shape[1] * frames.shape[2])
        joints_final = np.array(frames.shape)
            
        frame_count = frames.shape[0]
        joint_coords_total = frames.shape[1]
        joint_c = int(joint_coords_total / 3)

        avg_joints = np.empty((frame_count, 3))
        avg_joints[:, 0] = np.mean(frames[:,0::3], 1)
        avg_joints[:, 1] = np.mean(frames[:,1::3], 1)
        avg_joints[:, 2] = np.mean(frames[:,2::3], 1)
    
        frames_res = frames.reshape(frame_count, joint_c, 3)
        transl_joints = frames_res - avg_joints[:,np.newaxis,:]
        joint_distances = np.linalg.norm(transl_joints, axis=2).reshape(frame_count, joint_c)
        norms = np.reshape(np.mean(joint_distances, axis=1), (frame_count, 1))
        norms[norms==0.0] = 0.00001

        joints_final = np.divide(np.reshape(transl_joints,(frame_count, joint_coords_total)), norms)
        joints_final = joints_final.reshape(frame_count, joint_c, 3)

        return joints_final

    def __rotate_frames_ref_old(self, frames):
        frames_new = frames.copy()

        skel_ref = None

        if(self.__dataset['type'] == Dataset.NTU_RGB):
            skel_ref = SKELETON_REF_NTU
        elif(self.__dataset['type'] == Dataset.UTD_MHAD):
            skel_ref = SKELETON_REF_UTD

        #old shape is N * 25 * 3 -> new N * 75
        #frames_new = np.reshape(frames_new, (frames.shape[0], frames.shape[1] * frames.shape[2]))

        for i in range(frames.shape[0]):
            rot = Rotation.align_vectors(skel_ref, frames[i])
            frames_new[i, :, :] = frames[i] @ rot[0].as_matrix().T 

        return frames_new

    #def __augment_rotate(self, frames, times=10):

    def __rotate_frames_ref(self, frames, rot_type):
        frames_new = frames.copy()

        #old shape is N * 25 * 3 -> new N * 75
        #frames_new = np.reshape(frames_new, (frames.shape[0], frames.shape[1] * frames.shape[2]))

        for i in range(frames.shape[0]):
            
            if(rot_type == RotType.ROT_SEQ and i > 0):
                frames_new[i, :, :] = frames[i] @ R.T
            else:
                coord_x = np.copy(frames[i,16,:]) - frames[i,12,:]
                coord_y = np.copy(frames[i,1,:])

                coord_x = coord_x / np.linalg.norm(coord_x)
                coord_y = coord_y / np.linalg.norm(coord_y)

                rot_mat_fin = Rotation.from_rotvec(np.pi/2 * coord_y)
                coord_z = rot_mat_fin.as_matrix() @ coord_x
                coord_z = coord_z / np.linalg.norm(coord_z)

                coord_frame = np.vstack([coord_x, coord_y, coord_z]).T
                coord_system = gs(coord_frame, False, False)

                H = coord_system @ COORD_3D_BASIS.T
                U, S, Vt = np.linalg.svd(H)
                sig = np.sign(np.linalg.det(Vt @ U.T))
                dir_adj = np.eye(3)
                dir_adj[2,2] = sig

                R = Vt @ dir_adj @ U.T

                #R * data or data * R.T
                frames_new[i, :, :] = frames[i] @ R.T

        return frames_new


        


