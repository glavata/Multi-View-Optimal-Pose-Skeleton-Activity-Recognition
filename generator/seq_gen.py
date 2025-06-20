from keras.utils import Sequence
import os
import re
from enum import Enum
import numpy as np
import math
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from pathlib import Path
from utils.multi_view_util import Dataset, NormType, RotType
from . import data_format as dfo

class RepType(Enum):
    SEQUENCE = 1
    FULL_SEQ = 2 #TODO: ADD this option for PKU_MMD regression

MAX_LEN_INDICES = 75
MAX_LEN_SEQUENCE = 9000

DATASET_DIR = str(Path('..') / "datasets")

PKU_MMD_CLS_MULTI = [12, 16, 18, 21, 24, 26, 27]
PKU_MMD_CLS_SINGLE = list(x for x in range(1,51) if x not in PKU_MMD_CLS_MULTI)


class SkeletonSeqGenerator(Sequence):
  
    def __init__(self, dataset, batch_size=1, shuffle = True, max_body = 2):
        
        self.__dataset = dataset
        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.__dataset_name = str(dataset['type'].name)
        self.__max_body = max_body
        #self.__kalman_active = False
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

        #TODO:change to numpy
        idxs = [i for i in range(batch_start,batch_end)]
        list_IDs_temp = [self.__indexes[k] for k in idxs]

        try:
            X, seq_lens, extra_data = self.__prepare_file_batch(list_IDs_temp)    
        except Warning:
            print(1)
        
        y = self.labels_dataset[list_IDs_temp]

        return X, y, seq_lens, extra_data

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


       # else:
        #norm_mat = mat.reshape(mat.shape[0], joint_c, 3)

        #Translate joints to waist joint
        transl_mat = mat - mat[:, 0, :][:,np.newaxis,:]

        #ZERO VECTOR/ JOINT CHECK
        interp_mat, res = dfo.interpolate_missing_data(transl_mat)

        if(res == False):
            return None, False

        if(self.__dataset['norm_type'] == NormType.NORM_JOINT_DIFF):
            norm_mat = dfo.norm_frames_with_distances(interp_mat, self.joint_c, self.coord_c)
        elif(self.__dataset['norm_type'] == NormType.NORM_BONE_UNIT_VEC):
            norm_mat = dfo.norm_frames_bone_unit_vec(interp_mat)
        elif(self.__dataset['norm_type'] == NormType.NORM_NECK_TORSO):
            #TODO - Global const for enumeration of joints and number of joints for each dataset
            norm_mat = dfo.normalize_skeleton_neck_torso(interp_mat, 20, 1, self.joint_c, self.coord_c)
        elif(self.__dataset['norm_type'] == NormType.NORM_SKEL_REF):
            norm_mat = dfo.norm_frames_with_skel_ref(interp_mat, self.joint_c, self.coord_c)
        elif(self.__dataset['norm_type'] == NormType.NO_NORM):
            norm_mat = np.copy(interp_mat)


        #norm_mat = dfo.rotate_remove_roll(norm_mat)

        if(self.__dataset['rot_type'] == RotType.ROT_POSE):
            #norm_mat = self.__rotate_frames_ref(norm_mat, self.__dataset['rot_type'])
            norm_mat = dfo.rotate_frames_coord_sys(norm_mat)
        elif(self.__dataset['rot_type'] == RotType.ROT_SEQ):
            #norm_mat = self.__rotate_frames_ref(norm_mat, self.__dataset['rot_type'])
            #norm_mat = dfo.rotate_frames_seq_coord_sys(norm_mat) #TODO: RETURN
            pass
        elif(self.__dataset['rot_type'] == RotType.ROT_SEQ_REF):
            norm_mat = dfo.rotate_frames_seq_ref(norm_mat)
        elif(self.__dataset['rot_type'] == RotType.ROT_POSE_REF):
            norm_mat = dfo.rotate_frames_ref(norm_mat)

        return norm_mat, True

    def __prepare_file_batch(self,list_Id):
        
        X = np.zeros((0, self.act_size_max, self.coord_total))
        seq_lens = []
        ind_splits = np.zeros((0, MAX_LEN_INDICES), dtype=int)
        
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

            elif(self.__dataset['type'] == Dataset.PKUMMD):
                mat, pku_split = self.__pku_read_file(file_n)

            if(self.__dataset['type'] == Dataset.NTU_RGB):
                #TODO: IMPLEMENT FOR TWO BODIES
                if(sec_body_null == False):
                    mat_f = mat[0].reshape(mat[0].shape[0], self.joint_c, self.coord_c)
                    mat_s = mat[1].reshape(mat[1].shape[0], self.joint_c, self.coord_c)
                else:
                    pass
                    #mat = mat.reshape(mat.shape[0], self.joint_c, self.coord_c)

                mat, _ = self.__check_filter_zero_vec(mat)
                mat = mat.reshape(mat.shape[0], self.joint_c, self.coord_c)
                norm_mat, res = self.__format_mat(mat)
                if(res == False):
                    print("Unsalvageable activity sequence, continuing...")
                    continue

                norm_mat = np.reshape(norm_mat, (1, norm_mat.shape[0], norm_mat.shape[1] * norm_mat.shape[2]))
                
            elif(self.__dataset['type'] == Dataset.PKUMMD):
                #mat, pku_split
                #75 for both people, 1 for label, 1 for split for first person, 1 for split for sec person
                
                #Interpolate full sequence
                all_zero_vec_f = np.all(mat[:,0:75] == 0.0, axis=1)
                mat[all_zero_vec_f,0:75] = np.nan
                pd_frames = pd.DataFrame(mat[:,0:75])
                pd_frames = pd_frames.interpolate(method="spline", order=3, axis=0)
                mat[:, 0:75] = pd_frames.to_numpy()

                norm_mat = np.zeros((1, mat.shape[0], 153), dtype=mat.dtype)
                sec_person = True

                #check if all zeros at frame 500
                # if(np.all(mat[500, 75:150]==0)):
                #     sec_person = False
                # else:
                #     pd_frames = pd.DataFrame(mat[:,75:150])
                #     pd_frames = pd_frames.interpolate(method="spline", order=3, axis=0)
                #     mat[:, 75:150] = pd_frames.to_numpy()

                empty_spl_mat = np.ones((1, MAX_LEN_INDICES), dtype=int) * -1
                pku_split_new = np.copy(pku_split)

                for i in range(pku_split.shape[0] - 1):
                    st, end = pku_split[i], pku_split[i + 1]
                    if(st == end):
                        continue

                    cls_label = mat[st, 150]
                    mat_pku = mat[st:end]

                    if(cls_label == 0):
                        continue

                    #TODO: do not bother to format inbetween empty sequences??
                    # if(cls_label != 0):
                    #     _, offsets = self.__check_filter_zero_vec(mat_pku[:, 0:75]) #TODO: for second person also
                    #     mat[st:end, 150] = 0
                    #     st = st + offsets[0]
                    #     end = end - offsets[1]
                    #     mat[st:end, 150] = cls_label

                    #     pku_split[i] = st
                    #     pku_split[i + 1] = end
                    # else:
                    #     continue

                    if(end - st < 30):
                        to_pad = 30 - (end - st)
                        st_to_pad = int(to_pad / 2)
                        end_to_pad = to_pad - st_to_pad
                        st -= st_to_pad
                        end += end_to_pad
                        if(st < 0):
                            end = end + (-st)
                            st = 0
                        if(end >=  mat.shape[0]):
                            st = st - end_to_pad - 1
                            end = mat.shape[0] - 1

                        pku_split_new[i] = st
                        pku_split_new[i + 1] = end

                    mat_pku = mat[st:end]
                    mat_pku_f = mat_pku[:, 0:75]
                    mat_pku_s = mat_pku[:, 75:150]                    

                    mat_pku_f = mat_pku_f.reshape(mat_pku_f.shape[0], self.joint_c, self.coord_c)

                    norm_mat_f, res = self.__format_mat(mat_pku_f)
                    if(res == False):
                        print("Unsalvageable activity sequence, continuing...")
                        continue
                    
                    
                    norm_mat_f = np.reshape(norm_mat_f, (norm_mat_f.shape[0], norm_mat_f.shape[1] * norm_mat_f.shape[2]))
                    norm_mat[0, st:end, 0:75] = norm_mat_f
                                  
                    # if(sec_person and self.__max_body == 2):
                    #     raise NotImplementedError()
                    #     mat_pku_s = mat_pku_s.reshape(mat_pku_s.shape[0], self.joint_c, self.coord_c)

                    #     norm_mat_s, res = self.__format_mat(mat_pku_s)
                    #     if(res == False):
                    #         print("Unsalvageable activity sequence, continuing...")
                    #         continue

                    #     norm_mat_s = np.reshape(norm_mat_s, (norm_mat_s.shape[0], norm_mat_s.shape[1] * norm_mat_s.shape[2]))
                    #     norm_mat[0, st:end, 75:150] = norm_mat_s
                    
                    #If all went well, add the class label and start/end
                    norm_mat[0, st:end, 150] = cls_label


                empty_spl_mat[0, :pku_split_new.shape[0]] = pku_split_new
                ind_splits = np.concatenate([ind_splits, empty_spl_mat])

            seq_len = norm_mat.shape[1]

            new_mat = np.zeros((1, self.act_size_max, norm_mat.shape[2]))
            new_mat[0,:seq_len,:] = norm_mat
            X = np.concatenate((X, new_mat))

            seq_lens.append(seq_len)


        seq_lens = np.array(seq_lens, dtype=np.int32)
        if(self.__dataset['type'] == Dataset.PKUMMD):
            return X, seq_lens, ind_splits
        elif(self.__dataset['type'] == Dataset.NTU_RGB):
            return X, seq_lens, self.__view_unique_ids[list_Id]
        else:
            return X, seq_lens, None, None


    def __load_files(self):
        params = self.__dataset['param']
        self.joint_c = 25 
        self.coord_c = 3
        self.act_size_max = MAX_LEN_SEQUENCE

        dir_ = str(Path(DATASET_DIR) / self.__dataset_name) + os.sep

        files = []
        classes = []
        view_un_files = []

        #TODO: Empty checks

        if(self.__dataset['type'] == Dataset.NTU_RGB):
            self.act_size_max = 300

            unique_dict = {}
            unique_count = 0

            camera_ind = params['camera']
            sub_ind = params['subject']
            class_ind = params['classes']
            
            file_filter = None
            file_skip_path = dir_ + "samples_with_missing_skeletons.txt"
            if Path(file_skip_path).is_file():
                with open(file_skip_path) as f:
                    file_filter = f.read().splitlines() 

            for f in os.listdir(dir_):

                if(file_filter is not None and Path(f).stem in file_filter):
                    continue

                match =  re.search(r'S([0-9]+)C([0-9]+)P([0-9]+)R([0-9]+)A([0-9]+).skeleton', f)
                if(match == None):
                    continue

                st_n, cam, subj, rep, act = int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4)), int(match.group(5))
                un_str = f"{st_n}-{subj}-{rep}-{act}"

                if(sub_ind != 'all' and subj not in sub_ind):
                    continue

                if(camera_ind != 'all' and cam not in camera_ind):
                    continue

                if(class_ind != 'all' and act not in class_ind):
                    continue

                files.append(dir_ + f)
                classes.append(act)
                cur_un_id = -1

                if un_str in unique_dict:
                    cur_un_id = unique_dict[un_str]
                else:
                    unique_dict[un_str] = unique_count
                    unique_count +=1
                    cur_un_id = unique_count

                view_un_files.append(unique_dict[un_str])
            
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

            un_count = 0
            for f in range(len(target_files_list)):
                cur_file = target_files_list[f]

                #file blacklist (messed up labels) #TODO:create blacklist?
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

        y = np.array(classes)
        v = np.unique(y)
        z = len(v)

        indices = list(range(0,z))
        dict_tmp = dict(zip(v.tolist(),indices))

        self.labels_dataset = np.array([dict_tmp.get(i, -1) for i in y], dtype=np.int32)
        self.__files_dataset = files
        self.__dataset_length = self.labels_dataset.size

        if(self.__dataset['type'] == Dataset.NTU_RGB):
            ind_sort = np.argsort(view_un_files)
            self.__files_dataset = np.array(files)[ind_sort]
            self.labels_dataset = self.labels_dataset[ind_sort]

            self.__view_unique_ids = np.array(view_un_files)[ind_sort]



    def __ntu_rgb_read_file(self, file_n, act):

        file_h = open(file_n, 'r')

        frame_rows_dict = []
        frame_rows_dict_trdata = []
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
                        frame_rows_dict_trdata.append(False)

                    joint_c = int(file_h.readline())
                    
                    frame_row = np.empty((1,75))

                    for j in range(joint_c):
                        joint_info = str(file_h.readline()).split(' ')
                        x, y, z, t = float(joint_info[0]), float(joint_info[1]), float(joint_info[2]), float(joint_info[11])
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



    def __check_filter_zero_vec(self, mat):
        all_zero_vec = np.all(mat == 0.0, axis=1)

        start_ind = 0
        end_ind = all_zero_vec.shape[0]

        if(all_zero_vec[0] == True):
            start_ind = np.argwhere(np.diff(all_zero_vec) == True)[0][0] + 1

        if(all_zero_vec[-1] == True):
            end_ind = np.argwhere(np.diff(np.flip(all_zero_vec,axis=0)) == True)[0][0] + 1
            end_ind = all_zero_vec.shape[0] - end_ind
            
        assert start_ind < end_ind
        mat_new = mat[start_ind:end_ind]

        offsets = start_ind, all_zero_vec.shape[0] - end_ind

        return mat_new, offsets