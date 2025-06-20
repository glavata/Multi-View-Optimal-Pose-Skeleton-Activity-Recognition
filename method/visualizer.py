import os
import numpy as np
import cv2 as cv2
import re
import json
from pathlib import Path
from utils.skel_visualization import SkeletonVisualizer

def draw_dataset_generic(dataset_draw, filename_draw, type="gif"):

    sk = SkeletonVisualizer()

    folder_name = "graphics//{0}_{1}".format(filename_draw, type)

    os.makedirs(folder_name, exist_ok=True)

    X_act_tr, seq_lens_sampl_tr, y_act_tr = dataset_draw[0], dataset_draw[1], dataset_draw[2]

    if(type == 'gif'):
        for j in range(X_act_tr.shape[0]):
            filename = folder_name + "//" + str(j)
            sk.plot_anim(X_act_tr[j, :seq_lens_sampl_tr[j,0]], filename, False)

    pass

def draw_dataset_generator(generator, filename_draw, draw_type, cur_max_len_act, coords):

    sk = SkeletonVisualizer()

    folder_name = "graphics//{0}_{1}".format(filename_draw, draw_type)

    os.makedirs(folder_name, exist_ok=True)

    X_act_local_buffer = np.zeros((3,cur_max_len_act,coords), dtype=np.float64)
    seq_len_local_buffer = np.zeros((3), dtype=np.int32)

    counter = 0

    for node in generator:

        filename_cur = folder_name + os.sep + str(counter)
        if(draw_type == "mv_seq_uneq_dtw"):
            X_act_t, y_act_t, seq_lens_act_t, dtw_paths = node

            filename_cur = filename_cur + "_cls_{0}".format(y_act_t)
            sk.gen_views_non_parallel_simple(X_act_t, seq_lens_act_t, filename_cur, dtw_paths)
        elif(draw_type == "mv_seq_eq_full"):

            X_act_t, y_act_t, seq_lens_act_t, view_map, coords_sys, angles = node
            filename_cur = filename_cur + "_cls_{0}".format(y_act_t)
            sk.gen_views_parallel_full(X_act_t, seq_lens_act_t, view_map, coords_sys, angles, filename_cur)

        elif(draw_type == "gif_single"):
            X_act_t, y_act_t, angles = node
            filename_cur = filename_cur + "_cls_{0}".format(y_act_t)
            #sk.plot_anim(X_act_t, filename_cur, False, angles)

        elif(draw_type == "gif_triple"):
            X_act_t, y_act_t, coord_sys_t, view_map_t = node
            filename_cur = filename_cur + "_cls_{0}".format(y_act_t)
   
            sk.plot_anim_multiview(X_act_t, view_map_t, coord_sys_t, filename_cur)

        counter+=1

    pass
    # if(type == "mv_seq"):
    #     for j in range(X_act_tr.shape[0]):
    #         filename = folder_name + "//" + str(j)
    #         sk.plot_anim(X_act_tr[j, :seq_lens_sampl_tr[j,0]], filename, False)        
    #     sk.plot_views_parallel(X_act_tr[j, :seq_lens_sampl_tr[j,0]], filename, False)


def draw_dataset_hidden_states():

    sk = SkeletonVisualizer()

    for f in os.listdir('results//hmm_models//'):

        match =  re.search(r'([A-z]+)_hmm_model_data_(\d+)_states.json', f)
        if(match == None):
            continue
        
        folder_name = "graphics//{0}".format(Path(f).stem)

        os.makedirs(folder_name, exist_ok=True)

        fuse_type, n_states = match.group(1), int(match.group(2))
        with open('results//hmm_models//' + f, 'r') as file:
            hmm_models = json.load(file)
    
        m_count = 0
        for m in hmm_models:
            len_states = len(m['states']) - 2
            assert(len_states == n_states)

            skels = np.zeros((1, len_states, 75))
            for s in range(len_states):
                cur_state = m['states'][s]
                if(cur_state == None):
                    break

                filename_cur = "{}//{}_model".format(folder_name, str(m_count))
                
                for em in range(len(cur_state)):
                    mean, std = cur_state[em]
                    skels[0, s, em+3] = mean
                
            sk.gen_views_non_parallel_simple(skels, [n_states], filename_cur)

        
            m_count+=1
