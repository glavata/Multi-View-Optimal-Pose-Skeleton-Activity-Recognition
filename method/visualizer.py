import os
import numpy as np
import cv2 as cv2

from utils.skel_visualization import SkeletonVisualizer

def draw_dataset_generic(dataset_draw, filename_draw, type="gif"):

    sk = SkeletonVisualizer()

    folder_name = "graphics//{0}".format(filename_draw)

    os.makedirs(folder_name, exist_ok=True)

    X_act_tr, seq_lens_sampl_tr, y_act_tr = dataset_draw[0], dataset_draw[1], dataset_draw[2]

    for j in range(X_act_tr.shape[0]):
        filename = folder_name + "//" + str(j)
        if(type == 'gif'):
            sk.plot_anim(X_act_tr[j, :seq_lens_sampl_tr[j,0]], filename, False)

    pass

