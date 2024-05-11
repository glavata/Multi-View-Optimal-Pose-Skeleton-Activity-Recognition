from glob import glob
import cv2 as cv2
import joblib
import os
import numpy as np
from numpy import genfromtxt
from os.path import dirname, basename
from math import inf
from itertools import chain

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import Axes3D



links_NTU = [
    (3, 2),  (2, 20), (20, 4), (4,5), (5,6), (6,7), (6,22), (7,21),
    (20, 8), (8, 9), (9, 10), (10, 11), (10,24), (11,23),
    (20,1), (1,0), (0, 12), (12, 13), (13,14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19)
]

links_UTD = [
    (0, 1),(1, 2),(2, 3),
    (1, 4),(4, 5),(5, 6 ),(6, 7 ),(1, 8 ),
    (8, 9),(9, 10),(10,11),(3, 12),(12,13),
    (13,14),(14,15),(3,16 ),(16,17),(17,18), (18,19)
]

coord_3D = [[1, 0, 0], 
            [0, 1, 0], 
            [0, 0, -1]]

graphics_folder = "results//graphics//"

class SkeletonVisualizer():

    def __init__(self, links = links_NTU):
        self.links = links
        pass


    def initialize_plots_skel(self, skel, color_skel, fig=None, axes=None, ind=0):
        
        x_min = np.min(skel[:,0]) - 0.7
        x_max = np.max(skel[:,0]) + 0.7
        y_min = np.min(skel[:,1]) - 0.7
        y_max = np.max(skel[:,1]) + 0.7
        z_min = np.min(skel[:,2]) - 0.7
        z_max = np.max(skel[:,2]) + 0.7

        if(fig is None):
            fig = plt.figure()
            axes = fig.add_subplot(111, projection='3d')

        
        axes.view_init(elev=98, azim=-91)

        axes.dist = 7
        axes.set_xlabel('X')
        axes.set_ylabel('Y')
        axes.set_zlabel('Z')
        axes.set_xlim(left=x_max, right=x_min)
        axes.set_ylim(bottom=y_min, top=y_max)
        axes.set_zlim(bottom=z_max, top=z_min)
        axes.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        #axes.grid(False)
        #axes.axis('off')

        # Hide axes ticks
        #axes.set_xticks([])
        #axes.set_yticks([])
        #axes.set_zticks([])

        plots = {link: axes.plot([0], [0], [0], '{0}o-'.format(color_skel), markersize=2.5, \
                markeredgecolor='red' , markerfacecolor='red')[0] for link in self.links}
        return fig, axes, plots

    def plot_skeleton_multi(self, skels, coords, colors_skel, show_plot = False):
        
        #axes_arr = []
        #plots_arr = []


        glob_fig, glob_axes = plt.subplots(1, 3, subplot_kw={"projection": "3d"}, figsize=(10, 5))

        for i in range(len(skels)):
            skel = skels[i]
            fig, axes, plots = self.initialize_plots_skel(skels[i], colors_skel[i], glob_fig, glob_axes[i], i)
            #axes_glob = glob_fig.add_subplot(111, projection='3d')

            for link in self.links:
                top, bottom = link

                xs = np.array([skel[top, 0], skel[bottom, 0]])
                ys = np.array([skel[top, 1], skel[bottom, 1]])
                zs = np.array([skel[top, 2], skel[bottom, 2]])
                plots[link].set_xdata(xs)
                plots[link].set_ydata(ys)
                plots[link].set_3d_properties(zs)


            self.plot_coord_system(coords[i], axes)
            #glob_axis[i] = axes

        if(show_plot):
            plt.show()

        return tuple(plots.values())


    def plot_skeleton_single(self, skel, coords=None, color_skel='b'):

        _, axes, plots = self.initialize_plots_skel(skel, color_skel)
        
        for link in self.links:
            top, bottom = link

            xs = np.array([skel[top, 0], skel[bottom, 0]])
            ys = np.array([skel[top, 1], skel[bottom, 1]])
            zs = np.array([skel[top, 2], skel[bottom, 2]])
            plots[link].set_xdata(xs)
            plots[link].set_ydata(ys)
            plots[link].set_3d_properties(zs)
        
        if(coords is not None):
            self.plot_coord_system(coords, axes)
        # for i in range(len(coord_3D[0])):
        #     plt_tmp = axes.plot(np.array([coord_3D[0][i],0]), 
        #                         np.array([coord_3D[1][i],0]), 
        #                         np.array([coord_3D[2][i],0]), 'go-', markersize=2.5, \
        #     markeredgecolor='black' , markerfacecolor='black')[0]


        plt.show()

    def plot_coord_system(self, system, axes = None):
        
        plots = []

        if(axes is None):
            x_min = (system[0,0] + 0.7) * -1
            x_max = system[0,0] + 0.7
            y_min = (system[1,1] + 0.7) * -1
            y_max = system[1,1] + 0.7
            if(system.shape[1] == 3):
                z_min = (system[2,2] + 0.7) * -1
                z_max = system[2,2] + 0.7

            fig = plt.figure()
            axes = fig.add_subplot(111, projection='3d')
            axes.view_init(elev=98, azim=-91)

            axes.dist = 7
            axes.set_xlabel('X')
            axes.set_ylabel('Y')
            axes.set_zlabel('Z')

            axes.set_xlim(left=x_max, right=x_min)
            axes.set_ylim(bottom=y_min, top=y_max)

            if(system.shape[1] == 3):
                axes.set_zlim(bottom=z_max, top=z_min)

            axes.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        colors_line = ['r<--', 'g<--', 'b<--']

        for i in range(len(system[0])):
            plt_tmp = axes.plot(np.array([system[0][i],0]), 
                                np.array([system[1][i],0]), 
                                np.array([system[2][i],0]), colors_line[i], markersize=2.5, \
            markeredgecolor='black' , markerfacecolor='black')[0]
            plots.append(plt_tmp)

        if(axes is None):
            plt.show()

        return plots
        


    def plot_views_parallel(self):
        pass


    def plot_views_pose_sel(self):
        pass


    def plot_anim(self, skeleton_data, out_filename, show):
        
        skeleton_data = skeleton_data.reshape(skeleton_data.shape[0], 25, 3)
        fig, axes, plots = self.initialize_plots_skel(skeleton_data, color_skel='b')
        fr_text = axes.text(1, 1, 1,'',horizontalalignment='left',verticalalignment='top')


        def init():
            for link in self.links:
                plots[link].set_xdata(np.array([0]))
                plots[link].set_ydata(np.array([0]))
                plots[link].set_3d_properties(np.array([0]))

            fr_text.set_text(str(0))
            return tuple(plots.values()) + (fr_text,)

        def animate(i):
            skel = skeleton_data[i,:,:]

            for link in self.links:
                top, bottom = link
                xs = np.array([skel[top, 0], skel[bottom, 0]])
                ys = np.array([skel[top, 1], skel[bottom, 1]])
                zs = np.array([skel[top, 2], skel[bottom, 2]])
                plots[link].set_xdata(xs)
                plots[link].set_ydata(ys)
                plots[link].set_3d_properties(zs)

            fr_text.set_text(str(i))

            #plt.show()
            #fig.savefig(out_filename + "_" + str(i) + ".png")
            return tuple(plots.values()) + (fr_text,)

        interval = 33

        video = animation.FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=skeleton_data.shape[0],
            interval=interval,
            blit=True, repeat=False
        )

        writer = PillowWriter(fps=30)  
        video.save(out_filename + ".gif",writer=writer)
        #extra_args=['-vcodec', 'libx264']

        if show:
            plt.show()

        #plt.clf()
        plt.close('all')


    def plot_anim_multiview(self, skel_seqs, view_map, coord_sys, out_filename):

        v_count = skel_seqs.shape[0]

        glob_fig, glob_axes = plt.subplots(1, 3, subplot_kw={"projection": "3d"}, figsize=(10, 5))
        plots_arr = []
        plots_coord = []
        skel_seqs = skel_seqs.reshape(skel_seqs.shape[0],skel_seqs.shape[1], 25, 3)

        for j in range(v_count):
            skel = skel_seqs[:,0][j]
            fig, axes, plots = self.initialize_plots_skel(skel, 'r', glob_fig, glob_axes[j])
            plots_coord = self.plot_coord_system(coord_sys[0,0], glob_axes[j])
            plots_arr.append(plots)

        def init():
            for p in range(len(plots_arr)):
                pl = plots_arr[p]

                xs = np.array([np.array([0])])
                ys = np.array([np.array([0])])
                zs = np.array([np.array([0])])

                for link in self.links:
                    top, bottom = link

                    pl[link].set_xdata(xs)
                    pl[link].set_ydata(ys)
                    pl[link].set_3d_properties(zs)

                plots_coord[p].set_xdata(xs)
                plots_coord[p].set_ydata(ys)
                plots_coord[p].set_3d_properties(zs)

            #res = []

            #for p_tmp in plots_arr:
            #    res.extend(p_tmp.values())

            #res.extend(plots_coord)

            #return res

        def animate(i):
            color_mask = ['r'] * v_count
            color_mask[view_map[i]] = 'b'

            for j in range(len(skel_seqs[:,i])):
                skel = skel_seqs[:,i][j]
                
                for link in self.links:
                    top, bottom = link

                    xs = np.array([skel[top, 0], skel[bottom, 0]])
                    ys = np.array([skel[top, 1], skel[bottom, 1]])
                    zs = np.array([skel[top, 2], skel[bottom, 2]])
                    plots_arr[j][link].set_xdata(xs)
                    plots_arr[j][link].set_ydata(ys)
                    plots_arr[j][link].set_3d_properties(zs)


                plots_coord[j].set_xdata(np.array([coord_sys[j, i, 0, 0]]))
                plots_coord[j].set_ydata(np.array([coord_sys[j, i, 1, 1]]))
                plots_coord[j].set_3d_properties(np.array([coord_sys[j, i, 2, 2]]))

                #self.plot_coord_system(coord_sys[j, i], glob_axes[j])

            #res = []

            #for p_tmp in plots_arr:
                #res.extend(p_tmp.values())

            #res.extend(plots_coord)

            #return res


        interval = 33

        video = animation.FuncAnimation(
            glob_fig,
            animate,
            #init_func=init,
            frames=skel_seqs.shape[1],
            interval=interval,
            blit=False, repeat=False
        )

        writer = PillowWriter(fps=30)  
        video.save(graphics_folder + "{0}.gif".format(out_filename),writer=writer)

        plt.clf()
        plt.close('all')