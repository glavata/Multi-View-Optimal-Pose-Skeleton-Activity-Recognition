
import cv2 as cv2
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import PillowWriter
from mpl_toolkits.mplot3d import Axes3D


links_NTU = [
    (3, 2),  (2, 20), (20, 4), (4,5), (5,6), (6, 7), (7, 22), (7, 21),
    (20, 8), (8, 9), (9, 10), (10, 11), (11, 24), (11, 23),
    (20, 1), (1, 0), (0, 12), (12, 13), (13, 14), (14, 15),
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


    def initialize_plots_skel(self, skel, color_skel, fig=None, axes=None, hide_all=False):
        
        x_min = np.min(skel[:,0]) - 0.7
        x_max = np.max(skel[:,0]) + 0.7
        y_min = np.min(skel[:,1]) - 0.7
        y_max = np.max(skel[:,1]) + 0.7
        z_min = np.min(skel[:,2]) - 0.7
        z_max = np.max(skel[:,2]) + 0.7

        # x_min = np.min(skel[:,0]) - 0.4
        # x_max = np.max(skel[:,0]) + 0.4
        # y_min = np.min(skel[:,1]) - 0.3
        # y_max = np.max(skel[:,1]) + 0.3
        # z_min = np.min(skel[:,2]) - 0.4
        # z_max = np.max(skel[:,2]) + 0.4


        if(fig is None):
            fig = plt.figure()
            axes = fig.add_subplot(111, projection='3d')

        if(axes is None):
            raise "Axes parameter not set."
        
        axes.view_init(elev=98, azim=-91, roll=0)

        axes.dist = 7
        axes.set_xlabel('X')
        axes.set_ylabel('Y')
        axes.set_zlabel('Z')
        axes.set_xlim(left=x_max, right=x_min)
        axes.set_ylim(bottom=y_min, top=y_max)
        axes.set_zlim(bottom=z_max, top=z_min)
        axes.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        if(hide_all):
            axes.grid(False)
            axes.axis('off')

            #Hide axes ticks
            axes.set_xticks([])
            axes.set_yticks([])
            axes.set_zticks([])

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

        plt.show()

    def plot_coord_system(self, system, axes = None, show=False, coef=1, start_pos = None):
        
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

        colors_line = ['r<--', 'k<--', 'b<--'] #x, y, z
        start_pos_coords = np.zeros(3)
        if(start_pos is not None):
            start_pos_coords = start_pos

        for i in range(len(system[0])):
            plt_tmp = axes.plot(np.array([system[0][i] * coef +  start_pos_coords[0], start_pos_coords[0]]), 
                                np.array([system[1][i] * coef +  start_pos_coords[1], start_pos_coords[1]]), 
                                np.array([system[2][i] * coef +  start_pos_coords[2], start_pos_coords[2]]), colors_line[i], markersize=2.5, \
            markeredgecolor='black' , markerfacecolor='black')[0]
            plots.append(plt_tmp)

        if(show):
            plt.show()

        return plots
        
    def gen_views_parallel_full(self, skel_seqs, seq_lens, view_map, coords, angles, filename):
        v_count = skel_seqs.shape[0]
        max_c = seq_lens

        if(len(skel_seqs.shape) == 3):
            assert(skel_seqs.shape[-1] == 75)
            skel_seqs = skel_seqs.reshape(skel_seqs.shape[0],skel_seqs.shape[1], 25, 3)

        width_s = 400
        height_s = 400
        w_d = 40
        h_d = 40
        h_d_s = 20

        total_image_w = max_c * (width_s + w_d) - w_d
        total_image_h = v_count * (height_s + h_d) - h_d + h_d_s * 2

        image_res = np.ones((total_image_h, total_image_w, 3), dtype=np.uint8) * 255

        plt.ioff()

        for i in range(max_c):
            #glob_fig, glob_axes = plt.subplots(v_count, 1, subplot_kw={"projection": "3d"}, figsize=(5, 8))
            for j in range(v_count):

                skel = skel_seqs[j, i]

                color_cur = 'b'
                color_txt = 'blue'
                if(view_map[i] == j):
                    color_cur = 'g'
                    color_txt = 'green'

                fig, axes, plots = self.initialize_plots_skel(skel, color_cur)

                for link in self.links:
                    top, bottom = link

                    xs = np.array([skel[top, 0], skel[bottom, 0]])
                    ys = np.array([skel[top, 1], skel[bottom, 1]])
                    zs = np.array([skel[top, 2], skel[bottom, 2]])

                    plots[link].set_xdata(xs)
                    plots[link].set_ydata(ys)
                    plots[link].set_3d_properties(zs)
            
                if(coords is not None):
                    self.plot_coord_system(coords[j, i], axes, coef=0.3, start_pos=skel[20])

                if(angles is not None):
                    axes.text2D(0.25, 0.9, "{:.2f}, {:.2f}, {:.2f}".format(*angles[j, i]), color=color_txt, fontsize=16,  transform=axes.transAxes)
                
                axes.xaxis.labelpad=1

                #plt.tight_layout()
                fig.canvas.draw()
                img_plot = np.array(fig.canvas.renderer.buffer_rgba())
                img_ocv = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
                img_gr = cv2.cvtColor(img_ocv, cv2.COLOR_BGR2GRAY)
                plt.close()

                gray = (img_gr != 255).astype(np.uint8) # To invert the text to white
                coords_non_zero = cv2.findNonZero(gray) # Find all non-zero points (text)
                x, y, w, h = cv2.boundingRect(coords_non_zero)
                # x = max(0, x - 20)
                #y = max(0, y - 10)
                # w = min(img_gr.shape[1], w + 20)
                # h = min(img_gr.shape[0], h + 60)
                rect = img_ocv[y:y+h, x:x+w]
                prop_rect = rect.shape[1] / rect.shape[0]
                new_h = min(int(width_s / prop_rect), height_s)
                rect_res = cv2.resize(rect, (width_s, new_h), interpolation= cv2.INTER_LINEAR)

                tgt_s_y = h_d_s + j * height_s + j * h_d
                tgt_s_x = i * width_s + i * w_d

                image_res[tgt_s_y:tgt_s_y+new_h, tgt_s_x:tgt_s_x+width_s] = rect_res


        cv2.imwrite(filename + ".png", image_res)    
        plt.ion()


    #TOVA
    def gen_views_non_parallel_simple(self, skel_seqs, seq_lens, filename, dtw_paths=None):
        
        v_count = skel_seqs.shape[0]
        v_count_ind = np.arange(v_count)
        max_c = np.max(seq_lens)

        skel_seqs = skel_seqs.reshape(skel_seqs.shape[0],skel_seqs.shape[1], 25, 3)
        width_s = 70
        height_s = 120
        w_d = 10
        h_d = 100
        h_d_2 = 10

        total_image_w = max_c * (width_s + w_d) + w_d
        total_image_h = v_count * (height_s + h_d) + h_d_2 * 2 - h_d

        image_res = np.ones((total_image_h, total_image_w, 3), dtype=np.uint8) * 255

        if(dtw_paths is not None):
            assert len(dtw_paths) == v_count
            tgt_ind = int(v_count / 2)
            ind_center = int(v_count / 2)
            for d in range(v_count):
                if(dtw_paths[d] is None):
                    ind_center = d

            if(ind_center != tgt_ind):
                v_count_ind[tgt_ind] = ind_center
                v_count_ind[ind_center] = tgt_ind
                tmp_dtw = dtw_paths[tgt_ind]
                dtw_paths[tgt_ind] = dtw_paths[ind_center]
                dtw_paths[ind_center] = tmp_dtw

        plt.ioff()

        for i in range(max_c):
            #glob_fig, glob_axes = plt.subplots(v_count, 1, subplot_kw={"projection": "3d"}, figsize=(5, 8))
            for j in range(v_count_ind.shape[0]):
                ind_v = v_count_ind[j]
                if(seq_lens[ind_v] <= i):
                    continue

                skel = skel_seqs[ind_v, i]

                fig, axes, plots = self.initialize_plots_skel(skel, 'b', hide_all=True)

                for link in self.links:
                    top, bottom = link

                    xs = np.array([skel[top, 0], skel[bottom, 0]])
                    ys = np.array([skel[top, 1], skel[bottom, 1]])
                    zs = np.array([skel[top, 2], skel[bottom, 2]])

                    plots[link].set_xdata(xs)
                    plots[link].set_ydata(ys)
                    plots[link].set_3d_properties(zs)
            
                plt.tight_layout()
                fig.canvas.draw()
                img_plot = np.array(fig.canvas.renderer.buffer_rgba())
                img_ocv = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
                img_gr = cv2.cvtColor(img_ocv, cv2.COLOR_BGR2GRAY)
                plt.close()

                gray = (img_gr != 255).astype(np.uint8)
                coords = cv2.findNonZero(gray)
                x, y, w, h = cv2.boundingRect(coords)
                rect = img_ocv[y:y+h, x:x+w]
                prop_rect = rect.shape[1] / rect.shape[0]
                new_h = min(int(width_s / prop_rect), height_s)
                rect_res = cv2.resize(rect, (width_s, new_h), interpolation= cv2.INTER_LINEAR)
                tgt_s_y = h_d_2 + j * height_s + j * h_d + int(height_s / 2) - int(new_h / 2)
                tgt_s_x = w_d + i * width_s + i * w_d
                image_res[tgt_s_y:tgt_s_y+new_h, tgt_s_x:tgt_s_x+width_s] = rect_res

            #cv2.imshow("test",img_ocv)
            #cv2.waitKey(0)


        if(dtw_paths is not None):
            for v in range(len(dtw_paths)):
                if(v == tgt_ind):
                    continue
                for s in range(dtw_paths[v].shape[0]):
 
                    to_i, from_i = dtw_paths[v][s] #to_i is the common view
                    
                    if(v > tgt_ind):
                        y_p_from = h_d_2 + v * height_s + v * h_d - 10
                        y_p_to  = h_d_2 + (tgt_ind + 1)* height_s + tgt_ind * h_d + 10
                    elif(v < tgt_ind):
                        y_p_from  = h_d_2 + (v + 1)* height_s + v * h_d + 10
                        y_p_to = h_d_2 + tgt_ind * height_s + tgt_ind * h_d - 10

                    x_p_from = w_d + int(width_s / 2 + from_i * width_s + from_i * w_d)
                    x_p_to = w_d + int(width_s / 2 + to_i * width_s + to_i * w_d)

                    image_res = cv2.line(image_res, (x_p_from, y_p_from), (x_p_to, y_p_to), (0, 0, 0), 1)

        cv2.imwrite(filename + ".png", image_res)    
        plt.ion()


    #def plot_hidden_states(self, h_states, out_filename):



    def plot_anim(self, skeleton_data, out_filename, show, angles=None):
        
        skeleton_data = skeleton_data.reshape(skeleton_data.shape[0], 25, 3)
        fig, axes, plots = self.initialize_plots_skel(skeleton_data, color_skel='b')
        fr_text = axes.text(1, 1, 1,'',horizontalalignment='left',verticalalignment='top')
        angles_text = axes.text(0.05, 1, 1,'',horizontalalignment='left',verticalalignment='top')

        plt.ioff()

        def init():
            for link in self.links:
                plots[link].set_xdata(np.array([0]))
                plots[link].set_ydata(np.array([0]))
                plots[link].set_3d_properties(np.array([0]))

            fr_text.set_text(str(0))

            if(angles is not None):
                angles_text.set_text(str(0))

            return tuple(plots.values()) + (fr_text,) + (angles_text,)

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

            if(angles is not None):
                angles_text.set_text(f"{angles[i,0]:.2f} {angles[i,1]:.2f} {angles[i,2]:.2f}")

            #plt.show()
            #fig.savefig(out_filename + "_" + str(i) + ".png")
            return tuple(plots.values()) + (fr_text,) + (angles_text,)

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

        plt.ion()
        #plt.clf()
        plt.close('all')


    def plot_anim_multiview(self, skel_seqs, view_map, coord_sys, out_filename):

        v_count = skel_seqs.shape[0]

        glob_fig, glob_axes = plt.subplots(1, 3, subplot_kw={"projection": "3d"}, figsize=(15, 5))
        plots_arr = []
        plots_coord = []
        skel_seqs = skel_seqs.reshape(skel_seqs.shape[0],skel_seqs.shape[1], 25, 3)
        plt.ioff()
        #plt.tight_layout()


        def init():

            for j in range(v_count):
                skel = skel_seqs[:,0][j]
                fig, axes, plots = self.initialize_plots_skel(skel, 'r', glob_fig, glob_axes[j])
                plots_coord_tmp = self.plot_coord_system(coord_sys[0,0], glob_axes[j])
                plots_arr.append(plots)
                plots_coord.append(plots_coord_tmp)

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

                plots_coord[p][0].set_xdata([0,0])
                plots_coord[p][1].set_ydata([0,0])
                plots_coord[p][2].set_3d_properties([0,0])



        def animate(i):
            color_mask = ['r'] * v_count
            color_mask[view_map[i]] = 'g'

            for j in range(skel_seqs[:,i].shape[0]):
                skel = skel_seqs[:,i][j]
                
                for link in self.links:
                    top, bottom = link

                    xs = np.array([skel[top, 0], skel[bottom, 0]])
                    ys = np.array([skel[top, 1], skel[bottom, 1]])
                    zs = np.array([skel[top, 2], skel[bottom, 2]])
                    plots_arr[j][link].set_xdata(xs)
                    plots_arr[j][link].set_ydata(ys)
                    plots_arr[j][link].set_3d_properties(zs)
                    plots_arr[j][link].set_color(color_mask[j])


                plots_coord[j][0].set_xdata([coord_sys[j, i, 0, 0] * 0.5, 0])
                plots_coord[j][0].set_ydata([coord_sys[j, i, 1, 0] * 0.5, 0])
                plots_coord[j][0].set_3d_properties([coord_sys[j, i, 2, 0] * 0.5, 0])
                
                plots_coord[j][1].set_xdata([coord_sys[j, i, 0, 1] * 0.5, 0])
                plots_coord[j][1].set_ydata([coord_sys[j, i, 1, 1] * 0.5, 0])
                plots_coord[j][1].set_3d_properties([coord_sys[j, i, 2, 1], 0])

                plots_coord[j][2].set_xdata([coord_sys[j, i, 0, 2] * 0.5, 0])
                plots_coord[j][2].set_ydata([coord_sys[j, i, 1, 2] * 0.5, 0])
                plots_coord[j][2].set_3d_properties([coord_sys[j, i, 2, 2] * 0.5, 0])



        interval = 33

        video = animation.FuncAnimation(
            glob_fig,
            animate,
            init_func=init,
            frames=skel_seqs.shape[1],
            interval=interval,
            blit=False, repeat=False
        )

        writer = PillowWriter(fps=30)  
        video.save("{0}.gif".format(out_filename),writer=writer)
        
        plt.clf()
        plt.close('all')
        plt.ion()