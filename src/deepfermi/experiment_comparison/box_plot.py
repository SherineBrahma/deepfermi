import sys

import cv2
import matplotlib
matplotlib.use('Agg')
from matplotlib import ticker
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu

from perf_bullseye import PerfusionBullseye

sys.path.append('/data/brahma01/PtbPyTools/')
from visualisation.ahaBullseye.ahaHelperfunctions import segmentLabel
from visualisation.ahaBullseye.ahaHelperfunctions import PlottingSettings


def get_subplot(ncol, plot_list, title_list, range_list, cmap_list, figsize=None, suptitle='Sub-Plot'):
    
    subplot = plt.figure(figsize=figsize)
    nplots = plot_list.__len__()
    nrows = np.ceil(nplots / ncol).astype(int)
    subplot.suptitle(suptitle)
    gs = subplot.add_gridspec(nrows, ncol)
    for plt_count in range(nplots):
        plot_img = plot_list[plt_count]
        i = plt_count % ncol
        j = np.floor(plt_count / ncol).astype(int)
        axs = subplot.add_subplot(gs[j, i])
        im = axs.imshow(plot_img, cmap=cmap_list[plt_count])
        axs.set_title(title_list[plt_count])
        axs.axis('off')
        l_lim = range_list[plt_count][0]
        u_lim = round(range_list[plt_count][1],3)
        im.set_clim(l_lim, u_lim)
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        bin = 4
        tick_array =  np.round(np.linspace(l_lim, u_lim, bin),3)
        cb = plt.colorbar(im, cax=cax, ticks=tick_array, orientation='horizontal')
        cb.ax.set_xticklabels(tick_array)
    plt.close()
    
    return subplot

def main() -> None:
        
    # Folders to be read    
    plot_name_list = ['box_plot']
    complete_read_path = ['/data/brahma01/deepfermi/invivo/Experiments/Test_cross_val_5_fold_5/']
        
    # Save settings
    view_boundaries_flag = False
    view_diagnosis_flag = False
    perf_aha_path = Path('/data/brahma01/Datasets/perfusion_kcl/aha/')
    save_path = '/data/brahma01/deepfermi/invivo/Experiments/Test_cross_val_5_fold_5/'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    param_lim = [(0.0, None), (0.0, None), (0.0, None)]
    
    # Generating plot
    for i, path in enumerate(complete_read_path):
        
        # Loading arrays
        pid =  np.load(Path.joinpath(Path(path),'pid.npy'))
        eta =  np.load(Path.joinpath(Path(path),'eta_net.npy'))
        im_sig =  np.load(Path.joinpath(Path(path),'im_sig.npy'))
        seg =  np.load(Path.joinpath(Path(path),'seg.npy'))
        
        # Generating box-plot
        flow_wilcoxtest = {}
        delay_wilcoxtest = {}        
        for eta_load in ['eta_net', 'eta_lbfgs']:
            
            print("Patient-wise bull's eye plots for " + eta_load)
            
            # Load perfusion maps        
            eta =  np.load(Path.joinpath(Path(path), eta_load + '.npy'))
            
            # Generating patient perfusion maps
            ns = int(16)
            N = eta.shape[0]
            flow_segments = np.zeros(N*ns//3)
            delay_segments = np.zeros(N*ns//3)
            diagnosis_segments = np.zeros(N*ns//3)
            for p in tqdm(range(0, N, 3)):
                
                # Segmentation boundaries and Flow bulls-eye
                pdata_path = Path.joinpath(perf_aha_path, str(pid[p]) + '_STRESS_moco')
                mask = np.load(Path.joinpath(pdata_path, 'mask.npy'))
                # Create a kernel for binary dilation (e.g., a 3x3 square)
                kernel = np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]], dtype=np.uint8)
                boundaries = np.zeros(mask.shape)
                flow_bullseye_array = np.zeros(17)
                delay_bullseye_array = np.zeros(17)
                nseg_aha = 0
                for sl in range(2,-1,-1):
                    boundaries_sl = np.zeros(mask[...,sl].shape)
                    for nseg in np.trim_zeros(np.unique(mask[...,sl])):
                        # Extracting segmentation masks
                        aha_seg = np.zeros(mask[...,0].shape)
                        aha_seg[mask[...,sl]==nseg] = 1
                        
                        # Bulls-eye
                        # Flow
                        flow_temp = 60 * eta[p+sl,0][aha_seg!=0]                    
                        flow_temp = flow_temp[np.abs(flow_temp-np.median(flow_temp))<=3.5*np.median(np.abs(flow_temp-np.median(flow_temp)))]
                        flow_bullseye_array[nseg_aha] = flow_temp.mean()
                        # Delay
                        delay_temp = eta[p+sl,1][aha_seg!=0]                    
                        delay_temp = delay_temp[np.abs(delay_temp-np.median(delay_temp))<=3.5*np.median(np.abs(delay_temp-np.median(delay_temp)))]
                        delay_bullseye_array[nseg_aha] = delay_temp.mean()
                        # Increase count       
                        nseg_aha += 1
                        
                        # Segmentation boundaries
                        aha_seg = aha_seg.astype(np.uint8)
                        dilated = cv2.dilate(aha_seg, kernel, iterations=1)
                        boundaries_sl = boundaries_sl + (dilated - aha_seg)
                        boundaries_sl[aha_seg!=0] = 0
                        
                    if view_boundaries_flag == True:                        
                        matplotlib.use('TkAgg')
                        plt.figure()
                        plt.imshow(boundaries_sl)
                        plt.title('AHA Segments')
                        plt.axis('off')  # Optional: Turn off the axis
                        plt.show()
                        
                    boundaries_sl[boundaries_sl!=0] = 1
                    boundaries[...,sl] = boundaries_sl
                
                # Bulls-eye
                # Flow
                flow_max = 1.7
                plottingSettings = PlottingSettings(cmap = plt.cm.inferno,
                                                    vmin = 0,
                                                    vmax = flow_max,
                                                    show_segmentNumbers = True,
                                                    show_std = False,
                                                    closePlotAutomatically = True,
                                                    show_debuggingImages = False,
                                                    useEdgesToSetInnerPoints= False)
                flow_bullseye = PerfusionBullseye(pid[p], plottingSettings, flow_bullseye_array)
                
                # Delay
                delay_max = 3.7
                plottingSettings = PlottingSettings(cmap = plt.cm.inferno,
                                                    vmin = 0,
                                                    vmax = delay_max,
                                                    show_segmentNumbers = True,
                                                    show_std = False,
                                                    closePlotAutomatically = True,
                                                    show_debuggingImages = False,
                                                    useEdgesToSetInnerPoints= False)
                delay_bullseye = PerfusionBullseye(pid[p], plottingSettings, delay_bullseye_array)
                
                # For segmentation plots
                flow_segments[(p*ns)//3:((p*ns)//3)+ns] = flow_bullseye.perf_val[0:-1]
                delay_segments[(p*ns)//3:((p*ns)//3)+ns] = delay_bullseye.perf_val[0:-1]
                diagnosis_segments[(p*ns)//3:((p*ns)//3)+ns] = np.load(Path.joinpath(pdata_path, 'diagnosis.npy'))[0:-1]
                
                # Visualize loaded maps
                if view_diagnosis_flag==True:
                    diagnosis_bullseye = PerfusionBullseye(pid[p], plottingSettings, diagnosis_segments[(p*ns)//3:((p*ns)//3)+ns])
                    matplotlib.use('TkAgg')
                    plt.figure(figsize=(6,6))
                    diagnosis_bullseye.bulls_eye.figure.show()
            
            # Extracting flow and delay        
            flow_healthy = (flow_segments *  (1-diagnosis_segments))
            flow_healthy = flow_healthy[flow_healthy!=0]
            flow_ischemic = (flow_segments * diagnosis_segments)
            flow_ischemic = flow_ischemic[flow_ischemic!=0]        
            delay_healthy = (delay_segments *  (1-diagnosis_segments))
            delay_healthy = delay_healthy[delay_healthy!=0]
            delay_ischemic = (delay_segments * diagnosis_segments)
            delay_ischemic = delay_ischemic[delay_ischemic!=0]
            
            # Printing Median
            print('Flow Healthy: ' + str(np.median(flow_healthy)))
            print('Flow Ischemic: ' + str(np.median(flow_ischemic)))
            print('Delay Healthy: ' + str(np.median(delay_healthy)))
            print('Delay Ischemic: ' + str(np.median(delay_ischemic)))
            
            # Flow Box-plot
            pval_flow_diagnosis = round(mannwhitneyu(flow_healthy, flow_ischemic, method="exact").pvalue,6)
            flow_diag_dic = {'Healthy':flow_healthy, 'Ischemic':flow_ischemic}
            # matplotlib.use('TkAgg')
            flow_box_plot, axes = plt.subplots()
            axes.boxplot(flow_diag_dic.values(), widths=0.5, showfliers=False)        
            axes.scatter(np.random.normal(1, 0.1, flow_healthy.shape), flow_healthy, label="Healthy", color="black", alpha=0.5)
            axes.scatter(np.random.normal(2, 0.1, flow_ischemic.shape), flow_ischemic, label="Ischemic", color="black", alpha=0.5)
            axes.set_xticklabels(flow_diag_dic.keys())
            axes.set_ylim(bottom=None, top=2.0)
            axes.set_aspect(2)
            flow_bplot_title = 'Flow Box Plot (pvalue:' + str(pval_flow_diagnosis) + ')'
            plt.title(flow_bplot_title)
            flow_box_plot_name = 'flow_' + plot_name_list[i] + '_' + str(eta_load)
            flow_box_plot.savefig(Path.joinpath(Path(save_path), flow_box_plot_name ), dpi=300)
            
            # Delay Box-plot
            pval_delay_diagnosis = round(mannwhitneyu(delay_healthy, delay_ischemic, method="exact").pvalue,6)
            delay_diag_dic = {'Healthy':delay_healthy,'Ischemic':delay_ischemic}
            # matplotlib.use('TkAgg')
            delay_box_plot, axes = plt.subplots()
            axes.boxplot(delay_diag_dic.values(), widths=0.5, showfliers=False)        
            axes.scatter(np.random.normal(1, 0.1, delay_healthy.shape), delay_healthy, label="Healthy", color="black", alpha=0.5)
            axes.scatter(np.random.normal(2, 0.1, delay_ischemic.shape), delay_ischemic, label="Ischemic", color="black", alpha=0.5)
            axes.set_xticklabels(delay_diag_dic.keys())
            axes.set_aspect(0.8)
            delay_bplot_title = 'Delay Box Plot (pvalue:' + str(pval_delay_diagnosis) + ')'
            plt.title(delay_bplot_title)
            delay_box_plot_name = 'delay_' + plot_name_list[i] + '_' + str(eta_load)
            delay_box_plot.savefig(Path.joinpath(Path(save_path), delay_box_plot_name ), dpi=300)
            
            flow_wilcoxtest[eta_load] = {}
            flow_wilcoxtest[eta_load]['Healthy'] = flow_healthy
            flow_wilcoxtest[eta_load]['Ischemic'] = flow_ischemic
            delay_wilcoxtest[eta_load] = {}
            delay_wilcoxtest[eta_load]['Healthy'] = delay_healthy
            delay_wilcoxtest[eta_load]['Ischemic'] = delay_ischemic

if __name__ == "__main__":
    main()