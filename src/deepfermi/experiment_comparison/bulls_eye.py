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
    pdf_name_list = ['snr_15_bulls_eye.pdf']
    img_sub_dir_list = ['snr_15_bulls_eye']
    complete_read_path = ['/data/brahma01/deepfermi/invivo/Experiments/Test_Debug/']
        
    # Save settings
    view_boundaries_flag = False
    perf_aha_path = Path('/data/brahma01/Datasets/perfusion_kcl/aha/')
    save_path = '/data/brahma01/deepfermi/invivo/Experiments/Test_Debug/'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    param_lim = [(0.0, None), (0.0, None), (0.0, None)]
    
    # Generating plot
    for i, path in enumerate(complete_read_path):
        
        # Loading arrays
        pid =  np.load(Path.joinpath(Path(path),'pid.npy'))
        eta =  np.load(Path.joinpath(Path(path),'eta_net.npy'))
        im_sig =  np.load(Path.joinpath(Path(path),'im_sig.npy'))
        seg =  np.load(Path.joinpath(Path(path),'seg.npy'))
        
        for eta_load in ['eta_net', 'eta_lbfgs']:
            
            print("Patient-wise bull's eye plots for " + eta_load)
            
            # Load perfusion maps        
            eta =  np.load(Path.joinpath(Path(path), eta_load + '.npy'))
            
            # Generating patient perfusion maps        
            img_List = []
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
                
                # Scale range
                mfactor = 0.98
                # Flow
                flow_seg = seg[p:p+3]
                flow_seg[eta[p:p+3,0] > 6/60] = 0
                flow_max = 1.7 #  round(mfactor * 60 * (eta[p:p+3,0][flow_seg==1].mean()+ 3*eta[p:p+3,0][flow_seg==1].std()), 1)
                # Delay
                delay_seg = seg[p:p+3]
                delay_seg[eta[p:p+3,1] > 5] = 0
                delay_max = 3.7 #  round(mfactor * eta[p:p+3,1][delay_seg==1].mean()+ 3*eta[p:p+3,1][delay_seg==1].std(), 0)
                # Decay
                decay_seg = seg[p:p+3]
                decay_seg[eta[p:p+3,2] > 0.5] = 0
                decay_max = 0.2 #  round(mfactor * eta[p:p+3,2][decay_seg==1].mean()+ 3*eta[p:p+3,2][decay_seg==1].std(), 2)
                
                # Diagnosis
                diagnosis = [Image.open(Path.joinpath(pdata_path, 'diagnosis.png'))]
                range_diagnosis = [(None,None)]
                cmap_diagnosis = [None]
                title_diagnosis = ["Diagnosis"]
                ylabel_diagnosis = [""]
                
                # Bulls-eye
                # Flow
                plottingSettings = PlottingSettings(cmap = plt.cm.inferno,
                                                    vmin = 0,
                                                    vmax = flow_max,
                                                    show_segmentNumbers = True,
                                                    show_std = False,
                                                    closePlotAutomatically = True,
                                                    show_debuggingImages = False,
                                                    useEdgesToSetInnerPoints= False)
                flow_bullseye = PerfusionBullseye(pid[p], plottingSettings, flow_bullseye_array)
                flow_bullseye.bulls_eye.figure.savefig(Path.joinpath(Path(save_path), 'flow_bullseye'), dpi=300)
                plot_flow_bullseye = [Image.open(Path.joinpath(Path(save_path), 'flow_bullseye.png'))]
                range_flow_bullseye = [(None,None)]
                cmap_flow_bullseye = [None]
                title_flow_bullseye = ["Flow"]
                ylabel_flow_bullseye = [""]            
                # Delay
                plottingSettings = PlottingSettings(cmap = plt.cm.inferno,
                                                    vmin = 0,
                                                    vmax = delay_max,
                                                    show_segmentNumbers = True,
                                                    show_std = False,
                                                    closePlotAutomatically = True,
                                                    show_debuggingImages = False,
                                                    useEdgesToSetInnerPoints= False)
                delay_bullseye = PerfusionBullseye(pid[p], plottingSettings, delay_bullseye_array)
                delay_bullseye.bulls_eye.figure.savefig(Path.joinpath(Path(save_path), 'delay_bullseye'), dpi=300)
                plot_delay_bullseye = [Image.open(Path.joinpath(Path(save_path), 'delay_bullseye.png'))]
                range_delay_bullseye = [(None,None)]
                cmap_delay_bullseye = [None]
                title_delay_bullseye = ["Delay"]
                ylabel_delay_bullseye = [""]
                
                # For segmentation plots
                flow_segments[(p*ns)//3:((p*ns)//3)+ns] = flow_bullseye.perf_val[0:-1]
                delay_segments[(p*ns)//3:((p*ns)//3)+ns] = delay_bullseye.perf_val[0:-1]
                diagnosis_segments[(p*ns)//3:((p*ns)//3)+ns] = np.load(Path.joinpath(pdata_path, 'diagnosis.npy'))[0:-1]
                
                # Apical
                plot_apical = [im_sig[p].mean(-1)]
                range_apical = [(None,None)]
                cmap_apical = ['gray']
                title_apical = ["Apical"]
                # ylabel_apical = ["Apical", "", "", ""]
                
                # Mid
                plot_mid = [im_sig[p+1].mean(-1)]
                range_mid = [(None,None)]
                cmap_mid = ['gray']
                title_mid = ["Mid"]
                # ylabel_mid = ["Mid", "", "", ""]
                
                # Basal
                plot_basal = [im_sig[p+2].mean(-1)]
                range_basal = [(None,None)]
                cmap_basal = ['gray']
                title_basal = ["Basal"]
                # ylabel_basal = ["Basal", "", "", ""]
                
                # Accumulating plots
                plot_list = [*plot_apical, *plot_mid, *plot_basal, *plot_flow_bullseye, *plot_delay_bullseye, *diagnosis]
                range_list = [*range_apical, *range_mid, *range_basal, *range_flow_bullseye, *range_delay_bullseye, *range_diagnosis]
                cmap_list = [*cmap_apical, *cmap_mid, *cmap_basal, *cmap_flow_bullseye, *cmap_delay_bullseye, *cmap_diagnosis]
                title_list = [*title_apical, *title_mid, *title_basal, *title_flow_bullseye, *title_delay_bullseye, *title_diagnosis]
                # ylabel_list = [*ylabel_apical, *ylabel_mid, *ylabel_basal, *ylabel_bullseye, *ylabel_diagnosis]
                
                # Plot images
                # matplotlib.use('TkAgg')  
                figsize=(14, 3)
                suptitle = "Bull's Eye (Patient " + str(pid[p]) + ")"        
                ncol = 6
                subplot = plt.figure(figsize=figsize)
                nplots = plot_list.__len__()
                nrows = np.ceil(nplots / ncol).astype(int)
                subplot.suptitle(suptitle)
                gs = subplot.add_gridspec(nrows, ncol)
                for plt_count in range(nplots):
                    plot_img = plot_list[plt_count]
                    j = plt_count % ncol
                    k = np.floor(plt_count / ncol).astype(int)
                    axs = subplot.add_subplot(gs[k, j])
                    im = axs.imshow(plot_img, cmap=cmap_list[plt_count])
                    if (plt_count)<3 :
                        # Define a custom colormap
                        cmap_boundary = plt.get_cmap('bwr')
                        cmap_boundary.set_under('k', alpha=0)
                        plt.imshow(boundaries[...,plt_count], cmap=cmap_boundary, clim=[0.1, 1], alpha=1.0)
                    axs.set_title(title_list[plt_count], fontsize=16)
                    # axs.set_ylabel(ylabel_list[plt_count], fontsize=16)
                    plt.setp(axs.get_xticklabels(), visible=False)
                    plt.setp(axs.get_yticklabels(), visible=False)
                    axs.tick_params(axis='both', which='both',length=0)            
                    l_lim = range_list[plt_count][0]
                    u_lim = range_list[plt_count][1]
                    im.set_clim(l_lim, u_lim)
                    for pos in ['right', 'top', 'bottom', 'left']: 
                        plt.gca().spines[pos].set_visible(False)
                
                plt.close()            
                save_name = 'Patient_' + str(pid[p]) + '.png'
                save_data_dir = Path.joinpath(Path(save_path), img_sub_dir_list[i] + '_' + eta_load)
                Path(save_data_dir).mkdir(parents=True, exist_ok=True)
                subplot.savefig(Path.joinpath(save_data_dir, save_name), dpi=500)
                img_dir = str(Path.joinpath(save_data_dir, save_name))
                imga = Image.open(img_dir)
                imga.load()
                img = Image.new("RGB", imga.size, (255, 255, 255))
                img.paste(imga, mask=imga.split()[3])

                # Crop
                imw, imh = img.size
                if p==0:
                    img1=img
                else:
                    img_List.append(img)
                
            pdf_name = pdf_name_list[i]
            pdf_dir = str(Path.joinpath(Path(save_path), pdf_name))
            img1.save(pdf_dir, "PDF", resolution=100.0, save_all=True, append_images=img_List)

if __name__ == "__main__":
    main()