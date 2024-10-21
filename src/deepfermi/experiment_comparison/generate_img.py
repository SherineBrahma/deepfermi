import numpy as np
from pathlib import Path

from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
from matplotlib import ticker
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from PIL import Image

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
    pdf_name_list = ['eta']
    img_sub_dir_list = ['eta']
    complete_read_path = ['/data/brahma01/deepfermi/invivo/Experiments/Test_Seg_Outliers/']
        
    # Save settings
    # view_flag = True
    save_path = '/data/brahma01/deepfermi/invivo/Experiments/Test_Seg_Outliers/'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    param_lim = [(0.0, 1.7), (0, 4), (0.0, None)] # 4 # 2.5
    
    # Generating plot
    for i, path in enumerate(complete_read_path):
        
        # Loading arrays
        pid =  np.load(Path.joinpath(Path(path),'pid.npy'))
        eta =  np.load(Path.joinpath(Path(path),'eta_net.npy'))
        im_sig =  np.load(Path.joinpath(Path(path),'im_sig.npy'))
        seg =  np.load(Path.joinpath(Path(path),'seg.npy'))
        
        for eta_load in ['eta_net', 'eta_lbfgs']:
            eta =  np.load(Path.joinpath(Path(path), eta_load + '.npy'))        
            # Printing images
            img_List = []
            N = eta.shape[0]
            print('Generating images...')
            for j in tqdm(range(0, N, 3)):
                
                # Remove background
                eta[j:j+3,...][np.repeat(seg[j:j+3, None],3,axis=1)==0]=0
                
                # Perfusion map colormap
                pmap_colormap = 'viridis' # 'inferno' # 
                # Apical
                plot_apical = [60 * eta[j,0], eta[j,1], eta[j,2], im_sig[j].mean(-1)]
                range_apical = [(param_lim[0][0],param_lim[0][1]), (param_lim[1][0],param_lim[1][1]), (param_lim[2][0],param_lim[2][1]), (0,None)]
                cmap_apical = [pmap_colormap, pmap_colormap, pmap_colormap, 'gray']
                title_apical = ["Flow", "Delay", "Decay", "Signal Intensity"]
                ylabel_apical = ["Apical", "", "", ""]
                
                # Mid
                plot_mid = [60 * eta[j+1,0], eta[j+1,1], eta[j+1,2], im_sig[j+1].mean(-1)]
                range_mid = [(param_lim[0][0],param_lim[0][1]), (param_lim[1][0],param_lim[1][1]), (param_lim[2][0],param_lim[2][1]), (0,None)]
                cmap_mid = [pmap_colormap, pmap_colormap, pmap_colormap, 'gray']
                title_mid = ["", "", "", ""]
                ylabel_mid = ["Mid", "", "", ""]
                
                # Basal
                plot_basal = [60 * eta[j+2,0], eta[j+2,1], eta[j+2,2], im_sig[j+2].mean(-1)]
                range_basal = [(param_lim[0][0],param_lim[0][1]), (param_lim[1][0],param_lim[1][1]), (param_lim[2][0],param_lim[2][1]), (0,None)]
                cmap_basal = [pmap_colormap, pmap_colormap, pmap_colormap, 'gray']
                title_basal = ["", "", "", ""]
                ylabel_basal = ["Basal", "", "", ""]
                
                # Accumulating plots
                plot_list = [*plot_apical, *plot_mid, *plot_basal]
                range_list = [*range_apical, *range_mid, *range_basal]
                cmap_list = [*cmap_apical, *cmap_mid, *cmap_basal]
                title_list = [*title_apical, *title_mid, *title_basal]
                ylabel_list = [*ylabel_apical, *ylabel_mid, *ylabel_basal]
                
                # Plot images
                # matplotlib.use('TkAgg')  
                figsize=(12, 10)
                suptitle = 'Perfusion Maps (Patient ' + str(pid[j]) + ')'
                ncol = 4
                subplot = plt.figure(figsize=figsize)
                nplots = plot_list.__len__()
                nrows = np.ceil(nplots / ncol).astype(int)
                subplot.suptitle(suptitle)
                gs = subplot.add_gridspec(nrows, ncol)
                for plt_count in range(nplots):
                    plot_img = plot_list[plt_count]
                    if int(plt_count%4) != 3:
                        mask = (1-seg[j+int(np.floor(plt_count/4))])
                    o = int((np.ceil((plt_count+1)/4)*4)-1)
                    overlay_img = plot_list[o]             
                    k = plt_count % ncol
                    l = np.floor(plt_count / ncol).astype(int)
                    axs = subplot.add_subplot(gs[l, k])
                    im = axs.imshow(plot_img, cmap=cmap_list[plt_count])
                    axs.imshow(overlay_img, cmap=cmap_list[o], alpha=mask)
                    axs.set_title(title_list[plt_count], fontsize=16)
                    axs.set_ylabel(ylabel_list[plt_count], fontsize=16)
                    plt.setp(axs.get_xticklabels(), visible=False)
                    plt.setp(axs.get_yticklabels(), visible=False)
                    axs.tick_params(axis='both', which='both',length=0)            
                    l_lim = range_list[plt_count][0]
                    u_lim = range_list[plt_count][1]
                    im.set_clim(l_lim, u_lim)
                    for pos in ['right', 'top', 'bottom', 'left']: 
                        plt.gca().spines[pos].set_visible(False)
                    divider = make_axes_locatable(axs)
                    cax = divider.append_axes("bottom", size="5%", pad=0.05)
                    plt.colorbar(im, cax=cax, orientation="horizontal")
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=None)
                # plt.show()
                plt.close()
                save_name = 'Patient_' + str(pid[j]) + '.png'
                save_data_dir = Path.joinpath(Path(save_path), img_sub_dir_list[i] + "_" + eta_load)
                Path(save_data_dir).mkdir(parents=True, exist_ok=True)
                subplot.savefig(Path.joinpath(save_data_dir, save_name), dpi=500)
                img_dir = str(Path.joinpath(save_data_dir, save_name))
                imga = Image.open(img_dir)
                imga.load()
                img = Image.new("RGB", imga.size, (255, 255, 255))
                img.paste(imga, mask=imga.split()[3])

                # Crop
                imw, imh = img.size
                if j==0:
                    img1=img
                else:
                    img_List.append(img)         
                
            pdf_name = pdf_name_list[i] + "_" + eta_load + ".pdf"
            pdf_dir = str(Path.joinpath(Path(save_path), pdf_name))
            img1.save(pdf_dir, "PDF", resolution=100.0, save_all=True, append_images=img_List)

if __name__ == "__main__":
    main()