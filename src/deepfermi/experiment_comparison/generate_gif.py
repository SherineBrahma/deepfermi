import numpy as np
from pathlib import Path
import cv2

from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
import imageio
import shutil

def save_as_gif(img, 
                save_path, 
                save_name, 
                overlay = None, 
                cmap = 'gray', 
                clim = [None,None], 
                overlay_cmap = 'gray', 
                overlay_clim = [None, None], 
                overlay_alpha = 1, 
                dim = 0,
                seg=None):
    
    # Preliminary checks
    overlay_is_None = False
    if overlay is None:
        overlay_is_None = True
        overlay_alpha=0
        overlay=img
        
        
    # Segmentation boundaries
    if seg is not None:
        # Create a kernel for binary dilation (e.g., a 3x3 square)
        kernel = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]], dtype=np.uint8)
        boundaries_seg = np.zeros(seg.shape)
        seg = seg.astype(np.uint8) 
        dilated_seg = cv2.dilate(seg, kernel, iterations=1)
        boundaries_seg = boundaries_seg + (dilated_seg - seg)
        boundaries_seg[seg!=0] = 0
        boundaries_seg[boundaries_seg!=0] = 1
        
    # Generating GIFs    
    nframes = img.shape[dim]
    gif_dir = Path.joinpath(save_path, 'gif_cache')
    Path(gif_dir).mkdir(parents=True, exist_ok=True)
    for i in range(nframes):
        img_i = img[:, :, i]
        overlay_i = overlay[:, :, i]
        fig = plt.figure()
        im = plt.imshow(img_i, cmap = cmap)     
        if seg is not None:
            cmap_boundary = plt.get_cmap('bwr')
            cmap_boundary.set_under('k', alpha=0)
            plt.imshow(boundaries_seg, cmap=cmap_boundary, clim=[0.1, 1], alpha=1.0)
        im_o = plt.imshow(overlay_i, cmap = overlay_cmap, alpha=overlay_alpha)        
        if overlay_is_None == True:
            im.set_clim(clim[0], clim[1])
            plt.colorbar(im)
        else:
            im_o.set_clim(overlay_clim[0], overlay_clim[1])
            plt.colorbar(im_o)           
        plt.axis('off')                
        gif_name = str(i)
        gif_name =  str.zfill(gif_name, int(np.floor(np.log10(nframes)+1)))
        fig.savefig(Path.joinpath(gif_dir, gif_name), dpi=100, bbox_inches='tight')
        plt.close()
    filenames = list(sorted(Path(gif_dir).glob('*.png*')))
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(Path.joinpath(Path(save_path), save_name), images)
    shutil.rmtree(gif_dir)
            
            
def main() -> None:
    
    # Folders to be read    
    pdf_name_list = ['nspoke_540.pdf', 'nspoke_750.pdf', 'nspoke_1125.pdf']
    img_sub_dir_list = ['nspoke_540', 'nspoke_750', 'nspoke_1125']
    complete_read_path = ['/data/brahma01/DCEPerfusion/InVivo/Experiments/Test_xce_outliers/']
    
    # Save settings
    # view_flag = True
    save_path = '/data/brahma01/DCEPerfusion/InVivo/Experiments/Test_xce_outliers_gif/'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # # Crop Settings
    # xcrop_len, ycrop_len = 160, 160
    # img_dim = [320, 320, 30]    
    # xmid, ymid = img_dim[0]//2, img_dim[1]//2    
    # xmin, xmax  = xmid - xcrop_len//2, xmid + xcrop_len//2
    # ymin, ymax  = ymid - ycrop_len//2, ymid + ycrop_len//2
    
    # Generating GIFs
    for i, path in enumerate(complete_read_path):
        # Loading arrays
        data_dic = np.load(Path.joinpath(Path(path), 'test_data_dic.npz'), allow_pickle=True)
        pid =  data_dic['pat']
        img_sig =  data_dic['im_sig']
        wlen =  data_dic['wlen']
        seg =  data_dic['seg']        
        flow =  data_dic['eta_net'][:,0,...]
        delay =  data_dic['eta_net'][:,1,...]
        decay =  data_dic['eta_net'][:,2,...]
        
        # Determining meta information 
        N = pid.shape[0]
        slice_name = ['apical', 'mid', 'basal'] * int(N/3)
        
        # Printing images
        print('Generating images...')
        for j in tqdm(range(N)):
            
            # # Crop and calculate magnitude image
            # xf_plt = np.sqrt((xf[j,...,xmin:xmax,ymin:ymax, tindex]**2).sum(0))
            # xu_plt =  np.sqrt((xu[j,...,xmin:xmax,ymin:ymax, tindex]**2).sum(0))
            # xgen_plt =  np.sqrt((xgen[j,...,xmin:xmax,ymin:ymax, tindex]**2).sum(0))
            # xgen_epistemic_plt =  np.sqrt((xgen_epistemic[j,...,xmin:xmax,ymin:ymax, tindex]**2).sum(0))
            # xgen_aleatoric_plt =  np.sqrt((xgen_aleatoric[j,...,xmin:xmax,ymin:ymax, tindex]**2).sum(0))
            # xgen_total_uncertainty_plt = np.sqrt((xgen_total_uncertainty[j,...,xmin:xmax,ymin:ymax, tindex]**2).sum(0))
            # xgen_error_plt =  np.sqrt((xgen_error[j,...,xmin:xmax,ymin:ymax, tindex]**2).sum(0))
            
            # Extracting slices
            img_sig_plt =  img_sig[j,...,:wlen[j]]
            wlen_plt = wlen[j]
            seg_plt = seg[j]
            flow_plt =  flow[j]
            delay_plt = delay[j]
            decay_plt = decay[j]
            
            # Determining meta information             
            mask = 0.8*seg_plt
            
            # Saving image signal intensity GIF
            save_name = 'pid_' + str(pid[j]) + '_' + slice_name[j] + '.gif'
            decay_sub_dir = Path.joinpath(Path(save_path), 'img_sig')
            Path(decay_sub_dir).mkdir(parents=True, exist_ok=True)            
            save_as_gif(img_sig_plt,
                        save_path = decay_sub_dir, 
                        save_name = save_name,
                        cmap = 'gray', 
                        clim = [0,0.8],
                        dim = 2,
                        seg=seg_plt)
            
            
            # Saving Flow GIF
            save_name = 'pid_' + str(pid[j]) + '_' + slice_name[j] + '.gif'
            flow_sub_dir = Path.joinpath(Path(save_path), 'flow')
            Path(flow_sub_dir).mkdir(parents=True, exist_ok=True)            
            save_as_gif(img_sig_plt, 
                        save_path = flow_sub_dir, 
                        save_name = save_name, 
                        overlay = 60*flow_plt[...,None].repeat(wlen_plt,-1), 
                        cmap = 'gray', 
                        clim = [0,0.8], 
                        overlay_cmap = 'viridis', 
                        overlay_clim = [None, None], 
                        overlay_alpha = mask, 
                        dim = 2)
            
            # Saving delay GIF
            save_name = 'pid_' + str(pid[j]) + '_' + slice_name[j] + '.gif'
            delay_sub_dir = Path.joinpath(Path(save_path), 'delay')
            Path(delay_sub_dir).mkdir(parents=True, exist_ok=True)            
            save_as_gif(img_sig_plt, 
                        save_path = delay_sub_dir, 
                        save_name = save_name, 
                        overlay = delay_plt[...,None].repeat(wlen_plt,-1), 
                        cmap = 'gray', 
                        clim = [0,0.8], 
                        overlay_cmap = 'viridis', 
                        overlay_clim = [0,4], 
                        overlay_alpha = mask, 
                        dim = 2)
            
            # Saving decay GIF
            save_name = 'pid_' + str(pid[j]) + '_' + slice_name[j] + '.gif'
            decay_sub_dir = Path.joinpath(Path(save_path), 'decay')
            Path(decay_sub_dir).mkdir(parents=True, exist_ok=True)            
            save_as_gif(img_sig_plt, 
                        save_path = decay_sub_dir, 
                        save_name = save_name, 
                        overlay = decay_plt[...,None].repeat(wlen_plt,-1), 
                        cmap = 'gray', 
                        clim = [0,0.8], 
                        overlay_cmap = 'viridis', 
                        overlay_clim = [None, None], 
                        overlay_alpha = mask, 
                        dim = 2)
            
            a = 1

if __name__ == "__main__":
    main()