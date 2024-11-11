#%% Imports
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from scipy.stats import gamma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
import imageio
import shutil
from time import sleep
from tqdm import tqdm
from scipy.interpolate import griddata
import h5py
import pickle
import random

# Uses noise to vary between parameters
def assign_2D_qpar_val(seg, seg_indx, map, inter_sb_var, intra_sb_var):    
    qpar_val = torch.clip(torch.normal(inter_sb_var[0], inter_sb_var[1]), inter_sb_var[2], inter_sb_var[3])    
    seg_xdim = seg.shape[0]
    seg_ydim = seg.shape[1]
    for y in range(seg_ydim):
        for x in range(seg_xdim):
            if seg[y][x]==seg_indx:
                map[y][x]=qpar_val + torch.clip(torch.normal(intra_sb_var[0], intra_sb_var[1]), intra_sb_var[2], intra_sb_var[3])
    return map

# Purely T1 weighted signal

def t1_signal(t, t1_map):
    one = np.ones(t1_map.shape)
    t_len = t.shape[0]
    t = t * np.repeat(one[..., np.newaxis], t_len, axis=2)
    t1_map[t1_map==0]=np.inf
    t1 = np.repeat(t1_map[..., np.newaxis], t_len, axis=2) 
    m0 = 1
    mz = m0*(1-np.exp(-(t/t1)))
    c = 0.1
    output = c*mz
    return output

# Fermi Impulse Response function
def fermi_ir_func(t, eta):
    one = torch.ones(eta[0].shape, device=eta.device)
    t_len = t.shape[0]    
    t = t * one.unsqueeze(-1).repeat(1,1,t_len)
    flow_rate = eta[0].unsqueeze(-1)
    delay = eta[1].unsqueeze(-1)
    decay_rate = eta[2].unsqueeze(-1)        
    with torch.no_grad():
        unit_step = torch.heaviside(t-delay,torch.tensor(0.5, dtype=delay.dtype, device=delay.device))
    output = flow_rate*(1/(torch.exp((t-delay)*decay_rate)+1)) * unit_step
    return output

# Arterial Input Function
def aif_func(t, gpar):
    xdim = gpar[0].shape[0]
    ydim = gpar[0].shape[1]
    one = torch.ones(gpar[0].shape, device=gpar.device)
    t_len = t.shape[0]
    t_one = t * one.unsqueeze(-1).repeat(1,1,t_len)
    t = t.cpu()
    aplha_gamma = gpar[0].cpu()
    beta_gamma = gpar[1].cpu()
    delay_gamma = gpar[2].cpu()
    aif = torch.zeros(t_one.shape)
    for y in range(ydim):
        for x in range(xdim):
            if aplha_gamma[y][x]!=0 and delay_gamma[y][x]!=0 and beta_gamma[y][x]!=0:
                aif[y][x] = 0.06 * torch.tensor(gamma.pdf(t, aplha_gamma[y][x], delay_gamma[y][x], beta_gamma[y][x]))
    aif = aif.cuda()
    return aif

# Convolutional operator
def convolve(input, im_res):   
    # output =  im_res ⨂ input 
    x_len, y_len, t_len = im_res.shape
    im_res_flip = torch.flip(im_res, [2])
    output = torch.zeros(input.shape, dtype=input.dtype, device=input.device)
    for t_indx in range(t_len):
        output[..., t_indx] = torch.sum(im_res_flip[..., -(t_indx+1):] * input[...,:(t_indx+1)], dim=2)
    return output

# Generating GIF
def generate_gif(frames, save_path, gif_name='Untitled.gif', cmap = 'gray', clim = [0,1]):
    frames = np.asarray(frames)
    gif_dir = Path.joinpath(save_path, 'gif_cache')
    Path(gif_dir).mkdir(parents=True, exist_ok=True)
    print('generating gif...')
    for f_count in tqdm(range(frames.shape[2])):
        f_current = frames[...,f_count]
        fig = plt.figure()
        im = plt.imshow(f_current, cmap = cmap)
        plt.axis('off')
        im.set_clim(clim[0], clim[1])
        plt.colorbar(im)
        frame_name = str(f_count)
        frame_name =  str.zfill(frame_name, int(np.floor(np.log10(frames.shape[2])+1)))
        fig.savefig(Path.joinpath(gif_dir, frame_name), dpi=100)
        plt.close()
    filenames = list(sorted(Path(gif_dir).glob('*.png*')))
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(Path.joinpath(save_path, gif_name), images)
    shutil.rmtree(gif_dir)
    print('generating gif complete...')

# Visualize Dataset
def visualize_dataset(data_dic, save_path):
    
    rand_indx = random.randint(0, (data_dic.keys().__len__() - 1))

    eta = data_dic[list(data_dic.keys())[rand_indx]]['eta']
    im_sig = data_dic[list(data_dic.keys())[rand_indx]]['im_sig']
    ctc = data_dic[list(data_dic.keys())[rand_indx]]['ctc']
    aif = data_dic[list(data_dic.keys())[rand_indx]]['aif']
    seg = data_dic[list(data_dic.keys())[rand_indx]]['seg']

    #%% Perfusion Parameters Maps
    plot_list = [60 * eta[0], eta[1], eta[2]]
    title_list = ["F GND", "Tau GND", "k GND"]
    range_list = [(0,4), (0,3), (0,0.2)]
    cmap_list = ['viridis', 'viridis', 'viridis']
    no_of_plots = plot_list.__len__()
    no_of_col = 3
    no_of_rows = np.ceil(no_of_plots / no_of_col).astype(int)
    sub_plot_fig = plt.figure(figsize=(14, 6))
    sub_plot_fig.suptitle('Posterior Sampling')
    gs = sub_plot_fig.add_gridspec(no_of_rows, no_of_col)
    for plt_count in range(no_of_plots):
        plot_img = plot_list[plt_count]
        i = plt_count % no_of_col
        j = np.floor(plt_count / no_of_col).astype(int)
        axs = sub_plot_fig.add_subplot(gs[j, i])
        im = axs.imshow(plot_img, cmap=cmap_list[plt_count])
        axs.set_title(title_list[plt_count])
        axs.axis('off')
        l_lim = range_list[plt_count][0]
        u_lim = range_list[plt_count][1]    
        im.set_clim(l_lim, u_lim)
        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
    sub_plot_fig.savefig(Path.joinpath(save_path, 'PParam_Maps.png'), dpi=500)
    plt.close()

    # # Concentration-Time Curves
    # # Healthy Myocardium
    # healthy = list(zip(*np.where(seg==1)))[5]
    # # Defect Myocardium
    # defect = list(zip(*np.where(seg==71)))[5]
    # # Left ventrical blood pool
    # aif_loc = list(zip(*np.where(seg==5)))[5]
    # # Curves
    # pixel_picker = [healthy, defect]
    # pixel_class = ["Healthy", "Defect"]
    # fig = plt.figure() #figsize=(9, 7)
    # fig.suptitle('Concentration-Time Curves')
    # for loc, cls in zip(pixel_picker, pixel_class):
    #     gnd_ctc_label = cls + "Output"
    #     plt.plot(im_sig[loc[0], loc[1], :], label=gnd_ctc_label, linewidth=1.3)    
    # aif_vfact = 2 * (im_sig[healthy[0], healthy[1], ...].max()/im_sig[aif_loc[0], aif_loc[1], ...].max())
    # aif_label = str(aif_vfact.round(2)) + " * AIF"
    # plt.plot(im_sig[aif_loc[0], aif_loc[1], ...] * aif_vfact, color="blue", label=aif_label, linewidth=1.3)
    # plt.xlabel('time (s)')
    # plt.ylabel('concetration (l)')
    # plt.legend(loc="upper right")
    # fig.savefig(Path.joinpath(save_path, 'ctc.png'), dpi=500)
    # plt.close()

    # GIF
    # generate_gif(t1_sig, save_path, gif_name='t1_sig.gif', cmap = 'gray', clim = [0,0.8*t1_sig.max()])
    generate_gif(ctc, save_path, gif_name='ctc.gif', cmap = 'gray', clim = [0,0.8*ctc.max()])
    # generate_gif(aif, save_path, gif_name='aif.gif', cmap = 'gray', clim = [0,0.8*aif.max()])
    generate_gif(im_sig, save_path, gif_name='im_sig.gif', cmap = 'gray', clim = [0,0.8*im_sig.max()])
    
# Construct data dictionary of perfusion parameters
def construct_dictionary(file_path, param_range, dtype=torch.float64, device='cuda'):
    
    time = torch.arange(param_range["time"]["start"],
                     param_range["time"]["end"],
                     param_range["time"]["step_size"], dtype=dtype, device=device)
    
    data_dic = {}
    with h5py.File(file_path, "r") as f:       
        
        # Synthesizing data
        for sb in tqdm(range(list(f.keys()).__len__())):
            
            # # Extracting segmentation
            # # Segmentation Indentifiers
            # 1: Myocardium Left Ventrical
            # 71: Lesion Myocardium Left Ventrical
            # 5: AIF
            pid = list(f.keys())[sb]
            data_dic[pid] = {}
            seg = torch.tensor(f[pid]['segmentation'][()], device=device)
            
            # Creating place holder for perfusion parameters
            flow_rate_sim = torch.zeros(seg.shape, dtype=dtype, device=device)
            delay_sim = torch.zeros(seg.shape, dtype=dtype, device=device)
            decay_rate_sim = torch.zeros(seg.shape, dtype=dtype, device=device)
            # Myocardium healthy region
            flow_rate_sim = assign_2D_qpar_val(seg, 
                                               1,
                                               flow_rate_sim,
                                               param_range["myo_healthy"]["flow"][0], 
                                               param_range["myo_healthy"]["flow"][1])
            delay_sim = assign_2D_qpar_val(seg, 
                                               1,
                                               delay_sim,
                                               param_range["myo_healthy"]["delay"][0], 
                                               param_range["myo_healthy"]["delay"][1])
            decay_rate_sim = assign_2D_qpar_val(seg, 
                                               1,
                                               decay_rate_sim,
                                               param_range["myo_healthy"]["decay"][0], 
                                               param_range["myo_healthy"]["decay"][1])            
            # Myocardium ischemic region
            flow_rate_sim = assign_2D_qpar_val(seg, 
                                               71,
                                               flow_rate_sim,
                                               param_range["myo_ischemic"]["flow"][0], 
                                               param_range["myo_ischemic"]["flow"][1])
            delay_sim = assign_2D_qpar_val(seg, 
                                               71,
                                               delay_sim,
                                               param_range["myo_ischemic"]["delay"][0], 
                                               param_range["myo_ischemic"]["delay"][1])
            decay_rate_sim = assign_2D_qpar_val(seg, 
                                               71,
                                               decay_rate_sim,
                                               param_range["myo_ischemic"]["decay"][0], 
                                               param_range["myo_ischemic"]["decay"][1])  
            # Fermi impulse response    
            eta = torch.cat((flow_rate_sim.unsqueeze(0), delay_sim.unsqueeze(0), decay_rate_sim.unsqueeze(0)), 0)
            fermi_ir = fermi_ir_func(time, eta)
            
            # Arterial Input Function
            aif = torch.zeros(seg.shape, dtype=dtype, device=device)
            aplha_gamma = torch.zeros(seg.shape, dtype=dtype, device=device)
            beta_gamma = torch.zeros(seg.shape, dtype=dtype, device=device)
            delay_gamma = torch.zeros(seg.shape, dtype=dtype, device=device)
            # Assigning aif values [mean, variance, min_clip, max_clip]
            # Aplha gamma
            aplha_gamma = assign_2D_qpar_val(seg, 
                                             5, 
                                             aplha_gamma, 
                                             param_range["aif"]["alpha"][0], 
                                             param_range["aif"]["alpha"][1])
            beta_gamma = assign_2D_qpar_val(seg, 
                                            5, 
                                            beta_gamma, 
                                            param_range["aif"]["beta"][0], 
                                            param_range["aif"]["beta"][1])
            delay_gamma = assign_2D_qpar_val(seg, 
                                             5, 
                                             delay_gamma, 
                                             param_range["aif"]["delay"][0], 
                                             param_range["aif"]["delay"][1])
            # AIF in blood pool Left Ventrical
            gpar = torch.cat((aplha_gamma.unsqueeze(0), beta_gamma.unsqueeze(0), delay_gamma.unsqueeze(0)), 0)
            aif = aif_func(time, gpar)
            aif_seg = aif.clone()
            # Average AIF
            # basic_imshow(aif[...,65].cpu(), '/data/brahma01/DCEPerfusion/Debug3/')        
            bp_seg = seg.clone()
            bp_seg[bp_seg!=5]=0
            bp_seg[bp_seg==5]=1
            # bp_seg = np.repeat(bp_seg[..., np.newaxis], t_len, axis=2)
            aif_avg = torch.mean(aif[bp_seg==1,...],0)
            aif = aif_avg.unsqueeze(0).unsqueeze(0) * torch.ones(fermi_ir.shape, device=device)    
            # Concentration curve
            ctc = convolve(aif, fermi_ir)
            
            # Contrast enhanced signal
            im_sig = ctc + aif_seg
            
            # Inducing motion-outliers
            mask_indx = torch.randint(0, im_sig.shape[-1], (param_range["max_outliers"],))            
            # Bounding box area
            bbox = torch.zeros(seg.shape, device=device)
            bbox[seg==1]=1
            bbox[seg==71]=1
            bbox_x = bbox.sum(1)
            bbox_y = bbox.sum(0)
            bbox = (bbox_x.unsqueeze(1)) * (bbox_y.unsqueeze(0))
            bbox[bbox!=0]=1
            bbox_xlen, bbox_ylen = bbox.sum(0).max(), bbox.sum(1).max()
            # Translting affected frames
            myo_seg = torch.zeros(seg.shape, device=device)
            myo_seg[seg==1]=1
            myo_seg[seg==71]=1
            nx, ny, _ = aif_seg.shape
            for frame in mask_indx:
                xshift = int(0.5 * torch.rand(1, device=device) * bbox_xlen)
                yshift = int(0.5 * torch.rand(1, device=device) * bbox_ylen)
                outlier_frame = F.pad(aif_seg[..., frame], (nx//2,nx//2,ny//2,ny//2))
                outlier_frame = torch.roll(outlier_frame, shifts=(xshift, yshift), dims=(0, 1))
                outlier_frame = outlier_frame[nx//2:nx//2 + nx, ny//2:ny//2 + ny]
                outlier_frame = myo_seg * outlier_frame
                ctc[..., frame][outlier_frame!=0] = outlier_frame[outlier_frame!=0]
                im_sig[..., frame][outlier_frame!=0] = outlier_frame[outlier_frame!=0]
                
            # Affected ctc
            myo_seg = torch.zeros(seg.shape, device=device)
            myo_seg[seg==1]=1
            myo_seg[seg==71]=1
            
            # Creating list
            data_dic[pid]["im_sig"] = im_sig.cpu().numpy()
            data_dic[pid]["seg"] = myo_seg.cpu().numpy()
            data_dic[pid]["ctc"] = ctc.cpu().numpy()
            data_dic[pid]["aif"] = aif_avg.cpu().numpy()
            data_dic[pid]["time"] = time.cpu().numpy()
            data_dic[pid]["wlen"] = param_range["wlen"]
            data_dic[pid]["eta"] = eta.cpu().numpy()
        
        return data_dic
    
def dataset_split_indices(data_dic, shuffle=True):
    
    # The number of trainin samples
    npat = data_dic.__len__()
    ids = np.arange(npat)
    if shuffle==True:
        np.random.shuffle(ids)    
    # Dataset splits in percentage    
    train_split = 70
    val_split = 20
    test_split = 10
    nb_train = int(np.floor((train_split/100) * npat))
    nb_val = int(np.floor((val_split/100) * npat))
    nb_test = int(np.floor((test_split/100) * npat))
    # split indices    
    nb_train_min = 0
    nb_train_max = nb_train + nb_train_min
    nb_val_min = nb_train_max
    nb_val_max = nb_val + nb_val_min
    nb_test_min = nb_val_max
    nb_test_max = nb_test + nb_test_min
    
    return ids[nb_train_min:nb_train_max], ids[nb_val_min:nb_val_max], ids[nb_test_min:nb_test_max]

def main() -> None:
    
    # General settings    
    file_name = "quiero_cardiac_mrf_sim.h5"
    load_path = (Path(__file__).resolve().parent.parent.parent / 'data/XCAT_phantom')
    save_path = (Path(__file__).resolve().parent.parent.parent / 'data')
    save_name = "dce_perfusion_data"
    device = 'cuda'
    dtype = torch.float64
    time_step = 0.94
    time_start = 0
    time_end = 106
    wlen = 100
    max_outliers = 5
    
    # Parameter range settings
    param_range =  {}
    # Time (in seconds)
    param_range["time"] = {}
    param_range["time"]["step_size"] = time_step
    param_range["time"]["start"] = time_start
    param_range["time"]["end"] = time_end
    # Main-bolus window length
    param_range["wlen"] = wlen
    # Main-bolus window length
    param_range["max_outliers"] = max_outliers
    # Myocardium healthy region (inter, intra), (mean, std, min_clip, max_clip)
    param_range["myo_healthy"] = {}
    param_range["myo_healthy"]["flow"] = torch.tensor([[3/60, 0.005, 0, 5/60], [0, 0.005, 0, 5/60]], dtype=dtype, device=device)
    param_range["myo_healthy"]["delay"] = torch.tensor([[2, 0.1, 0, 3], [0, 0.1, 0, 3]], dtype=dtype, device=device)
    param_range["myo_healthy"]["decay"] = torch.tensor([[0.1, 0.01, 0, 0.5], [0, 0.01, 0, 0.5]], dtype=dtype, device=device)
    # Myocardium ischemic region (inter, intra), (mean, std, min_clip, max_clip)
    param_range["myo_ischemic"] = {}
    param_range["myo_ischemic"]["flow"] = torch.tensor([[1/60, 0.005, 0, 4/60], [0, 0.005, 0, 4/60]], dtype=dtype, device=device)
    param_range["myo_ischemic"]["delay"] = torch.tensor([[3, 0.1, 0, 5], [0, 0.1, 0, 5]], dtype=dtype, device=device)
    param_range["myo_ischemic"]["decay"] = torch.tensor([[0.05, 0.01, 0, 0.1], [0, 0.01, 0, 0.1]], dtype=dtype, device=device)
    # Arterial input function (inter, intra), (mean, std, min_clip, max_clip)
    param_range["aif"] = {}
    param_range["aif"]["alpha"] = torch.tensor([[3, 0.1, 2, 5], [0, 0.1, 2, 5]], dtype=dtype, device=device)
    param_range["aif"]["beta"] = torch.tensor([[1.1, 0.1, 1, 2], [0, 0.05, 1, 2]], dtype=dtype, device=device)
    param_range["aif"]["delay"] = torch.tensor([[(10*1)*time_step, (10*0.5)*time_step, (10*0.8)*time_step, (10*2)*time_step],
                                   [0, (10*0.1)*time_step, (10*0.8)*time_step, (10*2)*time_step]], dtype=dtype, device=device)
    
    # Construction of data dictionary
    file_path = Path.joinpath(load_path, file_name)
    data_dic = construct_dictionary(file_path, param_range, dtype, device)
    np.savez( Path.joinpath(save_path, save_name), **data_dic)
    
    # Suggested Training Partitions
    train_indices, val_indices, test_indices = dataset_split_indices(data_dic, shuffle=True)
    with open(Path.joinpath(save_path, save_name +'_info.txt'),'w') as file:
        file.write("%s %s\n" % ('Train: ', str(train_indices).strip('[]').replace("\n", " ").split()))
        file.write("%s %s\n" % ('Validation: ', str(val_indices).strip('[]').replace("\n", " ").split()))
        file.write("%s %s\n" % ('Test: ', str(test_indices).strip('[]').replace("\n", " ").split()))
    
    # Visualize a random sample
    visualize_dataset(data_dic, save_path)
    
if __name__ == "__main__":
    main()