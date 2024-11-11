import sys

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from termcolor import colored

import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.append('/data/brahma01/deepfermi/invivo/')
from fermi import *
from utils import interp_linear_1D, expand_dim

sys.path.append('/data/brahma01/PtbPyTools/')
from visualisation.ahaBullseye.ahaBullseye import Bullseye

# Approximated Delta response
def translate_ir_func(delay, t, t0=0, C=1000, neg_shift=0):
    t = t-t0
    t = torch.cat((-torch.flip(t, [0])[-neg_shift-1:-1], t))    
    delayed_dirac = t-delay
    C = C/torch.pi
    translate_ir = np.sqrt(C/np.pi)*torch.exp(-C*(delayed_dirac)**2)    
    translate_ir = translate_ir/translate_ir.max()
    return translate_ir

# Get the corner coordinates given a slice mask
def bbox_corners(slice_mask, crop_len):
    # Retrieving indices
    x, y, w, h = cv2.boundingRect(np.uint8(slice_mask))
    xmin, xmax  = y, y + h
    ymin, ymax  = x, x + w    
    # Padding logic
    if (crop_len-(xmax-xmin))%2==0:
        xpad_left, xpad_right = (int((crop_len - (xmax-xmin))/2), int((crop_len - (xmax-xmin))/2))
    else:
        if (crop_len-(xmax-xmin))>0:
            xpad_left, xpad_right = (int((crop_len - (xmax-xmin))/2) + 1, int((crop_len - (xmax-xmin))/2))
        else:
            xpad_left, xpad_right = (int((crop_len - (xmax-xmin))/2), int((crop_len - (xmax-xmin))/2) - 1)
    if (crop_len-(ymax-ymin))%2==0:
        ypad_left, ypad_right = (int((crop_len - (ymax-ymin))/2), int((crop_len - (ymax-ymin))/2))
    else:
        if (crop_len-(ymax-ymin))>0:
            ypad_left, ypad_right = (int((crop_len - (ymax-ymin))/2) + 1, int((crop_len - (ymax-ymin))/2))
        else:
            ypad_left, ypad_right = (int((crop_len - (ymax-ymin))/2), int((crop_len - (ymax-ymin))/2) - 1)
    # Padded indices             
    xmin, xmax  = xmin - xpad_left, xmax + xpad_right
    ymin, ymax  = ymin - ypad_left, ymax + ypad_right
    
    return xmin, xmax, ymin, ymax

# Post processing of segmentation masks to fill undesired holes
def fill_holes(myo_seg):
    myo_seg_filled = myo_seg.copy()
    TRight = np.array([[0,1,0],[1,1,0],[0,1,0]])
    TLeft = np.array([[0,1,0],[0,1,1],[0,1,0]])
    TTop = np.array([[0,1,0],[1,1,1],[0,0,0]])
    TBottom = np.array([[0,0,0],[1,1,1],[0,1,0]])
    for i in range(1, myo_seg.shape[0]-1):
        for j in range(1, myo_seg.shape[1]-1):
            for m in range(0, myo_seg.shape[-1]):                    
                P = np.array([[myo_seg[i-1,j+1,m],myo_seg[i,j+1,m],myo_seg[i+1,j+1,m]],
                            [myo_seg[i-1,j,m],myo_seg[i,j,m],myo_seg[i+1,j,m]],
                            [myo_seg[i-1,j-1,m],myo_seg[i,j-1,m],myo_seg[i+1,j-1,m]]])
                myo_seg_filled[i,j,m] = np.any([(TRight*P).sum()//3, (TLeft*P).sum()//3, (TTop*P).sum()//3, (TBottom*P).sum()//3]).astype('float64') if myo_seg[i,j,m]==0 else 1
    return myo_seg_filled

def get_segmentation(im_sig, aif, edit=False, pbullseye=None, plottingSettings=None,  storedConfig=None, show_debuggCTC=False):
    # Retrieving indices
    cdat = np.mean(im_sig, axis=0) # Time-averaged intensity
    cdat = cdat * 100 / np.max(cdat.flatten())
    cdat = np.split(cdat, 3, axis=2)
    cdat.insert(0, np.zeros_like(cdat[0])) # Including apex
    overlay = im_sig * 100 / np.max(im_sig.flatten())
    odat = np.split(overlay, 3, axis=3)    
    odat.insert(0, np.zeros_like(odat[0]))
    aif_bulleye = aif * 100 / np.max(im_sig.flatten())
    if edit==True:
        plt.close('all')
        bullseye = Bullseye(cdat, odat, pbullseye, plottingSettings,  storedConfig, aif=aif_bulleye)
        plt.close('all')        
    aha_seg = np.load(pbullseye+'mask.npy')
    myo_seg = aha_seg.copy()
    myo_seg[myo_seg!=0] = 1
    myo_seg = fill_holes(myo_seg)
    # Inspection of segmented concentration-curves
    if show_debuggCTC:        
            ctc_db = im_sig[:, myo_seg==1]
            plt.figure()
            plt.title("Concentration Time Curves")
            plt.plot(aif, label="aif", linewidth=1, color="black", linestyle="dashed")
            plt.plot(ctc_db, linewidth=1)
            plt.legend(loc="upper right")   
            plt.show()
    
    return myo_seg

def compensate_motion(im_sig, seg, im_sig_full, tpad, device='cuda'):    
    # Constructing vectors
    ctc_mc = torch.tensor(im_sig, device=device).moveaxis(0,-1)
    ctc_mc = ctc_mc[seg!=-1]
    im_sig_compensated = np.pad(im_sig,((0,tpad),(0,0),(0,0)),mode='constant')
                
    # Uncompensated motion detection 
    # Peak detection logic
    rshift_mc = torch.roll(ctc_mc,shifts=1,dims=1)
    lshift_mc = torch.roll(ctc_mc,shifts=-1,dims=1)
    shift_mc = (rshift_mc + lshift_mc)/2
    mc_fdiff = ((ctc_mc-shift_mc)**2).sum(0)
    mc_fdiff = mc_fdiff/ctc_mc.sum(0)
    mc_fdiff[0] = mc_fdiff[1] = mc_fdiff[-1] = mc_fdiff[-2] = 0
    mc_indx = (mc_fdiff == mc_fdiff.max()).nonzero().squeeze(1).cpu()
    
    # Debugging uncompensated motion detection
    if True==False:
        # Frame-wise difference
        matplotlib.use('TkAgg')
        plt.figure()
        plt.title("Frame-wise Difference")
        plt.plot(mc_fdiff.detach().cpu(), linewidth=1)
        plt.ylim(bottom=0)
        plt.legend(loc="upper right")   
        plt.show()
        
        # Concentration-curves
        ctc_mc_vis = torch.tensor(im_sig, device=device).moveaxis(0,-1)
        ctc_mc_vis = ctc_mc_vis[seg==1]
        matplotlib.use('TkAgg')
        plt.figure()
        plt.title("Concentration Time Curves")
        plt.plot(ctc_mc_vis.detach().swapaxes(0,1).cpu(), linewidth=1)
        plt.ylim(bottom=0, top=0.5)
        plt.legend(loc="upper right")   
        plt.show()
        
        # Traversing time-slice
        from matplotlib.widgets import Slider
        matplotlib.use('TkAgg')                
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)                
        # Display the initial frame (time frame 0)
        current_frame = 0
        img_display = ax.imshow(im_sig[current_frame], cmap='viridis', vmin=0, vmax=im_sig.max())
        # Create a slider for selecting the time frame
        num_frames = im_sig.shape[0]
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        slider = Slider(ax_slider, 'Time Frame', 0, num_frames - 1, valinit=current_frame, valstep=1)                
        # Function to update the displayed frame when the slider is changed
        def update(val):
            current_frame = int(slider.val)
            img_display.set_data(im_sig[current_frame])
            fig.canvas.draw_idle()                    
        slider.on_changed(update)  # Attach the update function to the slider
        plt.show()
        
    # Compensating motion
    nk = 0
    skip_indx = []
    while skip_indx.__len__() <= 10:             
        # Corrupted frame
        im_sig_corrupt = torch.tensor(im_sig[mc_indx,...], device=device)                
        # Interpolated signal
        im_sig_iterp = torch.tensor(im_sig[mc_indx-1:mc_indx+1,...].mean(0), device=device)
        im_sig_target = im_sig_iterp.unsqueeze(0).unsqueeze(0)                
        # Source
        im_sig_src = im_sig_src = torch.tensor(im_sig_full[mc_indx], device=device)
        im_sig_patches = im_sig_src.unfold(0, 70, 1).unfold(1, 70, 1).clone()
        im_sig_comp = im_sig_src.unfold(0, 70, 1).unfold(1, 70, 1).clone()                
        # Similarity logic
        error = (((im_sig_patches - im_sig_target))**2).sum(dim=(-1,-2))/(im_sig_target**2).sum()
        error_indx = (error==error.min()).nonzero().squeeze()
        # Compensated image frame if the replacing the frame reduces error
        im_sig_comp = im_sig_comp[error_indx[0], error_indx[1],...]
        if (((im_sig_iterp-im_sig_comp)[seg==1]**2).sum()/(im_sig_iterp[seg==1]**2).sum()) > 1.0:
            im_sig[mc_indx] = im_sig[mc_indx-2:mc_indx+2,...].mean(0)
            im_sig_compensated = np.pad(im_sig,((0,tpad),(0,0),(0,0)),mode='constant')
            print('Wrong heart phase: time-frame ' + str(mc_indx.item()) + ' is replaced with interpolated value')
            skip_indx.append(mc_indx.item())
            # Incrementing peak count
            nk+=1
        elif (((im_sig_iterp-im_sig_comp)[seg==1]**2).sum()/(im_sig_iterp[seg==1]**2).sum()) < (((im_sig_iterp-im_sig_corrupt)[seg==1]**2).sum()/(im_sig_iterp[seg==1]**2).sum()):
            im_sig[mc_indx] = im_sig_comp.detach().cpu()
            im_sig_compensated = np.pad(im_sig,((0,tpad),(0,0),(0,0)),mode='constant')
            print('Residual translational motion: patch ' + str(error_indx.cpu().numpy()) + ' is selected at time-frame ' + str(mc_indx.item()))
            skip_indx.append(mc_indx.item())
            # Incrementing peak count
            nk+=1
        else:
            skip_indx.append(mc_indx.item())
        
        # Peak detection logic
        im_sig_full 
        ctc_mc = torch.tensor(im_sig, device=device).moveaxis(0,-1)
        ctc_mc = ctc_mc[seg!=-1]
        rshift_mc = torch.roll(ctc_mc,shifts=1,dims=1)
        lshift_mc = torch.roll(ctc_mc,shifts=-1,dims=1)
        shift_mc = (rshift_mc + lshift_mc)/2
        mc_fdiff = ((ctc_mc-shift_mc)**2).sum(0)
        mc_fdiff = mc_fdiff/ctc_mc.sum(0)
        mc_fdiff[0] = mc_fdiff[1] = mc_fdiff[-1] = mc_fdiff[-2] = 0
        mc_max = mc_fdiff.clone()
        mc_max[skip_indx] = 0
        mc_indx = (mc_fdiff == mc_max.max()).nonzero().squeeze(1).cpu()
        
        # Debugging compensated motion
        if True==False:        
            # Motion corrupted
            matplotlib.use('TkAgg')
            plt.figure()
            plt.imshow((im_sig_corrupt).detach().cpu())  
            plt.show()            
            # Interpolated signal
            matplotlib.use('TkAgg')
            plt.figure()
            plt.imshow((im_sig_iterp).detach().cpu())  
            plt.show()            
            # Compensated frame
            im_sig_comp_ = im_sig_comp
            matplotlib.use('TkAgg')
            plt.figure()
            plt.imshow((im_sig_comp_).detach().cpu())  
            plt.show()            
            # Error
            matplotlib.use('TkAgg')
            plt.figure()
            plt.imshow((error).detach().cpu())  
            plt.show()
    print(colored("Peaks removed: " + str(nk), 'red'))
    
    return im_sig_compensated

def aif_delay_correction(mbolus, aif, time, tpad, device='cuda'):
    
    # Estimating delay for the aif
    # Initialization
    S = 10
    osamp = 20
    neg_shift = 10
    time_lbfgs = torch.tensor(time, device=device)
    delay_init = 1
    wlen_lbfgs = time_lbfgs.shape[0]
    time_lbfgs_t0 = time_lbfgs[0]/S
    time_lbfgs = time_lbfgs/S
    time_lbfgs_osamp = interp_linear_1D(time_lbfgs.unsqueeze(0), size=osamp*time_lbfgs.shape[-1])[0]
    # time_lbfgs_osamp[(mbolus_osamp[0,:] >= 0.1).nonzero()[0]]-time_lbfgs_osamp[(aifPreCorrection_osamp[0,:] >= 0.1).nonzero()[0]]            
    
    # Segmenting curves 
    aifPreCorrection = expand_dim(torch.tensor(aif, device=device), f_dim_pad=1)
    mbolus = expand_dim(torch.tensor(mbolus[:,2], device=device), f_dim_pad=1) # sl
    
    # Compensating offset in the time curves
    oTp = 5
    mbolus = F.relu(mbolus-mbolus[..., :oTp].mean(-1, keepdim=True))
    Comp_C = aifPreCorrection.max()/mbolus.max()     
    mbolus = Comp_C * mbolus
    upper = mbolus.argmax()
    
    # Oversampling curves (Linear)
    aifPreCorrection_osamp = interp_linear_1D(aifPreCorrection, size=osamp*aifPreCorrection.shape[-1])
    mbolus_osamp = interp_linear_1D(mbolus, size=osamp*mbolus.shape[-1])
    
    # lbfgs optimizer initialization
    global prev_iter
    prev_iter = -1
    # eta_lbfgs = S_op * eta_init
    delay_lbfgs = 1/S * torch.tensor(delay_init, device=device)
    # eta_lbfgs.requires_grad = True    
    delay_lbfgs.requires_grad = True 
    # lbfgs = optim.LBFGS([eta_lbfgs], lr=1 , history_size=10, max_eval=500, max_iter=500, line_search_fn="strong_wolfe")
    lbfgs = optim.LBFGS([delay_lbfgs], lr=1 , history_size=10, max_eval=500, max_iter=500, line_search_fn="strong_wolfe")
        
    def closure():
        # Initializations
        global prev_iter
        global aif_est_db
        global shift_ir_db
        # Option to add functionality per nr LBFGS iterations
        nr = 10
        if lbfgs.n_iter % nr==0 and (lbfgs.n_iter-prev_iter)!=0:
            prev_iter = lbfgs.n_iter                
            # Add additional functionality here
            # print(delay_lbfgs)
            pass
        
        # Start optimization
        lbfgs.zero_grad()
        shift_ir = translate_ir_func(delay_lbfgs , time_lbfgs_osamp, time_lbfgs_t0, C=2000, neg_shift=neg_shift*osamp)
        # Segmenting fermi impulse response
        shift_ir = shift_ir.squeeze(0).squeeze(0)
        
        # Convolution            
        mbolus_est = convolve(aifPreCorrection_osamp, shift_ir, neg_shift=neg_shift*osamp)
        comp_factor = aifPreCorrection_osamp.max()/mbolus_est.detach().max()
        mbolus_est = comp_factor * mbolus_est
        
        # Loss function
        C_mse = torch.sum(mbolus_osamp**2)
        objective = torch.sum(((mbolus_osamp[:,:upper*osamp] - mbolus_est[:,:upper*osamp]))**2)/C_mse
        objective.backward(retain_graph=True)   
        aif_est_db = mbolus_est.clone()
        shift_ir_db = shift_ir.clone()
        return objective
    
    print(colored("Estimating aif delay for correction...", 'red'))
    lbfgs.step(closure)
    
    # Correcting AIF delay
    del_delay = (((aif_est_db[0,:] >= 0.1).nonzero()[0]-(aifPreCorrection_osamp[0,:] >= 0.1).nonzero()[0])/osamp)
    thres_shift = 0.8       
    shifts = int(torch.round(del_delay-thres_shift+0.5) )
    aifCorrected = torch.roll(aifPreCorrection[0,:], shifts=(shifts), dims=(0)).detach().cpu()
    aifCorrected = np.pad(aifCorrected,((0, tpad)),mode='constant')
    
    # Debugging corrected AIF delay
    if True==False: 
        matplotlib.use('TkAgg')
        plt.figure()
        plt.title("Concentration Time Curves")
        plt.plot(aifPreCorrection[0,:].detach().cpu(), label="aif", linewidth=1, color="black", linestyle="dashed")
        plt.plot(mbolus[0,:].detach().cpu(), label="main bolus", linewidth=1, color="blue")
        plt.plot(aifCorrected.detach().cpu(), label="aif corrected", linewidth=1, color="green")
        plt.legend(loc="upper right")   
        plt.show()        
        
        matplotlib.use('TkAgg')
        plt.figure()
        plt.title("Concentration Time Curves")
        plt.plot(aifPreCorrection_osamp[0,:].detach().cpu(), label="aif", linewidth=1, color="black", linestyle="dashed")
        plt.plot(mbolus_osamp[0,:].detach().cpu(), label="main bolus", linewidth=1, color="green")            
        plt.plot(aif_est_db[0,:].detach().cpu(), label="estimated main bolus", linewidth=1, color="blue")
        plt.plot(shift_ir_db.detach().cpu(), label="impulse response", linewidth=1, color="orange")
        plt.legend(loc="upper right")   
        plt.show()
    return aifCorrected



def save_lehnert_results(save_path, 
                         npat, 
                         flow, 
                         decay, 
                         delay, 
                         offset, 
                         flow_avg, 
                         decay_avg, 
                         delay_avg, 
                         offset_avg, 
                         flow_bayes, 
                         decay_bayes, 
                         delay_bayes, 
                         offset_bayes, 
                         flow_bayes_avg, 
                         decay_bayes_avg, 
                         delay_bayes_avg, 
                         offset_bayes_avg):
    # Generating pdf files for classically estimated perfusion maps
    # Stacking myocardial slices
    # Point estimates
    flow_list = []
    decay_list = []
    delay_list = []
    offset_list = []
    flow_avg_list = []
    decay_avg_list = []
    delay_avg_list = []
    offset_avg_list = []
    for n in range(0,npat):
        flow_list.append(flow[n])
        decay_list.append(decay[n])
        delay_list.append(delay[n])
        offset_list.append(offset[n])    
        flow_avg_list.append(flow_avg[n])
        decay_avg_list.append(decay_avg[n])
        delay_avg_list.append(delay_avg[n])
        offset_avg_list.append(offset_avg[n])
    flow = np.concatenate(flow_list, axis=2)
    decay = np.concatenate(decay_list, axis=2)
    delay = np.concatenate(delay_list, axis=2)
    offset = np.concatenate(offset_list, axis=2)
    flow_avg = np.concatenate(flow_avg_list, axis=0)
    decay_avg = np.concatenate(decay_avg_list, axis=0)
    delay_avg = np.concatenate(delay_avg_list, axis=0)
    offset_avg = np.concatenate(offset_avg_list, axis=0)    
    # Classical perfusion estimates
    # Bayesian
    flow_bayes_list = []
    decay_bayes_list = []
    delay_bayes_list = []
    offset_bayes_list = []
    flow_bayes_avg_list = []
    decay_bayes_avg_list = []
    delay_bayes_avg_list = []
    offset_bayes_avg_list = []
    for n in range(0,npat):
        flow_bayes_list.append(flow_bayes[n])
        decay_bayes_list.append(decay_bayes[n])
        delay_bayes_list.append(delay_bayes[n])
        offset_bayes_list.append(offset_bayes[n])    
        flow_bayes_avg_list.append(flow_bayes_avg[n])
        decay_bayes_avg_list.append(decay_bayes_avg[n])
        delay_bayes_avg_list.append(delay_bayes_avg[n])
        offset_bayes_avg_list.append(offset_bayes_avg[n])
    flow_bayes = np.concatenate(flow_bayes_list, axis=2)
    decay_bayes = np.concatenate(decay_bayes_list, axis=2)
    delay_bayes = np.concatenate(delay_bayes_list, axis=2)
    offset_bayes = np.concatenate(offset_bayes_list, axis=2)
    flow_bayes_avg = np.concatenate(flow_bayes_avg_list, axis=0)
    decay_bayes_avg = np.concatenate(decay_bayes_avg_list, axis=0)
    delay_bayes_avg = np.concatenate(delay_bayes_avg_list, axis=0)
    offset_bayes_avg = np.concatenate(offset_bayes_avg_list, axis=0)
    # Creating pdf
    img_List = []
    for i in tqdm(range(0,npat)): 
        
        figsize=(14, 8)
        plot_list = [60 * flow[..., i], delay[..., i], decay[..., i], offset[..., i]]
        title_list = ["F : "+str(np.round(flow_avg[i], 3)), "Tau : "+str(np.round(delay_avg[i], 3)), "k : "+str(np.round(decay_avg[i], 3)), "off : "+str(np.round(offset_avg[i], 3))]
        rscaleF = 1
        rscaleTau = 0.4
        rscalek = 0.1
        range_list = [(None,None), (None,None), (None,None), (None,None)]
        cmap_list = ['viridis', 'viridis', 'viridis', 'viridis']
        pmaps_subplot = get_subplot(4, plot_list, title_list, range_list, cmap_list, figsize=figsize, suptitle='Perfusion Maps')
        save_name = 'perfusion_map_' + str(i) + '.png'
        save_data_dir = Path.joinpath(save_path, "qPerf_folder")
        Path(save_data_dir).mkdir(parents=True, exist_ok=True)            
        pmaps_subplot.savefig(Path.joinpath(save_data_dir, save_name), dpi=500)
        img_dir = str(Path.joinpath(save_data_dir, save_name)) 
        imga = Image.open(img_dir)
        imga.load()
        img = Image.new("RGB", imga.size, (255, 255, 255))
        img.paste(imga, mask=imga.split()[3])  # Alpha channel made opaque

        # Crop
        imw, imh = img.size # width, height
        # img = img.crop((imw/8-50, imh/8-30, 3*imw/8-155, imh/2)) # left, top, right, bottom
        if i==0:
            img1=img
        else:
            img_List.append(img)
        i += 1    
    pdf_name = 'qPerf.pdf'
    pdf_dir = str(Path.joinpath(save_path, pdf_name))
    img1.save(pdf_dir, "PDF", resolution=100.0, save_all=True, append_images=img_List)

    # Creating pdf
    img_List = []
    for i in tqdm(range(0,npat)): 
        
        figsize=(14, 8)
        plot_list = [60 * flow_bayes[..., i], delay_bayes[..., i], decay_bayes[..., i], offset_bayes[..., i]]
        title_list = ["F bayes : "+str(np.round(flow_bayes_avg[i], 3)), "Tau bayes : "+str(np.round(delay_bayes_avg[i], 3)), "k bayes : "+str(np.round(decay_bayes_avg[i], 3)), "off bayes : "+str(np.round(offset_bayes_avg[i], 3))]
        rscaleF = 1
        rscaleTau = 0.4
        rscalek = 0.1
        range_list = [(None,None), (None,None), (None,None), (None,None)]
        cmap_list = ['viridis', 'viridis', 'viridis', 'viridis']
        pmaps_subplot = get_subplot(4, plot_list, title_list, range_list, cmap_list, figsize=figsize, suptitle='Perfusion Maps')
        save_name = 'perfusion_map_' + str(i) + '.png'
        save_data_dir = Path.joinpath(save_path, "qPerf_bayes_folder")
        Path(save_data_dir).mkdir(parents=True, exist_ok=True)            
        pmaps_subplot.savefig(Path.joinpath(save_data_dir, save_name), dpi=500)
        img_dir = str(Path.joinpath(save_data_dir, save_name)) 
        imga = Image.open(img_dir)
        imga.load()
        img = Image.new("RGB", imga.size, (255, 255, 255))
        img.paste(imga, mask=imga.split()[3])  # Alpha channel made opaque

        # Crop
        imw, imh = img.size # width, height
        # img = img.crop((imw/8-50, imh/8-30, 3*imw/8-155, imh/2)) # left, top, right, bottom
        if i==0:
            img1=img
        else:
            img_List.append(img)
        i += 1    
    pdf_name = 'qPerf_bayes.pdf'
    pdf_dir = str(Path.joinpath(save_path, pdf_name))
    img1.save(pdf_dir, "PDF", resolution=100.0, save_all=True, append_images=img_List)
    
    
