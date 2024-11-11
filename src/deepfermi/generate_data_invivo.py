import os
import sys
import pickle

import matplotlib.cm as cm
import numpy as np
from termcolor import colored
from tqdm import tqdm
import dataio

from helper import bbox_corners, get_segmentation, compensate_motion, aif_delay_correction, save_lehnert_results

sys.path.append('/data/brahma01/deepfermi/invivo/')
from fermi import *

sys.path.append('/data/brahma01/PtbPyTools/')
from visualisation.ahaBullseye.ahaHelperfunctions import PlottingSettings

def main() -> None:
    
    # Settings
    dataset_name = "invivo_perfusion_data"
    save_path = Path('/data/brahma01/deepfermi/invivo/data/')
    pdat = '/data/brahma01/Datasets/perfusion_kcl/'
    pid_list = [36] #[10, 36, 37, 38, 39, 42, 44, 45, 46, 47, 54, 56, 58, 61, 62, 63, 71, 75, 76, 83, 86, 101, 102, 107, 108, 109]
    crop_len = 70
    tlen = 100
    n_myo_sl = 3
    show_debuggCTC = False
    motionCompensation = False
    seg_edit = {"10":False, 
                "36":True, 
                "37":False, 
                "38":False, 
                "39":False,
                "42":False, 
                "44":False, 
                "45":False, 
                "46":False, 
                "47":False, 
                "54":False, 
                "56":False, 
                "58":False, 
                "61":False, 
                "62":False, 
                "63":False, 
                "71":False, 
                "75":False, 
                "76":False, 
                "83":False, 
                "86":False, 
                "101":False, 
                "102":False, 
                "107":False, 
                "108":False, 
                "109":False}
    useSavedConfig = {"10":True,
                    "36":True, 
                    "37":True, 
                    "38":True, 
                    "39":True, 
                    "42":True, 
                    "44":True, 
                    "45":True, 
                    "46":True, 
                    "47":True, 
                    "54":True, 
                    "56":True, 
                    "58":True, 
                    "61":True, 
                    "62":True, 
                    "63":True, 
                    "71":True, 
                    "75":True, 
                    "76":True, 
                    "83":True, 
                    "86":True, 
                    "101":True, 
                    "102":True, 
                    "107":True, 
                    "108":True, 
                    "109":True}
    aifDelayCorrection = {"10":True,
                        "36":True, 
                        "37":True, 
                        "38":True, 
                        "39":True, 
                        "42":True, 
                        "44":True, 
                        "45":True, 
                        "46":True, 
                        "47":True, 
                        "54":True, 
                        "56":True, 
                        "58":True, 
                        "61":True, 
                        "62":True, 
                        "63":True, 
                        "71":True, 
                        "75":True, 
                        "76":True, 
                        "83":True, 
                        "86":True, 
                        "101":True, 
                        "102":True, 
                        "107":True, 
                        "108":True, 
                        "109":True}
    # Segmentation
    cmap = cm.get_cmap('inferno')
    pbullseye_main = pdat + 'aha/'
    # Settings for bulls eye plot
    plottingSettings = PlottingSettings(cmap = cmap,
                                        vmin = 0,
                                        vmax = 80.0,
                                        show_segmentNumbers = True,
                                        show_std = False,
                                        closePlotAutomatically = False,
                                        show_debuggingImages = False,
                                        useEdgesToSetInnerPoints= False)
    
    # Constructing placeholders for classical fits
    npat = pid_list.__len__()
    # Perfusion data
    im_sig = np.zeros((npat, tlen, crop_len, crop_len, n_myo_sl))
    seg = np.zeros((npat, crop_len, crop_len, n_myo_sl))
    aif = np.zeros((npat, tlen, n_myo_sl))
    time = np.zeros((npat, tlen))
    wlen = np.zeros((npat, n_myo_sl))
    snr = np.zeros((npat, n_myo_sl))
    # Point estimates
    # 2D maps
    flow = np.zeros((npat, crop_len, crop_len, n_myo_sl))
    decay = np.zeros((npat, crop_len, crop_len, n_myo_sl))
    delay = np.zeros((npat, crop_len, crop_len, n_myo_sl))
    offset = np.zeros((npat, crop_len, crop_len, n_myo_sl))
    # Average values
    flow_avg = np.zeros((npat, n_myo_sl))
    decay_avg = np.zeros((npat, n_myo_sl))
    delay_avg = np.zeros((npat, n_myo_sl))
    offset_avg = np.zeros((npat, n_myo_sl))
    # Bayesian estimates
    # 2D maps
    flow_bayes = np.zeros((npat, crop_len, crop_len, n_myo_sl))
    decay_bayes = np.zeros((npat, crop_len, crop_len, n_myo_sl))
    delay_bayes = np.zeros((npat, crop_len, crop_len, n_myo_sl))
    offset_bayes = np.zeros((npat, crop_len, crop_len, n_myo_sl))
    # Average values
    flow_bayes_avg = np.zeros((npat, n_myo_sl))
    decay_bayes_avg = np.zeros((npat, n_myo_sl))
    delay_bayes_avg = np.zeros((npat, n_myo_sl))
    offset_bayes_avg = np.zeros((npat, n_myo_sl))
    
    for p, pid in tqdm(enumerate(pid_list)):
        # Path
        print(colored('Patient '+ str(pid), 'green'))
        fperf = dataio.get_files(pdat + 'rec', inc_strg=[str(pid) + '_STRESS_moco_m10_k01_perf.npz', ], vis=1)[0]
        pbullseye = pbullseye_main + fperf.replace('_m10_k01_perf.npz', '/')
        if not os.path.exists(pbullseye):
            os.makedirs(pbullseye)
        storedConfig = None
        if useSavedConfig[str(pid)]==True:        
            fileName = os.path.abspath(pbullseye + 'ahaBullseye_config')
            with open(fileName, 'rb') as pickle_file:
                storedConfig = pickle.load(pickle_file)
        # Load
        f = np.load(pdat + 'rec/' + fperf, allow_pickle=True)
        # Standardizing    
        im_sig_crop = np.zeros(( f['t'].shape[0], crop_len, crop_len, n_myo_sl))
        for sl in range(0, n_myo_sl):
            
            # determine bbox limit
            slice_mask = np.mean(f['sl_mask'], axis=2)
            xmin, xmax, ymin, ymax = bbox_corners(slice_mask, crop_len)
            tpad = int(tlen-f['im_sig'].shape[0])
                                    
            # Cropping and extracting maps computed according to Lehnert et.al 2018 and 2019 
            # Signal Intensity
            im_sig_crop[..., sl] = f['im_sig'][:, xmin:xmax,ymin:ymax,sl]
            # Point estimates
            flow[p,..., sl] = f['qPerf'][0, xmin:xmax,ymin:ymax,sl]
            decay[p,..., sl] = f['qPerf'][1, xmin:xmax,ymin:ymax,sl]
            delay[p,..., sl] = f['qPerf'][2, xmin:xmax,ymin:ymax,sl]
            offset[p,..., sl] = f['qPerf'][3, xmin:xmax,ymin:ymax,sl]
            # Bayesian estimates
            flow_bayes[p,..., sl] = f['qPerf_bay'].mean(0)[0, xmin:xmax,ymin:ymax,sl]
            decay_bayes[p,..., sl] = f['qPerf_bay'].mean(0)[1, xmin:xmax,ymin:ymax,sl]
            delay_bayes[p,..., sl] = f['qPerf_bay'].mean(0)[2, xmin:xmax,ymin:ymax,sl]
            offset_bayes[p,..., sl] = f['qPerf_bay'].mean(0)[3, xmin:xmax,ymin:ymax,sl]
        
        # Myocardial segmentation
        myo_seg = get_segmentation(im_sig_crop,
                                   f['aif'], 
                                   edit=seg_edit[str(pid)], 
                                   pbullseye=pbullseye, 
                                   plottingSettings=plottingSettings,  
                                   storedConfig=storedConfig, show_debuggCTC=show_debuggCTC)
        seg[p,...] = myo_seg        
        
        # Classical perfusion estimates
        h = lambda x : x[x>0]
        # Point estimates
        # Average values
        flow_avg[p, ...] = np.stack((h(flow[p,...,0][myo_seg[...,0]>0]).mean(), h(flow[p,...,1][myo_seg[...,1]>0]).mean(), h(flow[p,...,2][myo_seg[...,2]>0]).mean()))    
        decay_avg[p, ...] = np.stack((h(decay[p,...,0][myo_seg[...,0]>0]).mean(), h(decay[p,...,1][myo_seg[...,1]>0]).mean(), h(decay[p,...,2][myo_seg[...,2]>0]).mean()))
        delay_avg[p, ...] = np.stack((h(delay[p,...,0][myo_seg[...,0]>0]).mean(), h(delay[p,...,1][myo_seg[...,1]>0]).mean(), h(delay[p,...,2][myo_seg[...,2]>0]).mean()))
        offset_avg[p, ...] = np.stack((h(offset[p,...,0][myo_seg[...,0]>0]).mean(), h(offset[p,...,1][myo_seg[...,1]>0]).mean(), h(offset[p,...,2][myo_seg[...,2]>0]).mean()))
        # Bayesian estimates
        # Average values
        flow_bayes_avg[p, ...] = np.stack((h(flow_bayes[p,...,0][myo_seg[...,0]>0]).mean(), h(flow_bayes[p,...,1][myo_seg[...,1]>0]).mean(), h(flow_bayes[p,...,2][myo_seg[...,2]>0]).mean()))    
        decay_bayes_avg[p, ...] = np.stack((h(decay_bayes[p,...,0][myo_seg[...,0]>0]).mean(), h(decay_bayes[p,...,1][myo_seg[...,1]>0]).mean(), h(decay_bayes[p,...,2][myo_seg[...,2]>0]).mean()))
        delay_bayes_avg[p, ...] = np.stack((h(delay_bayes[p,...,0][myo_seg[...,0]>0]).mean(), h(delay_bayes[p,...,1][myo_seg[...,1]>0]).mean(), h(delay_bayes[p,...,2][myo_seg[...,2]>0]).mean()))
        offset_bayes_avg[p, ...] = np.stack((h(offset_bayes[p,...,0][myo_seg[...,0]>0]).mean(), h(offset_bayes[p,...,1][myo_seg[...,1]>0]).mean(), h(offset_bayes[p,...,2][myo_seg[...,2]>0]).mean()))
        
        # Time-points
        time[p, ...] = np.pad(f['t'][:, 0], (0, tpad), mode='constant')
        wlen[p, ...] = np.stack((np.load(pbullseye+'tcut_0.npy'),np.load(pbullseye+'tcut_1.npy'),np.load(pbullseye+'tcut_2.npy')))
        
        # Time-points
        snr[p, ...] = np.stack((np.load(pbullseye+'snr_0.npy'),np.load(pbullseye+'snr_1.npy'),np.load(pbullseye+'snr_2.npy')))
        
        
        for sl in range(0, n_myo_sl):
            # Initialization
            device = 'cuda'
            tpad = int(tlen-f['im_sig'].shape[0])            
            # Motion compensation
            if motionCompensation==True:
                im_sig[p, ..., sl] = compensate_motion(im_sig_crop[..., sl], 
                                                       myo_seg[..., sl], 
                                                       f['im_sig'][..., sl], 
                                                       tpad, 
                                                       device='cuda')
            else:
                im_sig[p, ..., sl] = np.pad(im_sig_crop[..., sl],((0,tpad),(0,0),(0,0)),mode='constant')
                    
        # AIF delay Correction
        for sl in range(0, n_myo_sl):
            if aifDelayCorrection[str(pid)]==True:
                mbolus = (np.stack((np.load(pbullseye+'mbolus_0.npy'),
                            np.load(pbullseye+'mbolus_1.npy'),
                            np.load(pbullseye+'mbolus_2.npy')), 1) * np.max(im_sig_crop.flatten()))/100
                aif[p, ..., sl] = aif_delay_correction(mbolus, f['aif'], f['t'][:, 0], tpad)
            else:
                aif[p, ..., sl] = np.pad(f['aif'],((0,tpad)),mode='constant')
    
    dataset_path = Path.joinpath(save_path, dataset_name)

    # Concentration time curves    
    ctc = np.expand_dims(seg, axis=1) * im_sig

    # SNR value
    snr_avg = snr.mean()

    # Initializing files for recording
    open(Path.joinpath(save_path, 'snr.txt'), 'w').close()       
    with open(Path.joinpath(save_path, 'snr.txt'),'a') as file:
        for pid in pid_list:
            file.write("%s %d %s" % ('Patient', pid, ':   '))  
            file.write('   '.join(map(str, np.round(snr[p],3))))
            file.write('\n')
        file.write('=====================================')
        file.write('\n')
        file.write("%s %.3f" % ('Average snr :       ', np.round(snr_avg,3)))
        
    # Setting up dictionary
    dic = {}
    for p, pid in enumerate(pid_list):
        dic[str(pid)] = {}
        dic[str(pid)]['im_sig'] = im_sig[p]
        dic[str(pid)]['seg'] = seg[p]
        dic[str(pid)]['ctc'] = ctc[p]
        dic[str(pid)]['aif'] = aif[p]
        dic[str(pid)]['time'] = time[p]
        dic[str(pid)]['wlen'] = wlen[p]
    # Saving file
    np.savez(dataset_path, **dic)
    
    # Save previous results
    save_lehnert_results(save_path,
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
                         offset_bayes_avg)    

if __name__ == "__main__":
    main()























# # Fermi impusle response
# def translate_ir_func(delay, t, t0=0, C=1000, neg_shift=0):
#     t = t-t0
#     t = torch.cat((-torch.flip(t, [0])[-neg_shift-1:-1], t))    
#     delayed_dirac = t-delay
#     C = C/torch.pi
#     translate_ir = np.sqrt(C/np.pi)*torch.exp(-C*(delayed_dirac)**2)    
#     translate_ir = translate_ir/translate_ir.max()
#     return translate_ir

# def fill_holes(myo_seg):
#     myo_seg_filled = myo_seg.copy()
#     TRight = np.array([[0,1,0],[1,1,0],[0,1,0]])
#     TLeft = np.array([[0,1,0],[0,1,1],[0,1,0]])
#     TTop = np.array([[0,1,0],[1,1,1],[0,0,0]])
#     TBottom = np.array([[0,0,0],[1,1,1],[0,1,0]])
#     for i in range(1, myo_seg.shape[0]-1):
#         for j in range(1, myo_seg.shape[1]-1):
#             for m in range(0, myo_seg.shape[-1]):                    
#                 P = np.array([[myo_seg[i-1,j+1,m],myo_seg[i,j+1,m],myo_seg[i+1,j+1,m]],
#                             [myo_seg[i-1,j,m],myo_seg[i,j,m],myo_seg[i+1,j,m]],
#                             [myo_seg[i-1,j-1,m],myo_seg[i,j-1,m],myo_seg[i+1,j-1,m]]])
#                 myo_seg_filled[i,j,m] = np.any([(TRight*P).sum()//3, (TLeft*P).sum()//3, (TTop*P).sum()//3, (TBottom*P).sum()//3]).astype('float64') if myo_seg[i,j,m]==0 else 1
#     return myo_seg_filled

# def svd_approx(xin, k=4, kstart=0):

#     # x has shape (mb,ch,nx,ny,nt)
#     mb,nx,ny,nt = xin.shape
#     # Convert tensors to matrices
#     x = xin.reshape(mb,1,nx*ny,nt)
#     u, sdiag, v_H = torch.linalg.svd(x)
#     sdiag[...,k:] = 0
#     sdiag[...,:kstart] = 0
#     offset=((nx*ny)-nt)
#     sbar = torch.diag_embed(sdiag, offset=offset)[...,offset:]
#     xbar = u @ sbar @ v_H
#     xout = xbar.reshape(mb,nx,ny,nt)
#     return xout

# pdat = '/data/brahma01/Datasets/perfusion_kcl/'
# # pat = [10, 36, 37, 38, 39, 42, 44, 45, 46, 47, 54, 56, 58, 61, 62, 63, 71, 75, 76, 83, 86, 101, 102, 107, 108, 109]
# pat =  [75]
# # random.shuffle(pat)
# # pat_train = [46, 75, 109, 62, 45, 101, 10, 63 , 42 ,47, 107, 61, 39, 83, 37, 36, 71]
# # pat_val = [108 , 38 , 76 , 102, 44]
# # pat_test = [58 , 54 , 86 , 56]
# pat_train = [56, 75, 109, 62, 45, 101, 10, 54 , 42 ,47, 107, 61, 39, 83, 37, 58, 71]
# pat_val = [46 , 38 , 76 , 102, 44]
# pat_test = [36 , 63, 86 , 108]
# show_debuggCTC = False
# seg_edit = {"10":True, 
#             "36":True, 
#             "37":True, 
#             "38":True, 
#             "39":True,
#             "42":True, 
#             "44":True, 
#             "45":True, 
#             "46":True, 
#             "47":True, 
#             "54":True, 
#             "56":True, 
#             "58":True, 
#             "61":True, 
#             "62":True, 
#             "63":True, 
#             "71":True, 
#             "75":True, 
#             "76":True, 
#             "83":True, 
#             "86":True, 
#             "101":True, 
#             "102":True, 
#             "107":True, 
#             "108":True, 
#             "109":True}
# useSavedConfig = {"10":True,
#                 "36":True, 
#                 "37":True, 
#                 "38":True, 
#                 "39":True, 
#                 "42":True, 
#                 "44":True, 
#                 "45":True, 
#                 "46":True, 
#                 "47":True, 
#                 "54":True, 
#                 "56":True, 
#                 "58":True, 
#                 "61":True, 
#                 "62":True, 
#                 "63":True, 
#                 "71":True, 
#                 "75":True, 
#                 "76":True, 
#                 "83":True, 
#                 "86":True, 
#                 "101":True, 
#                 "102":True, 
#                 "107":True, 
#                 "108":True, 
#                 "109":True}

# aifDelayCorrection = {"10":True,
#                 "36":True, 
#                 "37":True, 
#                 "38":True, 
#                 "39":True, 
#                 "42":True, 
#                 "44":True, 
#                 "45":True, 
#                 "46":True, 
#                 "47":True, 
#                 "54":True, 
#                 "56":True, 
#                 "58":True, 
#                 "61":True, 
#                 "62":True, 
#                 "63":True, 
#                 "71":True, 
#                 "75":True, 
#                 "76":True, 
#                 "83":True, 
#                 "86":True, 
#                 "101":True, 
#                 "102":True, 
#                 "107":True, 
#                 "108":True, 
#                 "109":True}

# # Initialization
# npat = pat.__len__()
# len = 70
# tlen = 100
# n_myo_sl = 3
# im_sig = np.zeros((npat, tlen, len, len, n_myo_sl))
# seg = np.zeros((npat, len, len, n_myo_sl))
# aif = np.zeros((npat, tlen, n_myo_sl))
# time = np.zeros((npat, tlen))
# wlen = np.zeros((npat, n_myo_sl))
# snr = np.zeros((npat, n_myo_sl))

# # Classical fit
# # Point estimates
# # 2D maps
# flow = np.zeros((npat, len, len, n_myo_sl))
# decay = np.zeros((npat, len, len, n_myo_sl))
# delay = np.zeros((npat, len, len, n_myo_sl))
# offset = np.zeros((npat, len, len, n_myo_sl))
# # Average values
# flow_avg = np.zeros((npat, n_myo_sl))
# decay_avg = np.zeros((npat, n_myo_sl))
# delay_avg = np.zeros((npat, n_myo_sl))
# offset_avg = np.zeros((npat, n_myo_sl))
# # Bayesian estimates
# # 2D maps
# flow_bayes = np.zeros((npat, len, len, n_myo_sl))
# decay_bayes = np.zeros((npat, len, len, n_myo_sl))
# delay_bayes = np.zeros((npat, len, len, n_myo_sl))
# offset_bayes = np.zeros((npat, len, len, n_myo_sl))
# # Average values
# flow_bayes_avg = np.zeros((npat, n_myo_sl))
# decay_bayes_avg = np.zeros((npat, n_myo_sl))
# delay_bayes_avg = np.zeros((npat, n_myo_sl))
# offset_bayes_avg = np.zeros((npat, n_myo_sl))

# # Segmentation
# cmap = cm.get_cmap('inferno')
# pbullseye_main = pdat + 'aha/'
# # Settings for bulls eye plot
# plottingSettings = PlottingSettings(cmap = cmap,
#                                     vmin = 0,
#                                     vmax = 80.0,
#                                     show_segmentNumbers = True,
#                                     show_std = False,
#                                     closePlotAutomatically = False,
#                                     show_debuggingImages = False,
#                                     useEdgesToSetInnerPoints= False)
    
# for p in tqdm(range(0, npat)):
#     # Path
#     print(colored('Patient '+ str(pat[p]), 'green'))
#     fperf = dataio.get_files(pdat + 'rec', inc_strg=[str(pat[p]) + '_STRESS_moco_m10_k01_perf.npz', ], vis=1)[0]
#     pbullseye = pbullseye_main + fperf.replace('_m10_k01_perf.npz', '/')
#     if not os.path.exists(pbullseye):
#         os.makedirs(pbullseye)
#     storedConfig = None
#     if useSavedConfig[str(pat[p])]==True:        
#         fileName = os.path.abspath(pbullseye + 'ahaBullseye_config')
#         with open(fileName, 'rb') as pickle_file:
#             storedConfig = pickle.load(pickle_file)
#     # Load
#     f = np.load(pdat + 'rec/' + fperf, allow_pickle=True)
#     # Standardizing    
#     im_sig_crop = np.zeros(( f['t'].shape[0], len, len, n_myo_sl))
#     for sl in range(0, n_myo_sl):
        
#         bullseye_mask = np.mean(f['sl_mask'], axis=2)
#         x, y, w, h = cv2.boundingRect(np.uint8(bullseye_mask))
#         # Retrieving indices                      
#         xmin, xmax  = y, y + h
#         ymin, ymax  = x, x + w
        
#         # Padding logic
#         if (len-(xmax-xmin))%2==0:
#             xpad_left, xpad_right = (int((len - (xmax-xmin))/2), int((len - (xmax-xmin))/2))
#         else:
#             if (len-(xmax-xmin))>0:
#                 xpad_left, xpad_right = (int((len - (xmax-xmin))/2) + 1, int((len - (xmax-xmin))/2))
#             else:
#                 xpad_left, xpad_right = (int((len - (xmax-xmin))/2), int((len - (xmax-xmin))/2) - 1)
#         if (len-(ymax-ymin))%2==0:
#             ypad_left, ypad_right = (int((len - (ymax-ymin))/2), int((len - (ymax-ymin))/2))
#         else:
#             if (len-(ymax-ymin))>0:
#                 ypad_left, ypad_right = (int((len - (ymax-ymin))/2) + 1, int((len - (ymax-ymin))/2))
#             else:
#                 ypad_left, ypad_right = (int((len - (ymax-ymin))/2), int((len - (ymax-ymin))/2) - 1)
#         tpad = int(tlen-f['im_sig'].shape[0])
#         # Padded indices             
#         xmin, xmax  = xmin - xpad_left, xmax + xpad_right
#         ymin, ymax  = ymin - ypad_left, ymax + ypad_right
                                
#         # Crop maps
#         im_sig_crop[..., sl] = f['im_sig'][:, xmin:xmax,ymin:ymax,sl]
        
#         # Classical perfusion estimates
#         # Point estimates
#         # 2D maps
#         flow[p,..., sl] = f['qPerf'][0, xmin:xmax,ymin:ymax,sl]
#         decay[p,..., sl] = f['qPerf'][1, xmin:xmax,ymin:ymax,sl]
#         delay[p,..., sl] = f['qPerf'][2, xmin:xmax,ymin:ymax,sl]
#         offset[p,..., sl] = f['qPerf'][3, xmin:xmax,ymin:ymax,sl]
#         # Bayesian estimates
#         # 2D maps
#         flow_bayes[p,..., sl] = f['qPerf_bay'].mean(0)[0, xmin:xmax,ymin:ymax,sl]
#         decay_bayes[p,..., sl] = f['qPerf_bay'].mean(0)[1, xmin:xmax,ymin:ymax,sl]
#         delay_bayes[p,..., sl] = f['qPerf_bay'].mean(0)[2, xmin:xmax,ymin:ymax,sl]
#         offset_bayes[p,..., sl] = f['qPerf_bay'].mean(0)[3, xmin:xmax,ymin:ymax,sl]
    
#     # Myocardial segmentation
#     cdat = np.mean(im_sig_crop, axis=0) # Time-averaged intensity
#     cdat = cdat * 100 / np.max(cdat.flatten())
#     cdat = np.split(cdat, 3, axis=2)
#     cdat.insert(0, np.zeros_like(cdat[0])) # Including apex
#     overlay = im_sig_crop * 100 / np.max(im_sig_crop.flatten())
#     odat = np.split(overlay, 3, axis=3)    
#     odat.insert(0, np.zeros_like(odat[0]))
#     aif_bulleye = f['aif'] * 100 / np.max(im_sig_crop.flatten())
#     if seg_edit[str(pat[p])]==True:
#         plt.close('all')
#         bullseye = Bullseye(cdat, odat, pbullseye, plottingSettings,  storedConfig, aif=aif_bulleye)
#         plt.close('all')        
#     aha_seg = np.load(pbullseye+'mask.npy')
#     myo_seg = aha_seg.copy()
#     myo_seg[myo_seg!=0] = 1
#     myo_seg = fill_holes(myo_seg)    
#     seg[p,...] = myo_seg
    
#     if show_debuggCTC:        
#         ctc_db = im_sig_crop[:, myo_seg==1]
#         plt.figure()
#         plt.title("Concentration Time Curves")
#         plt.plot(f['aif'], label="aif", linewidth=1, color="black", linestyle="dashed")
#         plt.plot(ctc_db, linewidth=1)
#         plt.legend(loc="upper right")   
#         plt.show()
    
#     # Classical perfusion estimates
#     h = lambda x : x[x>0]
#     # Point estimates
#     # Average values
#     flow_avg[p, ...] = np.stack((h(flow[p,...,0][myo_seg[...,0]>0]).mean(), h(flow[p,...,1][myo_seg[...,1]>0]).mean(), h(flow[p,...,2][myo_seg[...,2]>0]).mean()))    
#     decay_avg[p, ...] = np.stack((h(decay[p,...,0][myo_seg[...,0]>0]).mean(), h(decay[p,...,1][myo_seg[...,1]>0]).mean(), h(decay[p,...,2][myo_seg[...,2]>0]).mean()))
#     delay_avg[p, ...] = np.stack((h(delay[p,...,0][myo_seg[...,0]>0]).mean(), h(delay[p,...,1][myo_seg[...,1]>0]).mean(), h(delay[p,...,2][myo_seg[...,2]>0]).mean()))
#     offset_avg[p, ...] = np.stack((h(offset[p,...,0][myo_seg[...,0]>0]).mean(), h(offset[p,...,1][myo_seg[...,1]>0]).mean(), h(offset[p,...,2][myo_seg[...,2]>0]).mean()))
#     # Bayesian estimates
#     # Average values
#     flow_bayes_avg[p, ...] = np.stack((h(flow_bayes[p,...,0][myo_seg[...,0]>0]).mean(), h(flow_bayes[p,...,1][myo_seg[...,1]>0]).mean(), h(flow_bayes[p,...,2][myo_seg[...,2]>0]).mean()))    
#     decay_bayes_avg[p, ...] = np.stack((h(decay_bayes[p,...,0][myo_seg[...,0]>0]).mean(), h(decay_bayes[p,...,1][myo_seg[...,1]>0]).mean(), h(decay_bayes[p,...,2][myo_seg[...,2]>0]).mean()))
#     delay_bayes_avg[p, ...] = np.stack((h(delay_bayes[p,...,0][myo_seg[...,0]>0]).mean(), h(delay_bayes[p,...,1][myo_seg[...,1]>0]).mean(), h(delay_bayes[p,...,2][myo_seg[...,2]>0]).mean()))
#     offset_bayes_avg[p, ...] = np.stack((h(offset_bayes[p,...,0][myo_seg[...,0]>0]).mean(), h(offset_bayes[p,...,1][myo_seg[...,1]>0]).mean(), h(offset_bayes[p,...,2][myo_seg[...,2]>0]).mean()))
    
#     # Time-points
#     time[p, ...] = np.pad(f['t'][:, 0], (0, tpad), mode='constant')
#     wlen[p, ...] = np.stack((np.load(pbullseye+'tcut_0.npy'),np.load(pbullseye+'tcut_1.npy'),np.load(pbullseye+'tcut_2.npy')))
    
#     # Time-points
#     snr[p, ...] = np.stack((np.load(pbullseye+'snr_0.npy'),np.load(pbullseye+'snr_1.npy'),np.load(pbullseye+'snr_2.npy')))
    
#     # Motion compensation
#     motionCompensation = False # True
#     for sl in range(0, n_myo_sl):
#         # Initialization
#         device = 'cuda'
#         tpad = int(tlen-f['im_sig'].shape[0])
#         im_sig[p, ..., sl] = np.pad(im_sig_crop[..., sl],((0,tpad),(0,0),(0,0)),mode='constant')
#         # Uncompensated motion detection
#         if motionCompensation==True:
#             # Peak detection logic   
#             seg_mc = torch.tensor(myo_seg[..., sl], device=device)
#             ctc_mc = torch.tensor(im_sig_crop[..., sl], device=device).moveaxis(0,-1)
#             ctc_mc = ctc_mc[seg_mc!=-1]
#             rshift_mc = torch.roll(ctc_mc,shifts=1,dims=1)
#             lshift_mc = torch.roll(ctc_mc,shifts=-1,dims=1)
#             shift_mc = (rshift_mc + lshift_mc)/2
#             mc_fdiff = ((ctc_mc-shift_mc)**2).sum(0)
#             mc_fdiff = mc_fdiff/ctc_mc.sum(0)
#             mc_fdiff[0] = mc_fdiff[1] = mc_fdiff[-1] = mc_fdiff[-2] = 0
#             mc_indx = (mc_fdiff == mc_fdiff.max()).nonzero().squeeze(1).cpu()
            
#             # Debugging uncompensated motion detection
#             if True==False:
#                 # Frame-wise difference
#                 matplotlib.use('TkAgg')
#                 plt.figure()
#                 plt.title("Frame-wise Difference")
#                 plt.plot(mc_fdiff.detach().cpu(), linewidth=1)
#                 plt.ylim(bottom=0)
#                 plt.legend(loc="upper right")   
#                 plt.show()
                
#                 # Concentration-curves
#                 ctc_mc_vis = torch.tensor(im_sig_crop[..., sl], device=device).moveaxis(0,-1)
#                 ctc_mc_vis = ctc_mc_vis[seg_mc==1]
#                 matplotlib.use('TkAgg')
#                 plt.figure()
#                 plt.title("Concentration Time Curves")
#                 plt.plot(ctc_mc_vis.detach().swapaxes(0,1).cpu(), linewidth=1)
#                 plt.ylim(bottom=0, top=0.5)
#                 plt.legend(loc="upper right")   
#                 plt.show()
                
#                 # Traversing time-slice
#                 from matplotlib.widgets import Slider
#                 matplotlib.use('TkAgg')                
#                 fig, ax = plt.subplots()
#                 plt.subplots_adjust(bottom=0.25)                
#                 # Display the initial frame (time frame 0)
#                 current_frame = 0
#                 img_display = ax.imshow(im_sig_crop[current_frame,..., sl], cmap='viridis', vmin=0, vmax=im_sig_crop[..., sl].max())
#                 # Create a slider for selecting the time frame
#                 num_frames = im_sig_crop.shape[0]
#                 ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
#                 slider = Slider(ax_slider, 'Time Frame', 0, num_frames - 1, valinit=current_frame, valstep=1)                
#                 # Function to update the displayed frame when the slider is changed
#                 def update(val):
#                     current_frame = int(slider.val)
#                     img_display.set_data(im_sig_crop[current_frame,..., sl])
#                     fig.canvas.draw_idle()                    
#                 slider.on_changed(update)  # Attach the update function to the slider
#                 plt.show()
                
#             # Compensating motion
#             nk = 0
#             skip_indx = []
#             while skip_indx.__len__() <= 10:             
#                 # Corrupted frame
#                 im_sig_mc = torch.tensor(im_sig_crop[..., sl][mc_indx,...], device=device)                
#                 # Interpolated signal
#                 im_sig_iterp = torch.tensor(im_sig_crop[..., sl][mc_indx-1:mc_indx+1,...].mean(0), device=device)
#                 im_sig_target = im_sig_iterp.unsqueeze(0).unsqueeze(0)                
#                 # Source
#                 im_sig_src = torch.tensor(f['im_sig'][mc_indx, ..., sl], device=device)
#                 im_sig_patches = im_sig_src.unfold(0, 70, 1).unfold(1, 70, 1).clone()
#                 im_sig_comp = im_sig_src.unfold(0, 70, 1).unfold(1, 70, 1).clone()                
#                 # Similarity logic
#                 error = (((im_sig_patches - im_sig_target))**2).sum(dim=(-1,-2))/(im_sig_target**2).sum()
#                 error_indx = (error==error.min()).nonzero().squeeze()
#                 # Compensated image frame if the replacing the frame reduces error
#                 im_sig_comp = im_sig_comp[error_indx[0], error_indx[1],...]
#                 if (((im_sig_iterp-im_sig_comp)[seg_mc==1]**2).sum()/(im_sig_iterp[seg_mc==1]**2).sum()) > 1.0:
#                     im_sig_crop[mc_indx,..., sl] = im_sig_crop[..., sl][mc_indx-2:mc_indx+2,...].mean(0)
#                     im_sig[p, ..., sl] = np.pad(im_sig_crop[..., sl],((0,tpad),(0,0),(0,0)),mode='constant')
#                     print('Wrong heart phase: time-frame ' + str(mc_indx.item()) + ' is replaced with interpolated value')
#                     skip_indx.append(mc_indx.item())
#                     # Incrementing peak count
#                     nk+=1
#                 elif (((im_sig_iterp-im_sig_comp)[seg_mc==1]**2).sum()/(im_sig_iterp[seg_mc==1]**2).sum()) < (((im_sig_iterp-im_sig_mc)[seg_mc==1]**2).sum()/(im_sig_iterp[seg_mc==1]**2).sum()):
#                     im_sig_crop[mc_indx,..., sl] = im_sig_comp.detach().cpu()
#                     im_sig[p, ..., sl] = np.pad(im_sig_crop[..., sl],((0,tpad),(0,0),(0,0)),mode='constant')
#                     print('Residual translational motion: patch ' + str(error_indx.cpu().numpy()) + ' is selected at time-frame ' + str(mc_indx.item()))
#                     skip_indx.append(mc_indx.item())
#                     # Incrementing peak count
#                     nk+=1
#                 else:
#                     skip_indx.append(mc_indx.item())
                
#                 # Peak detection logic   
#                 seg_mc = torch.tensor(myo_seg[..., sl], device=device)
#                 ctc_mc = torch.tensor(im_sig_crop[..., sl], device=device).moveaxis(0,-1)
#                 ctc_mc = ctc_mc[seg_mc!=-1]
#                 rshift_mc = torch.roll(ctc_mc,shifts=1,dims=1)
#                 lshift_mc = torch.roll(ctc_mc,shifts=-1,dims=1)
#                 shift_mc = (rshift_mc + lshift_mc)/2
#                 mc_fdiff = ((ctc_mc-shift_mc)**2).sum(0)
#                 mc_fdiff = mc_fdiff/ctc_mc.sum(0)
#                 mc_fdiff[0] = mc_fdiff[1] = mc_fdiff[-1] = mc_fdiff[-2] = 0
#                 mc_max = mc_fdiff.clone()
#                 mc_max[skip_indx] = 0
#                 mc_indx = (mc_fdiff == mc_max.max()).nonzero().squeeze(1).cpu()
                
#                 # Debugging compensated motion
#                 if True==False:        
#                     # Motion corrupted
#                     matplotlib.use('TkAgg')
#                     plt.figure()
#                     plt.imshow((im_sig_mc).detach().cpu())  
#                     plt.show()            
#                     # Interpolated signal
#                     matplotlib.use('TkAgg')
#                     plt.figure()
#                     plt.imshow((im_sig_iterp).detach().cpu())  
#                     plt.show()            
#                     # Compensated frame
#                     im_sig_comp_ = im_sig_comp
#                     matplotlib.use('TkAgg')
#                     plt.figure()
#                     plt.imshow((im_sig_comp_).detach().cpu())  
#                     plt.show()            
#                     # Error
#                     matplotlib.use('TkAgg')
#                     plt.figure()
#                     plt.imshow((error).detach().cpu())  
#                     plt.show()
#             print(colored("Peaks removed: " + str(nk), 'red'))
                
#     # AIF delay Correction
#     for sl in range(0, n_myo_sl):
#         if aifDelayCorrection[str(pat[p])]==True:
#             # Load main bolus
#             mbolus = (np.stack((np.load(pbullseye+'mbolus_0.npy'),
#                                     np.load(pbullseye+'mbolus_1.npy'),
#                                     np.load(pbullseye+'mbolus_2.npy')), 1) * np.max(im_sig_crop.flatten()))/100
            
#             # Estimating delay for the aif
#             # Initialization
#             device = 'cuda'
#             S = 10
#             osamp = 20
#             neg_shift = 10
#             time_lbfgs = torch.tensor(f['t'][:, 0], device=device)
#             delay_init = 1
#             wlen_lbfgs = time_lbfgs.shape[0]
#             time_lbfgs_t0 = time_lbfgs[0]/S
#             time_lbfgs = time_lbfgs/S
#             time_lbfgs_osamp = interp_linear_1D(time_lbfgs.unsqueeze(0), size=osamp*time_lbfgs.shape[-1])[0]
#             # time_lbfgs_osamp[(mbolus_osamp[0,:] >= 0.1).nonzero()[0]]-time_lbfgs_osamp[(aifPreCorrection_osamp[0,:] >= 0.1).nonzero()[0]]            
            
#             # Segmenting curves 
#             aifPreCorrection = expand_dim(torch.tensor(f['aif'], device=device), f_dim_pad=1)
#             mbolus = expand_dim(torch.tensor(mbolus[:,2], device=device), f_dim_pad=1) # sl
            
#             # Compensating offset in the time curves
#             oTp = 5
#             mbolus = F.relu(mbolus-mbolus[..., :oTp].mean(-1, keepdim=True))
#             Comp_C = aifPreCorrection.max()/mbolus.max()     
#             mbolus = Comp_C * mbolus
#             upper = mbolus.argmax()
            
#             # Oversampling curves (Linear)
#             aifPreCorrection_osamp = interp_linear_1D(aifPreCorrection, size=osamp*aifPreCorrection.shape[-1])
#             mbolus_osamp = interp_linear_1D(mbolus, size=osamp*mbolus.shape[-1])
            
#             # lbfgs optimizer initialization
#             prev_iter = -1
#             # eta_lbfgs = S_op * eta_init
#             delay_lbfgs = 1/S * torch.tensor(delay_init, device=device)
#             # eta_lbfgs.requires_grad = True    
#             delay_lbfgs.requires_grad = True 
#             # lbfgs = optim.LBFGS([eta_lbfgs], lr=1 , history_size=10, max_eval=500, max_iter=500, line_search_fn="strong_wolfe")
#             lbfgs = optim.LBFGS([delay_lbfgs], lr=1 , history_size=10, max_eval=500, max_iter=500, line_search_fn="strong_wolfe")
                
#             def closure():
#                 # Initializations
#                 global prev_iter
#                 global aif_est_db
#                 global shift_ir_db
#                 # Option to add functionality per nr LBFGS iterations
#                 nr = 10
#                 if lbfgs.n_iter % nr==0 and (lbfgs.n_iter-prev_iter)!=0:
#                     prev_iter = lbfgs.n_iter                
#                     # Add additional functionality here
#                     # print(delay_lbfgs)
#                     pass
                
#                 # Start optimization
#                 lbfgs.zero_grad()
#                 shift_ir = translate_ir_func(delay_lbfgs , time_lbfgs_osamp, time_lbfgs_t0, C=2000, neg_shift=neg_shift*osamp)
#                 # Segmenting fermi impulse response
#                 shift_ir = shift_ir.squeeze(0).squeeze(0)
                
#                 # Convolution            
#                 mbolus_est = convolve(aifPreCorrection_osamp, shift_ir, neg_shift=neg_shift*osamp)
#                 comp_factor = aifPreCorrection_osamp.max()/mbolus_est.detach().max()
#                 mbolus_est = comp_factor * mbolus_est
                
#                 # Loss function
#                 C_mse = torch.sum(mbolus_osamp**2)
#                 objective = torch.sum(((mbolus_osamp[:,:upper*osamp] - mbolus_est[:,:upper*osamp]))**2)/C_mse
#                 objective.backward(retain_graph=True)   
#                 aif_est_db = mbolus_est.clone()
#                 shift_ir_db = shift_ir.clone()
#                 return objective
            
#             print(colored("Estimating aif delay for correction...", 'red'))
#             lbfgs.step(closure)
            
#             # Correcting AIF delay
#             del_delay = (((aif_est_db[0,:] >= 0.1).nonzero()[0]-(aifPreCorrection_osamp[0,:] >= 0.1).nonzero()[0])/osamp)
#             thres_shift = 0.8       
#             shifts = int(torch.round(del_delay-thres_shift+0.5) )
#             aifCorrected = torch.roll(aifPreCorrection[0,:], shifts=(shifts), dims=(0)).detach().cpu()
#             aif[p, ..., sl] = np.pad(aifCorrected,((0,tpad)),mode='constant')
            
#             # Debugging corrected AIF delay
#             if True==False: 
#                 matplotlib.use('TkAgg')
#                 plt.figure()
#                 plt.title("Concentration Time Curves")
#                 plt.plot(aifPreCorrection[0,:].detach().cpu(), label="aif", linewidth=1, color="black", linestyle="dashed")
#                 plt.plot(mbolus[0,:].detach().cpu(), label="main bolus", linewidth=1, color="blue")
#                 plt.plot(aifCorrected.detach().cpu(), label="aif corrected", linewidth=1, color="green")
#                 plt.legend(loc="upper right")   
#                 plt.show()        
                
#                 matplotlib.use('TkAgg')
#                 plt.figure()
#                 plt.title("Concentration Time Curves")
#                 plt.plot(aifPreCorrection_osamp[0,:].detach().cpu(), label="aif", linewidth=1, color="black", linestyle="dashed")
#                 plt.plot(mbolus_osamp[0,:].detach().cpu(), label="main bolus", linewidth=1, color="green")            
#                 plt.plot(aif_est_db[0,:].detach().cpu(), label="estimated main bolus", linewidth=1, color="blue")
#                 plt.plot(shift_ir_db.detach().cpu(), label="impulse response", linewidth=1, color="orange")
#                 plt.legend(loc="upper right")   
#                 plt.show()
#         else:
#             aif[p, ..., sl] = np.pad(f['aif'],((0,tpad)),mode='constant')

# ## Saving Dataset Dictionary
# dataset_name = "invivo_perfusion_data"
# save_path = Path('/data/brahma01/DCEPerfusion/InVivo/')
# dataset_path = Path.joinpath(save_path, dataset_name)

# # Concentration time curves    
# ctc = np.expand_dims(seg, axis=1) * im_sig

# # SNR value
# snr_avg = snr.mean()

# # Initializing files for recording
# open(Path.joinpath(save_path, 'snr.txt'), 'w').close()       
# with open(Path.joinpath(save_path, 'snr.txt'),'a') as file:
#     for p in range(pat.__len__()):
#         file.write("%s %d %s" % ('Patient', pat[p], ':   '))  
#         file.write('   '.join(map(str, np.round(snr[p],3))))
#         file.write('\n')
#     file.write('=====================================')
#     file.write('\n')
#     file.write("%s %.3f" % ('Average snr :       ', np.round(snr_avg,3)))

# # Constructing dataset
# # Train
# npat_train = pat_train.__len__()
# im_sig_train = np.zeros((npat_train, tlen, len, len, n_myo_sl))
# seg_train = np.zeros((npat_train, len, len, n_myo_sl))
# ctc_train = np.zeros((npat_train, tlen, len, len, n_myo_sl))
# aif_train = np.zeros((npat_train, tlen, n_myo_sl))
# time_train = np.zeros((npat_train, tlen))
# wlen_train = np.zeros((npat_train, n_myo_sl))
# for p in range(0, npat_train):
#     i = pat.index(pat_train[p])
#     im_sig_train[p,...]=im_sig[i,...]
#     seg_train[p,...]=seg[i,...]
#     aif_train[p,...]=aif[i]
#     time_train[p,...]=time[i]
#     wlen_train[p,...]=wlen[i,...]
#     ctc_train[p,...]=ctc[i,...]
    
# # Val
# npat_val = pat_val.__len__()
# im_sig_val = np.zeros((npat_val, tlen, len, len, n_myo_sl))
# seg_val = np.zeros((npat_val, len, len, n_myo_sl))
# ctc_val = np.zeros((npat_val, tlen, len, len, n_myo_sl))
# aif_val = np.zeros((npat_val, tlen, n_myo_sl))
# time_val = np.zeros((npat_val, tlen))
# wlen_val = np.zeros((npat_val, n_myo_sl))
# for p in range(0, npat_val):
#     i = pat.index(pat_val[p])
#     im_sig_val[p,...]=im_sig[i,...]
#     seg_val[p,...]=seg[i,...]
#     aif_val[p,...]=aif[i]
#     time_val[p,...]=time[i]
#     wlen_val[p,...]=wlen[i,...]
#     ctc_val[p,...]=ctc[i,...]
    
# # Test
# npat_test = pat_test.__len__()
# im_sig_test = np.zeros((npat_test, tlen, len, len, n_myo_sl))
# seg_test = np.zeros((npat_test, len, len, n_myo_sl))
# ctc_test = np.zeros((npat_test, tlen, len, len, n_myo_sl))
# aif_test = np.zeros((npat_test, tlen, n_myo_sl))
# time_test = np.zeros((npat_test, tlen))
# wlen_test = np.zeros((npat_test, n_myo_sl))
# for p in range(0, npat_test):
#     i = pat.index(pat_test[p])
#     im_sig_test[p,...]=im_sig[i,...]
#     seg_test[p,...]=seg[i,...]
#     aif_test[p,...]=aif[i]
#     time_test[p,...]=time[i]
#     wlen_test[p,...]=wlen[i,...]
#     ctc_test[p,...]=ctc[i,...]
    
# # Setting up dictionary
# dic = {}
# # Training Dataset
# dic['train'] = {}
# dic['train']['pat_train']=pat_train
# dic['train']['im_sig_train']=im_sig_train
# dic['train']['seg_train']=seg_train
# dic['train']['ctc_train']=ctc_train
# dic['train']['aif_train']=aif_train
# dic['train']['time_train']=time_train
# dic['train']['wlen_train']=wlen_train
# # Validation Dataset
# dic['val'] = {}
# dic['val']['pat_val']=pat_val
# dic['val']['im_sig_val']=im_sig_val
# dic['val']['seg_val']=seg_val
# dic['val']['ctc_val']=ctc_val
# dic['val']['aif_val']=aif_val
# dic['val']['time_val']=time_val
# dic['val']['wlen_val']=wlen_val
# # Test Dataset
# dic['test'] = {}
# dic['test']['pat_test']=pat_test
# dic['test']['im_sig_test']=im_sig_test
# dic['test']['seg_test']=seg_test
# dic['test']['ctc_test']=ctc_test
# dic['test']['aif_test']=aif_test
# dic['test']['time_test']=time_test
# dic['test']['wlen_test']=wlen_test
# # Saving file
# np.savez(dataset_path, **dic)

# # Generating pdf files for classically estimated perfusion maps
# # Stacking myocardial slices
# # Point estimates
# flow_list = []
# decay_list = []
# delay_list = []
# offset_list = []
# flow_avg_list = []
# decay_avg_list = []
# delay_avg_list = []
# offset_avg_list = []
# for n in range(0,npat):
#     flow_list.append(flow[n])
#     decay_list.append(decay[n])
#     delay_list.append(delay[n])
#     offset_list.append(offset[n])    
#     flow_avg_list.append(flow_avg[n])
#     decay_avg_list.append(decay_avg[n])
#     delay_avg_list.append(delay_avg[n])
#     offset_avg_list.append(offset_avg[n])
# flow = np.concatenate(flow_list, axis=2)
# decay = np.concatenate(decay_list, axis=2)
# delay = np.concatenate(delay_list, axis=2)
# offset = np.concatenate(offset_list, axis=2)
# flow_avg = np.concatenate(flow_avg_list, axis=0)
# decay_avg = np.concatenate(decay_avg_list, axis=0)
# delay_avg = np.concatenate(delay_avg_list, axis=0)
# offset_avg = np.concatenate(offset_avg_list, axis=0)    
# # Classical perfusion estimates
# # Bayesian
# flow_bayes_list = []
# decay_bayes_list = []
# delay_bayes_list = []
# offset_bayes_list = []
# flow_bayes_avg_list = []
# decay_bayes_avg_list = []
# delay_bayes_avg_list = []
# offset_bayes_avg_list = []
# for n in range(0,npat):
#     flow_bayes_list.append(flow_bayes[n])
#     decay_bayes_list.append(decay_bayes[n])
#     delay_bayes_list.append(delay_bayes[n])
#     offset_bayes_list.append(offset_bayes[n])    
#     flow_bayes_avg_list.append(flow_bayes_avg[n])
#     decay_bayes_avg_list.append(decay_bayes_avg[n])
#     delay_bayes_avg_list.append(delay_bayes_avg[n])
#     offset_bayes_avg_list.append(offset_bayes_avg[n])
# flow_bayes = np.concatenate(flow_bayes_list, axis=2)
# decay_bayes = np.concatenate(decay_bayes_list, axis=2)
# delay_bayes = np.concatenate(delay_bayes_list, axis=2)
# offset_bayes = np.concatenate(offset_bayes_list, axis=2)
# flow_bayes_avg = np.concatenate(flow_bayes_avg_list, axis=0)
# decay_bayes_avg = np.concatenate(decay_bayes_avg_list, axis=0)
# delay_bayes_avg = np.concatenate(delay_bayes_avg_list, axis=0)
# offset_bayes_avg = np.concatenate(offset_bayes_avg_list, axis=0)
# # Creating pdf
# img_List = []
# for i in tqdm(range(0,npat)): 
    
#     figsize=(14, 8)
#     plot_list = [60 * flow[..., i], delay[..., i], decay[..., i], offset[..., i]]
#     title_list = ["F : "+str(np.round(flow_avg[i], 3)), "Tau : "+str(np.round(delay_avg[i], 3)), "k : "+str(np.round(decay_avg[i], 3)), "off : "+str(np.round(offset_avg[i], 3))]
#     rscaleF = 1
#     rscaleTau = 0.4
#     rscalek = 0.1
#     range_list = [(None,None), (None,None), (None,None), (None,None)]
#     cmap_list = ['viridis', 'viridis', 'viridis', 'viridis']
#     pmaps_subplot = get_subplot(4, plot_list, title_list, range_list, cmap_list, figsize=figsize, suptitle='Perfusion Maps')
#     save_name = 'perfusion_map_' + str(i) + '.png'
#     save_data_dir = Path.joinpath(save_path, "qPerf_folder")
#     Path(save_data_dir).mkdir(parents=True, exist_ok=True)            
#     pmaps_subplot.savefig(Path.joinpath(save_data_dir, save_name), dpi=500)
#     img_dir = str(Path.joinpath(save_data_dir, save_name)) 
#     imga = Image.open(img_dir)
#     imga.load()
#     img = Image.new("RGB", imga.size, (255, 255, 255))
#     img.paste(imga, mask=imga.split()[3])  # Alpha channel made opaque

#     # Crop
#     imw, imh = img.size # width, height
#     # img = img.crop((imw/8-50, imh/8-30, 3*imw/8-155, imh/2)) # left, top, right, bottom
#     if i==0:
#         img1=img
#     else:
#         img_List.append(img)
#     i += 1    
# pdf_name = 'qPerf.pdf'
# pdf_dir = str(Path.joinpath(save_path, pdf_name))
# img1.save(pdf_dir, "PDF", resolution=100.0, save_all=True, append_images=img_List)

# # Creating pdf
# img_List = []
# for i in tqdm(range(0,npat)): 
    
#     figsize=(14, 8)
#     plot_list = [60 * flow_bayes[..., i], delay_bayes[..., i], decay_bayes[..., i], offset_bayes[..., i]]
#     title_list = ["F bayes : "+str(np.round(flow_bayes_avg[i], 3)), "Tau bayes : "+str(np.round(delay_bayes_avg[i], 3)), "k bayes : "+str(np.round(decay_bayes_avg[i], 3)), "off bayes : "+str(np.round(offset_bayes_avg[i], 3))]
#     rscaleF = 1
#     rscaleTau = 0.4
#     rscalek = 0.1
#     range_list = [(None,None), (None,None), (None,None), (None,None)]
#     cmap_list = ['viridis', 'viridis', 'viridis', 'viridis']
#     pmaps_subplot = get_subplot(4, plot_list, title_list, range_list, cmap_list, figsize=figsize, suptitle='Perfusion Maps')
#     save_name = 'perfusion_map_' + str(i) + '.png'
#     save_data_dir = Path.joinpath(save_path, "qPerf_bayes_folder")
#     Path(save_data_dir).mkdir(parents=True, exist_ok=True)            
#     pmaps_subplot.savefig(Path.joinpath(save_data_dir, save_name), dpi=500)
#     img_dir = str(Path.joinpath(save_data_dir, save_name)) 
#     imga = Image.open(img_dir)
#     imga.load()
#     img = Image.new("RGB", imga.size, (255, 255, 255))
#     img.paste(imga, mask=imga.split()[3])  # Alpha channel made opaque

#     # Crop
#     imw, imh = img.size # width, height
#     # img = img.crop((imw/8-50, imh/8-30, 3*imw/8-155, imh/2)) # left, top, right, bottom
#     if i==0:
#         img1=img
#     else:
#         img_List.append(img)
#     i += 1    
# pdf_name = 'qPerf_bayes.pdf'
# pdf_dir = str(Path.joinpath(save_path, pdf_name))
# img1.save(pdf_dir, "PDF", resolution=100.0, save_all=True, append_images=img_List)
 
# # # For debubugging purpose
# # plt.figure()
# # plt.imshow(aha_seg[...,2])

# # basic_imshow(bbox, '/data/brahma01/DCEPerfusion/InVivo/', fig_name='bbox')

# # t_steps = np.sort((np.roll(f['t'],1,axis=0) - f['t'])[:,0])

# # for k in f.keys():
# #     print(k)
    
# # # aif
# # basic_plot('/data/brahma01/DCEPerfusion/InVivo/', x= f['aif'] , fig_name='aif')

# # # im_sig_myo    
# # basic_plot('/data/brahma01/DCEPerfusion/InVivo/', x= f['im_sig_myo'][...,0] , fig_name='im_sig_myo')

# # # im_sig
# # basic_imshow(f['im_sig'][39,:,:,0], '/data/brahma01/DCEPerfusion/InVivo/', fig_name='im_sig')

# # # Plotting change in convolutions with change in delay
# # import imageio
# # import shutil
# # matplotlib.use('Agg')
# # ctc_est_frames = ctc[0,...,2] # im_sig[0,...,2] # 
# # # Start making GIF
# # gif_frame_data = ctc_est_frames
# # gif_dir = Path('/data/brahma01/DCEPerfusion/InVivo/gif_cache')
# # Path(gif_dir).mkdir(parents=True, exist_ok=True)
# # for i in range(gif_frame_data.shape[0]):
# #     fig = plt.figure()
# #     plt.imshow(gif_frame_data[i,:,:])
# #     gif_frame_name = str(i)
# #     gif_frame_name =  str.zfill(gif_frame_name, int(np.floor(np.log10(gif_frame_data.shape[0])+1)))
# #     fig.savefig(Path.joinpath(gif_dir, gif_frame_name), dpi=100)
# #     plt.close()
# # filenames = list(sorted(Path(gif_dir).glob('*.png*')))
# # images = []
# # for filename in filenames:
# #     images.append(imageio.imread(filename))
# # imageio.mimsave('/data/brahma01/DCEPerfusion/InVivo/GIF.gif', images)
# # shutil.rmtree(gif_dir)


# # p4 = f['qPerf'][4,...,0]
# # p5 = f['qPerf'][5,...,0]
# # p6 = f['qPerf'][6,...,0]
# # p7 = f['qPerf'][7,...,0]

# # basic_imshow(p0, '/data/brahma01/DCEPerfusion/InVivo/Experiments/Debug/', figsize=(10,8), fig_name='p0', dpi=300)
# # basic_imshow(p1, '/data/brahma01/DCEPerfusion/InVivo/Experiments/Debug/', figsize=(10,8), fig_name='p1', dpi=300)
# # basic_imshow(p2, '/data/brahma01/DCEPerfusion/InVivo/Experiments/Debug/', figsize=(10,8), fig_name='p2', dpi=300)
# # basic_imshow(p3, '/data/brahma01/DCEPerfusion/InVivo/Experiments/Debug/', figsize=(10,8), fig_name='p3', dpi=300)

# # basic_imshow(p4, '/data/brahma01/DCEPerfusion/InVivo/Experiments/Debug/', figsize=(10,8), fig_name='p4', dpi=300)
# # basic_imshow(p5, '/data/brahma01/DCEPerfusion/InVivo/Experiments/Debug/', figsize=(10,8), fig_name='p5', dpi=300)
# # basic_imshow(p6, '/data/brahma01/DCEPerfusion/InVivo/Experiments/Debug/', figsize=(10,8), fig_name='p6', dpi=300)
# # basic_imshow(p7, '/data/brahma01/DCEPerfusion/InVivo/Experiments/Debug/', figsize=(10,8), fig_name='p7', dpi=300)  

# # basic_imshow(pb0, '/data/brahma01/DCEPerfusion/InVivo/Experiments/Debug/', figsize=(10,8), fig_name='pb0', dpi=300)
# # basic_imshow(pb1, '/data/brahma01/DCEPerfusion/InVivo/Experiments/Debug/', figsize=(10,8), fig_name='pb1', dpi=300)
# # basic_imshow(pb2, '/data/brahma01/DCEPerfusion/InVivo/Experiments/Debug/', figsize=(10,8), fig_name='pb2', dpi=300)
# # basic_imshow(pb3, '/data/brahma01/DCEPerfusion/InVivo/Experiments/Debug/', figsize=(10,8), fig_name='pb3', dpi=300)