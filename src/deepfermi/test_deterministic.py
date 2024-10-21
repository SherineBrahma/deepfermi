from load import *
from tqdm import tqdm
from network.Unet import Unet
from network.DeepFermi_net import DeepFermi
# from network.B_DeepFermi_net import DeepFermi
from unet_fermi_manager import UNet_FermiManager
import torch
from utils import *
from network.translate_iDS import *
import itertools
from save import *
from unet_report import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path
from read_config import *
import argparse
import os
from PIL import Image
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
import cv2
from scipy import interpolate
from perf_bullseye import PerfusionBullseye
import sys
sys.path.append('/data/brahma01/PtbPyTools/')
from visualisation.ahaBullseye.ahaHelperfunctions import segmentLabel
from visualisation.ahaBullseye.ahaHelperfunctions import PlottingSettings

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Read Config File %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parser = argparse.ArgumentParser()
parser.add_argument('--config_path',default='/data/brahma01/DCEPerfusion/InVivo/TestInVivo.yaml',type=str)
parser.add_argument('--project_name', default=None,type=str)
parser.add_argument('--test_SNR_ctc', default=None,type=float)
parser.add_argument('--read_project_name', default=None,type=str)
args = parser.parse_args()
config = ReadConfig(args.config_path)
mode = config.mode
system = config.system
assert mode=="testing", "Only mode allowed during testing is 'testing'"
assert system=="deterministic", "Only test for deterministic systems are included here"
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Loading Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print('Loading npz dataset as pytorch tensors...')
dsplit = {'train':config.n_train, 'val':config.n_val, 'test':config.n_test}
dataset_path = config.dataset_path
data_dic = load_npz(dataset_path, dsplit)

# Converting numpy tensors to torch tensors
for k in data_dic.keys():
    if isinstance(data_dic[k], list):
        data_dic[k] = [torch.tensor(val).clone().detach() for val in data_dic[k]]
    else:
        data_dic[k] = torch.tensor(data_dic[k], dtype=torch.float).clone().detach()
                    
# Stacking myocardial slices    
for im_sig, aif, seg, ctc, wlen in [('im_sig_train', 'aif_train', 'seg_train', 'ctc_train', 'wlen_train'),
                               ('im_sig_val', 'aif_val', 'seg_val', 'ctc_val', 'wlen_val'),
                               ('im_sig_test', 'aif_test', 'seg_test', 'ctc_test', 'wlen_test')]:
    im_sig_list = []
    aif_list = []
    seg_list = []
    ctc_list = []
    wlen_list = []  
    for i in range(0,data_dic[im_sig].shape[0]):
        im_sig_list.append(data_dic[im_sig][i])
        aif_list.append(data_dic[aif][i])
        seg_list.append(data_dic[seg][i])
        ctc_list.append(data_dic[ctc][i])
        wlen_list.append(data_dic[wlen][i])
    data_dic[im_sig] = torch.cat(im_sig_list, dim=3).permute((3,1,2,0))
    data_dic[aif] = torch.cat(aif_list, dim=1).permute((1,0))
    data_dic[seg] = torch.cat(seg_list, dim=2).permute((2,0,1))
    data_dic[ctc] = torch.cat(ctc_list, dim=3).permute((3,1,2,0))
    data_dic[wlen] = torch.cat(wlen_list).to(int)
    
# Repeating common attributes
for pat, time in [('pat_train', 'time_train'), ('pat_val', 'time_val'), ('pat_test', 'time_test')]:
    data_dic[pat] = list(itertools.chain.from_iterable(zip(data_dic[pat], data_dic[pat], data_dic[pat])))
    data_dic[time]= data_dic[time].repeat_interleave(3, dim=0)
    
# Adding noise to data
test_SNR_ctc  = config.test_SNR_ctc = config.test_SNR_ctc if args.test_SNR_ctc == None else args.test_SNR_ctc
inject_perturbation = config.inject_perturbation
inject_noise = config.inject_noise
inject_motion = True
if inject_perturbation == True:
    for aif, im_sig, ctc, seg, wlen  in [('aif_train', 'im_sig_train', 'ctc_train', 'seg_train', 'wlen_train'), 
                                   ('aif_val', 'im_sig_val', 'ctc_val', 'seg_val', 'wlen_val'), 
                                   ('aif_test', 'im_sig_test', 'ctc_test', 'seg_test', 'wlen_test') ]:
        
        # Extracting data
        aif_data = data_dic[aif]
        im_sig_data = data_dic[im_sig]
        ctc_data = data_dic[ctc]
        seg_data = data_dic[seg]
        wlen_data = data_dic[wlen]
        
        # Creating myocardial segmentation
        myo_seg = torch.zeros(seg_data.shape)
        myo_seg[seg_data==1]=1
        myo_seg[seg_data==71]=1
        
        if inject_noise == True:
            # Calculating noise power
            nb, nx, ny, _ = ctc_data.shape
            oTp = 5
            signal = ctc_data-ctc_data[...,0:oTp].mean(-1, keepdim=True)
            PN = torch.zeros((nb))
            for i in range(signal.shape[0]):
                signal[i][...,wlen_data[i]:] = 0
                PN[i] = signal[i][seg_data[i]==1][...,0:oTp].var()
            nx = ny = 120
            PS = (signal**2).sum(axis=(-1))/expand_dim(wlen_data,b_dim_pad=2)
            SNR = ((PS.sum(axis=(1,2))/(nx * ny))/PN).mean()
            print('Mean SNR before adding noise: ' + str(SNR.item()))
            
            # UPDATE: New SNR value
            snr_new = 10*torch.log10(((PS.sum(axis=(1,2))/myo_seg.sum(axis=(1,2)))/PN))
            print(snr_new.mean())
            
            PN_add = torch.max(((PS.sum(axis=(1,2))/(nx * ny))/test_SNR_ctc)-PN,PN)
            if PN[((PS.sum(axis=(1,2))/(nx * ny))/test_SNR_ctc)-PN<0].numel()!= 0:
                print(colored("Cannot add negative noise as the SNR of " + str(PN[((PS.sum(axis=(1,2))/(nx * ny))/test_SNR_ctc)-PN<0].numel()) 
                            + "/" + str(PN.numel()) + " signal vectors are already lower than the entered '" + str(test_SNR_ctc) 
                            + "'. Leaving them unaffected.", 'yellow'))            
            mean_normal = 0 * torch.ones([2, *signal.shape])
            std_normal = torch.sqrt(expand_dim(PN_add, f_dim_pad=1, b_dim_pad=3) * torch.ones([2, *signal.shape]))
            noise_normal = torch.normal(mean_normal, std_normal)       
            noise_rician = (noise_normal**2).sum(axis=0).sqrt()
            
            # Adding noise to ctc
            data_dic[ctc] = ctc_data + (myo_seg.unsqueeze(-1)*noise_rician)
            
            # Adding noise to aif
            data_dic[aif] = aif_data + noise_rician.mean((1,2))
            
            # Adding noise to im_sig
            data_dic[im_sig] = im_sig_data + noise_rician
            
            # AIF SNR
            PS = (aif_data**2).sum(axis=-1)/wlen_data
            print((PS.sum()/(nb * nx * ny))/PN)
        
        if inject_motion == True:
            # Calculating noise power
            nb, nx, ny, _ = ctc_data.shape
        
        # # 108 -> 210:211;
        # pixel = np.arange(570,580)
        # ctc_seg = data_dic[ctc][myo_seg==1]
        # # aif_seg = data_dic[aif]
        # matplotlib.use('TkAgg')
        # plt.figure()
        # plt.title("Concentration Time Curves")
        # plt.plot(ctc_seg[pixel,:].swapaxes(0,1).detach().cpu(), linewidth=1)
        # # plt.plot(aif_seg[pixel,:].mean(0).detach().cpu(), color='black', linestyle='dashed', linewidth=1, label='aif')
        # #plt.ylim(bottom=0, top=0.4)
        # plt.legend(loc="upper right")   
        # plt.show()
    
# Loading mask for outlying time-points
filter_outliers = config.filter_outliers
dic_mask_od = np.load('/data/brahma01/DCEPerfusion/InVivo/mask_od.npz', allow_pickle=True)
for pat, im_sig, ctc, split, mask_od, wlen in [('pat_train', 'im_sig_train', 'ctc_train', 'train', 'mask_od_train', 'wlen_train'), ('pat_val', 'im_sig_val', 'ctc_val', 'val', 'mask_od_val', 'wlen_val'), ('pat_test', 'im_sig_test', 'ctc_test', 'test', 'mask_od_test', 'wlen_test')]:    
    mask_od_list = []
    for i in range(0,data_dic[pat].__len__()):        
        if data_dic[pat][i] == dic_mask_od[pat][i]:
            mask_od_list.append(torch.tensor(dic_mask_od[split][i]))
    if filter_outliers==True:
        data_dic[mask_od] = torch.stack(mask_od_list, dim=0)
        
        # Exclude outliers and interpolate training-data
        im_sig_train = outlier_fill(data_dic[im_sig], data_dic[mask_od], data_dic[wlen])
        ctc_train = outlier_fill(data_dic[ctc], data_dic[mask_od], data_dic[wlen])
        
        
    else:
        data_dic[mask_od] = torch.ones(torch.stack(mask_od_list, dim=0).shape)
# Changing the background for ctc 
for aif, ctc, seg, time, wlen  in zip(['aif_train', 'aif_val', 'aif_test'], ['ctc_train', 'ctc_val', 'ctc_test'], ['seg_train', 'seg_val', 'seg_test'], ['time_train', 'time_val', 'time_test'], ['wlen_train', 'wlen_val', 'wlen_test']):
    
    # Extracting aif, ctc and segmentation
    aif_data = data_dic[aif]
    ctc_data = data_dic[ctc]
    seg_data = data_dic[seg]
    time_data = data_dic[time]
    wlen_data = data_dic[wlen]
    
    # General initializing
    S = 10
    S_op = expand_dim(torch.tensor([1,1/S,S], device=config.device), f_dim_pad=1, b_dim_pad=2)
    
    # Initializing background perfusion parameters
    flow_bkg = config.eta_bkg_ref[0]
    delay_bkg = config.eta_bkg_ref[1]
    decay_bkg = config.eta_bkg_ref[2]
    eta_bkg = S_op * expand_dim(torch.tensor([flow_bkg, delay_bkg, decay_bkg], device=config.device), f_dim_pad=1, b_dim_pad=2)
    
    for i in range(ctc_data.shape[0]):
    
        # Extracting slice aif, ctc and segmentation
        wlen_bkg = wlen_data[i:i+1]
        aif_bkg = torch.tensor(aif_data[i:i+1, ...,0:wlen_bkg].unsqueeze(1).unsqueeze(1), device=config.device)
        ctc_full = ctc_data[i:i+1, ...,0:wlen_bkg]
        seg_bkg = seg_data[i:i+1]        
        time_bkg = torch.tensor(time_data[i:i+1, 0:wlen_bkg]/S, device=config.device)        
        
        # Compensating offset in the time curves
        oTp = 5
        # aif_bkg = F.relu(aif_bkg-aif_bkg[...,0:oTp].mean(-1, keepdim=True))
        aif_bkg = aif_bkg-aif_bkg[...,0:oTp].mean(-1, keepdim=True)
        
        # Interpolating vectors
        aif_bkg = interp_linear_1D(aif_bkg, size=config.test_osamp*aif_bkg.shape[-1])
        time_bkg = interp_linear_1D(time_bkg.unsqueeze(0), size=config.test_osamp*time_bkg.shape[-1])[0]
        
        # Generating concentration curves
        fermi_ir = fermi_ir_func(eta_bkg, time_bkg.squeeze())    
        ctc_bkg = convolve(aif_bkg, fermi_ir)[...,::config.test_osamp]/config.test_osamp
        
        # Assigning background concentration curves
        ctc_full[seg_bkg==0] = ctc_bkg.squeeze().cpu()
        ctc_data[i:i+1, ...,0:wlen_bkg] = ctc_full
    data_dic[ctc] = ctc_data
      
print('Loading complete...')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Testing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

project_name = config.project_name if args.project_name == None else args.project_name
read_project_name = config.read_project_name if args.read_project_name == None else args.read_project_name
save_path = Path.joinpath(Path(config.save_path), project_name)
read_path = Path.joinpath(Path(config.read_dir), read_project_name)

# General configurations
test_slice_indx = config.test_slice_indx
test_osamp = config.test_osamp
test_nsamps = config.test_nsamps
device = config.device

# Testing dataset split
testing_split = config.testing_split
test_data_dic = {}
slice = np.arange(data_dic['im_sig_' + testing_split].shape[0]) # np.arange(8,9) # 
test_data_dic['pat'] = torch.tensor(data_dic['pat_' + testing_split])[slice]
test_data_dic['im_sig'] = data_dic['im_sig_' + testing_split][slice]
test_data_dic['seg'] = data_dic['seg_' + testing_split][slice]
test_data_dic['ctc'] = data_dic['ctc_' + testing_split][slice]
test_data_dic['aif'] = data_dic['aif_' + testing_split][slice]
test_data_dic['time'] = data_dic['time_' + testing_split][slice]
test_data_dic['wlen'] = data_dic['wlen_' + testing_split][slice]
test_data_dic['mask_od'] = data_dic['mask_od_' + testing_split][slice]

# # Dilate segmentation
# dilation = True
# if dilation == True:
#     k_dilate = torch.tensor([ [1, 1, 1],
#                         [1, 1, 1],
#                         [1, 1, 1] ], device=test_data_dic['seg'].device, dtype=test_data_dic['seg'].dtype)
#     test_data_dic['seg'] = torch.clamp(torch.nn.functional.conv2d(test_data_dic['seg'].unsqueeze(1), k_dilate.unsqueeze(0).unsqueeze(0), padding=(1, 1)), 0, 1).squeeze(1)
#     test_data_dic['ctc'] = test_data_dic['seg'].unsqueeze(-1) * test_data_dic['im_sig']

# Flags
test_network = config.test_network
gen_perfusion_maps = config.gen_perfusion_maps
plot_conc_curves = config.plot_conc_curves
estimation_eval = config.estimation_eval
bullseye_eval = config.bullseye_eval
train_curves = config.train_curves
outlier_tolerance_eval = config.outlier_tolerance_eval

# Fermi specific 
fermi_params = {}        
fermi_params['osamp'] = test_osamp

# Saving functions
save = Save(save_path)

# Reporting functions
report = ReportUtils(save_path, save, fermi_params, mode='testing', device=device)

# Obtaining reports    
if all([test_network])==True:
        
    for test_unet, test_train_params in zip(config.test_unet_list, config.test_train_params_list):

        # Loading settings that were set while training
        test_train_params_path = Path.joinpath(read_path, test_train_params+ '.txt')
        with open(test_train_params_path, 'r') as file:
            train_params = {'dummy': 'dummy'}
            for line in file:
                line = 'train_params' + "['" + line.replace(" : ", "']=")
                line = line.replace(" \n", "")
                try:
                    exec(line)
                except:
                    try:
                        line = line.replace("]=", "]='")
                        line = line + "'"
                        exec(line)
                    except:
                        break                      
                    
        # Initializing training parameters
        img_dim = train_params['img_dim']
        nu = train_params['nu']
        max_iter_lbfgs = train_params['max_iter_lbfgs']
        max_eval_lbfgs = train_params['max_eval_lbfgs']
        
        # Initializing networks
        cnn = Unet(dim=3, ncin=2, ncout=12, nstage=3, nconv_stage=2, nfilters=16, res_connect=False)
        unet = DeepFermi(cnn, osamp=config.test_osamp, nu=nu, max_iter_lbfgs=max_iter_lbfgs, max_eval_lbfgs=max_eval_lbfgs, mode=mode, learn_lambda=False).cuda()
        unet_state_dic = torch.load(Path.joinpath(read_path, test_unet))
        unet.load_state_dict(unet_state_dic, strict=True)
        
        # Initializing files for recording         
        open(Path.joinpath(save_path, 'Eval_Metrics.txt'), 'w').close()

        #  Testing network
        if test_network == True:
            # Lbfgs results
            print('Testing lbfgs..')                      
            eta_lbfgs, test_lbfgs_stat = report.test_lbfgs(test_data_dic)
            eta_lbfgs = eta_lbfgs.cpu()
            # save(eta_lbfgs, save_obj_type='numpy', save_obj_name='eta_lbfgs')
            with open(Path.joinpath(save_path, 'Eval_Metrics.txt'),'a') as file:
                file.write("%s %f\n" % ('CTC NRMSE LBFGS: ', test_lbfgs_stat['ctc_NRMSE']))
                file.write("%s %f\n" % ('CTC NRMSE LBFGS Std per slice: ', test_lbfgs_stat['ctc_NRMSE_slice_std']))
                file.write("%s %f\n" % ('Time taken LBFGS: ', test_lbfgs_stat['time_taken']))
                file.write("%s %f\n" % ('Time taken LBFGS Std per slice: ', test_lbfgs_stat['time_taken_slice_std']))
                
            # Network results
            print('Testing network..')            
            eta_net, test_net_stat = report.test_network(test_data_dic, unet)
            seg = test_data_dic['seg']
            eta_net[seg.unsqueeze(1).repeat(1,3,1,1)==0] = 0
            eta_net = eta_net.cpu()
            # save(eta_net, save_obj_type='numpy', save_obj_name='eta_net')
            with open(Path.joinpath(save_path, 'Eval_Metrics.txt'),'a') as file:
                file.write("%s %f\n" % ('CTC NRMSE Net: ', test_net_stat['ctc_NRMSE']))
                file.write("%s %f\n" % ('CTC NRMSE Net Std per slice: ', test_net_stat['ctc_NRMSE_slice_std']))
                file.write("%s %f\n" % ('Time taken Net: ', test_net_stat['time_taken']))
                file.write("%s %f\n" % ('Time taken Net Std per slice: ', test_net_stat['time_taken_slice_std']))
                
            # Saving estimated map
            test_data_dic['eta_lbfgs'] = eta_lbfgs
            test_data_dic['eta_net'] = eta_net
            # Saving file
            np.savez(Path.joinpath(save_path, 'test_data_dic'), **test_data_dic)
            
            # Wilcoxon statistical test
            # ctc
            dctc = (test_lbfgs_stat['ctc_NRMSE_slice'] - test_net_stat['ctc_NRMSE_slice'])
            dres_ctc = wilcoxon(dctc)
            # time
            dtime = (np.subtract(test_lbfgs_stat['time_taken_slice'],test_net_stat['time_taken_slice']))      
            dres_time = wilcoxon(dtime)      
            # Writing statistical test result            
            with open(Path.joinpath(save_path, 'Eval_Metrics.txt'),'a') as file:
                file.write("%s %f\n" % ('CTC wicoxon pvalue: ', dres_ctc.pvalue))
                file.write("%s %f\n" % ('Time Taken wicoxon pvalue: ', dres_time.pvalue))
                
            # Generating pdf of perfusion maps
            pat = test_data_dic['pat']
            for eta_name, eta in [('eta_lbfgs', eta_lbfgs), ('eta_net', eta_net)]:
                
                N = eta.shape[0]
                img_List = []
                print('Generating ' + eta_name + ' maps..')
                for i in tqdm(range(N)):
                                            
                    # Crop maps
                    # Flow
                    flow = eta[i,0,...]
                    # Delay
                    delay = eta[i,1,...]
                    # Decay
                    decay = eta[i,2,...]
                    # im_sig
                    im_sig = (test_data_dic['im_sig'][i,...].mean(-1) * 10) / test_data_dic['im_sig'][i,...].mean(-1).max()
                    
                    # LBFGS eta solution from svd approximated ctc
                    figsize=(14, 8)
                    plot_list = [60 * flow, delay, decay, im_sig]
                    title_list = ["F", "Tau", "k", "DCE MR Image"]
                    range_list = [(None, None), (None, None), (None, None), (0, 10)]
                    cmap_list = ['viridis', 'viridis', 'viridis', 'inferno']
                    suptitle = 'Perfusion Maps (Patient ' + str(pat[i].item()) + ')'
                    pmaps_subplot = get_subplot(4, plot_list, title_list, range_list, cmap_list, figsize=figsize, suptitle=suptitle)
                    save_name = 'perfusion_map_' + str(i) + '.png'
                    save_data_dir = Path.joinpath(save_path, "perfusion_maps" + '_' + eta_name )
                    Path(save_data_dir).mkdir(parents=True, exist_ok=True)            
                    pmaps_subplot.savefig(Path.joinpath(save_data_dir, save_name), dpi=500)
                    img_dir = str(Path.joinpath(save_data_dir, save_name)) 
                    imga = Image.open(img_dir)
                    imga.load()
                    img = Image.new("RGB", imga.size, (255, 255, 255))
                    img.paste(imga, mask=imga.split()[3])  # Alpha channel made opaque

                    # Crop
                    imw, imh = img.size
                    if i==0:
                        img1=img
                    else:
                        img_List.append(img)
                    i += 1
                    
                pdf_name = eta_name + '_Perfusion_maps.pdf'
                pdf_dir = str(Path.joinpath(save_path, pdf_name))
                img1.save(pdf_dir, "PDF", resolution=100.0, save_all=True, append_images=img_List)
        
if estimation_eval==True:
    
    # Load arrays
    test_data_dic = np.load(Path.joinpath(save_path, 'test_data_dic.npz'), allow_pickle=True)
    pat = test_data_dic['pat']
    im_sig = test_data_dic['im_sig']
    seg = test_data_dic['seg']
    
    for eta_load in config.load_eta_list:
        
        print('Patient-wise perfusion maps for ' + eta_load)
            
        # Load perfusion maps        
        eta = test_data_dic[eta_load]
    
        # Load AHA segments, bulls-eye and diagnosis
        perf_aha_path = Path('/data/brahma01/Datasets/perfusion_kcl/aha/')
        
        # Generating patient perfusion maps        
        img_List = []
        for p in tqdm(range(0, pat.__len__(), 3)):
            
            # Scale range
            mfactor = 0.98
            # Flow
            flow_seg = seg[p:p+3].copy()
            flow_seg[eta[p:p+3,0] > 6/60] = 0
            flow_max =  round(mfactor * 60 * (eta[p:p+3,0][flow_seg==1].mean()+ 2*eta[p:p+3,0][flow_seg==1].std()), 1) # 3.0 # 
            # Delay
            delay_seg = seg[p:p+3].copy()
            delay_seg[eta[p:p+3,1] > 5] = 0
            delay_max = 3.8 # round(mfactor * eta[p:p+3,1][delay_seg==1].mean()+ 2*eta[p:p+3,1][delay_seg==1].std(), 1) # 0.5 # 3.5 # 
            # Decay
            decay_seg = seg[p:p+3].copy()
            decay_seg[eta[p:p+3,2] > 0.5] = 0
            decay_max = None # round(mfactor * eta[p:p+3,2][decay_seg==1].mean()+ 2*eta[p:p+3,2][decay_seg==1].std(), 2) # 0.04 #0.15 #  
            
            # Perfusion map colormap
            pmap_colormap = 'viridis' # 'inferno' # 
            # Apical
            plot_apical = [60 * eta[p,0], eta[p,1], eta[p,2], im_sig[p].mean(-1)]
            range_apical = [(0,flow_max), (0,delay_max), (0,decay_max), (0,None)]
            cmap_apical = [pmap_colormap, pmap_colormap, pmap_colormap, 'gray']
            title_apical = ["Flow", "Delay", "Decay", "Signal Intensity"]
            ylabel_apical = ["Apical", "", "", ""]
            
            # Mid
            plot_mid = [60 * eta[p+1,0], eta[p+1,1], eta[p+1,2], im_sig[p+1].mean(-1)]
            range_mid = [(0,flow_max), (0,delay_max), (0,decay_max), (0,None)]
            cmap_mid = [pmap_colormap, pmap_colormap, pmap_colormap, 'gray']
            title_mid = ["", "", "", ""]
            ylabel_mid = ["Mid", "", "", ""]
            
            # Basal
            plot_basal = [60 * eta[p+2,0], eta[p+2,1], eta[p+2,2], im_sig[p+2].mean(-1)]
            range_basal = [(0,flow_max), (0,delay_max), (0,decay_max), (0,None)]
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
            suptitle = 'Perfusion Maps (Patient ' + str(pat[p]) + ')'        
            ncol = 4
            subplot = plt.figure(figsize=figsize)
            nplots = plot_list.__len__()
            nrows = np.ceil(nplots / ncol).astype(int)
            subplot.suptitle(suptitle)
            gs = subplot.add_gridspec(nrows, ncol)
            for plt_count in range(nplots):
                plot_img = plot_list[plt_count]
                if int(plt_count%4) != 3:
                    mask = (1-seg[p+int(np.floor(plt_count/4))])
                o = int((np.ceil((plt_count+1)/4)*4)-1)
                overlay_img = plot_list[o]             
                i = plt_count % ncol
                j = np.floor(plt_count / ncol).astype(int)
                axs = subplot.add_subplot(gs[j, i])
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
            save_name = 'Patient_' + str(pat[p]) + '.png'
            save_data_dir = Path.joinpath(save_path, "patient_eval_verbose_" + eta_load)
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
            
        pdf_name = eta_load + '_patient_perfusion_maps_verbose.pdf'
        pdf_dir = str(Path.joinpath(save_path, pdf_name))
        img1.save(pdf_dir, "PDF", resolution=100.0, save_all=True, append_images=img_List)
        
        
        # p = 0
        # seg_filtered = seg[p:p+3]
        # seg_filtered[eta[p:p+3,1,...]<0.1] = 0
        # flow_voxels = 60*eta[p:p+3,0,...][seg_filtered==1]
        # delay_voxels = eta[p:p+3,1,...][seg_filtered==1]
        # covar = ((flow_voxels-flow_voxels.mean())*(delay_voxels-delay_voxels.mean())).mean()
        # # Plot Clusters
        # matplotlib.use('TkAgg') 
        # lbfgs_cluster_fig = plt.figure()
        # # plt.scatter(flow_voxels, delay_voxels, label="Healthy", color="black")
        # plt.scatter(flow_voxels, delay_voxels, s=1)
        # plt.title(f'LBFGS (Covariance : {np.round(covar,4)})')
        # plt.xlabel('Flow')
        # plt.ylabel('Delay')
        # plt.legend(loc="upper right")
        # plt.show()
        # # plt.close()
        # # lbfgs_cluster_fig.savefig(Path.joinpath(save_path, 'lbfgs_cluster_fig'), dpi=500)
        
if bullseye_eval==True:
    
    # Load arrays
    test_data_dic = np.load(Path.joinpath(save_path, 'test_data_dic.npz'), allow_pickle=True)
    pat = test_data_dic['pat']
    im_sig = test_data_dic['im_sig']
    seg = test_data_dic['seg']
    
    flow_wilcoxtest = {}
    delay_wilcoxtest = {}
    
    for eta_load in config.load_eta_list:
        
        print("Patient-wise bull's eye plots for " + eta_load)
            
        # Load perfusion maps        
        eta = test_data_dic[eta_load]
    
        # Load AHA segments, bulls-eye and diagnosis
        perf_aha_path = Path('/data/brahma01/Datasets/perfusion_kcl/aha/')
        
        # Generating patient perfusion maps        
        img_List = []
        ns = int(16)
        flow_segments = np.zeros(pat.__len__()*ns//3)
        delay_segments = np.zeros(pat.__len__()*ns//3)
        diagnosis_segments = np.zeros(pat.__len__()*ns//3)
        for p in tqdm(range(0, pat.__len__(), 3)):
            
            # Segmentation boundaries and Flow bulls-eye
            pdata_path = Path.joinpath(perf_aha_path, str(pat[p]) + '_STRESS_moco')
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
                    
                # matplotlib.use('TkAgg')
                # figure = plt.figure()
                # plt.imshow(boundaries_sl)
                # plt.title('AHA Segments')
                # plt.axis('off')  # Optional: Turn off the axis
                # plt.show()
                    
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
            flow_bullseye = PerfusionBullseye(pat[p], plottingSettings, flow_bullseye_array)
            flow_bullseye.bulls_eye.figure.savefig(Path.joinpath(save_path, 'flow_bullseye'), dpi=300)
            plot_flow_bullseye = [Image.open(Path.joinpath(save_path, 'flow_bullseye.png'))]
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
            delay_bullseye = PerfusionBullseye(pat[p], plottingSettings, delay_bullseye_array)
            delay_bullseye.bulls_eye.figure.savefig(Path.joinpath(save_path, 'delay_bullseye'), dpi=300)
            plot_delay_bullseye = [Image.open(Path.joinpath(save_path, 'delay_bullseye.png'))]
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
            suptitle = "Bull's Eye (Patient " + str(pat[p]) + ")"        
            ncol = 6
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
                # if (plt_count%5)!=4 :
                #     divider = make_axes_locatable(axs)
                #     cax = divider.append_axes("bottom", size="5%", pad=0.05)
                #     plt.colorbar(im, cax=cax, orientation="horizontal")
            # plt.show()
            plt.close()            
            save_name = 'Patient_' + str(pat[p]) + '.png'
            save_data_dir = Path.joinpath(save_path, "flow_bullseye_" + eta_load)
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
            
        pdf_name = eta_load + '_flow_bullseye.pdf'
        pdf_dir = str(Path.joinpath(save_path, pdf_name))
        img1.save(pdf_dir, "PDF", resolution=100.0, save_all=True, append_images=img_List)
        
        # Correlation plot between flow and delay
        # matplotlib.use('TkAgg')             
        correlation_fig = plt.figure(figsize=(6, 6))
        correlation_coefficient = np.corrcoef(flow_segments, delay_segments)[0, 1]
        plt.scatter(flow_segments, delay_segments, label=f'Correlation: {correlation_coefficient:.2f}')
        plt.xlabel('Flow')
        plt.ylabel('Delay')
        plt.legend()
        plt.title('Segment-wise Correlation Scatter Plot')
        correlation_fig_name = 'correlation_fig_' + str(eta_load)
        correlation_fig.savefig(Path.joinpath(save_path, correlation_fig_name), dpi=300)
        
        # Extracting flow and delay        
        flow_healthy = (flow_segments *  (1-diagnosis_segments))
        flow_healthy = flow_healthy[flow_healthy!=0]
        flow_ischemic = (flow_segments * diagnosis_segments)
        flow_ischemic = flow_ischemic[flow_ischemic!=0]        
        delay_healthy = (delay_segments *  (1-diagnosis_segments))
        delay_healthy = delay_healthy[delay_healthy!=0]
        delay_ischemic = (delay_segments * diagnosis_segments)
        delay_ischemic = delay_ischemic[delay_ischemic!=0]
        
        # Flow Box-plot
        pval_flow_diagnosis = round(mannwhitneyu(flow_healthy, flow_ischemic, method="exact").pvalue,6)
        flow_diag_dic = {'Healthy':flow_healthy,'Ischemic':flow_ischemic}
        # matplotlib.use('TkAgg')
        flow_box_plot, axes = plt.subplots()
        axes.boxplot(flow_diag_dic.values(), widths=0.5, showfliers=False)        
        axes.scatter(np.random.normal(1, 0.1, flow_healthy.shape), flow_healthy, label="Healthy", color="black", alpha=0.5)
        axes.scatter(np.random.normal(2, 0.1, flow_ischemic.shape), flow_ischemic, label="Ischemic", color="black", alpha=0.5)
        # axes.set_ylim(None, 2.0)
        axes.set_xticklabels(flow_diag_dic.keys())
        # axes.set_ylim(bottom=None, top=1.6)
        axes.set_aspect(2)
        flow_bplot_title = 'Flow Box Plot (pvalue:' + str(pval_flow_diagnosis) + ')'
        plt.title(flow_bplot_title)
        flow_box_plot_name = 'flow_box_plot_' + str(eta_load)
        flow_box_plot.savefig(Path.joinpath(save_path, flow_box_plot_name ), dpi=300)
        
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
        delay_box_plot_name = 'delay_box_plot_' + str(eta_load)
        delay_box_plot.savefig(Path.joinpath(save_path, delay_box_plot_name ), dpi=300)
        
        flow_wilcoxtest[eta_load] = {}
        flow_wilcoxtest[eta_load]['Healthy'] = flow_healthy
        flow_wilcoxtest[eta_load]['Ischemic'] = flow_ischemic
        delay_wilcoxtest[eta_load] = {}
        delay_wilcoxtest[eta_load]['Healthy'] = delay_healthy       
        delay_wilcoxtest[eta_load]['Ischemic'] = delay_ischemic
    
    # LBFGS 
    # Healthy
    lbfgs_healthy = np.stack((flow_wilcoxtest['eta_lbfgs']['Healthy'],delay_wilcoxtest['eta_lbfgs']['Healthy']), axis=1)
    lbfgs_healthy_clabels = 0*np.ones(lbfgs_healthy.shape[0])
    # Ischemic
    lbfgs_ischemic = np.stack((flow_wilcoxtest['eta_lbfgs']['Ischemic'],delay_wilcoxtest['eta_lbfgs']['Ischemic']), axis=1)
    lbfgs_ischemic_clabels = 1*np.ones(lbfgs_ischemic.shape[0])
    # Calculate the Silhouette Score
    lbfgs_perf = np.concatenate((lbfgs_healthy,lbfgs_ischemic), axis=0)
    lbfgs_perf_clabels = np.concatenate((lbfgs_healthy_clabels,lbfgs_ischemic_clabels), axis=0)
    lbfgs_dunn_index = silhouette_score(lbfgs_perf, lbfgs_perf_clabels, metric='euclidean') # silhouette_score dunn_index silhouette_samples
    # lbfgs_dunn_index = np.mean(dunn_index(lbfgs_perf, lbfgs_perf_clabels, diameter_def='centroid')) mahalanobis
    # Plot Clusters
    matplotlib.use('TkAgg') 
    lbfgs_cluster_fig = plt.figure()
    plt.scatter(flow_wilcoxtest['eta_lbfgs']['Healthy'], delay_wilcoxtest['eta_lbfgs']['Healthy'], label="Healthy", color="black")
    plt.scatter(flow_wilcoxtest['eta_lbfgs']['Ischemic'], delay_wilcoxtest['eta_lbfgs']['Ischemic'], label="Ischemic", color="green")
    plt.title(f'LBFGS (Silhouette Score : {round(lbfgs_dunn_index,4)})')
    plt.xlabel('Flow')
    plt.ylabel('Delay')
    plt.legend(loc="upper right")
    # plt.show()
    plt.close()
    lbfgs_cluster_fig.savefig(Path.joinpath(save_path, 'lbfgs_cluster_fig'), dpi=500)
    
    # LBFGS-OD
    # Healthy
    net_healthy = np.stack((flow_wilcoxtest['eta_net']['Healthy'],delay_wilcoxtest['eta_net']['Healthy']), axis=1)
    net_healthy_clabels = 1*np.ones(net_healthy.shape[0])
    # Ischemic
    net_ischemic = np.stack((flow_wilcoxtest['eta_net']['Ischemic'],delay_wilcoxtest['eta_net']['Ischemic']), axis=1)
    net_ischemic_clabels = 2*np.ones(net_ischemic.shape[0])
    # Calculate the Silhouette Score
    net_perf = np.concatenate((net_healthy,net_ischemic), axis=0)
    net_perf_clabels = np.concatenate((net_healthy_clabels,net_ischemic_clabels), axis=0)
    net_dunn_index = silhouette_score(net_perf, net_perf_clabels, metric='euclidean') # silhouette_score dunn_index silhouette_samples
    # lbfgs_od_dunn_index = np.mean(dunn_index(lbfgs_od_perf, lbfgs_od_perf_clabels, diameter_def='centroid'))
    # Plot Clusters
    matplotlib.use('TkAgg') 
    net_cluster_fig = plt.figure()
    plt.scatter(flow_wilcoxtest['eta_net']['Healthy'], delay_wilcoxtest['eta_net']['Healthy'], label="Healthy", color="black")
    plt.scatter(flow_wilcoxtest['eta_net']['Ischemic'], delay_wilcoxtest['eta_net']['Ischemic'], label="Ischemic", color="green")
    plt.title(f'DeepFermi-Net (Silhouette Score : {round(net_dunn_index,4)})')
    plt.xlabel('Flow')
    plt.ylabel('Delay')
    plt.legend(loc="upper right")
    # plt.show()
    plt.close()
    net_cluster_fig.savefig(Path.joinpath(save_path, 'net_cluster_fig'), dpi=500)
    

if train_curves==True:
    
    # Step-size
    step = 5
    pretrain_nn_path = Path('/data/brahma01/DCEPerfusion/InVivo/Experiments/09_Deterministic_MANN/')
    
    # Loading model-agnostic pre-training curves
    pretrain_ssup_loss_train = np.load(Path.joinpath(pretrain_nn_path, 'ssup_loss_train.npy'))[::step]
    pretrain_ssup_loss_val = np.load(Path.joinpath(pretrain_nn_path, 'ssup_loss_val.npy'))[::step]
    pretrain_it_vect = np.load(Path.joinpath(pretrain_nn_path, 'it_vect.npy'))[::step]
    
    # Loading model-based training curves
    ssup_loss_train = np.load(Path.joinpath(read_path, 'ssup_loss_train.npy'))[::step]
    ssup_loss_val = np.load(Path.joinpath(read_path, 'ssup_loss_val.npy'))[::step]
    it_vect = np.load(Path.joinpath(read_path, 'it_vect.npy'))[::step]
    
    # Padding 
    pre_ulim = np.load(Path.joinpath(pretrain_nn_path, 'it_vect.npy'))[-1]    
    it_vect = it_vect + pre_ulim
    ssup_loss_train = np.pad(ssup_loss_train, (pretrain_ssup_loss_train.__len__(), 0), 'constant', constant_values=(np.nan, 0))
    ssup_loss_val = np.pad(ssup_loss_val, (pretrain_ssup_loss_val.__len__(), 0), 'constant', constant_values=(np.nan, 0))
    it_vect = np.concatenate((pretrain_it_vect, it_vect))
    
    # Plotting Curves
    ssup_loss_fig = plt.figure()
    ssup_loss_fig.suptitle('Self-supervised Loss curves')
    plt.plot(pretrain_it_vect, pretrain_ssup_loss_train, linewidth=1, color='black')
    plt.plot(pretrain_it_vect, pretrain_ssup_loss_val, '--', linewidth=1, color='red')
    plt.plot(it_vect, ssup_loss_train, label="Training", linewidth=1, color='black')
    plt.plot(it_vect, ssup_loss_val, '--', label="Validation", linewidth=1, color='red')
    plt.axvspan(0, 100000, alpha=0.08, color='blue')
    plt.axvspan(100000, 200000, alpha=0.08, color='yellow')
    plt.xlabel('Iterations')
    plt.ylabel('Self-supervised Loss')
    plt.legend(loc="upper right")
    plt.xlim(left=0, right=200000) # None
    plt.ylim(bottom=0, top=0.8)
    ssup_loss_fig.savefig(Path.joinpath(save_path, 'ssup_loss_fig'), dpi=500)
    plt.close()
    
if outlier_tolerance_eval==True:
    
    # Initialization
    test_data_dic = np.load(Path.joinpath(save_path, 'test_data_dic.npz'), allow_pickle=True)
    osamp = 20
    S = 10
    S_op = expand_dim(torch.tensor([1,1/S,S]), f_dim_pad=1, b_dim_pad=2).to(device)
    sl = 8
    
    ctc_dic = {}    
    for eta_load in config.load_eta_list:
        
        print('Generating maps for ' + eta_load) 
        
        # Windowing
        wlen_aug = 10
        wlen = test_data_dic['wlen'][sl] + 5
        aif = torch.tensor(test_data_dic['aif'][sl:sl+1,0:wlen], device=device)
        ctc = torch.tensor(test_data_dic['ctc'][sl:sl+1,...,0:wlen], device=device)
        time = torch.tensor(test_data_dic['time'][sl:sl+1,0:wlen], device=device)
        seg = torch.tensor(test_data_dic['seg'][sl:sl+1], device=device)
        mask_od = torch.tensor(test_data_dic['mask_od'][sl,0:wlen])
        # seg = torch.tensor(np.moveaxis(np.load('/data/brahma01/Datasets/perfusion_kcl/aha/36_STRESS_moco/mask_CTC.npy', allow_pickle=True),-1,0), device='cuda')
        
        # Estimate perfusion parameters
        aif = aif.unsqueeze(1).unsqueeze(1) * torch.ones(ctc.shape, device=aif.device)
        eta = torch.tensor(test_data_dic[eta_load][sl:sl+1], device=device)
        
        # Segmenting curves            
        aif_seg = aif[seg==1]
        ctc_seg = ctc[seg==1]
        
        ctc_seg_uncomp = ctc[seg==1]
        
        # Compensating offset in the time curves
        oTp = 5
        aif_seg = aif_seg-aif_seg[...,0:oTp].mean(-1, keepdim=True)
        ctc_seg = ctc_seg-ctc_seg[...,0:oTp].mean(-1, keepdim=True)
        
        # Oversampling curves (Linear)
        time = (time-time[:,0])
        time_osamp = interp_linear_1D(time, size=osamp*time.shape[-1])/S
        aif_osamp = interp_linear_1D(aif_seg, size=osamp*aif_seg.shape[-1])
        
        # Self-supervised loss
        neg_shift = (2*osamp)
        fermi_ir = fermi_ir_func(S_op * eta, time_osamp.squeeze(), C=500, neg_shift=neg_shift)
        fermi_ir = fermi_ir[seg==1]
        ctc_est = convolve(aif_osamp, fermi_ir, neg_shift=neg_shift)[..., ::osamp]/osamp        
        ctc_dic['ctc_' + eta_load] = ctc_est
    # Logic for selecting criterion
    # Small Delay
    eta_ROI = eta[:,1,...][seg==1]
    cond = eta_ROI < 10000.0
    indices = cond.nonzero().squeeze()
    
    # eta_ROI = eta[:,1,...][seg==1]
    # cond = (eta_ROI > 1.0) & (eta_ROI < 5.0)
    # indices = cond.nonzero().squeeze()
    
    sl = 0
    bins = 500
    # hist_eta_net = test_data_dic['eta_net'][sl,1,...].flatten()
    # hist_eta_lbfgs = test_data_dic['eta_lbfgs'][sl,1,...].flatten()
    hist_eta_net = test_data_dic['eta_net'][:,1,...].flatten()
    hist_eta_lbfgs = test_data_dic['eta_lbfgs'][:,1,...].flatten()
    hist_eta_net = hist_eta_net[hist_eta_net!=0]
    hist_eta_lbfgs = hist_eta_lbfgs[hist_eta_lbfgs!=0]
    bins = np.histogram(np.hstack((hist_eta_net,hist_eta_lbfgs)), bins=bins)[1]
    matplotlib.use('TkAgg')
    figure = plt.figure()
    plt.hist(list(hist_eta_lbfgs), bins=bins, color='red', alpha=0.2, label='lbfgs')
    plt.hist(list(hist_eta_net), bins=bins, color='teal', alpha=0.2, label='DeepFermi')
    plt.legend(loc="upper right")
    plt.title('Histogram')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
    # random_index = torch.randint(0, len(indices), (1,))
    # indices = indices[random_index]
    
    # matplotlib.use('TkAgg') 
    # ctc_fig = plt.figure()
    # plt.plot(aif_seg.mean(0).detach().cpu(), label="aif", linewidth=1, color="black", linestyle="dashed")
    # plt.plot(ctc_dic['ctc_eta_lbfgs'][indices].mean(0).detach().cpu(), label="lbfgs", linewidth=1, color="red")
    # plt.plot(ctc_dic['ctc_eta_net'][indices].mean(0).detach().cpu(), label="DeepFermi", linewidth=1, color="blue")
    # plt.plot(ctc_seg[indices].mean(0).detach().cpu(), linewidth=1, color="green")
    # plt.legend(loc="upper right")
    # plt.show()

    eta[:,1,...][seg!=1]=10
    eta[:,1,...][eta[:,1,...]<0.1]= 50    
    matplotlib.use('TkAgg')
    figure = plt.figure()
    plt.imshow(eta[0,1,...].detach().cpu())  # Assuming 'seg_i' is a grayscale image
    plt.title('AHA Segments')
    plt.axis('off')  # Optional: Turn off the axis
    plt.show()
    
    matplotlib.use('TkAgg')
    figure = plt.figure()
    error_net = torch.norm((ctc_seg-ctc_dic['ctc_eta_net'])[indices], dim=0).detach().cpu()
    error_lbfgs = torch.norm((ctc_seg-ctc_dic['ctc_eta_lbfgs'])[indices], dim=0).detach().cpu()
    plt.plot(time.squeeze(0).detach().cpu(), error_lbfgs, label="lbfgs", color="red")
    plt.plot(time.squeeze(0).detach().cpu(), error_net, label="DeepFermi", color="blue")
    plt.title('Fitting Error')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()
    
    matplotlib.use('TkAgg')
    figure = plt.figure()
    plt.hist(list(error_net.numpy()), bins=20, color='teal')
    plt.title('Histogram')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
    
    matplotlib.use('TkAgg')
    figure = plt.figure()
    ind = np.arange((error_net.__len__()))
    np.random.shuffle(ind)
    ind = np.sort(ind[0:40])
    z = (error_net[ind] - error_net[ind].mean()) / error_net[ind].std()
    time_z = time.squeeze(0)[ind].detach().cpu()
    plt.plot(time_z, z, label="z-score", color="black")
    if 29 in ind:
        plt.title('Z-score (29)')
    else:
        plt.title('Z-score')        
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()
    
    measured_mean = ctc_seg[indices].mean(0).detach().cpu()
    measured_std = ctc_seg[indices].std(0).detach().cpu()
    uncomp_mean = ctc_seg_uncomp[indices].mean(0).detach().cpu()
    uncomp_std = ctc_seg_uncomp[indices].std(0).detach().cpu()    
    lbfgs_mean = ctc_dic['ctc_eta_lbfgs'][indices].mean(0).detach().cpu()
    lbfgs_std = ctc_dic['ctc_eta_lbfgs'][indices].std(0).detach().cpu()
    net_mean = ctc_dic['ctc_eta_net'][indices].mean(0).detach().cpu()
    net_std = ctc_dic['ctc_eta_net'][indices].std(0).detach().cpu()
    lbfgs_error = torch.norm((ctc_seg[indices]-ctc_dic['ctc_eta_lbfgs'][indices]), dim=0).detach().cpu()/torch.norm((ctc_seg[indices])).detach().cpu()
    net_error = torch.norm((ctc_seg[indices]-ctc_dic['ctc_eta_net'][indices]), dim=0).detach().cpu()/torch.norm((ctc_seg[indices])).detach().cpu()
    time_axis = time.squeeze(0).detach().cpu()
    fit_timepoint = time_axis[mask_od==1]
    fit_measured_ctc = measured_mean[mask_od==1]
    outlier_timepoint = time_axis[0:wlen-wlen_aug][mask_od[0:wlen-wlen_aug]==0]
    outlier_measured_ctc = measured_mean[0:wlen-wlen_aug][mask_od[0:wlen-wlen_aug]==0]
    linewidth = 1.5
    matplotlib.use('TkAgg') 
    ctc_fig = plt.figure()
    plt.plot(time_axis, aif_seg.mean(0).detach().cpu(), label="aif", linewidth=linewidth, color="black", linestyle="dashed")
    plt.plot(time_axis, measured_mean, '--', label="measured", linewidth=1, color="green")
    plt.scatter(fit_timepoint, fit_measured_ctc, color='green', marker='o', s=30, zorder=2)
    plt.scatter(outlier_timepoint, outlier_measured_ctc, color='green', marker='x', s=30, zorder=2)
    plt.fill_between(time_axis, measured_mean-measured_std, measured_mean+measured_std, color="green", alpha=0.05)
    # plt.plot(time.squeeze(0).detach().cpu(), uncomp_mean, label="measured", linewidth=1, color="green")
    # plt.fill_between(time.squeeze(0).detach().cpu(), uncomp_mean-uncomp_std, uncomp_mean+uncomp_std, color="green", alpha=0.05)
    plt.plot(time_axis, lbfgs_mean, label="lbfgs", linewidth=linewidth, color="red")
    plt.fill_between(time_axis, lbfgs_mean-lbfgs_std, lbfgs_mean+lbfgs_std, color="red", alpha=0.05)
    plt.plot(time_axis, net_mean, label="DeepFermi", linewidth=linewidth, color="blue")
    plt.fill_between(time_axis, net_mean-net_std, net_mean+net_std, color="blue", alpha=0.05)
    # plt.plot(time_axis,lbfgs_error, label="lbfgs", linewidth=linewidth, color="purple")
    # plt.plot(time_axis,net_error, label="DeepFermi", linewidth=linewidth, color="violet")
    plt.legend(loc="upper right")
    plt.xlim(left=0, right=28.22)
    plt.ylim(bottom=-0.03, top=0.22)
    # save(ctc_fig, save_obj_type='fig', save_obj_name='conc_curves_fig') 
    # plt.close()
    plt.show()
    
    sumsampl = np.arange(0,1)
    measured = ctc_seg[indices[sumsampl]].swapaxes(0,1).detach().cpu()
    lbfgs = ctc_dic['ctc_eta_lbfgs'][indices[sumsampl]].swapaxes(0,1).detach().cpu()
    net = ctc_dic['ctc_eta_net'][indices[sumsampl]].swapaxes(0,1).detach().cpu()
    matplotlib.use('TkAgg') 
    ctc_fig = plt.figure()
    plt.plot(time.squeeze(0).detach().cpu(), aif_seg.mean(0).detach().cpu(), label="aif", linewidth=1, color="black", linestyle="dashed")
    plt.plot(time.squeeze(0).detach().cpu(), measured, '--', linewidth=1)
    plt.plot(time.squeeze(0).detach().cpu(), lbfgs, linewidth=1, color="red")
    plt.plot(time.squeeze(0).detach().cpu(), net, linewidth=1, color="blue")
    plt.legend(loc="upper right")
    plt.show()
            
if plot_conc_curves==True:
    
    aif_uncorrupted = test_data_dic['aif_uncorrupted'].to(device)
    aif_corrupted = test_data_dic['aif'].to(device)
    aif = aif_corrupted.unsqueeze(1).unsqueeze(1) * torch.ones(test_data_dic['ctc'].shape, device=aif_corrupted.device)
    seg_full = test_data_dic['seg_full'][test_slice_indx, ...]
    
    # Choosing slice for ctc 
    ctc_uncorrupted = test_data_dic['ctc_uncorrupted'][test_slice_indx]
    ctc_corrupted = test_data_dic['ctc'][test_slice_indx]
    # Segmenting areas            
    ctc_uncorrupted_healthy = ctc_uncorrupted[seg_full==1]
    ctc_corrupted_healthy = ctc_corrupted[seg_full==1]
    ctc_uncorrupted_ischemic = ctc_uncorrupted[seg_full==71]
    ctc_corrupted_ischemic = ctc_corrupted[seg_full==71]
    
    # Choosing pixels
    healthy_pixel = random.randint(0, ctc_corrupted_healthy.shape[0])
    ischemic_pixel = random.randint(0, ctc_corrupted_ischemic.shape[0])
    
    # Plotting curves
    conc_curves_fig = plt.figure()
    plt.plot(aif_corrupted[test_slice_indx,...].detach().cpu(), label="corrupted aif", linewidth=1, color="red")
    plt.plot(aif_uncorrupted[test_slice_indx,...].detach().cpu(), label="aif", linewidth=1, color="red", linestyle="dashed")
    plt.plot(ctc_corrupted_healthy[healthy_pixel,...].detach().cpu(), label="corrupted healthy ctc", linewidth=1, color="blue")
    plt.plot(ctc_uncorrupted_healthy[healthy_pixel,...].detach().cpu(), label="healthy ctc", linewidth=1, color="blue", linestyle="dashed")
    plt.plot(ctc_corrupted_ischemic[ischemic_pixel,...].detach().cpu(), label="corrupted ischemic ctc", linewidth=1, color="green")
    plt.plot(ctc_uncorrupted_ischemic[ischemic_pixel,...].detach().cpu(), label="ischemic ctc", linewidth=1, color="green", linestyle="dashed")
    plt.legend(loc="upper right")   
    save(conc_curves_fig, save_obj_type='fig', save_obj_name='corrupted_conc_curves_fig')    
    
    for eta_load in config.load_eta_list:
        
        print('Plotting concentration curves ' + eta_load + ' maps..')
        
        # Load perfusion maps
        eta = np.load(Path.joinpath(save_path, eta_load + '.npy'))
        
        # Construction
        eta = torch.tensor(eta).to(device)               
        ctc = report.ctc_plot(eta, aif)[test_slice_indx]
        
        # Segmenting areas            
        ctc_healthy = ctc[seg_full==1]
        ctc_ischemic = ctc[seg_full==71]
        
        # Plotting curves
        conc_curves_fig = plt.figure()
        plt.plot(aif_corrupted[test_slice_indx,...].detach().cpu(), label="corrupted aif", linewidth=1, color="black")
        plt.plot(aif_uncorrupted[test_slice_indx,...].detach().cpu(), label="aif", linewidth=1, color="black", linestyle="dashed")
        plt.plot(ctc_healthy[healthy_pixel,...].detach().cpu(), label="estimated healthy ctc", linewidth=1, color="blue")
        plt.plot(ctc_uncorrupted_healthy[healthy_pixel,...].detach().cpu(), label="healthy ctc", linewidth=1, color="blue", linestyle="dashed")
        plt.plot(ctc_ischemic[ischemic_pixel,...].detach().cpu(), label="estimated ischemic ctc", linewidth=1, color="red")
        plt.plot(ctc_uncorrupted_ischemic[ischemic_pixel,...].detach().cpu(), label="ischemic ctc", linewidth=1, color="red", linestyle="dashed")
        plt.legend(loc="upper right")
        save_obj_name =  eta_load +  '_corrupted_conc_curves_fig'
        save(conc_curves_fig, save_obj_type='fig', save_obj_name=save_obj_name)
