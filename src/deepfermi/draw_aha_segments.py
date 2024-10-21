import numpy as np
import pickle
import cv2
import os
import sys
import matplotlib.cm as cm
from utils import *
import dataio

sys.path.append('/data/brahma01/PtbPyTools/')
from visualisation.ahaBullseye.ahaHelperfunctions import PlottingSettings
from visualisation.ahaBullseye.ahaBullseye import Bullseye


pdat = '/data/brahma01/Datasets/perfusion_kcl/'

pbullseye_main = pdat + 'aha/'

# Patient number
pat_num = 63
cmap = cm.get_cmap('inferno') # matplotlib.colors.Colormap('plasma', N=256) # None #'inferno' # None#'plasma'

# Get images
fperf = dataio.get_files(pdat + 'rec', inc_strg=[str(pat_num) + '_STRESS_moco_m10_k01_perf.npz', ], vis=1)[0]
f = np.load(pdat + 'rec/' + fperf, allow_pickle=True)

# for k in f.keys():
#     print(k)
    
# # im_sig
# basic_imshow(f['im_sig'][53,:,:,0], '/data/brahma01/Datasets/perfusion_kcl/', fig_name='im_sig')

# # qPerf_bay
# basic_imshow(f['qPerf_bay'][53, 0, :,:,0], '/data/brahma01/Datasets/perfusion_kcl/', fig_name='qPerf_bay')

# # aif_cof
# basic_plot('/data/brahma01/Datasets/perfusion_kcl/', x=f['aif_cof'][0,0]['coefs'][0,:,3] , fig_name='aif_cof')

# # fit_res
# basic_plot('/data/brahma01/Datasets/perfusion_kcl/', x=f['fit_res'][0,0][...,0] , fig_name='fit_res')

# # im_sig_myo
# basic_plot('/data/brahma01/Datasets/perfusion_kcl/', x=f['im_sig_myo'][:,0,0] , fig_name='im_sig_myo')

# # jac
# basic_plot('/data/brahma01/Datasets/perfusion_kcl/', x= *f['jac'][0,0][:,0] , fig_name='jac')

# # sl_mask
# basic_imshow(f['sl_mask'][...,0], '/data/brahma01/Datasets/perfusion_kcl/', fig_name='sl_mask_0')
# basic_imshow(f['sl_mask'][...,1], '/data/brahma01/Datasets/perfusion_kcl/', fig_name='sl_mask_1')
# basic_imshow(f['sl_mask'][...,2], '/data/brahma01/Datasets/perfusion_kcl/', fig_name='sl_mask_2')

# # qPerf
# basic_imshow(f['qPerf'][0,:,:,0], '/data/brahma01/Datasets/perfusion_kcl/', fig_name='qPerf_0')
# basic_imshow(f['qPerf'][1,:,:,0], '/data/brahma01/Datasets/perfusion_kcl/', fig_name='qPerf_1')
# basic_imshow(f['qPerf'][2,:,:,0], '/data/brahma01/Datasets/perfusion_kcl/', fig_name='qPerf_2')
# basic_imshow(f['qPerf'][3,:,:,0], '/data/brahma01/Datasets/perfusion_kcl/', fig_name='qPerf_3')

# # qPerf_tik
# basic_imshow(f['qPerf_tik'][0,:,:,0], '/data/brahma01/Datasets/perfusion_kcl/', fig_name='qPerf_tik_0')
# basic_imshow(f['qPerf_tik'][1,:,:,0], '/data/brahma01/Datasets/perfusion_kcl/', fig_name='qPerf_tik_1')
# basic_imshow(f['qPerf_tik'][2,:,:,0], '/data/brahma01/Datasets/perfusion_kcl/', fig_name='qPerf_tik_2')
# basic_imshow(f['qPerf_tik'][3,:,:,0], '/data/brahma01/Datasets/perfusion_kcl/', fig_name='qPerf_tik_3')

# Folder for results of AHA saved
pbullseye = pbullseye_main + fperf.replace('_m10_k01_perf.npz', '/')
if not os.path.exists(pbullseye):
    os.mkdir(pbullseye)

# Get corner points for myocardium - DO NOT MODIFY
bullseye_mask = np.mean(f['sl_mask'], axis=2)
x, y, w, h = cv2.boundingRect(np.uint8(bullseye_mask))
disp_x = [y - 10, y + h + 10]
disp_y = [x - 10, x + w + 10]

# basic_imshow(bullseye_mask, '/data/brahma01/DCEPerfusion/InVivo/', fig_name='bullseye_mask')

# Visualise average intensity
cdat = np.mean(f['im_sig'][:,disp_x[0]:disp_x[1], disp_y[0]:disp_y[1],:], axis=0) # Time-averaged
cdat = cdat * 100 / np.max(cdat.flatten())
cdat = np.split(cdat, 3, axis=2)
cdat.insert(0, np.zeros_like(cdat[0])) # Including apex

# Settings for bulls eye plot
plottingSettings = PlottingSettings(cmap = cmap,
                                    vmin = 0,
                                    vmax = 80.0,
                                    show_segmentNumbers = True,
                                    show_std = False,
                                    closePlotAutomatically = False,
                                    show_debuggingImages = False,
                                    useEdgesToSetInnerPoints= False)

useSavedConfig = False

storedConfig = None
if useSavedConfig:
    fileName = os.path.abspath(pbullseye + 'ahaBullseye_config')
    with open(fileName, 'rb') as pickle_file:
        storedConfig = pickle.load(pickle_file)
        
        
bullseye = Bullseye(cdat, pbullseye, plottingSettings,  storedConfig)

