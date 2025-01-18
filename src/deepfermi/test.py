import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
from pathlib import Path
import pickle
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader

from config import TestConfig, TrainConfig, Mode
from data_loading import DatasetDCEPerfusion, Transform, collate
from network.Unet import Unet
from network.DeepFermi_net import DeepFermi

def main() -> None:
    
    # Reading command line
    parser = argparse.ArgumentParser()
    default_config_path = str(Path(__file__).resolve().parent.parent.parent / 'config/test_config.yaml')
    parser.add_argument('--config_path', default=default_config_path,type=str)    
    parser.add_argument('--project_name', default=None,type=str)
    parser.add_argument('--read_project_name', default=None,type=str)
    parser.add_argument('--dataset_file_name', default=None,type=str)
    parser.add_argument('--SNR_ctc',default=None,type=int)
    parser.add_argument('--build_dataset_flag', default=None, choices=(True, False), type=eval)
    parser.add_argument('--mode', default=None,type=str)
    parser.add_argument('--load_unet', default=None,type=str)
    args = parser.parse_args()
    
    # Reading testing configuration file
    test_config_path = args.config_path
    print(test_config_path)
    test_cfg = TestConfig.from_yaml(test_config_path)
    
    # # Overriding configuration if command line input provided
    test_cfg.info.project_name = test_cfg.info.project_name if args.project_name == None else args.project_name
    test_cfg.info.read_project_name = test_cfg.info.read_project_name if args.read_project_name == None else args.read_project_name
    test_cfg.test_params.dataset.file_name = test_cfg.test_params.dataset.file_name if args.dataset_file_name == None else args.dataset_file_name
    test_cfg.test_params.dataset.SNR_ctc = test_cfg.test_params.dataset.SNR_ctc if args.SNR_ctc == None else args.SNR_ctc
    test_cfg.test_params.dataset.build_dataset_flag = test_cfg.test_params.dataset.build_dataset_flag if args.build_dataset_flag == None else args.build_dataset_flag
    test_cfg.test_params.mode = Mode(test_cfg.test_params.mode.value) if args.mode == None else Mode(args.mode)    
    test_cfg.test_params.load_unet = test_cfg.test_params.load_unet if args.load_unet == None else args.load_unet
    
    # Preliminary checks
    assert test_cfg.test_params.mode.value=='testing', "Only 'testing' mode is allowed"
    
    # Reading saved config of the experiment to be read
    config_path = Path.joinpath(Path(test_cfg.paths.read + test_cfg.info.read_project_name), 'train_config.yaml')
    cfg = TrainConfig.from_yaml(config_path)
    
    # Dataset
    file_path = Path.joinpath(Path(test_cfg.paths.dataset), test_cfg.test_params.dataset.file_name)
    # Save a reference to the class itself
    build_dataset = test_cfg.test_params.dataset.build_dataset_flag
    test_dataset_path = Path.joinpath(Path(test_cfg.paths.dataset),'test_dataset.pkl')
    
    if build_dataset == True:
        transform = None
        test_dataset = DatasetDCEPerfusion.construct_from_npz(
            file_path,
            transform,
            pid_to_load=test_cfg.test_params.ntest,
            aug_dataset_flag = False,
            config=test_cfg,
            od_enable=test_cfg.test_params.clean_outliers
            )
        # Save dataset
        with open(test_dataset_path, 'wb') as f:
            pickle.dump(test_dataset, f)
    else:
        # Load dataset
        transform = None
        with open(test_dataset_path, 'rb') as f:
            test_dataset = pickle.load(f)
            test_dataset.transform = transform
            
    # Update eta_bkg_ref
    test_dataset = DatasetDCEPerfusion._update_precond_bkg_ref(test_dataset, test_cfg)
            
    # Network Initialization
    cnn = Unet(dim=3, 
               ncin=cfg.train_params.network.ncin, 
               ncout=cfg.train_params.network.ncout, 
               nstage=cfg.train_params.network.nstage, 
               nconv_stage=cfg.train_params.network.nconv_stage, 
               nfilters=cfg.train_params.network.nfilters, 
               res_connect=False, 
               bias=False)
    unet = DeepFermi(cnn, 
                     osamp=cfg.train_params.network.osamp, 
                     nu=cfg.train_params.network.nu, 
                     max_iter_lbfgs=cfg.train_params.network.max_iter_lbfgs, 
                     max_eval_lbfgs=cfg.train_params.network.max_eval_lbfgs, 
                     mode=test_cfg.test_params.mode.value, 
                     learn_lambda=cfg.train_params.network.learn_lambda).cuda()
    
    # Load Network from checkpoint
    network_path = Path.joinpath(Path(test_cfg.paths.save), test_cfg.info.read_project_name)
    print('Loading network...')
    load_unet = test_cfg.test_params.load_unet
    unet_state_dic = torch.load(Path.joinpath(network_path, load_unet))
    # unet_state_dic.pop('lambda_reg')
    # unet_state_dic.pop('max_iter_lbfgs')
    # unet_state_dic.pop('max_eval_lbfgs')
    unet.load_state_dict(unet_state_dic, strict=False)
    
    # Parameters
    device = test_cfg.test_params.device.value
    clean_outliers = test_cfg.test_params.clean_outliers
    morph_flag = test_cfg.test_params.morph_flag
    is_erosion_not_dilate = test_cfg.test_params.is_erosion_not_dilate 
    
    # Segmentation       
    if morph_flag == True:
        morph_kernel = torch.tensor([[1, 1, 1],
                                     [1, 1, 1],
                                     [1, 1, 1]], dtype=test_dataset.seg.dtype)
        morph_conv = torch.nn.functional.conv2d(test_dataset.seg.unsqueeze(1), 
                                                morph_kernel.unsqueeze(0).unsqueeze(0), 
                                                padding=(1, 1))
        morph_thresh = morph_kernel.numel()-0.01 if is_erosion_not_dilate==True else 0
        test_dataset.seg = torch.heaviside(morph_conv- morph_thresh, 
                                           torch.tensor(0, dtype=test_dataset.seg.dtype)
                                           ).squeeze(1)
        test_dataset.ctc = test_dataset.seg.unsqueeze(-1) * test_dataset.im_sig
        
    # Old AIF
    # old_aif = np.load("/data/brahma01/deepfermi/invivo/old_aif.npy")
    # print((test_dataset.aif-old_aif).sum())
    # test_dataset.aif = torch.tensor(old_aif)
    
    # Load data
    test_dataloader = DataLoader(test_dataset,
                                    batch_size=1,
                                    num_workers=2,
                                    collate_fn=collate,
                                    prefetch_factor=4,
                                    pin_memory=True)
    
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('TkAgg')
    # plt.figure()
    # for i in range(test_dataset.aif.shape[0]):
    #     plt.title("AIF")
    #     plt.plot( test_dataset.aif.cpu()[i], linewidth=1)
    # plt.show()
    
    # matplotlib.use('TkAgg')
    # plt.figure()
    # for i in range(old_aif.shape[0]):
    #     plt.title("AIF")
    #     plt.plot( old_aif[8], linewidth=1)
    # plt.show()
    
    # Constructing placeholders
    N, nx, ny, _ = test_dataset.im_sig.shape
    eta_net = torch.zeros((N, 3, nx, ny), device=device)
    eta_lbfgs = torch.zeros((N, 3, nx, ny), device=device)
    time_taken = torch.zeros(N, device=device)
    
    # For timing performance
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    # import matplotlib
    # import matplotlib.pyplot as plt
    # matplotlib.use('TkAgg')
    # i = 3
    # mbolus = test_dataset.mbolus[i]
    # aif = test_dataset.aif[i]
    # plt.figure()
    # plt.title("Concentration Time Curves")
    # plt.plot(mbolus.squeeze(0).cpu(), label="mbolus", linewidth=1, color="blue", linestyle="solid")
    # plt.plot(aif.squeeze(0).cpu(), label="aif", linewidth=1, color="red", linestyle="solid")
    # plt.legend(loc="upper right")   
    # plt.show()
    
    # Testing network
    print('Testing network')
    with torch.no_grad():   
        for i, batch in tqdm(enumerate(test_dataloader), total=N):
            
            # Unpack batched tuple
            wlen_test = batch.wlen
            im_sig_test = batch.im_sig[...,0:wlen_test].to(device)
            ctc_test = batch.ctc[...,0:wlen_test].to(device)
            aif_test = batch.aif[...,0:wlen_test].to(device)
            time_test = batch.time[...,0:wlen_test].to(device)            
            seg_test = batch.seg.to(device)
            eta_lbfgs_test = batch.eta_pretrain.to(device)
            mask_od_test = batch.mask_od
            
            # Filtering time-points
            if clean_outliers==True:
                indx_dc = np.arange(wlen_test)
                indx_dc = indx_dc[mask_od_test[0,:wlen_test]==1]
            else:
                indx_dc = np.arange(wlen_test)
            
            # Apply network
            start.record()
            eta_net_test = unet(im_sig_test.unsqueeze(1), seg_test, aif=aif_test, ctc=ctc_test, time=time_test, indx_dc=indx_dc)
            end.record()
            torch.cuda.synchronize()            
            
            # Results
            eta_net[i] = eta_net_test
            eta_lbfgs[i] = eta_lbfgs_test
            time_taken[i] = (start.elapsed_time(end)/1000)
    
    # Transfering tensors to cpu
    pid = test_dataset.pid
    im_sig = test_dataset.im_sig.cpu()
    ctc = test_dataset.ctc.cpu()
    aif = test_dataset.aif.cpu()
    time = test_dataset.time.cpu()
    wlen = test_dataset.wlen.cpu()
    seg = test_dataset.seg.cpu()
    mbolus = test_dataset.mbolus.cpu()
    mask_od = test_dataset.mask_od.cpu()
    eta_net = eta_net.cpu()
    eta_lbfgs = eta_lbfgs.cpu()
    time_taken = time_taken.cpu()
    
    # Saving tensors
    save_path = test_cfg.paths.save + '/' + test_cfg.info.project_name
    Path(save_path).mkdir(parents=True, exist_ok=True)
    np.save(Path.joinpath(Path(save_path), "pid.npy"), pid)
    np.save(Path.joinpath(Path(save_path), "im_sig.npy"), im_sig)
    np.save(Path.joinpath(Path(save_path), "ctc.npy"), ctc)
    np.save(Path.joinpath(Path(save_path), "aif.npy"), aif)
    np.save(Path.joinpath(Path(save_path), "time.npy"), time)
    np.save(Path.joinpath(Path(save_path), "wlen.npy"), wlen)    
    np.save(Path.joinpath(Path(save_path), "seg.npy"), seg)
    np.save(Path.joinpath(Path(save_path), "mbolus.npy"), mbolus)
    np.save(Path.joinpath(Path(save_path), "mask_od.npy"), mask_od)
    np.save(Path.joinpath(Path(save_path), "eta_net.npy"), eta_net)    
    np.save(Path.joinpath(Path(save_path), "eta_lbfgs.npy"), eta_lbfgs)
    np.save(Path.joinpath(Path(save_path), "time_taken.npy"), time_taken)

if __name__ == "__main__":
    main()