# from imports import *
# from analyze import *

# from save import *
# from unet_report import *
# from termcolor import colored
# from unet_train import *
# from fermi import *
# from utils import *
# import torchvision.transforms.functional as TF
# import torchvision.transforms as T

import time as exec_time
import warnings
warnings.filterwarnings("ignore")

from einops import rearrange
import numpy as np
from termcolor import colored
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from data_loading import collate
from tracker import Tracker
from utils import secs2time
from train import unet_eval, unet_pretrain, unet_train

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TRAINING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

class TrainingManager():

    def __init__(self,
                 cfg,
                 train_dataset,
                 val_dataset,
                 unet
                 ):

        # Training Parameter Dictionary
        self.cfg = cfg
        self.project_name = cfg.info.project_name
        self.mode = cfg.train_params.mode.value
        self.save_path = cfg.paths.save
        self.unet = unet
        self.device = cfg.train_params.device.value
        # self.aug_dataset_flag = self.cfg.train_params.aug_dataset_flag
        
        # Fermi operator parameters
        self.fermi_params = {}        
        S = cfg.train_params.pre_scale_factor
        self.fermi_params['S'] = S
        self.fermi_params['S_op'] = rearrange(torch.tensor([1,1/S,S], device=self.device), 'np -> 1 np 1 1')
        self.fermi_params['SH_op'] = rearrange(torch.tensor([1,S,1/S], device=self.device), 'np -> np 1 1')
        self.fermi_params['osamp'] = cfg.train_params.osamp
        
        # Datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Initializing Training Components
        self.val_step_size = cfg.train_params.val_step_size

        # Optimizers
        self.unet_optmzr = torch.optim.Adam(self.unet.parameters(),
                                            lr=cfg.train_params.optimizer.unet_lr,
                                            weight_decay=cfg.train_params.optimizer.unet_wd
                                            )
            
        # Tracker
        self.tracker = Tracker(save_path=str(Path.joinpath(Path(cfg.paths.save), cfg.info.project_name)))

    def model_eval(self, dataset, dsplit):
            
        print(colored('TEST ON ' + dsplit.upper() + ' SET', 'red'))      
        eval_dataloader = DataLoader(dataset,
                                     batch_size=self.cfg.train_params.mb,
                                     shuffle=True,
                                     pin_memory=True)
        
        # Evaluation requires no grad computation
        with torch.no_grad():
            
            ssup_loss = torch.zeros(eval_dataloader.__len__())
            avg_lbfgs_iter = torch.zeros(eval_dataloader.__len__())
            for i, sample_batched in enumerate(eval_dataloader):
                
                # Unpack batched tuple
                im_sig_batch = sample_batched[0].to(self.device)
                ctc_batch = sample_batched[1].to(self.device)
                aif_batch = sample_batched[2].to(self.device)
                time_batch = sample_batched[3].to(self.device)
                wlen_batch = sample_batched[4].to(self.device)
                seg_batch = sample_batched[5].to(self.device)
                
                # Evaluating
                ssup_loss[i] = unet_eval(im_sig_batch,
                                         aif_batch, 
                                         ctc_batch, 
                                         seg_batch, 
                                         time_batch, 
                                         wlen_batch, 
                                         self.unet, 
                                         self.fermi_params)
                avg_lbfgs_iter[i] = self.unet.lbfgs_iter
        
        # Averaging over the dataset
        ssup_loss = ssup_loss.mean()
        avg_lbfgs_iter = avg_lbfgs_iter.mean()
        
        # Printing evaluated value      
        print(colored('{} : {}'.format(dsplit, ssup_loss), 'red'))
        
        return ssup_loss, avg_lbfgs_iter

    def model_train(self):
        
        # the number of trainin samples
        N_train = self.train_dataset.__len__()

        # Make mini-batch size available to the object
        mb = self.cfg.train_params.mb

        # the number of existing mini-batches of size mb (floor in order to avoid to access indices which do not exist;)
        nepochs = self.cfg.train_params.nepochs
        nmb = np.int(np.floor(N_train / mb))
        
        # counter for the backprops;
        n_back_props = nmb * nepochs

        # how often to intermediately check the training/validation error;
        val_step_size = self.val_step_size

        # measure time to see how long the model has been training,
        t0_train = exec_time.time()
        
        # Iterating through epochs
        itr = 0
        for ke in range(nepochs):
            
            # Augumented training dataset to be loaded
            train_dataset_to_load = self.train_dataset.transformed_dataset()
            val_dataset_to_load = self.val_dataset
                        
            # Load data
            train_dataloader = DataLoader(train_dataset_to_load,
                                          batch_size=self.cfg.train_params.mb,
                                          shuffle=True,
                                          num_workers=2,
                                          collate_fn=collate,
                                          prefetch_factor=4,
                                          pin_memory=True)
        
            for iter_batch, batch in enumerate(train_dataloader):
                
                # Unpack batched tuple
                im_sig_batch = batch.im_sig.to(self.device)
                ctc_batch = batch.ctc.to(self.device)
                aif_batch = batch.aif.to(self.device)
                time_batch = batch.time.to(self.device)
                wlen_batch = batch.wlen
                seg_batch = batch.seg.to(self.device)
                mbolus_batch = batch.mbolus.to(self.device)
                eta_pretrain_batch = batch.eta_pretrain.to(self.device)
                mask_od_batch = batch.mask_od.to(self.device)
                
                # Number of time points
                Idst = np.arange(wlen_batch)
                indx_len = Idst.__len__()
                # Subgroup data time points into t_train and t_dc
                ssup_split = self.cfg.train_params.ssup_split
                np.random.shuffle(Idst)
                indx_nn = np.sort(Idst[0:int(ssup_split*indx_len)])
                indx_dc = np.sort(Idst[int(ssup_split*indx_len):-1])                
                
                # Evaluation of network and recording of training parameters
                if itr % val_step_size == 0:
                    ssup_loss_train, avg_lbfgs_iter_train = self.model_eval(train_dataset_to_load, 'Train')
                    ssup_loss_val, avg_lbfgs_iter_val = self.model_eval(val_dataset_to_load, 'Val')                    
                    self.tracker.update_and_save(itr,
                                                ssup_loss_train,
                                                avg_lbfgs_iter_train,
                                                ssup_loss_val,
                                                avg_lbfgs_iter_val,
                                                self.unet)
                    self.tracker.save_ssup_plot()
                    self.tracker.save_lambda_reg_plot()
                    self.tracker.save_avg_lbfgs_iter_plot()
                    with torch.no_grad():
                        self.tracker.save_samp_plot(train_dataset_to_load, self.unet)
                        
                # measure time to get an estimate of how long the training will last
                t0_bp = exec_time.time()

                model_trained_epochs = 'network-training: epoch {} of {}; mini-batch {} out of {}; backprop {}  of {}'.format(
                    ke + 1, nepochs, iter_batch, nmb, itr + 1, n_back_props)
                print(colored(model_trained_epochs, 'yellow'))
                        
                # Network Training
                # Training of DeepFermi
                # Network Training
                if self.mode == 'pre_training':
                    loss = unet_pretrain(im_sig_batch, 
                                         aif_batch, 
                                         ctc_batch, 
                                         seg_batch, 
                                         time_batch, 
                                         wlen_batch, 
                                         indx_nn, 
                                         eta_pretrain_batch, 
                                         mbolus_batch, 
                                         self.unet, 
                                         self.unet_optmzr,
                                         itr)
                elif self.mode == 'fine_tuning':
                    loss = unet_train(im_sig_batch, 
                                      aif_batch, 
                                      ctc_batch, 
                                      seg_batch, 
                                      time_batch, 
                                      wlen_batch, 
                                      indx_nn, 
                                      indx_dc, 
                                      mbolus_batch, 
                                      self.unet,
                                      self.unet_optmzr,
                                      self.fermi_params)
                
                # Training Reporting
                with torch.no_grad():
                    print(colored("ssup loss {}".format(loss.cpu()), 'magenta'))
                
                # Measure the time again and substract how long one weight-update took and also
                t1_bp = exec_time.time() - t0_bp
                t1_train = exec_time.time() - t0_train

                # Print the time in readable format
                est_time = secs2time(t1_bp * n_back_props)
                trained_time = secs2time(t1_train)
                model_trained_time = 'estimated training time: {} ; already trained: {};'.format(est_time, trained_time)
                print(colored(model_trained_time,
                              'cyan'))
                
                # Increment iteration
                itr += 1