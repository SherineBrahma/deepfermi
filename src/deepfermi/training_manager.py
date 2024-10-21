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
        self.tracker = Tracker(save_path=(cfg.paths.save+cfg.info.project_name))

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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        # Number of training samples
        N_train = self.data_dic['im_sig_train'].shape[0]
        Ids = np.arange(N_train)
        train_tsplit = self.train_params['train_tsplit']

        # Make mini-batch size available to the object
        self.mb = mb

        # the number of existing mini-batches of size mb (floor in order to avoid to access indices which do not exist;)
        self.nmb = np.int(np.floor(N_train / self.mb))
        
        # how often to intermediately check the training/validation error;
        val_step_size = self.val_step_size

        # iteration vector (backprops), training and validation loss vectors to be filled;
        if self.train_params['train_from_ckpt']==True and self.train_params['backprop_ckpt']>0:
            
            # Logic for loading the arrays uptil the last checkpoint iteration            
            it_vect_end = np.load(Path.joinpath(self.save_path, 'it_vect_load.npy'))[-1]           
            bp_ckpt_index = self.train_params['backprop_ckpt']//val_step_size            
            load_upper = bp_ckpt_index if bp_ckpt_index <= it_vect_end else it_vect_end
            
            # Loading arrays
            self.it_vect = list(np.load(Path.joinpath(self.save_path, 'it_vect_load.npy'))[:load_upper+1])
            self.ssup_loss_train = list(np.load(Path.joinpath(self.save_path, 'ssup_loss_train_load.npy'))[:load_upper+1])
            self.avg_lbfgs_iter_train = list(np.load(Path.joinpath(self.save_path, 'avg_lbfgs_iter_train_load.npy'))[:load_upper+1])
            self.ssup_loss_val = list(np.load(Path.joinpath(self.save_path, 'ssup_loss_val_load.npy'))[:load_upper+1])
            self.avg_lbfgs_iter_val = list(np.load(Path.joinpath(self.save_path, 'avg_lbfgs_iter_val_load.npy'))[:load_upper+1])
            self.lambda_reg = list(np.load(Path.joinpath(self.save_path, 'lambda_reg_load.npy'))[:load_upper+1])
            
            # counter for the backprops
            backprops_counter = self.it_vect[bp_ckpt_index]
            
            # Epoch start
            epoch_start = backprops_counter//N_train
            
        else:
            # Initializing arrays
            self.it_vect = []
            self.ssup_loss_train = []
            self.avg_lbfgs_iter_train = []
            self.ssup_loss_val = []
            self.avg_lbfgs_iter_val = []        
            self.lambda_reg = []

            # counter for the backprops
            backprops_counter = 0
            
            # Epoch start
            epoch_start = backprops_counter//N_train
            
        # counter for the backprops
        n_backprop = self.nmb * n_epochs

        # measure time to see how long the model has been training,
        t0_train = exe_time.time()
        
        # Training data
        im_sig_train = self.data_dic['im_sig_train'].clone()
        ctc_train = self.data_dic['ctc_train'].clone()
        seg_train = self.data_dic['seg_train'].clone()
        time_train = self.data_dic['time_train'].clone()
        wlen_train = self.data_dic['wlen_train'].clone()
        aif_train = self.data_dic['aif_train'].clone()
        mbolus_train = self.data_dic['mbolus_train'].clone()
        eta_svd_train = self.data_dic['eta_svd_train'].clone()
        mask_od_train = self.data_dic['mask_od_train'].clone()
        if self.aug_data == True:
            aif_aug_train = self.data_dic['aif_aug_train'].clone()
            # time_aug_train = self.data_dic['time_aug_train'].clone()
            eta_aug_train = self.data_dic['eta_aug_train'].clone()
            mask_od_aug_train = self.data_dic['mask_od_aug_train'].clone()
            
        # Exclude outliers and interpolate training-data
        im_sig_train = outlier_fill(im_sig_train, mask_od_train, wlen_train)
        ctc_train = outlier_fill(ctc_train, mask_od_train, wlen_train)
        
        # i = 2
        # ctc = ctc_train[i]
        # seg = seg_train[i]
        # ctc_seg = ctc[seg==1]
        # ctc_seg =  ctc_seg[torch.randperm(ctc[seg==1].size()[0])]        
        # matplotlib.use('TkAgg')
        # plt.figure()
        # plt.title("Concentration Time Curves")
        # plt.plot(ctc_seg[0:40].swapaxes(0,1).detach().cpu(), linewidth=1, linestyle="dashed")
        # plt.ylim((0, 0.4))
        # # plt.plot(anomaly_inject(im_sig)[seg==1].swapaxes(0,1).detach().cpu(), label="im_sig original", linewidth=1, color="red", linestyle="dashed")
        # plt.legend(loc="upper right")   
        # plt.show()
    
        # Pre-allocating memory
        print('Pre-allocating memory')
        # Input tensors for pre-allocation
        im_sig_pre_alloc = torch.tensor(im_sig_train[Ids[0:self.mb]], device=self.device)
        seg_pre_alloc = torch.tensor(seg_train[Ids[0:self.mb]], device=self.device)
        ctc_pre_alloc = torch.tensor(ctc_train[Ids[0:self.mb]], device=self.device)
        time_pre_alloc = torch.tensor(time_train[Ids[0:self.mb]], device=self.device)
        aif_pre_alloc = torch.tensor(aif_train[Ids[0:self.mb]], device=self.device)
        if self.aug_data == True:
            aug_indx = torch.randint(1, N_train, (1,))
            aif_pre_alloc = torch.tensor(aif_train[Ids[0:self.mb],aug_indx], device=self.device)
        # Forward pass
        pre_alloc_out = self.unet(im_sig_pre_alloc.unsqueeze(1), seg_pre_alloc, aif=aif_pre_alloc, ctc=ctc_pre_alloc, time=time_pre_alloc)        
        # Backward pass
        pre_alloc_out.norm().backward()
        # Zeroing gradient
        self.unet.zero_grad()

        # iteratve through epochs
        for ke in range(epoch_start, n_epochs):

            # Shuffle the set of training indices;
            np.random.shuffle(Ids)
            
            # Apply data-augumentation            
            for i in range(N_train):
                rot_angle = random.choice([-90., 0., 90., 180.])
                hfilp = random.choice([TF.hflip, torch.nn.Identity()])
                im_sig_train[i] = hfilp(TF.rotate(im_sig_train[i].moveaxis(-1, -3), angle=rot_angle)).moveaxis(-3, -1)
                ctc_train[i] = hfilp(TF.rotate(ctc_train[i].moveaxis(-1, -3), angle=rot_angle)).moveaxis(-3, -1)
                seg_train[i] = hfilp(TF.rotate(seg_train[i].unsqueeze(0), angle=rot_angle))
                if self.mode=='pre_training':
                    if self.aug_data == True:
                        eta_aug_train[i] = hfilp(TF.rotate(eta_aug_train[i], angle=rot_angle))
                    else:
                        eta_svd_train[i] = hfilp(TF.rotate(eta_svd_train[i], angle=rot_angle))
                    
                # Randomly choose augumented perfusion data
                if self.aug_data == True:
                    reject_list = [(0, 9), (0,10), (0,11), (3,9), (3,10), (3,11)]
                    indx = np.arange(0, N_train)
                    aug_indx = np.zeros(indx.shape, dtype=int)
                    for i in range(N_train):
                        aug_indx[i] = np.random.randint(0, N_train)
                        while (indx[i], aug_indx[i]) in reject_list:
                            aug_indx[i] = np.random.randint(0, N_train)                        
                    aif_train = aif_aug_train[indx,aug_indx].clone()
                    # time_train = time_aug_train[indx,aug_indx].clone()
                    eta_svd_train = eta_aug_train[indx,aug_indx].clone()
                    mask_od_train = mask_od_aug_train[indx,aug_indx].clone()
                    
                # for i in range(eta_aug_train.shape[0]):
                #     for j in range(eta_aug_train.shape[0]):
                #         if torch.isnan((eta_aug_train[i,j].sum())):
                #             print('i : ' + str(i) + ', j : ' + str(j) + ', value : ' + str((eta_aug_train[i,j].sum()).item()))
                
                # tensor = eta_aug_train[3:,:,1,...].flatten()
                # tensor = tensor[tensor!=0]
                # matplotlib.use('TkAgg')
                # plt.figure()
                # plt.hist(tensor.numpy(), bins=5000, color='blue', alpha=0.7)
                # plt.title('Histogram of Delay')
                # plt.xlabel('Values')
                # plt.ylabel('Frequency')
                # plt.show()
                

            # Iterate through mini-batches
            for kb in range(self.nmb):

                # get the kb-th minibatch of the data;
                im_sig = torch.tensor(im_sig_train[Ids[kb * self.mb:(kb + 1) * self.mb], :], device=self.device)                
                ctc = torch.tensor(ctc_train[Ids[kb * self.mb:(kb + 1) * self.mb], :], device=self.device)
                seg = torch.tensor(seg_train[Ids[kb * self.mb:(kb + 1) * self.mb], :], device=self.device)
                time = torch.tensor(time_train[Ids[kb * self.mb:(kb + 1) * self.mb], :], device=self.device)
                wlen = wlen_train[Ids[kb * self.mb:(kb + 1) * self.mb]]
                aif = torch.tensor(aif_train[Ids[kb * self.mb:(kb + 1) * self.mb], :], device=self.device)
                mbolus = torch.tensor(mbolus_train[Ids[kb * self.mb:(kb + 1) * self.mb], :], device=self.device)
                mask_od = mask_od_train[Ids[kb * self.mb:(kb + 1) * self.mb], :]
                eta_svd = torch.tensor(eta_svd_train[Ids[kb * self.mb:(kb + 1) * self.mb], :], device=self.device)
                               
                # # Number of time points        
                # Idst = np.arange(wlen)
                # Idst = Idst[mask_od[0,:wlen]==1]
                # indx_len = Idst.__len__()                
                # # Subgroup data time points into t_train and t_dc 
                # np.random.shuffle(Idst)
                # indx_nn = torch.tensor(np.sort(Idst[0:int(train_tsplit*indx_len)]), device=self.device)
                # indx_dc = torch.tensor(np.sort(Idst[int(train_tsplit*indx_len):-1]), device=self.device)
                
                # Number of time points
                Idst = np.arange(wlen)
                indx_len = Idst.__len__()
                # Subgroup data time points into t_train and t_dc
                np.random.shuffle(Idst)
                # Clean NN input
                Idst_nn = np.sort(Idst[0:int(train_tsplit*indx_len)])
                # mask_od_nn = mask_od[0,:wlen][Idst_nn]
                # indx_nn = torch.tensor(Idst_nn[mask_od_nn==1], device=self.device)
                indx_nn = Idst_nn # torch.tensor(Idst_nn, device=self.device)
                # Keep label outliers
                Idst_dc = np.sort(Idst[int(train_tsplit*indx_len):-1])
                indx_dc = Idst_dc # torch.tensor(Idst_dc, device=self.device)
                
                # measure time to get an estimate of how long the training will last
                t0_bp = exe_time.time()
                # Evaluation of network and recording of training parameters
                if backprops_counter % val_step_size == 0:
                    print(colored('TEST ON TRAINING SET', 'red'))
                    ssup_loss_train, avg_lbfgs_iter_train = self.eval(dsplit='train')
                    print(colored('net_train self-supervised loss: {}'.format(ssup_loss_train), 'red'))
                    
                    print(colored('TEST ON VALIDATION TEST', 'red'))
                    ssup_loss_val, avg_lbfgs_iter_val = self.eval(dsplit='val')
                    print(colored('net_val self-supervised loss: {}'.format(ssup_loss_val), 'red'))

                    # Append the measures to the corresponding vectors;
                    self.it_vect.append(backprops_counter)
                    self.ssup_loss_train.append(ssup_loss_train)
                    self.avg_lbfgs_iter_train.append(avg_lbfgs_iter_train)
                    self.ssup_loss_val.append(ssup_loss_val)
                    self.avg_lbfgs_iter_val.append(avg_lbfgs_iter_val)

                    # Save curves and models
                    self.save(self.ssup_loss_train, save_obj_type='var', save_obj_name='ssup_loss_train')
                    self.save(self.avg_lbfgs_iter_train, save_obj_type='var', save_obj_name='avg_lbfgs_iter_train')
                    self.save(self.ssup_loss_val, save_obj_type='var', save_obj_name='ssup_loss_val')
                    self.save(self.avg_lbfgs_iter_val, save_obj_type='var', save_obj_name='avg_lbfgs_iter_val')
                    self.save(self.it_vect, save_obj_type='var', save_obj_name='it_vect')
                    self.save(self.unet, save_obj_type='model', save_obj_name='unet')
                    unet_name = 'unet_iter_' + str(backprops_counter)
                    self.save(self.unet, save_obj_type='model', save_obj_name=unet_name, sub_folder='unet_model')
                    
                    self.lambda_reg.append(self.unet.lambda_reg.item())
                    self.save(self.lambda_reg, save_obj_type='var', save_obj_name='lambda_reg')
                    
                    # self.lambda_reg.append(self.unet.lambda_reg[:,:, 0, 0].detach().cpu())
                    # self.save(torch.cat(self.lambda_reg, dim=0), save_obj_type='var', save_obj_name='lambda_reg')
                    
                    # # Save plots
                    self.report.ssup_loss_curve()
                    self.report.avg_lbfgs_iter_curve()
                    with torch.no_grad():
                        self.report.post_samp_plot(self.data_dic, self.unet, dsplit='train')
                        self.report.post_samp_plot(self.data_dic, self.unet, dsplit='val')
                        self.report.post_samp_plot(self.data_dic, self.unet, dsplit='test', mno_increment=True)
                
                # Network Training
                if self.mode == 'pre_training':
                    loss = self.train_utils.unet_pre_train(im_sig, aif, ctc, seg, time, wlen, indx_nn, eta_svd, mbolus)
                elif self.mode == 'fine_tuning':
                    loss = self.train_utils.unet_loss(im_sig, aif, ctc, seg, time, wlen, indx_nn, indx_dc, mbolus)

                # Training Reporting
                print(colored("loss {}".format(loss.cpu()), 'magenta'))
                # Measure the time again and substract how long one weight-update took and also
                t1_bp = exe_time.time() - t0_bp
                t1_train = exe_time.time() - t0_train

                # Print the time in readable format
                est_time = secs2time(t1_bp * n_backprop)
                trained_time = secs2time(t1_train)
                model_trained_epochs = 'network-training: epoch {} of {}; mini-batch {} out of {}; backprop {}  of {}'.format(ke + 1, n_epochs, kb+1, self.nmb, backprops_counter+1, n_backprop)
                print(colored(model_trained_epochs, 'yellow'))
                model_trained_time = 'estimated training time: {} ; already trained: {};'.format(est_time, trained_time)
                print(colored(model_trained_time, 'cyan'))

                if backprops_counter % val_step_size == 0 and backprops_counter != 0:
                    self.save(model_trained_epochs+'\n', save_obj_type='text', save_obj_name='training_time.txt', mode='a')
                    self.save(model_trained_time+'\n', save_obj_type='text', save_obj_name='training_time.txt', mode='a')
                elif backprops_counter % val_step_size == 0 and backprops_counter == 0:
                    self.save(model_trained_epochs + '\n', save_obj_type='text', save_obj_name='training_time.txt',  mode='w')
                    self.save(model_trained_time + '\n', save_obj_type='text', save_obj_name='training_time.txt', mode='w')

                backprops_counter += 1
