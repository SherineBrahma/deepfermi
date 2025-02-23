import itertools
import random
import time as exe_time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from einops import rearrange
from fermi import FermiLBFGSSolver, convolve, fermi_ir_func, shift_aif
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from tqdm import trange
from utils import Interp_Linear_1D


class DatasetDCEPerfusion(Dataset):
    """
    Dataset class for dynamic contrast-enhanced (DCE) perfusion data.

    Attributes:
        pid (list): List of patient IDs.
        im_sig (tensor): Image signature data.
        ctc (tensor): Concentration-time curve data.
        aif (tensor): Arterial input function data.
        time (tensor): Time data for the measurements.
        wlen (tensor): Wavelength data.
        seg (tensor): Segmentation data.
        eta_gnd (tensor): Ground truth perfusion data.
        mbolus (tensor): Main bolus data for the perfusion.
        eta_pretrain (tensor): Pretraining estimates of perfusion.
        mask_od (tensor): Mask for outlier detection.
        transform (callable): Transform function to preprocess data.
        aug_data_dic (dict): Augmented data dictionary.
    """

    def __init__(self,
                 pid,
                 im_sig,
                 ctc,
                 aif,
                 time,
                 wlen,
                 seg,
                 eta_gnd,
                 mbolus,
                 eta_pretrain,
                 mask_od,
                 aug_data_dic,
                 transform):
        """
        Initializes the dataset with the given parameters.
        """

        self.pid = pid
        self.im_sig = im_sig
        self.ctc = ctc
        self.aif = aif
        self.time = time
        self.wlen = wlen
        self.seg = seg
        self.eta_gnd = eta_gnd
        self.mbolus = mbolus
        self.eta_pretrain = eta_pretrain
        self.mask_od = mask_od
        self.transform = transform
        self.aug_data_dic = aug_data_dic

    def __len__(self) -> int:
        """
        Returns the total number of slices in the dataset.
        
        Returns:
            int: The total number of slices in the dataset.
        """

        total_slices = self.pid.__len__()

        return total_slices

    def __getitem__(self, idx):
        """
        Returns a specific sample from the dataset given an index.
        
        Parameters:
            idx (int): The index of the sample to retrieve.
        
        Returns:
            tuple: A tuple containing the transformed data.
        """

        return (self.im_sig[idx],
                self.ctc[idx],
                self.aif[idx],
                self.time[idx],
                self.wlen[idx],
                self.seg[idx],
                self.mbolus[idx],
                self.eta_pretrain[idx],
                self.mask_od[idx])

    def transformed_dataset(self):
        """
        Applies transformation to the dataset and returns a new transformed 
        dataset.
        
        Returns:
            DatasetDCEPerfusion: The transformed dataset.
        """

        # Transforming dataset
        transformed_array = self.transform(self.im_sig,
                                           self.ctc,
                                           self.aif,
                                           self.time,
                                           self.seg,
                                           self.eta_gnd,
                                           self.eta_pretrain,
                                           self.mask_od,
                                           self.aug_data_dic)
        # Crop data
        im_sig = transformed_array[0]
        ctc = transformed_array[1]
        aif = transformed_array[2]
        time = transformed_array[3]
        seg = transformed_array[4]
        eta_gnd = transformed_array[5]
        eta_pretrain = transformed_array[6]
        mask_od = transformed_array[7]

        return DatasetDCEPerfusion(self.pid,
                                   im_sig,
                                   ctc,
                                   aif,
                                   time,
                                   self.wlen,
                                   seg,
                                   eta_gnd,
                                   self.mbolus,
                                   eta_pretrain,
                                   mask_od,
                                   self.aug_data_dic,
                                   self.transform)

    @classmethod
    def construct_from_npz(cls,
                           file_path,
                           transform,
                           pid_to_load,
                           config,
                           aug_dataset_flag=False,
                           od_enable=True,
                           device='cuda'):
        """
        Constructs a DatasetDCEPerfusion object from an NPZ file.

        Parameters:
            file_path (str): Path to the NPZ file.
            transform (callable): Transform function to preprocess data.
            pid_to_load (list): List of patient IDs to load.
            config (TrainConfig): Configuration object containing settings for 
                                  the dataset.
            aug_dataset_flag (bool, optional): Flag to indicate whether to 
                                               augment the dataset. Defaults 
                                               to False.
            od_enable (bool, optional): Flag to enable outlier detection.
                                        Defaults to True.
            device (str, optional): The device to store the tensors on. 
                                    Defaults to 'cuda'.

        Returns:
            DatasetDCEPerfusion: The constructed dataset.
        """

        # General settings
        osamp = (
            config.train_params.osamp
            if hasattr(config, 'train_params')
            else config.test_params.osamp
            )
        eta_bkg_ref = (
            config.train_params.dataset.eta_bkg_ref
            if hasattr(config, 'train_params')
            else config.test_params.dataset.eta_bkg_ref
            )
        cross_val_flag = (
            config.train_params.cross_val_flag
            if hasattr(config, 'train_params')
            else False
            )
        crop_dim = (
            config.train_params.dataset.crop_dim
            if hasattr(config, 'train_params')
            else config.test_params.dataset.crop_dim
            )
        dtype = torch.float32

        # Extracting pre-processing parameters
        if cross_val_flag is True:
            k_val = config.train_params.cross_val_k
            fold = config.train_params.cross_val_fold
            combined_pid_list = (config.train_params.nval
                                 + config.train_params.ntrain)
            kf = KFold(n_splits=k_val)
            if pid_to_load is config.train_params.ntrain:
                print('Building Training Set...')
                pid_to_load = [
                    combined_pid_list[i]
                    for i in list(kf.split(combined_pid_list))[fold-1][0]
                    ]
            elif pid_to_load is config.train_params.nval:
                print('Building Validation Set...')
                pid_to_load = [
                    combined_pid_list[i]
                    for i in list(kf.split(combined_pid_list))[fold-1][1]
                    ]

        # Construct dataset dictionary
        data_dic = DatasetDCEPerfusion._read_from_npz(file_path, pid_to_load)

        # Converting to tensor data-types
        # Initilize place-holders
        pid_list = []
        im_sig_list = []
        ctc_list = []
        aif_list = []
        time_list = []
        wlen_list = []
        seg_list = []
        eta_gnd_list = []
        for pid, data in data_dic.items():
            # Extract data
            im_sig = torch.tensor(data['im_sig'], dtype=dtype)
            ctc = torch.tensor(data['ctc'], dtype=dtype)
            aif = torch.tensor(data['aif'], dtype=dtype)
            time = torch.tensor(data['time'], dtype=dtype)
            wlen = torch.tensor(data['wlen'], dtype=torch.int)
            seg = torch.tensor(data['seg'], dtype=dtype)
            eta_gnd = torch.tensor(data['eta'], dtype=dtype)
            # Populating the place-holders
            pid_list.append(pid)
            im_sig_list.append(im_sig)
            ctc_list.append(ctc)
            aif_list.append(aif)
            time_list.append(time)
            wlen_list.append(wlen)
            seg_list.append(seg)
            eta_gnd_list.append(eta_gnd)
        # Stacking the list to construct tensors
        pid = pid_list
        im_sig = torch.stack(im_sig_list)
        ctc = torch.stack(ctc_list)
        aif = torch.stack(aif_list)
        time = torch.stack(time_list)
        wlen = torch.stack(wlen_list)
        seg = torch.stack(seg_list)
        eta_gnd = torch.stack(eta_gnd_list)

        # Preprocessing steps
        # Data-preprocessing
        # Cropping dataset
        ctc, seg, im_sig, eta_gnd = DatasetDCEPerfusion._crop_dim(ctc,
                                                                  seg,
                                                                  im_sig,
                                                                  eta_gnd,
                                                                  crop_dim)
        # Precondition ctc background
        ctc = DatasetDCEPerfusion._precond_bkg(aif,
                                               ctc,
                                               seg,
                                               time,
                                               wlen,
                                               osamp,
                                               eta_bkg_ref,
                                               device)
        # Calculate main bolus
        mbolus = DatasetDCEPerfusion._calc_mbolus(im_sig, seg)
        # Determining estimates for pretraining
        # Calculate conventional LBFGS estimates without deep learning
        lbfgs_est_data = DatasetDCEPerfusion._calc_lbfgs_estimates(
            pid,
            ctc,
            aif,
            mbolus,
            seg,
            time,
            wlen,
            osamp,
            od_enable=od_enable
            )
        eta_pretrain, mask_od, aif, _ = lbfgs_est_data
        # Precondition eta_pretrain background
        eta_pretrain = DatasetDCEPerfusion._precond_eta_pretrain(eta_pretrain,
                                                                 seg,
                                                                 eta_bkg_ref)
        # Augumented dataset generation
        if aug_dataset_flag is True:
            print('Augumenting Dataset...')
            # Calculate conventional LBFGS estimates without deep learning
            aug_data = DatasetDCEPerfusion._aug_dataset(
                pid,
                ctc,
                aif,
                mbolus,
                seg,
                time,
                wlen,
                osamp
                )
            eta_pretrain_aug, mask_od_aug, aif_aug, time_aug = aug_data
            # Precondition eta_pretrain background
            eta_pretrain_aug = DatasetDCEPerfusion._precond_eta_pretrain_aug(
                eta_pretrain_aug,
                seg,
                eta_bkg_ref
                )
            # Construct
            aug_data_dic = {}
            aug_data_dic['eta_pretrain_aug'] = eta_pretrain_aug
            aug_data_dic['mask_od_aug'] = mask_od_aug
            aug_data_dic['aif_aug'] = aif_aug
            aug_data_dic['time_aug'] = time_aug
        else:
            aug_data_dic = None

        return DatasetDCEPerfusion(pid,
                                   im_sig,
                                   ctc,
                                   aif,
                                   time,
                                   wlen,
                                   seg,
                                   eta_gnd,
                                   mbolus,
                                   eta_pretrain,
                                   mask_od,
                                   aug_data_dic,
                                   transform)

    @staticmethod
    def _read_from_npz(file_path,
                       pid_to_load=None):

        # Loading and constructing data dictionary
        data = np.load(file_path, allow_pickle=True)
        data_dic = {}
        for pid in data.keys():
            if pid in pid_to_load:
                data_dic[pid] = data[pid].item()

        return data_dic

    @staticmethod
    def _stack_myo_slices(pid,
                          im_sig,
                          ctc,
                          aif,
                          time,
                          wlen,
                          seg):

        # Stack the slices
        im_sig = rearrange(
            im_sig, 'npat nt nx ny nslc -> (npat nslc) nx ny nt'
            )
        ctc = rearrange(
            ctc, 'npat nt nx ny nslc -> (npat nslc) nx ny nt'
            )
        aif = rearrange(
            aif, 'npat nt nslc -> (npat nslc) nt'
            )
        wlen = rearrange(
            wlen, 'npat nslc -> (npat nslc)'
            )
        seg = rearrange(
            seg, 'npat nx ny nslc -> (npat nslc) nx ny'
            )

        # Repeat common attributes for each slices
        pid = list(itertools.chain.from_iterable(zip(pid, pid, pid)))
        time = time.repeat_interleave(3, dim=0)

        return pid, im_sig, ctc, aif, time, wlen, seg

    @staticmethod
    def _crop_dim(ctc,
                  seg,
                  im_sig,
                  eta_gnd,
                  crop_dim):

        # General
        crop_dim_x, crop_dim_y = crop_dim[0], crop_dim[1]
        nb, npar, _, _ = eta_gnd.shape
        _, _, _, nt = im_sig.shape
        eta_gnd_data = torch.zeros(nb,
                                   npar,
                                   crop_dim_x,
                                   crop_dim_y,
                                   dtype=eta_gnd.dtype)
        im_sig_data = torch.zeros(nb,
                                  crop_dim_x,
                                  crop_dim_y,
                                  nt,
                                  dtype=im_sig.dtype)
        ctc_data = torch.zeros(nb,
                               crop_dim_x,
                               crop_dim_y,
                               nt,
                               dtype=ctc.dtype)
        seg_data = torch.zeros(nb,
                               crop_dim_x,
                               crop_dim_y,
                               dtype=seg.dtype)
        for i in range(seg.shape[0]):
            # Bounding box
            # Construction
            bbox = torch.zeros(seg[i, ...].shape)
            bbox[seg[i, ...] == 1] = 1
            bbox[seg[i, ...] == 71] = 1
            bbox_x = bbox.sum(1)
            bbox_y = bbox.sum(0)
            bbox = (bbox_x.unsqueeze(1)) * (bbox_y.unsqueeze(0))
            bbox[bbox != 0] = 1
            # Retrieving indices
            xmin, xmax = np.where(bbox_x)[0].min(), np.where(bbox_x)[0].max()
            ymin, ymax = np.where(bbox_y)[0].min(), np.where(bbox_y)[0].max()
            # Padding logic
            if (crop_dim_x-(xmax-xmin)) % 2 == 0:
                xpad_left, xpad_right = (int((crop_dim_x - (xmax-xmin))/2),
                                         int((crop_dim_x - (xmax-xmin))/2))
            else:
                if (crop_dim_x-(xmax-xmin)) > 0:
                    xpad_left, xpad_right = (int((crop_dim_x - (xmax-xmin))/2)
                                             + 1,
                                             int((crop_dim_x - (xmax-xmin))/2))
                else:
                    xpad_left, xpad_right = (int((crop_dim_x - (xmax-xmin))/2),
                                             int((crop_dim_x - (xmax-xmin))/2)
                                             - 1)
            if (crop_dim_y-(ymax-ymin)) % 2 == 0:
                ypad_left, ypad_right = (int((crop_dim_y - (ymax-ymin))/2),
                                         int((crop_dim_y - (ymax-ymin))/2))
            else:
                if (crop_dim_y-(ymax-ymin)) > 0:
                    ypad_left, ypad_right = (int((crop_dim_y - (ymax-ymin))/2)
                                             + 1,
                                             int((crop_dim_y - (ymax-ymin))/2))
                else:
                    ypad_left, ypad_right = (int((crop_dim_y - (ymax-ymin))/2),
                                             int((crop_dim_y - (ymax-ymin))/2)
                                             - 1)
            # Padded indices
            xmin, xmax = xmin - xpad_left, xmax + xpad_right
            ymin, ymax = ymin - ypad_left, ymax + ypad_right

            # Crop maps
            eta_gnd_data[i, ...] = eta_gnd[i, :, xmin:xmax, ymin:ymax]
            im_sig_data[i, ...] = im_sig[i, xmin:xmax, ymin:ymax, :]
            ctc_data[i, ...] = ctc[i, xmin:xmax, ymin:ymax, :]
            seg_data[i, ...] = seg[i, xmin:xmax, ymin:ymax]
        return ctc_data, seg_data, im_sig_data, eta_gnd_data

    @staticmethod
    def _precond_bkg(aif,
                     ctc,
                     seg,
                     time,
                     wlen,
                     osamp,
                     eta_bkg_ref,
                     device='cuda'):
        """
        Updates the background of the concentration-time curves (CTC) to match
        changes in the perfusion parameters background using the specified
        reference values. This function ensures that any modifications to the
        perfusion background are reflected in the corresponding CTC.
        """

        # General initializing
        S = 10
        S_op = rearrange(
            torch.tensor([1, 1/S, S], device=device),
            'np -> 1 np 1 1'
            )

        # Initializing background perfusion parameters
        flow_bkg = eta_bkg_ref[0]
        delay_bkg = eta_bkg_ref[1]
        decay_bkg = eta_bkg_ref[2]
        eta_bkg = S_op * rearrange(
            torch.tensor([flow_bkg, delay_bkg, decay_bkg], device=device),
            'np -> 1 np 1 1'
            )

        for i in range(ctc.shape[0]):

            # Extracting slice aif, ctc and segmentation
            wlen_bkg = wlen[i:i+1]
            aif_bkg = torch.tensor(
                aif[i:i+1, ..., 0:wlen_bkg].unsqueeze(1).unsqueeze(1),
                device=device
                )
            ctc_full = ctc[i:i+1, ..., 0:wlen_bkg]
            seg_bkg = seg[i:i+1]
            time_bkg = torch.tensor(
                time[i:i+1, 0:wlen_bkg]/S, device=device
                )

            # Compensating offset in the time curves
            oTp = 5
            aif_bkg = aif_bkg-aif_bkg[..., 0:oTp].mean(-1, keepdim=True)

            # Interpolating vectors
            aif_bkg = Interp_Linear_1D(
                aif_bkg, size=osamp*aif_bkg.shape[-1]
                )
            time_bkg = Interp_Linear_1D(
                time_bkg.unsqueeze(0),
                size=(osamp * time_bkg.shape[-1])
                )[0]

            # Generating concentration curves
            fermi_ir = fermi_ir_func(eta_bkg, time_bkg.squeeze())
            ctc_bkg = convolve(aif_bkg, fermi_ir)[..., ::osamp]/osamp

            # Assigning background concentration curves
            ctc_full[seg_bkg == 0] = ctc_bkg.squeeze().cpu()
            ctc[i:i+1, ..., 0:wlen_bkg] = ctc_full
        return ctc

    @staticmethod
    def _calc_mbolus(im_sig, seg):
        # Calculating the main bolus to be used for outlier injection during
        # training
        mbolus = torch.zeros((im_sig.shape[0], im_sig.shape[-1]))
        for i in range(0, im_sig.shape[0]):
            seg_i = seg[i, ...].to(torch.torch.uint8).numpy()
            im_sig_i = im_sig[i, ...]
            contours, _ = cv2.findContours(
                seg_i, cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
                )
            fill = (1, 1, 1)
            mask_i = torch.tensor(
                cv2.fillPoly(seg_i.copy(), contours, fill),
                dtype=im_sig_i.dtype
                )
            lv_mask = (mask_i - seg_i).to(torch.float).unsqueeze(-1)
            lv_sig = (lv_mask * im_sig_i)
            max_index = (
                lv_sig.mean(-1) == torch.max(lv_sig.mean(-1))
                ).nonzero()
            mbolus[i] = im_sig_i[max_index[0][0], max_index[0][1], ...]
        return mbolus

    @staticmethod
    def _calc_lbfgs_estimates(pid,
                              ctc,
                              aif,
                              mbolus,
                              seg,
                              time,
                              wlen,
                              osamp,
                              flow_init=0.05,
                              delay_init=2,
                              decay_init=0.1,
                              od_enable=True,
                              device='cuda'):

        # General initializations
        time_taken = []
        in_device = ctc.device
        ctc = ctc.to(device)
        aif = aif.to(device)
        mbolus = mbolus.to(device)
        seg = seg.to(device)
        time = time.to(device)
        mask_od = torch.zeros(time.shape, device=device)
        mb, xdim, ydim = ctc.shape[0], ctc.shape[1], ctc.shape[2]
        eta_dim = [mb, 3, xdim, ydim]
        eta = seg.unsqueeze(1) * torch.ones(eta_dim, device=device)
        init_val = rearrange(
            torch.tensor([flow_init, delay_init, decay_init],
                         device=device),
            'np -> 1 np 1 1'
            )
        fermi_lbfgs_solver = FermiLBFGSSolver(
            osamp=osamp, od_enable=od_enable
            ).to(device)
        pbar = trange(mb, desc='PID', leave=True)
        for i in pbar:
            pbar.set_description(
                'Progress (Current PID = ' + str(pid[i]) + ')'
                )
            pbar.refresh()
            wlen_i = wlen[i]
            ctc_i = ctc[i:i+1, ..., 0:wlen_i]
            seg_i = seg[i:i+1, ...]
            # Aligining AIF
            aif_i = aif[i:i+1, ..., 0:wlen_i]
            time_i = time[i, 0:wlen_i]
            mbolus_i = mbolus[i:i+1, 0:wlen_i]
            aif_i = shift_aif(aif_i, mbolus_i, time_i, osamp)
            aif_i_2D = (
                aif_i.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                * torch.ones(ctc_i.shape, device=aif_i.device)
                )
            with torch.no_grad():
                eta_init = init_val * eta[i:i+1, ...].clone()
                t0 = exe_time.time()
                eta[i, ...], mask_od[i, 0:wlen_i] = fermi_lbfgs_solver(
                    eta_init,
                    ctc_i,
                    aif_i_2D,
                    seg_i,
                    time_i)
                time_taken.append(exe_time.time()-t0)
                aif[i, 0:wlen_i] = aif_i
            pbar.set_description('Progress')
            pbar.refresh()
        eta = eta.to(in_device)
        mask_od = mask_od.to(in_device)
        aif = aif.to(in_device)
        torch.cuda.empty_cache()

        return eta, mask_od, aif, np.mean(time_taken)

    @staticmethod
    def _precond_eta_pretrain(eta, seg, eta_bkg_ref):
        # Change the background of the perfusion parameters
        for i in range(seg.shape[0]):
            seg_bkg = seg[i:i+1]
            eta_bkg = eta[i:i+1]
            eta_bkg[:, 0, ...][seg_bkg == 0] = eta_bkg_ref[0]
            eta_bkg[:, 1, ...][seg_bkg == 0] = eta_bkg_ref[1]
            eta_bkg[:, 2, ...][seg_bkg == 0] = eta_bkg_ref[2]
            eta[i:i+1] = eta_bkg

        return eta

    @staticmethod
    def _aug_dataset(pid,
                     ctc,
                     aif,
                     mbolus,
                     seg,
                     time,
                     wlen,
                     osamp,
                     flow_init=0.05,
                     delay_init=2,
                     decay_init=0.1,
                     device='cuda'):

        # General initializations
        in_device = ctc.device
        ctc = ctc.to(device)
        aif = aif.to(device)
        mbolus = mbolus.to(device)
        seg = seg.to(device)
        time = time.to(device)
        mb, xdim, ydim, tdim = (ctc.shape[0],
                                ctc.shape[1],
                                ctc.shape[2],
                                ctc.shape[3])
        eta_dim = [mb, mb, 3, xdim, ydim]
        eta_aug = (seg.unsqueeze(1).unsqueeze(1)
                   * torch.ones(eta_dim, device=device))
        mask_od = torch.zeros([mb, mb, tdim], device=device)
        aif_aug = torch.zeros([mb, mb, tdim], device=device)
        time_aug = torch.zeros([mb, mb, tdim], device=device)
        init_val = rearrange(
            torch.tensor([flow_init, delay_init, decay_init], device=device),
            'np -> 1 np 1 1'
            )
        fermi_lbfgs_solver = FermiLBFGSSolver(
            osamp=osamp, od_enable=True
            ).to(device)
        pbar_outer = trange(mb, desc='PID', leave=True, position=0)
        for i in pbar_outer:
            pbar_msg = 'Progress (Current Outer PID = ' + str(pid[i]) + ')'
            pbar_outer.set_description(pbar_msg)
            pbar_outer.refresh()
            # for i in tqdm(range(mb), desc=" outer", position=0):
            wlen_i = wlen[i]
            ctc_i = ctc[i:i+1, ..., 0:wlen_i]
            seg_i = seg[i:i+1, ...]
            time_i = time[i, 0:wlen_i]
            mbolus_i = mbolus[i:i+1, 0:wlen_i]
            # Calculated augmented pairs
            pbar_inner = trange(mb, desc='PID ', leave=False, position=1)
            for j in pbar_inner:
                pbar_inner_msg = (
                    'AIF Exchange Progress (Current Inner PID = '
                    + str(pid[i])
                    + ')'
                    )
                pbar_inner.set_description(pbar_inner_msg)
                pbar_inner.refresh()
                aif_j = aif[j:j+1, ..., 0:wlen_i]
                aif_ij = shift_aif(aif_j, mbolus_i, time_i, osamp)
                aif_ij_2D = (aif_ij.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                             * torch.ones(aif_ij.shape, device=aif_ij.device))
                with torch.no_grad():
                    eta_init = init_val * eta_aug[i:i+1, j, ...].clone()
                    deconv_data = fermi_lbfgs_solver(
                        eta_init,
                        ctc_i,
                        aif_ij_2D,
                        seg_i,
                        time_i
                        )
                    eta_aug[i, j, ...], mask_od[i, j, 0:wlen_i] = deconv_data
                    aif_aug[i, j, 0:wlen_i] = aif_ij
                    time_aug[i, j, 0:wlen_i] = time_i

                    # import dataio
                    # import matplotlib
                    # import matplotlib.pyplot as plt
                    # matplotlib.use('TkAgg')
                    # plt.figure()
                    # plt.title("Concentration Time Curves")
                    # plt.plot(time_i.cpu(),
                    #          mbolus_i.squeeze(0).cpu(),
                    #          label="mbolus",
                    #          linewidth=1, color="blue",
                    #          linestyle="solid")
                    # plt.plot(time_i.cpu(),
                    #          aif_ij.squeeze(0).cpu(),
                    #          label="aif_shifted",
                    #          linewidth=1,
                    #          color="red",
                    #          linestyle="solid")
                    # plt.plot(time_i.cpu(),
                    #          aif_j.squeeze(0).cpu(),
                    #          label="aif",
                    #          linewidth=1,
                    #          color="black",
                    #          linestyle="solid")
                    # plt.legend(loc="upper right")
                    # plt.show()

            pbar_outer.set_description('Progress')
            pbar_outer.refresh()

        eta_aug = eta_aug.to(in_device)
        mask_od = mask_od.to(in_device)
        aif_aug = aif_aug.to(in_device)
        time_aug = time_aug.to(in_device)
        torch.cuda.empty_cache()

        return eta_aug, mask_od, aif_aug, time_aug

    @staticmethod
    def _precond_eta_pretrain_aug(eta,
                                  seg,
                                  eta_bkg_ref):
        """
        Modifies the background of the pretraining perfusion parameters using
        the specified reference. The reference values should be distinct from
        the actual perfusion parameters to help the network properly
        differentiate between the foreground and background during training.
        """
        # Change the background of the perfusion parameters
        for i in range(seg.shape[0]):
            # Extracting slice aif, ctc and segmentation
            seg_bkg = seg[i:i+1].unsqueeze(1).repeat(1,
                                                     eta.shape[1],
                                                     1,
                                                     1)
            eta_bkg = eta[i:i+1]
            eta_bkg[:, :, 0, ...][seg_bkg == 0] = eta_bkg_ref[0]
            eta_bkg[:, :, 1, ...][seg_bkg == 0] = eta_bkg_ref[1]
            eta_bkg[:, :, 2, ...][seg_bkg == 0] = eta_bkg_ref[2]
            eta[i:i+1] = eta_bkg

        return eta

    @staticmethod
    def update_precond_bkg_ref(dataset,
                               config,
                               aug_dataset_flag=False,
                               device='cuda'):

        # General settings
        osamp = (
            config.train_params.osamp
            if hasattr(config, 'train_params')
            else config.test_params.osamp
            )
        eta_bkg_ref = (
            config.train_params.dataset.eta_bkg_ref
            if hasattr(config, 'train_params')
            else config.test_params.dataset.eta_bkg_ref
            )

        # Initialization
        aif = dataset.aif
        ctc = dataset.ctc
        seg = dataset.seg
        time = dataset.time
        wlen = dataset.wlen
        eta_pretrain = dataset.eta_pretrain
        
        # Precondition ctc background
        dataset.ctc = DatasetDCEPerfusion._precond_bkg(aif,
                                                       ctc,
                                                       seg,
                                                       time,
                                                       wlen,
                                                       osamp,
                                                       eta_bkg_ref,
                                                       device)
        # Precondition eta_pretrain background
        dataset.eta_pretrain = DatasetDCEPerfusion._precond_eta_pretrain(
            eta_pretrain,
            seg,
            eta_bkg_ref
            )
        # Augumented dataset generation
        if aug_dataset_flag is True:
            eta_pretrain_aug = dataset.aug_data_dic['eta_pretrain_aug']
            # Precondition augumented eta_pretrain background
            eta_pretrain_aug = DatasetDCEPerfusion._precond_eta_pretrain_aug(
                eta_pretrain_aug,
                seg,
                eta_bkg_ref
                )
            dataset.aug_data_dic['eta_pretrain_aug'] = eta_pretrain_aug

        return dataset


class Transform(nn.Module):
    """
    Class for applying transformations (like augmentation) to the dataset.
    """

    def __init__(self, transform_cfg):
        super(Transform, self).__init__()

        # General Setting
        self.aug_dataset_flag = transform_cfg.train_params.aug_dataset_flag

    def __call__(self,
                 im_sig,
                 ctc,
                 aif,
                 time,
                 seg,
                 eta_gnd,
                 eta_pretrain,
                 mask_od,
                 aug_data_dic):

        # General
        if self.aug_dataset_flag is True:
            eta_pretrain_aug = aug_data_dic['eta_pretrain_aug']
            aif_aug = aug_data_dic['aif_aug']
            mask_od_aug = aug_data_dic['mask_od_aug']
            time_aug = aug_data_dic['time_aug']

            # Avoid eta_pretrain_aug values with NaN values
            if not hasattr(self, 'reject_list'):
                self.reject_list = []
                for i in range(eta_pretrain_aug.shape[0]):
                    for j in range(eta_pretrain_aug.shape[0]):
                        if torch.isnan((eta_pretrain_aug[i, j].sum())):
                            print(
                                'Reject List: i = '
                                + str(i)
                                + ', j ='
                                + str(j)
                                + ', value = '
                                + str((eta_pretrain_aug[i, j].sum()).item()))
                            self.reject_list.append((i, j))

        # Apply data-augumentation
        N_train = im_sig.shape[0]
        indx = np.arange(0, N_train)
        aug_indx = np.zeros(indx.shape, dtype=int)
        with torch.no_grad():
            for i in range(N_train):
                rot_angle = random.choice([-90., 0., 90., 180.])
                hfilp = random.choice([TF.hflip, torch.nn.Identity()])
                im_sig[i] = hfilp(
                    TF.rotate(im_sig[i].moveaxis(-1, -3),
                              angle=rot_angle)
                    ).moveaxis(-3, -1)
                ctc[i] = hfilp(
                    TF.rotate(ctc[i].moveaxis(-1, -3), angle=rot_angle)
                    ).moveaxis(-3, -1)
                seg[i] = hfilp(
                    TF.rotate(seg[i].unsqueeze(0), angle=rot_angle)
                    )
                eta_pretrain[i] = hfilp(
                    TF.rotate(eta_pretrain[i], angle=rot_angle)
                    )
                eta_gnd[i] = hfilp(
                    TF.rotate(eta_gnd[i], angle=rot_angle)
                    )
                if self.aug_dataset_flag is True:
                    eta_pretrain_aug[i] = hfilp(
                        TF.rotate(eta_pretrain_aug[i],
                                  angle=rot_angle)
                        )
                    aug_indx[i] = np.random.randint(0, N_train)
                    while (indx[i], aug_indx[i]) in self.reject_list:
                        aug_indx[i] = np.random.randint(0, N_train)

            # Randomly choose augumented perfusion data
            if self.aug_dataset_flag is True:
                eta_pretrain = eta_pretrain_aug[indx, aug_indx].clone()
                aif = aif_aug[indx, aug_indx].clone()
                time = time_aug[indx, aug_indx].clone()
                mask_od = mask_od_aug[indx, aug_indx].clone()

        return im_sig, ctc, aif, time, seg, eta_gnd, eta_pretrain, mask_od


def collate(batch):
    """
    Combines a batch of data into a single `BatchRadCineSENSE2D` object.
    """
    return BatchRadCineSENSE2D(batch)


class BatchRadCineSENSE2D():
    """
    Class to store and process a batch of RadCine SENSE 2D data.
    """

    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.im_sig = torch.stack(transposed_data[0], 0)
        self.ctc = torch.stack(transposed_data[1], 0)
        self.aif = torch.stack(transposed_data[2], 0)
        self.time = torch.stack(transposed_data[3], 0)
        self.wlen = torch.stack(transposed_data[4], 0)
        self.seg = torch.stack(transposed_data[5], 0)
        self.mbolus = torch.stack(transposed_data[6], 0)
        self.eta_pretrain = torch.stack(transposed_data[7], 0)
        self.mask_od = torch.stack(transposed_data[8], 0)

    def pin_memory(self):
        self.im_sig = self.im_sig.pin_memory()
        self.ctc = self.ctc.pin_memory()
        self.aif = self.aif.pin_memory()
        self.time = self.time.pin_memory()
        self.wlen = self.wlen.pin_memory()
        self.seg = self.seg.pin_memory()
        self.mbolus = self.mbolus.pin_memory()
        self.eta_pretrain = self.eta_pretrain.pin_memory()
        self.mask_od = self.mask_od.pin_memory()

        return self
