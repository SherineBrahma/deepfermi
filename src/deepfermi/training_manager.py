import time as exec_time
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from data_loading import collate
from einops import rearrange
from termcolor import colored
from torch.utils.data import DataLoader
from tracker import Tracker
from train import unet_eval, unet_pretrain, unet_train
from utils import secs2time

warnings.filterwarnings('ignore')


class TrainingManager:
    """Manages the training and evaluation process of DeepFermi."""

    def __init__(self, cfg, train_dataset, val_dataset, unet):
        """Initializes the training manager with configuration parameters for
        the training process."""

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
        S_op = rearrange(
            torch.tensor([1, 1 / S, S], device=self.device),
            'np -> 1 np 1 1'
            )
        SH_op = rearrange(
            torch.tensor([1, S, 1 / S], device=self.device),
            'np -> np 1 1'
            )
        self.fermi_params['S'] = S
        self.fermi_params['S_op'] = S_op
        self.fermi_params['SH_op'] = SH_op
        self.fermi_params['osamp'] = cfg.train_params.osamp

        # Datasets
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        # Initializing Training Components
        self.val_step_size = cfg.train_params.val_step_size

        # Optimizers
        lr = cfg.train_params.optimizer.unet_lr
        weight_decay = cfg.train_params.optimizer.unet_wd
        self.unet_optmzr = torch.optim.Adam(
            self.unet.parameters(),
            lr=lr,
            weight_decay=weight_decay
            )

        # Tracker
        save_path = str(
            Path.joinpath(Path(cfg.paths.save), cfg.info.project_name)
            )
        self.tracker = Tracker(save_path=save_path)

    def model_eval(self, dataset, dsplit) -> Tuple[int, int]:
        """Evaluates the model on a given dataset and returns the loss and
        LBFGS iteration.

        Parameters:
            dataset (Dataset): The dataset to evaluate the model on.
            dsplit (str): The dataset split ('Train' or 'Val').

        Returns:
            tuple: The evaluation loss and average LBFGS iteration count.
        """

        print(colored('TEST ON ' + dsplit.upper() + ' SET', 'red'))
        eval_dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.train_params.mb,
            shuffle=True,
            pin_memory=True
            )

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
                ssup_loss[i] = unet_eval(
                    im_sig_batch,
                    aif_batch,
                    ctc_batch,
                    seg_batch,
                    time_batch,
                    wlen_batch,
                    self.unet,
                    self.fermi_params
                )
                avg_lbfgs_iter[i] = self.unet.lbfgs_iter

        # Averaging over the dataset
        ssup_loss = ssup_loss.mean()
        avg_lbfgs_iter = avg_lbfgs_iter.mean()

        # Printing evaluated value
        print(colored(f'{dsplit} : {ssup_loss}', 'red'))

        return ssup_loss, avg_lbfgs_iter

    def model_train(self) -> None:
        """Trains the model by iterating through epochs and mini-batches,
        performing backpropagation and validation at specified intervals.

        This method handles:
        - Loading training and validation data.
        - Augmenting the training data.
        - Performing forward and backward passes for training.
        - Evaluating the model periodically.
        - Logging and saving training progress.

        Returns:
            None
        """

        # the number of trainin samples
        N_train = self.train_dataset.__len__()

        # Make mini-batch size available to the object
        mb = self.cfg.train_params.mb

        # The number of existing mini-batches of size mb (floor in order to
        # avoid to access indices which do not exist;)
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
            train_dataloader = DataLoader(
                train_dataset_to_load,
                batch_size=self.cfg.train_params.mb,
                shuffle=True,
                num_workers=2,
                collate_fn=collate,
                prefetch_factor=4,
                pin_memory=True,
            )

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
                mask_od_batch = batch.mask_od

                # Check for NaN values in any of the ground-truth labels
                if torch.isnan(eta_pretrain_batch).any():
                    continue

                # All time points indices
                indx = np.arange(wlen_batch)
                indx_len = indx.__len__()

                # Subgroup data time points into t_nn and t_dc
                ssup_split = self.cfg.train_params.ssup_split
                np.random.shuffle(indx)
                indx_nn = np.sort(indx[0: int(ssup_split * indx_len)])
                indx_dc = np.sort(indx[int(ssup_split * indx_len): -1])

                # Discarding the motion-artifact affected time-points
                # for training the network-prior
                indx_nn = np.setdiff1d(
                    indx_nn, (mask_od_batch == 0).nonzero()[:, 1]
                    )

                # Evaluation of network and recording of training parameters
                if itr % val_step_size == 0:
                    ssup_loss_train, avg_lbfgs_iter_train = self.model_eval(
                        train_dataset_to_load,
                        'Train'
                        )
                    ssup_loss_val, avg_lbfgs_iter_val = self.model_eval(
                        val_dataset_to_load,
                        'Val'
                        )
                    self.tracker.update_and_save(
                        itr,
                        ssup_loss_train,
                        avg_lbfgs_iter_train,
                        ssup_loss_val,
                        avg_lbfgs_iter_val,
                        self.unet
                    )
                    self.tracker.save_ssup_plot()
                    self.tracker.save_lambda_reg_plot()
                    self.tracker.save_avg_lbfgs_iter_plot()
                    with torch.no_grad():
                        self.tracker.save_samp_plot(
                            train_dataset_to_load,
                            self.unet
                            )

                # Measure time to get an estimate of how long the training
                # will last
                t0_bp = exec_time.time()

                model_trained_epochs = (
                    f'network-training: epoch {ke + 1} of {nepochs}; '
                    f'mini-batch {iter_batch} out of {nmb}; '
                    f'backprop {itr + 1} of {n_back_props}'
                )
                print(colored(model_trained_epochs, 'yellow'))

                # DeepFermi Training
                loss = None
                if self.mode == 'pre_training':
                    loss = unet_pretrain(
                        im_sig_batch,
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
                        itr,
                    )
                elif self.mode == 'fine_tuning':
                    loss = unet_train(
                        im_sig_batch,
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
                        self.fermi_params,
                    )

                # Training Reporting
                if loss is not None:
                    with torch.no_grad():
                        print(colored(f'ssup loss {loss.cpu()}', 'magenta'))
                else:
                    print('Error: Loss not computed due to invalid mode.')

                # Measure the time again and substract how long one
                # weight-update took and also
                t1_bp = exec_time.time() - t0_bp
                t1_train = exec_time.time() - t0_train

                # Print the time in readable format
                est_time = secs2time(t1_bp * n_back_props)
                trained_time = secs2time(t1_train)
                model_trained_time = (f'estimated training time: {est_time}; '
                                      f'already trained: {trained_time};')
                print(colored(model_trained_time, 'cyan'))

                # Increment iteration
                itr += 1
