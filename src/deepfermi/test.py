import argparse
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from config import Mode, TestConfig, TrainConfig
from data_loading import DatasetDCEPerfusion, collate
from network.DeepFermi_net import DeepFermi
from network.Unet import Unet
from torch.utils.data import DataLoader
from tqdm import tqdm


def parse_override_arguments() -> argparse.Namespace:
    """
    Parses command line for selected arguments for overriding the
    configuration file.

    Returns:
        args (argparse.Namespace): The parsed arguments containing
        user-provided arguments for overriding.
    """

    # Reading command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default=None, type=str)
    parser.add_argument('--project_name', default=None, type=str)
    parser.add_argument('--read_project_name', default=None, type=str)
    parser.add_argument('--dataset_file_name', default=None, type=str)
    parser.add_argument('--SNR_ctc', default=None, type=int)
    parser.add_argument('--mode', default=None, type=Mode)
    parser.add_argument('--load_unet', default=None, type=str)
    parser.add_argument('--build_dataset_flag', default=None,
                        choices=(True, False), type=eval)
    args = parser.parse_args()

    return args


def read_cfg(config_path, override_args) -> Tuple[TestConfig, TrainConfig]:
    """
    Reads a YAML configuration file and applies overrides from command-line
    arguments.

    This function loads the configuration from the specified YAML file and
    updates its parameters with values from the provided command-line
    arguments. It ensures that only valid modes are set and returns the updated
    configuration objects for testing and training.

    Parameters:
        config_path (str): The relative or absolute path to the YAM
                           configuration file.
        override_args: Command-line arguments that may contain configuration
                       overrides.

    Returns:
        TestConfig: The updated configuration object containing both the
                    loaded and overridden values.
        TrainConfig: The configuration object for training, loaded from a
                     saved experiment file.

    Raises:
        AssertionError: If the mode in the configuration is set to a value
        other than the allowed mode during testing.
    """

    # Reading configuration file
    deepfermi_dir = Path(__file__).resolve().parent.parent.parent
    test_config_path = str(deepfermi_dir / config_path)
    print(test_config_path)
    test_cfg = TestConfig.from_yaml(test_config_path)

    # Overriding configuration if command line input provided
    test_cfg.info.project_name = (
        override_args.project_name
        or test_cfg.info.project_name
        )
    test_cfg.info.read_project_name = (
        override_args.read_project_name
        or test_cfg.info.read_project_name
        )
    test_cfg.test_params.dataset.file_name = (
        override_args.dataset_file_name
        or test_cfg.test_params.dataset.file_name
        )
    test_cfg.test_params.dataset.SNR_ctc = (
        override_args.SNR_ctc
        or test_cfg.test_params.dataset.SNR_ctc
        )
    test_cfg.test_params.dataset.build_dataset_flag = (
        override_args.build_dataset_flag
        or test_cfg.test_params.dataset.build_dataset_flag
        )
    test_cfg.test_params.mode = (
        override_args.mode
        or test_cfg.test_params.mode
        )
    test_cfg.test_params.load_unet = (
        override_args.load_unet
        or test_cfg.test_params.load_unet
        )

    # Preliminary checks
    msg = (
        "Only mode allowed during testing "
        "is 'testing'"
        )
    assert test_cfg.test_params.mode.value in [
        'testing',
    ], msg

    # Reading saved config of the experiment to be read
    config_path = Path.joinpath(
        Path(test_cfg.paths.read + test_cfg.info.read_project_name),
        'train_config.yaml'
        )
    cfg = TrainConfig.from_yaml(config_path)

    return test_cfg, cfg


def load_dataset_obj(test_cfg) -> Tuple[DatasetDCEPerfusion]:
    """
    Constructs or loads the test dataset based on the configuration settings.

    If the dataset build flag is set, this function will construct new testing
    dataset objects from the specified data file. If the build flag is not
    set, it will load previously saved dataset objects. Additionally, the
    background values in the dataset are updated according to the specified
    reference values in the configuration.

    Parameters:
        test_cfg (TestConfig): The configuration object containing settings
                               for loading or building the dataset.

    Returns:
        DatasetDCEPerfusion: The test dataset object.
    """

    # Dataset
    file_path = Path.joinpath(Path(test_cfg.paths.dataset),
                              test_cfg.test_params.dataset.file_name)
    # Save a reference to the class itself
    build_dataset = test_cfg.test_params.dataset.build_dataset_flag
    test_dataset_path = Path.joinpath(
        Path(test_cfg.paths.dataset),
        'test_dataset.pkl'
        )
    if build_dataset is True:
        transform = None
        test_dataset = DatasetDCEPerfusion.construct_from_npz(
            file_path,
            transform,
            pid_to_load=test_cfg.test_params.ntest,
            aug_dataset_flag=False,
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

    # Replacing background values with specified values
    test_dataset = DatasetDCEPerfusion.update_precond_bkg_ref(test_dataset,
                                                              test_cfg)

    return test_dataset


def load_network(test_cfg, cfg) -> DeepFermi:
    """
    Loads DeepFermi network model from a configuration and checkpoint.

    Parameters:
        cfg (TrainConfig): The configuration object containing network
                            parameters and checkpoint settings.

    Returns:
        DeepFermi: The constructed and loaded neural network model.
    """

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
                     max_iter_lbfgs=cfg.train_params.network.max_iter_lbfgs,
                     max_eval_lbfgs=cfg.train_params.network.max_eval_lbfgs,
                     mode=test_cfg.test_params.mode.value,
                     learn_lambda=cfg.train_params.network.learn_lambda).cuda()
    
    # Load Network from checkpoint
    network_path = Path.joinpath(Path(test_cfg.paths.save),
                                 test_cfg.info.read_project_name)
    print('Loading network...')
    load_unet = test_cfg.test_params.load_unet
    unet_state_dic = torch.load(Path.joinpath(network_path, load_unet))
    unet.load_state_dict(unet_state_dic, strict=False)

    return unet


def apply_morphology(test_dataset, is_erosion_not_dilate):
    """
    Applies morphological operations (erosion or dilation) to the segmentation
    mask in the dataset to simulate variations of the mask.

    Parameters:
        test_dataset (DatasetDCEPerfusion): The dataset object containing the
                                            segmentation and image signals.
        is_erosion_not_dilate (bool): A flag to determine whether erosion
                                      (True) or dilation (False) should be
                                      applied.

    Returns:
        DatasetDCEPerfusion: The updated dataset with modified segmentation
                             mask and CTC based on the morphological operation
                             applied.
    """
    
    morph_kernel = torch.tensor([[1, 1, 1],
                                 [1, 1, 1],
                                 [1, 1, 1]], dtype=test_dataset.seg.dtype)
    # pylint: disable=not-callable
    morph_conv = torch.nn.functional.conv2d(
        test_dataset.seg.unsqueeze(1),
        morph_kernel.unsqueeze(0).unsqueeze(0), padding=(1, 1)
        )
    morph_thresh = (morph_kernel.numel() - 0.01
                    if is_erosion_not_dilate is True
                    else 0)
    test_dataset.seg = torch.heaviside(
        morph_conv - morph_thresh,
        torch.tensor(0, dtype=test_dataset.seg.dtype)
        ).squeeze(1)
    test_dataset.ctc = test_dataset.seg.unsqueeze(-1) * test_dataset.im_sig

    return test_dataset


def test_network_module(test_dataset, unet, test_cfg) -> Tuple[torch.Tensor,
                                                               torch.Tensor,
                                                               torch.Tensor]:
    """
    Tests the neural network on the provided dataset and returns the output
    and other relevant information.

    Parameters:
        test_dataset (DatasetDCEPerfusion): The test dataset object.
        unet (torch.nn.Module): The trained DeepFermi model to be tested.
        test_cfg (TestConfig): Configuration object containing test parameters.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - `eta_net` (torch.Tensor): The estimated perfusion parameters from
              DeepFermi.
            - `eta_lbfgs` (torch.Tensor): The estimated perfusion parameters
              from the LBFGS optimization without deep learning components.
            - `time_taken` (torch.Tensor): The time taken for each test sample
              to be processed by DeepFermi.
    """

    # Training options
    device = test_cfg.test_params.device.value
    clean_outliers = test_cfg.test_params.clean_outliers
    morph_flag = test_cfg.test_params.morph_flag
    is_erosion_not_dilate = test_cfg.test_params.is_erosion_not_dilate

    # Segmentation
    if morph_flag is True:
        test_dataset = apply_morphology(test_dataset,
                                        is_erosion_not_dilate)

    # Load data
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 num_workers=2,
                                 collate_fn=collate,
                                 prefetch_factor=4,
                                 pin_memory=True)

    # Constructing placeholders
    N, nx, ny, _ = test_dataset.im_sig.shape
    eta_net = torch.zeros((N, 3, nx, ny), device=device)
    eta_lbfgs = torch.zeros((N, 3, nx, ny), device=device)
    time_taken = torch.zeros(N, device=device)

    # For timing performance
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # Testing network
    print('Testing network')
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_dataloader), total=N):

            # Unpack batched tuple
            wlen_test = batch.wlen
            im_sig_test = batch.im_sig[..., 0:wlen_test].to(device)
            ctc_test = batch.ctc[..., 0:wlen_test].to(device)
            aif_test = batch.aif[..., 0:wlen_test].to(device)
            time_test = batch.time[..., 0:wlen_test].to(device)
            seg_test = batch.seg.to(device)
            eta_lbfgs_test = batch.eta_pretrain.to(device)
            mask_od_test = batch.mask_od

            # Filtering time-points
            if clean_outliers is True:
                indx_dc = np.arange(wlen_test)
                indx_dc = indx_dc[mask_od_test[0, :wlen_test] == 1]
            else:
                indx_dc = np.arange(wlen_test)

            # Apply network
            start.record()
            eta_net_test = unet(im_sig_test.unsqueeze(1),
                                seg_test,
                                aif=aif_test,
                                ctc=ctc_test,
                                time=time_test,
                                indx_dc=indx_dc)
            end.record()
            torch.cuda.synchronize()

            # Results
            eta_net[i] = eta_net_test
            eta_lbfgs[i] = eta_lbfgs_test
            time_taken[i] = (start.elapsed_time(end)/1000)

    return eta_net, eta_lbfgs, time_taken


def main() -> None:
    """
    Main function to test the network and save the results.

    This function handles the full process of testing the neural network on
    the provided dataset, including reading the configuration, initializing
    the dataset and network, performing network testing, and saving the results
    to disk.

    1. Reads configuration and overrides from command-line arguments.
    2. Initializes the test dataset and the network model (UNet).
    3. Runs the network on the test dataset and collects the outputs.
    4. Transfers the results from GPU to CPU.
    5. Saves the results to a specified path in `.npy` format.
    """

    # Reading arguments for overriding config file
    override_args = parse_override_arguments()

    # Building configuration object
    config_path = 'config/test_config.yaml'
    test_cfg, cfg = read_cfg(config_path, override_args)

    # Initialize dataset object
    test_dataset = load_dataset_obj(test_cfg)

    # Initialize network
    unet = load_network(test_cfg, cfg)

    # Testing the network
    eta_net, eta_lbfgs, time_taken = test_network_module(test_dataset,
                                                         unet,
                                                         test_cfg)

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
    eta_gnd = test_dataset.eta_gnd.cpu()

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
    np.save(Path.joinpath(Path(save_path), "eta_gnd.npy"), eta_gnd)


if __name__ == "__main__":
    main()