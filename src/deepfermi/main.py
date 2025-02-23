import argparse
import pickle
from pathlib import Path
from typing import Tuple

import torch
import yaml
from config import Mode, TrainConfig
from data_loading import DatasetDCEPerfusion, Transform
from network.DeepFermi_net import DeepFermi
from network.Unet import Unet
from training_manager import TrainingManager


def parse_override_arguments() -> argparse.Namespace:
    """
    Parses command line for selected arguments for overriding the
    configuration file.

    Returns:
        args (argparse.Namespace): The parsed arguments containing
        user-provided arguments for overriding.
    """

    # Reading command lines
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', default=None, type=str)
    parser.add_argument('--read_project_name', default=None, type=str)
    parser.add_argument('--dataset_file_name', default=None, type=str)
    parser.add_argument('--mode', default=None, type=Mode)
    parser.add_argument('--cross_val_k', default=None, type=int)
    parser.add_argument('--cross_val_fold', default=None, type=int)
    parser.add_argument('--unet_lr', default=None, type=float)
    parser.add_argument('--unet_wd', default=None, type=float)
    parser.add_argument('--build_dataset_flag', default=None,
                        choices=(True, False), type=eval)
    parser.add_argument('--train_from_ckpt', default=None,
                        choices=(True, False), type=eval)
    parser.add_argument('--cross_val_flag', default=None,
                        choices=(True, False), type=eval)
    args = parser.parse_args()

    return args


def read_cfg(config_path, override_args) -> TrainConfig:
    """
    Reads a YAML configuration file and overrides parameters if provided as
    command-line arguments.

    The function loads the configuration from the specified path and updates
    fields with values from the command-line arguments. Basic validity checks
    are performed before returning the updated configuration object.

    Parameters:
        config_path (str): The path to the YAML configuration file.
        override_args: The command-line arguments containing configuration
                       overrides.

    Returns:
        TrainConfig: The updated configuration object, including both the
                     loaded and overridden parameters.

    Raises:
        AssertionError: If the mode is set to 'fine_tuning' but not one of
                        the allowed values ('pre_training' or 'fine_tuning').
    """

    # Reading configuration file
    cfg = TrainConfig.from_yaml(config_path)

    # Overriding configuration if command line input provided
    cfg.info.project_name = (
        override_args.project_name
        or cfg.info.project_name)
    cfg.train_params.dataset.file_name = (
        override_args.dataset_file_name
        or cfg.train_params.dataset.file_name
        )
    cfg.train_params.dataset.build_dataset_flag = (
        override_args.build_dataset_flag
        or cfg.train_params.dataset.build_dataset_flag
        )
    cfg.train_params.mode = (
        override_args.mode
        or cfg.train_params.mode
        )
    cfg.train_params.network.train_from_ckpt = (
        override_args.train_from_ckpt
        or cfg.train_params.network.train_from_ckpt
        )
    cfg.train_params.cross_val_flag = (
        override_args.cross_val_flag
        or cfg.train_params.cross_val_flag
        )
    cfg.train_params.cross_val_k = (
        override_args.cross_val_k
        or cfg.train_params.cross_val_k
        )
    cfg.train_params.cross_val_fold = (
        override_args.cross_val_fold
        or cfg.train_params.cross_val_fold
        )
    cfg.train_params.optimizer.unet_lr = (
        override_args.unet_lr
        or cfg.train_params.optimizer.unet_lr
        )
    cfg.train_params.optimizer.unet_wd = (
        override_args.unet_wd
        or cfg.train_params.optimizer.unet_wd
        )
    cfg.update_yaml()

    # Preliminary checks
    msg = (
        "Only mode allowed during training "
        "is 'pre_training' and 'fine_tuning'"
        )
    assert cfg.train_params.mode.value in [
        'pre_training',
        'fine_tuning',
    ], msg

    return cfg


def load_dataset_obj(cfg) -> Tuple[DatasetDCEPerfusion, DatasetDCEPerfusion]:
    """
    Constructs new training and validation dataset objects from specifid data
    file in the configuration if the build flag is set, or it loads pre-saved
    dataset objects.

    Parameters:
        cfg (TrainConfig): The configuration object that for training.

    Returns:
        Tuple[DatasetDCEPerfusion, DatasetDCEPerfusion]: A tuple containing
        the training dataset and validation dataset
    """

    # Save a reference to the class itself
    file_path = Path.joinpath(Path(cfg.paths.dataset),
                              cfg.train_params.dataset.file_name)
    build_dataset = cfg.train_params.dataset.build_dataset_flag
    if cfg.train_params.cross_val_flag is True:
        train_dataset_path = Path.joinpath(
            Path(cfg.paths.dataset),
            'train_dataset_cross_val_'
            + str(cfg.train_params.cross_val_k)
            + '_fold_'
            + str(cfg.train_params.cross_val_fold)
            + '.pkl',
        )
        val_dataset_path = Path.joinpath(
            Path(cfg.paths.dataset),
            'val_dataset_cross_val_'
            + str(cfg.train_params.cross_val_k)
            + '_fold_'
            + str(cfg.train_params.cross_val_fold)
            + '.pkl',
        )
    else:
        train_dataset_path = Path.joinpath(Path(cfg.paths.dataset),
                                           'train_dataset.pkl')
        val_dataset_path = Path.joinpath(Path(cfg.paths.dataset),
                                         'val_dataset.pkl')
    if build_dataset is True:
        transform = Transform(cfg)
        train_dataset = DatasetDCEPerfusion.construct_from_npz(
            file_path,
            transform,
            pid_to_load=cfg.train_params.ntrain,
            config=cfg,
            aug_dataset_flag=cfg.train_params.aug_dataset_flag,
        )
        val_dataset = DatasetDCEPerfusion.construct_from_npz(
            file_path,
            transform,
            pid_to_load=cfg.train_params.nval,
            aug_dataset_flag=False,
            config=cfg
        )
        # Save dataset
        with open(train_dataset_path, 'wb') as f:
            pickle.dump(train_dataset, f)
        with open(val_dataset_path, 'wb') as f:
            pickle.dump(val_dataset, f)
    else:
        # Load dataset
        transform = Transform(cfg)
        with open(train_dataset_path, 'rb') as f:
            train_dataset = pickle.load(f)
            train_dataset.transform = transform
        with open(val_dataset_path, 'rb') as f:
            val_dataset = pickle.load(f)
            val_dataset.transform = transform

    # Replacing background values with specified values
    train_dataset = DatasetDCEPerfusion.update_precond_bkg_ref(
        train_dataset, cfg, aug_dataset_flag=cfg.train_params.aug_dataset_flag
    )
    val_dataset = DatasetDCEPerfusion.update_precond_bkg_ref(val_dataset, cfg)

    return train_dataset, val_dataset


def load_network(cfg) -> DeepFermi:
    """
    Loads DeepFermi network model from a configuration and checkpoint if
    provided.

    Parameters:
        cfg (TrainConfig): The configuration object containing network
                            parameters and checkpoint settings.

    Returns:
        DeepFermi: The constructed and optionally loaded neural network model.
    """

    cnn = Unet(
        dim=3,
        ncin=cfg.train_params.network.ncin,
        ncout=cfg.train_params.network.ncout,
        nstage=cfg.train_params.network.nstage,
        nconv_stage=cfg.train_params.network.nconv_stage,
        nfilters=cfg.train_params.network.nfilters,
        res_connect=False,
        bias=False,
    )
    unet = DeepFermi(
        cnn,
        osamp=cfg.train_params.network.osamp,
        max_iter_lbfgs=cfg.train_params.network.max_iter_lbfgs,
        max_eval_lbfgs=cfg.train_params.network.max_eval_lbfgs,
        mode=cfg.train_params.mode.value,
        learn_lambda=cfg.train_params.network.learn_lambda,
    ).cuda()

    # Load Network from checkpoint
    network_path = Path.joinpath(Path(cfg.paths.save), cfg.info.project_name)
    if cfg.train_params.network.train_from_ckpt is True:
        print('Loading network...')
        load_unet = cfg.train_params.network.load_unet
        unet_state_dic = torch.load(Path.joinpath(network_path, load_unet))
        unet_state_dic.pop('lambda_reg')
        unet_state_dic.pop('max_iter_lbfgs')
        unet_state_dic.pop('max_eval_lbfgs')
        unet.load_state_dict(unet_state_dic, strict=False)

    return unet


def record_train_params(cfg, unet) -> None:
    """
    Records training configuration and network parameters in text files.

    Parameters:
        cfg (TrainConfig): The configuration object containing paths,
                           settings, and training parameters.
        unet (DeepFermi): The network model for which the parameters will
                          be recorded.
    """

    # Record training configurations
    save_path = Path.joinpath(Path(cfg.paths.save), cfg.info.project_name)
    save_file = Path.joinpath(save_path, 'train_config.yaml')
    with open(save_file, 'w', encoding='utf-8') as file:
        yaml.dump(cfg.yaml_config, file, default_flow_style=None)
    save_file = Path.joinpath(save_path, 'network_params.txt')
    with open(save_file, 'w', encoding='utf-8') as file:
        table, total_params = cfg.train_params.network.parameters(unet)
        file.write('DeepFermi parameter breakdown:\n')
        file.write(str(table))
        file.write(f'\n Total Trainable Params: {total_params} \n')


def main() -> None:
    """
    Executes the full training pipeline: parses arguments, loads
    configuration, initializes datasets and the network, records
    parameters, and starts training.

    Returns:
        None
    """

    # Reading arguments for overriding config file
    override_args = parse_override_arguments()

    # Building configuration object
    config_path = 'config/train_config.yaml'
    cfg = read_cfg(config_path, override_args)

    # Initialize dataset object
    train_dataset, val_dataset = load_dataset_obj(cfg)

    # Initialize network
    unet = load_network(cfg)

    # Record training configurations
    record_train_params(cfg, unet)

    # Initialization of the unit that controls different
    # components while training
    tm = TrainingManager(cfg, train_dataset, val_dataset, unet)

    # Start training
    print('Training Started')
    tm.model_train()


if __name__ == '__main__':
    main()
