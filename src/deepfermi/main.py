import argparse
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from pathlib import Path
import pickle
import torch
import yaml

from config import TrainConfig, Mode
from data_loading import DatasetDCEPerfusion, Transform
from network.Unet import Unet
from network.DeepFermi_net import DeepFermi
from training_manager import TrainingManager

def main() -> None:
    
    # Reading command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', default=None,type=str)
    parser.add_argument('--read_project_name', default=None,type=str)
    parser.add_argument('--dataset_file_name', default=None,type=str)
    parser.add_argument('--build_dataset_flag', default=None, choices=(True, False), type=eval)
    parser.add_argument('--mode', default=None,type=str)
    parser.add_argument('--train_from_ckpt', default=None, choices=(True, False), type=eval)
    parser.add_argument('--cross_val_flag', default=None, choices=(True, False), type=eval)
    parser.add_argument('--cross_val_k', default=None,type=int)
    parser.add_argument('--cross_val_fold', default=None,type=int)
    parser.add_argument('--unet_lr', default=None,type=float)
    parser.add_argument('--unet_wd', default=None,type=float)
    args = parser.parse_args()
    
    # Reading configuration file
    config_path = "config/train_config.yaml"
    cfg = TrainConfig.from_yaml(config_path)
    
    # Overriding configuration if command line input provided
    cfg.info.project_name = cfg.info.project_name if args.project_name == None else args.project_name
    cfg.train_params.dataset.file_name = cfg.train_params.dataset.file_name if args.dataset_file_name == None else args.dataset_file_name
    cfg.train_params.dataset.build_dataset_flag = cfg.train_params.dataset.build_dataset_flag if args.build_dataset_flag == None else args.build_dataset_flag
    cfg.train_params.mode = Mode(cfg.train_params.mode.value) if args.mode == None else Mode(args.mode)
    cfg.train_params.network.train_from_ckpt = cfg.train_params.network.train_from_ckpt if args.train_from_ckpt == None else args.train_from_ckpt    
    cfg.train_params.cross_val_flag = cfg.train_params.cross_val_flag if args.cross_val_flag == None else args.cross_val_flag
    cfg.train_params.cross_val_k = cfg.train_params.cross_val_k if args.cross_val_k == None else args.cross_val_k
    cfg.train_params.cross_val_fold = cfg.train_params.cross_val_fold if args.cross_val_fold == None else args.cross_val_fold
    cfg.train_params.optimizer.unet_lr = cfg.train_params.optimizer.unet_lr if args.unet_lr == None else args.unet_lr
    cfg.train_params.optimizer.unet_wd = cfg.train_params.optimizer.unet_wd if args.unet_wd == None else args.unet_wd
    cfg.update_yaml()
    
    # Preliminary checks
    if cfg.train_params.mode.value=='fine_tuning':
        assert cfg.train_params.mode.value in ['pre_training', 'fine_tuning'], "Only mode allowed during testing is 'pre_training' and 'fine_tuning'"
        
    # Dataset
    file_path = Path.joinpath(Path(cfg.paths.dataset), cfg.train_params.dataset.file_name)
    # Save a reference to the class itself
    build_dataset = cfg.train_params.dataset.build_dataset_flag
    if cfg.train_params.cross_val_flag==True:
        train_dataset_path = Path.joinpath(Path(cfg.paths.dataset),'train_dataset_cross_val_'+ str(cfg.train_params.cross_val_k) + '_fold_' + str(cfg.train_params.cross_val_fold) +'.pkl')
        val_dataset_path = Path.joinpath(Path(cfg.paths.dataset),'val_dataset_cross_val_'+ str(cfg.train_params.cross_val_k) + '_fold_' + str(cfg.train_params.cross_val_fold) +'.pkl')
    else:
        train_dataset_path = Path.joinpath(Path(cfg.paths.dataset),'train_dataset.pkl')
        val_dataset_path = Path.joinpath(Path(cfg.paths.dataset),'val_dataset.pkl')
    if build_dataset == True:
        transform = Transform(cfg)
        train_dataset = DatasetDCEPerfusion.construct_from_npz(
            file_path,
            transform,
            pid_to_load=cfg.train_params.ntrain,
            config=cfg,
            aug_dataset_flag = cfg.train_params.aug_dataset_flag
            )
        val_dataset = DatasetDCEPerfusion.construct_from_npz(
            file_path,
            transform,
            pid_to_load=cfg.train_params.nval,
            aug_dataset_flag = False,
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
            
    # Update eta_bkg_ref
    train_dataset = DatasetDCEPerfusion._update_precond_bkg_ref(train_dataset, cfg, aug_dataset_flag = cfg.train_params.aug_dataset_flag)
    val_dataset = DatasetDCEPerfusion._update_precond_bkg_ref(val_dataset, cfg)
    
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
                     mode=cfg.train_params.mode.value, 
                     learn_lambda=cfg.train_params.network.learn_lambda).cuda()
    
    # Load Network from checkpoint
    network_path = Path.joinpath(Path(cfg.paths.save), cfg.info.project_name)    
    if cfg.train_params.network.train_from_ckpt == True:
        print('Loading network...')
        load_unet = cfg.train_params.network.load_unet
        unet_state_dic = torch.load(Path.joinpath(network_path, load_unet))
        unet_state_dic.pop('lambda_reg')
        unet_state_dic.pop('max_iter_lbfgs')
        unet_state_dic.pop('max_eval_lbfgs')
        unet.load_state_dict(unet_state_dic, strict=False)
    
    # Initialization of the unit that controls different components while training
    tm = TrainingManager(cfg, 
                         train_dataset, 
                         val_dataset, 
                         unet)
    
    # Record training configurations
    save_path = Path.joinpath(Path(cfg.paths.save), cfg.info.project_name)
    save_file = Path.joinpath(save_path, 'train_config.yaml')
    with open(save_file, 'w') as file:
        yaml.dump(cfg.yaml_config, file, default_flow_style=None)
    save_file = Path.joinpath(save_path, 'network_params.txt')
    with open(save_file, 'w') as file:
        table, total_params = cfg.train_params.network.parameters(unet)
        file.write(unet._get_name() +  " parameter breakdown:\n")
        file.write(str(table))
        file.write(f"\n Total Trainable Params: {total_params} \n")
    
    # Start training
    print('Training Started')
    tm.model_train()    

if __name__ == "__main__":
    main()
    