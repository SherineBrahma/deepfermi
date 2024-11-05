from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Type

import yaml
from prettytable import PrettyTable
from typeguard import check_type

def construct_yaml_obj(obj: Type, data: Dict[str, Any]):
    
    field_types = obj.__dict__
    constructor_args = {}
    
    for key, value in data.items():
        
        if key in field_types:
            expected_type = field_types[key]
            if is_dataclass(expected_type):
                # If the expected type is a data class (and not a direct type like int, str, etc.)
                constructor_args[key] = construct_yaml_obj(expected_type, value)
            else:
                if not isinstance(expected_type, Enum):
                    try:                           
                        check_type(value, type(expected_type))
                        obj.__setattr__(key, value)
                    except Exception as e:
                        error_message = 'Error: expected type of ' + key + ' is ' + type(expected_type).__name__ + ', got ' + type(value).__name__ + '.'
                        raise TypeError(error_message)      
                else:
                            obj.__setattr__(key, eval(type(expected_type).__name__ )(value))
        else:
            raise KeyError(f"Unexpected key '{key}' for class '{type(obj).__name__}'")
    
    return obj

@dataclass
class GeneralInfo:
    project_name: str = 'Debug'
    read_project_name: str = ''
    project_description: str = 'Developing Code'    
    
@dataclass
class Paths:
    dataset: str = str(Path(__file__).resolve().parent.parent.parent / 'src/deepfermi/data/')
    read: str = str(Path(__file__).resolve().parent.parent.parent / 'src/deepfermi/Experiments/')
    save: str = str(Path(__file__).resolve().parent.parent.parent / 'src/deepfermi/Experiments/')
        
    def __setattr__(self, attr, val):
        val = str(Path(__file__).resolve().parent.parent.parent / val)
        super(Paths, self).__setattr__(attr, val)
    
@dataclass
class Dataset:
    file_name: str = 'dataset'
    build_dataset_flag: str = True
    SNR_ctc: int = 15
    img_dim: List[int] = field(default_factory=lambda: [120, 120])
    crop_dim: List[int] = field(default_factory=lambda: [120, 120])
    eta_bkg_ref: List[int] = field(default_factory=lambda: [0.001667, 0.0, 0.01])

class Mode(Enum):
    PRE_TRAINING = 'pre_training'
    FINE_TUNING = 'fine_tuning'
    TESTING = 'testing'
    
class Device(Enum):
    CUDA = 'cuda'
    CPU = 'cpu'

@dataclass
class Network:    
    # Architecture
    ncin: int = 2
    nfilters: int = 16    
    nstage: int = 3
    nconv_stage: int = 2
    ncout: int = 2
    learn_lambda: bool = True
    dropout: float = 0.0
    # Data-consistency configurations
    osamp: int = 20
    nu: int = 1
    max_iter_lbfgs: int = 10000
    max_eval_lbfgs: int = 10000
    # Check-pointing
    train_from_ckpt: bool = True
    load_unet: str = 'unet_load'
    backprop_ckpt: int = 0
    
    @staticmethod
    def parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in model.named_parameters():
            # if not parameter.requires_grad: continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        return table, total_params
    
@dataclass
class Optimizer:
    unet_lr: float = 10.e-4
    unet_wd: float = 10.e-12

@dataclass
class TrainParams:
    dataset: Dataset = Dataset()
    mode: Mode = Mode('pre_training')
    network: Network = Network()
    optimizer: Optimizer = Optimizer()
    device: Device = Device('cuda')
    ntrain: List[str] = field(default_factory=lambda: ['56'])
    nval: List[str] = field(default_factory=lambda: ['46'])
    ntest: List[str] = field(default_factory=lambda: ['36'])
    nepochs: int = 5
    val_step_size: int = 50
    mb: int = 1
    ssup_split: float = 0.3
    adv_mb: int = 1
    adv_itr: int = 1
    cross_val_flag: bool = False
    cross_val_k: int = 5
    cross_val_fold: int = 1
    aug_dataset_flag: bool = True
    osamp: int = 5
    pre_scale_factor: int = 10

@dataclass
class TrainConfig:
    info: GeneralInfo = GeneralInfo()
    paths: Paths = Paths()
    train_params: TrainParams = TrainParams()
    yaml_config: dict | None = None
    
    @classmethod
    def from_yaml(cls, config_path: str) -> TrainConfig:
        
        config_path = str(Path(__file__).resolve().parent.parent.parent / "config/train_config.yaml")
        with open(Path(config_path), "r") as stream:
            try:
                yaml_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)            
        instance = construct_yaml_obj(cls, yaml_config)()
        instance.yaml_config = yaml_config
           
        return instance
    
    def update_yaml(self):
        yaml_config_temp = self.__dict__.copy()
        del yaml_config_temp['yaml_config']
        self.yaml_config = dict(yaml_config_temp)
    
    def __str__(self):
        return yaml.dump(self.yaml_config, default_flow_style=None)
    
@dataclass
class TestParams:
    dataset: Dataset = Dataset()
    mode: Mode = Mode('testing')
    device: Device = Device('cuda')
    load_unet: str = 'unet'
    osamp: int = 5
    pre_scale_factor: int = 10
    nsamp: int = 1
    calib_nsamp: int = 1
    ntest: List[str] = field(default_factory=lambda: ['P3'])    
    clean_outliers: bool = False
    morph_flag: bool = False
    is_erosion_not_dilate: bool = True
    
@dataclass
class TestConfig:
    # General
    info: GeneralInfo = GeneralInfo()
    # Paths
    paths: Paths = Paths()
    # Paths
    test_params: TestParams = TestParams()
    
    @classmethod
    def from_yaml(cls, config_path: str) -> TestConfig:
        
        with open(Path(config_path), "r") as stream:
            try:
                yaml_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)            
        instance = construct_yaml_obj(cls, yaml_config)()
        instance.yaml_config = yaml_config
           
        return instance
    
    def __str__(self):
        return yaml.dump(self.yaml_config, default_flow_style=None)