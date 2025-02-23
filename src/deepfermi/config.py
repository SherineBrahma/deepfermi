from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Type

import yaml
from prettytable import PrettyTable
from typeguard import check_type


def construct_yaml_obj(obj: Type, data: Dict[str, Any]):
    """
    Constructs an object from a dictionary of data, setting its attributes
    based on the provided values.

    Recursively constructs dataclass attributes and assigns values, checking
    types for each attribute. If an enum is encountered, the corresponding
    enum value is assigned.

    Parameters:
        obj (Type): The target object type (dataclass or class).
        data (Dict[str, Any]): A dictionary where keys are attribute names
                               and values are data to set.

    Returns:
        Type: The object with attributes set from the data.

    Raises:
        KeyError: If an unexpected key is found in data.
        TypeError: If the type of a value does not match the expected type.
    """

    field_types = obj.__dict__
    constructor_args = {}
    for key, value in data.items():

        if key in field_types:
            expected_type = field_types[key]
            if is_dataclass(expected_type):
                # If the expected type is a data class (and not a direct type
                # like int, str, etc.)
                constructor_args[key] = construct_yaml_obj(expected_type,
                                                           value)
            else:
                if not isinstance(expected_type, Enum):
                    try:
                        check_type(value, type(expected_type))
                        setattr(obj, key, value)
                    except Exception as exc:
                        error_message = ('Error: expected type of ' +
                                         key +
                                         ' is ' +
                                         type(expected_type).__name__ +
                                         ', got ' +
                                         type(value).__name__ +
                                         '.')
                        raise TypeError(error_message) from exc
                else:
                    setattr(obj,
                            key,
                            eval(type(expected_type).__name__)(value))
        else:
            raise KeyError(f"Unexpected key '{key}'" +
                           "for class '{type(obj).__name__}'")

    return obj


@dataclass
class GeneralInfo:
    project_name: str = 'Debug'
    read_project_name: str = ''
    project_description: str = 'Developing Code'    


@dataclass
class Paths:
    home_dir = str(Path(__file__).resolve().parent.parent.parent)
    dataset: str = home_dir + '/src/deepfermi/data/'
    read: str = home_dir + '/src/deepfermi/Experiments/'
    save: str = home_dir + '/src/deepfermi/Experiments/'

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
    eta_bkg_ref: List[int] = field(default_factory=lambda: [0.001667,
                                                            0.0, 0.01])


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
    """
    Holds the configuration for training, including paths, parameters, and
    YAML data.

    Attributes:
        info (GeneralInfo): Information about the training.
        paths (Paths): File paths related to datasets and outputs.
        train_params (TrainParams): Training parameters.
        yaml_config (Optional[dict]): The loaded YAML configuration.

    Methods:
        from_yaml(config_path: str) -> TrainConfig: Loads configuration from
                                       a YAML file.
        update_yaml() -> None: Updates `yaml_config` based on the object's
                               attributes.
        __str__() -> str: Returns a string representation of the YAML
                          configuration.
    """

    info: GeneralInfo = GeneralInfo()
    paths: Paths = Paths()
    train_params: TrainParams = TrainParams()
    yaml_config: dict | None = None

    @classmethod
    def from_yaml(cls, config_path: str) -> TrainConfig:

        home_dir = Path(__file__).resolve().parent.parent.parent
        config_path = str(home_dir / "config/train_config.yaml")
        with open(Path(config_path), "r", encoding="utf-8") as stream:
            try:
                yaml_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        instance = construct_yaml_obj(cls, yaml_config)()
        instance.yaml_config = yaml_config

        return instance

    def update_yaml(self) -> None:
        yaml_config_temp = self.__dict__.copy()
        del yaml_config_temp['yaml_config']
        self.yaml_config = dict(yaml_config_temp)

    def __str__(self) -> str:
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
    """
    Holds configuration for testing, including general info, paths, and test
    parameters.

    Attributes:
        info (GeneralInfo): General information for the test.
        paths (Paths): File paths for datasets and outputs.
        test_params (TestParams): Testing parameters.

    Methods:
        from_yaml(config_path: str) -> TestConfig: Loads configuration from a
                                                   YAML file.
        __str__() -> str: Returns a string representation of the YAML
                          configuration.
    """

    info: GeneralInfo = GeneralInfo()
    paths: Paths = Paths()
    test_params: TestParams = TestParams()

    @classmethod
    def from_yaml(cls, config_path: str) -> TestConfig:

        with open(Path(config_path), "r", encoding="utf-8") as stream:
            try:
                yaml_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)            
        instance = construct_yaml_obj(cls, yaml_config)()
        instance.yaml_config = yaml_config

        return instance

    def __str__(self) -> str:
        return yaml.dump(self.yaml_config, default_flow_style=None)
