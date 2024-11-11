# Arguments
project_name=$1
read_project_name=$2
dataset_file_name=$3
cuda_device=$4

# Activate environment
conda activate brahma01_env_1
cd /data/brahma01/deepfermi/invivo/script/

# Perfunsion Quantification Simulation Experiments
experiment_folder='/data/brahma01/deepfermi/invivo/Experiments/'

# Testing script
CUDA_VISIBLE_DEVICES=$cuda_device python ../test.py --config_path='/data/brahma01/deepfermi/invivo/config/test_config.yaml' --project_name=$project_name  --read_project_name=$read_project_name --dataset_file_name=$dataset_file_name --build_dataset_flag False --mode='testing' --load_unet='unet' 
