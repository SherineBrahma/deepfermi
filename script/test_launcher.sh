# Arguments
project_name=$1
read_project_name=$2
dataset_file_name=$3
cuda_device=$4
environment=$5
code_directory=$6
config_path=$7

# Activate environment
conda activate $environment
cd $code_directory

# Testing script
CUDA_VISIBLE_DEVICES=$cuda_device python test.py --config_path=$config_path --project_name=$project_name  --read_project_name=$read_project_name --dataset_file_name=$dataset_file_name --build_dataset_flag True --mode='testing' --load_unet='unet' 
