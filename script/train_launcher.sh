# Arguments
pretraining_folder=$1
model_based_training_folder=$2
dataset_file_name=$3
cross_val_k=$4
cross_val_fold=$5
cuda_device=$6
environment=$7
code_directory=$8
script_directory=$9
experiments_directory=${10}

# Activate environment
conda activate $environment
cd $code_directory

# # Model-agnostic pretraining
CUDA_VISIBLE_DEVICES=$cuda_device python main.py --project_name=$pretraining_folder  --dataset_file_name=$dataset_file_name --build_dataset_flag True --mode='pre_training' --train_from_ckpt False  --cross_val_flag True --cross_val_k=$cross_val_k --cross_val_fold=$cross_val_fold --unet_lr=10e-4 --unet_wd=10e-8
# Model-based training
mkdir $experiments_directory/$model_based_training_folder
cp $experiments_directory/$pretraining_folder/unet $experiments_directory/$model_based_training_folder/unet_load
CUDA_VISIBLE_DEVICES=$cuda_device python main.py --project_name=$model_based_training_folder  --dataset_file_name=$dataset_file_name --build_dataset_flag False --mode='fine_tuning' --train_from_ckpt True  --cross_val_flag True --cross_val_k=$cross_val_k --cross_val_fold=$cross_val_fold --unet_lr=10e-5 --unet_wd=10e-12
