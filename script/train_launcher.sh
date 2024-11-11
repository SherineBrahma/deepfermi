# Arguments
pretraining_folder=$1
model_based_training_folder=$2
dataset_file_name=$3
cross_val_k=$4
cross_val_fold=$5
cuda_device=$6

# Activate environment
conda activate brahma01_env_1
cd /data/brahma01/deepfermi/invivo/script/

# Perfunsion Quantification Simulation Experiments
experiment_folder='/data/brahma01/deepfermi/invivo/Experiments/'

# # Model-agnostic pretraining
CUDA_VISIBLE_DEVICES=$cuda_device python ../main.py --project_name=$pretraining_folder  --dataset_file_name=$dataset_file_name --build_dataset_flag False --mode='pre_training' --train_from_ckpt False  --cross_val_flag True --cross_val_k=$cross_val_k --cross_val_fold=$cross_val_fold --unet_lr=10e-4 --unet_wd=10e-8
# Model-based training
mkdir $experiment_folder/$model_based_training_folder
cp $experiment_folder/$pretraining_folder/model/unet/unet_iter_300000 $experiment_folder/$model_based_training_folder/unet_load
CUDA_VISIBLE_DEVICES=$cuda_device python ../main.py --project_name=$model_based_training_folder  --dataset_file_name=$dataset_file_name --build_dataset_flag False --mode='fine_tuning' --train_from_ckpt True  --cross_val_flag True --cross_val_k=$cross_val_k --cross_val_fold=$cross_val_fold --unet_lr=10e-5 --unet_wd=10e-8