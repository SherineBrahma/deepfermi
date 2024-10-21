# Arguments
pretraining_folder=$1
model_based_training_folder=$2
cuda_device=$3

# echo 'sleeping'
# sleep 15
# echo 'starting'

# Activate environment
cd /data/general/ANACONDA/anaconda3
source bin/activate
conda activate brahma01_env_1
cd /data/brahma01/DCEPerfusion/InVivo

# Perfunsion Quantification Invivo Experiments
experiment_folder='/data/brahma01/DCEPerfusion/InVivo/Experiments/'

# Model-agnostic pretraining
CUDA_VISIBLE_DEVICES=$cuda_device python main.py --config_path='/data/brahma01/DCEPerfusion/InVivo/config_files/Deterministic_MANN.yaml' --project_name=$pretraining_folder
# Model-based training
mkdir $experiment_folder/$model_based_training_folder
cp $experiment_folder/$pretraining_folder/unet_model/unet_iter_300000 $experiment_folder/$model_based_training_folder/unet_load
CUDA_VISIBLE_DEVICES=$cuda_device python main.py --config_path='/data/brahma01/DCEPerfusion/InVivo/config_files/Deterministic_PINN.yaml' --project_name=$model_based_training_folder