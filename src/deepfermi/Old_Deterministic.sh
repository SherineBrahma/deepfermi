# Perfunsion Quantification Simulation Experiments
experiment_folder='/data/brahma01/DCEPerfusion/InVivo/Experiments/'

# # # FULL TRAINING
pretraining_folder='22_Deterministic_MANN'
model_based_training_folder='23_Deterministic_PINN'
# Model-agnostic pretraining
# CUDA_VISIBLE_DEVICES=3 python main.py --config_path='/data/brahma01/DCEPerfusion/InVivo/config_files/Deterministic_MANN.yaml' --project_name=$pretraining_folder
# Model-based training
mkdir $experiment_folder/$model_based_training_folder
cp $experiment_folder/$pretraining_folder/unet_model/unet_iter_300000 $experiment_folder/$model_based_training_folder/unet_load
CUDA_VISIBLE_DEVICES=3 python main.py --config_path='/data/brahma01/DCEPerfusion/InVivo/config_files/Deterministic_PINN.yaml' --project_name=$model_based_training_folder