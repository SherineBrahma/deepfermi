# Project Directory
environment='deepfermi'
deepfermi_home_dir=$(dirname "$(dirname "$(realpath "$0")")")
code_directory=$deepfermi_home_dir/src/deepfermi/
script_directory=$deepfermi_home_dir/script/
experiments_directory=$deepfermi_home_dir/experiments/

# DeepFermi Training 
# # Arguments
SCREEN_NAME='deepfermi_train'
pretraining_folder='02_01_model_agnostic'
model_based_training_folder='02_02_model_based'
dataset_file_name='dce_perfusion_data.npz'
cross_val_k=2
cross_val_fold=1
cuda_device=0
# # Start screen and start training
if screen -ls | grep -q "\b${SCREEN_NAME}\b"; then
  echo "Screen already exists!"
  screen -S $SCREEN_NAME -X stuff "cd $script_directory; source train_launcher.sh $pretraining_folder $model_based_training_folder $dataset_file_name $cross_val_k $cross_val_fold $cuda_device $environment $code_directory $script_directory $experiments_directory^M"
else 
  screen -S $SCREEN_NAME
  screen -S $SCREEN_NAME -X stuff "cd $script_directory; source train_launcher.sh $pretraining_folder $model_based_training_folder $dataset_file_name $cross_val_k $cross_val_fold $cuda_device $environment $code_directory $script_directory $experiments_directory^M"
fi
