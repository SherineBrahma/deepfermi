# Project Directory
project_directory='/data/brahma01/deepfermi/invivo/script/'

# # With Vanilla Encoder
# # # Training Arguments
# SCREEN_NAME='Normalized'
# pretraining_folder='01_MANN_Normalized'
# model_based_training_folder='02_PINN_Normalized'
# dataset_file_name='invivo_perfusion_data.npz'
# cross_val_k=5
# cross_val_fold=1
# cuda_device=0
# # Start screen and start training
# if screen -ls | grep -q "\b${SCREEN_NAME}\b"; then
#   echo "Screen already exists!"
#   screen -S $SCREEN_NAME -X stuff "cd $project_directory; source train_launcher.sh $pretraining_folder $model_based_training_folder $dataset_file_name $cross_val_k $cross_val_fold $cuda_device^M"
# else 
#   screen -S $SCREEN_NAME
#   screen -S $SCREEN_NAME -X stuff "source train_launcher.sh $pretraining_folder $model_based_training_folder $dataset_file_name $cross_val_k $cross_val_fold $cuda_device^M"
# fi

# # With Vanilla Encoder
# # # Training Arguments
# SCREEN_NAME='Un_Normalized'
# pretraining_folder='01_MANN_Un_Normalized'
# model_based_training_folder='02_PINN_Un_Normalized'
# dataset_file_name='invivo_perfusion_data.npz'
# cross_val_k=5
# cross_val_fold=1
# cuda_device=1
# # Start screen and start training
# if screen -ls | grep -q "\b${SCREEN_NAME}\b"; then
#   echo "Screen already exists!"
#   screen -S $SCREEN_NAME -X stuff "cd $project_directory; source train_launcher.sh $pretraining_folder $model_based_training_folder $dataset_file_name $cross_val_k $cross_val_fold $cuda_device^M"
# else 
#   screen -S $SCREEN_NAME
#   screen -S $SCREEN_NAME -X stuff "source train_launcher.sh $pretraining_folder $model_based_training_folder $dataset_file_name $cross_val_k $cross_val_fold $cuda_device^M"
# fi

# # With Vanilla Encoder
# # # Training Arguments
# SCREEN_NAME='Normalized_ReLU'
# pretraining_folder='01_MANN_Normalized_ReLU'
# model_based_training_folder='02_PINN_Normalized_ReLU'
# dataset_file_name='invivo_perfusion_data.npz'
# cross_val_k=5
# cross_val_fold=1
# cuda_device=2
# # Start screen and start training
# if screen -ls | grep -q "\b${SCREEN_NAME}\b"; then
#   echo "Screen already exists!"
#   screen -S $SCREEN_NAME -X stuff "cd $project_directory; source train_launcher.sh $pretraining_folder $model_based_training_folder $dataset_file_name $cross_val_k $cross_val_fold $cuda_device^M"
# else 
#   screen -S $SCREEN_NAME
#   screen -S $SCREEN_NAME -X stuff "source train_launcher.sh $pretraining_folder $model_based_training_folder $dataset_file_name $cross_val_k $cross_val_fold $cuda_device^M"
# fi





# With Vanilla Encoder
# # Training Arguments
SCREEN_NAME='cross_val_5_fold_1'
pretraining_folder='01_MANN_cross_val_5_fold_1'
model_based_training_folder='02_PINN_cross_val_5_fold_1'
dataset_file_name='invivo_perfusion_data.npz'
cross_val_k=5
cross_val_fold=1
cuda_device=0
# Start screen and start training
if screen -ls | grep -q "\b${SCREEN_NAME}\b"; then
  echo "Screen already exists!"
  screen -S $SCREEN_NAME -X stuff "cd $project_directory; source train_launcher.sh $pretraining_folder $model_based_training_folder $dataset_file_name $cross_val_k $cross_val_fold $cuda_device^M"
else 
  screen -S $SCREEN_NAME
  screen -S $SCREEN_NAME -X stuff "source train_launcher.sh $pretraining_folder $model_based_training_folder $dataset_file_name $cross_val_k $cross_val_fold $cuda_device^M"
fi

# With Vanilla Encoder
# # Training Arguments
SCREEN_NAME='cross_val_5_fold_2'
pretraining_folder='03_MANN_cross_val_5_fold_2'
model_based_training_folder='04_PINN_cross_val_5_fold_2'
dataset_file_name='invivo_perfusion_data.npz'
cross_val_k=5
cross_val_fold=2
cuda_device=1
# Start screen and start training
if screen -ls | grep -q "\b${SCREEN_NAME}\b"; then
  echo "Screen already exists!"
  screen -S $SCREEN_NAME -X stuff "cd $project_directory; source train_launcher.sh $pretraining_folder $model_based_training_folder $dataset_file_name $cross_val_k $cross_val_fold $cuda_device^M"
else 
  screen -S $SCREEN_NAME
  screen -S $SCREEN_NAME -X stuff "source train_launcher.sh $pretraining_folder $model_based_training_folder $dataset_file_name $cross_val_k $cross_val_fold $cuda_device^M"
fi

# With Vanilla Encoder
# # Training Arguments
SCREEN_NAME='cross_val_5_fold_3'
pretraining_folder='05_MANN_cross_val_5_fold_3'
model_based_training_folder='06_PINN_cross_val_5_fold_3'
dataset_file_name='invivo_perfusion_data.npz'
cross_val_k=5
cross_val_fold=3
cuda_device=2
# Start screen and start training
if screen -ls | grep -q "\b${SCREEN_NAME}\b"; then
  echo "Screen already exists!"
  screen -S $SCREEN_NAME -X stuff "cd $project_directory; source train_launcher.sh $pretraining_folder $model_based_training_folder $dataset_file_name $cross_val_k $cross_val_fold $cuda_device^M"
else 
  screen -S $SCREEN_NAME
  screen -S $SCREEN_NAME -X stuff "source train_launcher.sh $pretraining_folder $model_based_training_folder $dataset_file_name $cross_val_k $cross_val_fold $cuda_device^M"
fi

# With Vanilla Encoder
# # Training Arguments
SCREEN_NAME='cross_val_5_fold_4'
pretraining_folder='07_MANN_cross_val_5_fold_4'
model_based_training_folder='08_PINN_cross_val_5_fold_4'
dataset_file_name='invivo_perfusion_data.npz'
cross_val_k=5
cross_val_fold=4
cuda_device=3
# Start screen and start training
if screen -ls | grep -q "\b${SCREEN_NAME}\b"; then
  echo "Screen already exists!"
  screen -S $SCREEN_NAME -X stuff "cd $project_directory; source train_launcher.sh $pretraining_folder $model_based_training_folder $dataset_file_name $cross_val_k $cross_val_fold $cuda_device^M"
else 
  screen -S $SCREEN_NAME
  screen -S $SCREEN_NAME -X stuff "source train_launcher.sh $pretraining_folder $model_based_training_folder $dataset_file_name $cross_val_k $cross_val_fold $cuda_device^M"
fi

# With Vanilla Encoder
# # Training Arguments
SCREEN_NAME='cross_val_5_fold_5'
pretraining_folder='09_MANN_cross_val_5_fold_5'
model_based_training_folder='10_PINN_cross_val_5_fold_5'
dataset_file_name='invivo_perfusion_data.npz'
cross_val_k=5
cross_val_fold=5
cuda_device=3
# Start screen and start training
if screen -ls | grep -q "\b${SCREEN_NAME}\b"; then
  echo "Screen already exists!"
  screen -S $SCREEN_NAME -X stuff "cd $project_directory; source train_launcher.sh $pretraining_folder $model_based_training_folder $dataset_file_name $cross_val_k $cross_val_fold $cuda_device^M"
else 
  screen -S $SCREEN_NAME
  screen -S $SCREEN_NAME -X stuff "source train_launcher.sh $pretraining_folder $model_based_training_folder $dataset_file_name $cross_val_k $cross_val_fold $cuda_device^M"
fi
