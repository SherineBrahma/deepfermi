# Project Directory
project_directory='/data/brahma01/deepfermi/invivo/script/'

# Testing Fold 1
SCREEN_NAME='Test_cross_val_5_fold_1'
project_name='Test_cross_val_5_fold_1'
read_project_name='02_PINN_cross_val_5_fold_1'
dataset_file_name='invivo_perfusion_data.npz'
cuda_device=1
# Start screen and start training
if screen -ls | grep -q "\b${SCREEN_NAME}\b"; then
  echo "Screen already exists!"
  screen -S $SCREEN_NAME -X stuff "cd $project_directory; source test_launcher.sh $project_name $read_project_name $dataset_file_name $cuda_device^M"
else 
  screen -S $SCREEN_NAME
  screen -S $SCREEN_NAME -X stuff "source test_launcher.sh $project_name $read_project_name $dataset_file_name $nspokes $cuda_device^M"
fi

# Testing Fold 2
SCREEN_NAME='Test_cross_val_5_fold_2'
project_name='Test_cross_val_5_fold_2'
read_project_name='04_PINN_cross_val_5_fold_2'
dataset_file_name='invivo_perfusion_data.npz'
cuda_device=2
# Start screen and start training
if screen -ls | grep -q "\b${SCREEN_NAME}\b"; then
  echo "Screen already exists!"
  screen -S $SCREEN_NAME -X stuff "cd $project_directory; source test_launcher.sh $project_name $read_project_name $dataset_file_name $cuda_device^M"
else 
  screen -S $SCREEN_NAME
  screen -S $SCREEN_NAME -X stuff "source test_launcher.sh $project_name $read_project_name $dataset_file_name $nspokes $cuda_device^M"
fi

# Testing Fold 3
SCREEN_NAME='Test_cross_val_5_fold_3'
project_name='Test_cross_val_5_fold_3'
read_project_name='06_PINN_cross_val_5_fold_3'
dataset_file_name='invivo_perfusion_data.npz'
cuda_device=1
# Start screen and start training
if screen -ls | grep -q "\b${SCREEN_NAME}\b"; then
  echo "Screen already exists!"
  screen -S $SCREEN_NAME -X stuff "cd $project_directory; source test_launcher.sh $project_name $read_project_name $dataset_file_name $cuda_device^M"
else 
  screen -S $SCREEN_NAME
  screen -S $SCREEN_NAME -X stuff "source test_launcher.sh $project_name $read_project_name $dataset_file_name $nspokes $cuda_device^M"
fi

# Testing Fold 4
SCREEN_NAME='Test_cross_val_5_fold_4'
project_name='Test_cross_val_5_fold_4'
read_project_name='08_PINN_cross_val_5_fold_4'
dataset_file_name='invivo_perfusion_data.npz'
cuda_device=1
# Start screen and start training
if screen -ls | grep -q "\b${SCREEN_NAME}\b"; then
  echo "Screen already exists!"
  screen -S $SCREEN_NAME -X stuff "cd $project_directory; source test_launcher.sh $project_name $read_project_name $dataset_file_name $cuda_device^M"
else 
  screen -S $SCREEN_NAME
  screen -S $SCREEN_NAME -X stuff "source test_launcher.sh $project_name $read_project_name $dataset_file_name $nspokes $cuda_device^M"
fi

# Testing Fold 5
SCREEN_NAME='Test_cross_val_5_fold_5'
project_name='Test_cross_val_5_fold_5'
read_project_name='10_PINN_cross_val_5_fold_5'
dataset_file_name='invivo_perfusion_data.npz'
cuda_device=2
# Start screen and start training
if screen -ls | grep -q "\b${SCREEN_NAME}\b"; then
  echo "Screen already exists!"
  screen -S $SCREEN_NAME -X stuff "cd $project_directory; source test_launcher.sh $project_name $read_project_name $dataset_file_name $cuda_device^M"
else 
  screen -S $SCREEN_NAME
  screen -S $SCREEN_NAME -X stuff "source test_launcher.sh $project_name $read_project_name $dataset_file_name $nspokes $cuda_device^M"
fi