# Project Directory
environment='deepfermi'
deepfermi_home_dir=$(dirname "$(dirname "$(realpath "$0")")")
code_directory=$deepfermi_home_dir/src/deepfermi/
script_directory=$deepfermi_home_dir/script/
experiments_directory=$deepfermi_home_dir/experiments/

# Testing pretrained deepfermi network with outliers retained
SCREEN_NAME='test_deepfermi_pretrained_outlier_retained'
project_name='01_02_test_deepfermi_pretrained_outlier_retained'
read_project_name='01_01_deepfermi_pretrained'
dataset_file_name='dce_perfusion_data.npz'
config_path=$deepfermi_home_dir/config/test_config_outlier_retained.yaml
cuda_device=1
# Start screen and start training
if screen -ls | grep -q "\b${SCREEN_NAME}\b"; then
  echo "Screen already exists!"
  screen -S $SCREEN_NAME -X stuff "cd $script_directory; source test_launcher.sh $project_name $read_project_name $dataset_file_name $cuda_device $environment $code_directory $config_path^M"
else
  screen -S $SCREEN_NAME
  screen -S $SCREEN_NAME -X stuff "cd $script_directory; source test_launcher.sh $project_name $read_project_name $dataset_file_name $nspokes $cuda_device $environment $code_directory $config_path^M"
fi
sleep 5
screen -d $SCREEN_NAME

# Adding a delay to avoid overwriting of testing data with ahd without outliers
sleep 1m

# Testing pretrained deepfermi network with outliers removed
SCREEN_NAME='test_deepfermi_pretrained_outlier_removed'
project_name='01_03_test_deepfermi_pretrained_outlier_removed'
read_project_name='01_01_deepfermi_pretrained'
dataset_file_name='dce_perfusion_data.npz'
config_path=$deepfermi_home_dir/config/test_config_outlier_removed.yaml
cuda_device=1
# Start screen and start training
if screen -ls | grep -q "\b${SCREEN_NAME}\b"; then
  echo "Screen already exists!"
  screen -S $SCREEN_NAME -X stuff "cd $script_directory; source test_launcher.sh $project_name $read_project_name $dataset_file_name $cuda_device $environment $code_directory $config_path^M"
else
  screen -S $SCREEN_NAME
  screen -S $SCREEN_NAME -X stuff "cd $script_directory; source test_launcher.sh $project_name $read_project_name $dataset_file_name $nspokes $cuda_device $environment $code_directory $config_path^M"
fi
sleep 5
screen -d $SCREEN_NAME
