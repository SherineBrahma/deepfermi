# Project Directory
project_directory='/data/brahma01/DCEPerfusion/InVivo/'
# Training Arguments
SCREEN_NAME='30_31_Invivo'
pretraining_folder='30_Deterministic_MANN'
model_based_training_folder='31_Deterministic_PINN'
cuda_device=3
# Start screen and start training
if screen -ls | grep -q "\b${SCREEN_NAME}\b"; then
  echo "Screen already exists!"
  screen -S $SCREEN_NAME -X stuff "cd $project_directory; source DeterministicTraining.sh $pretraining_folder $model_based_training_folder $cuda_device^M"
else 
  screen -S $SCREEN_NAME
  screen -S $SCREEN_NAME -X stuff "source DeterministicTraining.sh $pretraining_folder $model_based_training_folder $cuda_device^M"
fi
