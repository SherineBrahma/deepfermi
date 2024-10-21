# Test model-based perfusion quantification
mkdir /data/brahma01/DCEPerfusion/InVivo/Experiments/Test_18
CUDA_VISIBLE_DEVICES=0 python test_deterministic.py --config_path='/data/brahma01/DCEPerfusion/InVivo/TestInVivo.yaml' --read_project_name='18_Deterministic_PINN' --project_name='Test_18'

