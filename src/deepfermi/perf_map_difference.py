import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu

# NRMSE
def nrmse(x1, x2):
    x = np.linalg.norm(x1-x2)/np.linalg.norm(x2)
    return x

# SNR_Change
# Load perfusion maps
path_1 = Path('/data/brahma01/DCEPerfusion/InVivo/Experiments/Test_test/')
test_data_dic = dict(np.load(Path.joinpath(path_1, 'test_data_dic.npz'), allow_pickle=True))
eta_net_1 = test_data_dic['eta_net'].copy()
eta_lbfgs_1 = test_data_dic['eta_lbfgs'].copy()
seg_lbfgs_1 = test_data_dic['seg'].copy()

# Load perfusion maps
path_2 = Path('/data/brahma01/DCEPerfusion/InVivo/Experiments/Test_SNR_0_5/')
test_data_dic = dict(np.load(Path.joinpath(path_2, 'test_data_dic.npz'), allow_pickle=True))
eta_net_2 = test_data_dic['eta_net'].copy()
eta_lbfgs_2 = test_data_dic['eta_lbfgs'].copy()
seg_lbfgs_2 = test_data_dic['seg'].copy()

# Save difference perfusion maps
delta_eta_lbfgs = np.sqrt((eta_lbfgs_1-eta_lbfgs_2)**2)
delta_eta_net = np.sqrt((eta_net_1-eta_net_2)**2)
test_data_dic['eta_lbfgs'] = np.sqrt((eta_lbfgs_1-eta_lbfgs_2)**2)
test_data_dic['eta_net'] = np.sqrt((eta_net_1-eta_net_2)**2) 
# Saving file
save_folder = 'Test_test_SNR_0_5_change' # 'Test_test_SNR_0_5_change'
save_path = Path.joinpath(Path('/data/brahma01/DCEPerfusion/InVivo/Experiments/'), save_folder)
Path(save_path).mkdir(parents=True, exist_ok=True)
np.savez(Path.joinpath(save_path, 'test_data_dic'), **test_data_dic)

# General initialization
assert (seg_lbfgs_1==seg_lbfgs_2).all(), 'Segmentation tensor should match'
seg = seg_lbfgs_1 = seg_lbfgs_2

# Test to check if changing SNR changed lbfgs
# delta_flow_lbfgs = mannwhitneyu(eta_lbfgs_1[:,0,...][seg==1].flatten(), eta_lbfgs_2[:,0,...][seg==1].flatten())
# delta_delay_lbfgs = mannwhitneyu(eta_lbfgs_1[:,1,...][seg==1].flatten(), eta_lbfgs_2[:,1,...][seg==1].flatten())
# delta_decay_lbfgs = mannwhitneyu(eta_lbfgs_1[:,2,...][seg==1].flatten(), eta_lbfgs_2[:,2,...][seg==1].flatten())

delta_flow_lbfgs = nrmse(eta_lbfgs_1[:,0,...][seg==1], eta_lbfgs_2[:,0,...][seg==1])
delta_delay_lbfgs = nrmse(eta_lbfgs_1[:,1,...][seg==1], eta_lbfgs_2[:,1,...][seg==1])
delta_decay_lbfgs = nrmse(eta_lbfgs_1[:,2,...][seg==1], eta_lbfgs_2[:,2,...][seg==1])

# Test to check if changing SNR changed net
# delta_flow_net = mannwhitneyu(eta_net_1[:,0,...][seg==1].flatten(), eta_net_2[:,0,...][seg==1].flatten())
# delta_delay_net = mannwhitneyu(eta_net_1[:,1,...][seg==1].flatten(), eta_net_2[:,1,...][seg==1].flatten())
# delta_decay_net = mannwhitneyu(eta_net_1[:,2,...][seg==1].flatten(), eta_net_2[:,2,...][seg==1].flatten())

delta_flow_net = nrmse(eta_net_1[:,0,...][seg==1], eta_net_2[:,0,...][seg==1])
delta_delay_net = nrmse(eta_net_1[:,1,...][seg==1], eta_net_2[:,1,...][seg==1])
delta_decay_net = nrmse(eta_net_1[:,2,...][seg==1], eta_net_2[:,2,...][seg==1])

# # Wilcoxon's test to compare 
# # Flow
# delta_flow = (delta_eta_lbfgs[:,0,...] - delta_eta_net[:,0,...])[seg==1].flatten()
# res_flow = wilcoxon(delta_flow)
# # Delay
# delta_delay = (delta_eta_lbfgs[:,1,...] - delta_eta_net[:,1,...])[seg==1].flatten()
# res_delay = wilcoxon(delta_delay)
# # Decay
# delta_decay = (delta_eta_lbfgs[:,2,...] - delta_eta_net[:,2,...])[seg==1].flatten()
# res_decay = wilcoxon(delta_decay)

print('Impact on adding SNR.')
print('Change in flow value.')
print('LBFGS: ' + str(delta_flow_lbfgs) + ', DeepFermi: ' + str(delta_flow_net))
print('Change in delay value.')
print('LBFGS: ' + str(delta_delay_lbfgs) + ', DeepFermi: ' + str(delta_delay_net))
print('Change in decay value.')
print('LBFGS: ' + str(delta_decay_lbfgs) + ', DeepFermi: ' + str(delta_decay_net))


#############################################################################

# Outlier_Change
# Load perfusion maps
path_1 = Path('/data/brahma01/DCEPerfusion/InVivo/Experiments/Test_test/')
test_data_dic = dict(np.load(Path.joinpath(path_1, 'test_data_dic.npz'), allow_pickle=True))
eta_net_1 = test_data_dic['eta_net'].copy()
eta_lbfgs_1 = test_data_dic['eta_lbfgs'].copy()
seg_lbfgs_1 = test_data_dic['seg'].copy()

# Load perfusion maps
path_2 = Path('/data/brahma01/DCEPerfusion/InVivo/Experiments/Test_outliers/')
test_data_dic = dict(np.load(Path.joinpath(path_2, 'test_data_dic.npz'), allow_pickle=True))
eta_net_2 = test_data_dic['eta_net'].copy()
eta_lbfgs_2 = test_data_dic['eta_lbfgs'].copy()
seg_lbfgs_2 = test_data_dic['seg'].copy()

# General initialization
assert (seg_lbfgs_1==seg_lbfgs_2).all(), 'Segmentation tensor should match'
seg = seg_lbfgs_1 = seg_lbfgs_2

# Save difference perfusion maps
delta_eta_lbfgs = np.sqrt((eta_lbfgs_1-eta_lbfgs_2)**2)
delta_eta_net = np.sqrt((eta_net_1-eta_net_2)**2)
test_data_dic['eta_lbfgs'] = delta_eta_lbfgs
test_data_dic['eta_net'] = delta_eta_net
# Saving file
save_folder = 'Test_test_outlier_change'
save_path = Path.joinpath(Path('/data/brahma01/DCEPerfusion/InVivo/Experiments/'), save_folder)
Path(save_path).mkdir(parents=True, exist_ok=True)
np.savez(Path.joinpath(save_path, 'test_data_dic'), **test_data_dic)

# Test to check if changing SNR changed lbfgs
# delta_flow_lbfgs = mannwhitneyu(eta_lbfgs_1[:,0,...][seg==1].flatten(), eta_lbfgs_2[:,0,...][seg==1].flatten())
# delta_delay_lbfgs = mannwhitneyu(eta_lbfgs_1[:,1,...][seg==1].flatten(), eta_lbfgs_2[:,1,...][seg==1].flatten())
# delta_decay_lbfgs = mannwhitneyu(eta_lbfgs_1[:,2,...][seg==1].flatten(), eta_lbfgs_2[:,2,...][seg==1].flatten())

delta_flow_lbfgs = nrmse(eta_lbfgs_1[:,0,...][seg==1], eta_lbfgs_2[:,0,...][seg==1])
delta_delay_lbfgs = nrmse(eta_lbfgs_1[:,1,...][seg==1], eta_lbfgs_2[:,1,...][seg==1])
delta_decay_lbfgs = nrmse(eta_lbfgs_1[:,2,...][seg==1], eta_lbfgs_2[:,2,...][seg==1])


# Test to check if changing SNR changed net
# delta_flow_net = mannwhitneyu(eta_net_1[:,0,...][seg==1].flatten(), eta_net_2[:,0,...][seg==1].flatten())
# delta_delay_net = mannwhitneyu(eta_net_1[:,1,...][seg==1].flatten(), eta_net_2[:,1,...][seg==1].flatten())
# delta_decay_net = mannwhitneyu(eta_net_1[:,2,...][seg==1].flatten(), eta_net_2[:,2,...][seg==1].flatten())

delta_flow_net = nrmse(eta_net_1[:,0,...][seg==1], eta_net_2[:,0,...][seg==1])
delta_delay_net = nrmse(eta_net_1[:,1,...][seg==1], eta_net_2[:,1,...][seg==1])
delta_decay_net = nrmse(eta_net_1[:,2,...][seg==1], eta_net_2[:,2,...][seg==1])

print('Impact on including outliers.')
print('Change in flow value.')
print('LBFGS: ' + str(delta_flow_lbfgs) + ', DeepFermi: ' + str(delta_flow_net))
print('Change in delay value.')
print('LBFGS: ' + str(delta_delay_lbfgs) + ', DeepFermi: ' + str(delta_delay_net))
print('Change in decay value.')
print('LBFGS: ' + str(delta_decay_lbfgs) + ', DeepFermi: ' + str(delta_decay_net))