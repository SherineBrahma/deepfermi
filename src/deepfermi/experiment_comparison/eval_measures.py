import numpy as np
from pathlib import Path

from skimage import measure

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main() -> None:
    
    # Folders to be read    
    label_list = ['10_PINN_20']
    complete_read_path = ['/data/brahma01/DCEPerfusionUQ/Simulation/Experiments/Test_snr_15/']
    
    # # Data array paths    
    # complete_read_path = ['/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_1125_test_full/', 
    #                       '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_1125_test_1125/',
    #                       '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_1125_test_750/',
    #                       '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_1125_test_540/',
    #                       '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_750_test_full/', 
    #                       '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_750_test_1125/',
    #                       '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_750_test_750/',
    #                       '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_750_test_540/',
    #                       '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_540_test_full/', 
    #                       '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_540_test_1125/',
    #                       '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_540_test_750/',
    #                       '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_540_test_540/']
    
    # Save settings
    # view_flag = True    
    save_path = '/data/brahma01/DCEPerfusionUQ/Simulation/Experiments/Test_snr_15/'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Generating plot
    eval_metrics_dic = {}
    eval_metrics_dic['Variable'] = []
    eval_metrics_dic['Flow NMAE'] = []
    eval_metrics_dic['Delay NMAE'] = []
    eval_metrics_dic['Decay NMAE'] = []
    eval_metrics_dic['Flow Calib. Error'] = []
    eval_metrics_dic['Delay Calib. Error'] = []
    eval_metrics_dic['Decay Calib. Error'] = []
    open(Path.joinpath(Path(save_path), 'eval_metrics.txt'), 'w').close()
    for i, path in enumerate(complete_read_path):
        # Loading arrays
        eta_gnd =  np.load(Path.joinpath(Path(path),'eta.npy'))
        eta_gen_error =  np.load(Path.joinpath(Path(path),'eta_gen_error.npy'))
        seg =  np.load(Path.joinpath(Path(path),'seg.npy'))
        prob_flow_gnd_in_CI = np.load(Path.joinpath(Path(path),'prob_flow_gnd_in_CI.npy'))
        prob_flow_gen_in_CI = np.load(Path.joinpath(Path(path),'prob_flow_gen_in_CI.npy'))        
        prob_delay_gnd_in_CI = np.load(Path.joinpath(Path(path),'prob_delay_gnd_in_CI.npy'))
        prob_delay_gen_in_CI = np.load(Path.joinpath(Path(path),'prob_delay_gen_in_CI.npy'))        
        prob_decay_gnd_in_CI = np.load(Path.joinpath(Path(path),'prob_decay_gnd_in_CI.npy'))
        prob_decay_gen_in_CI = np.load(Path.joinpath(Path(path),'prob_decay_gen_in_CI.npy'))
        
        # Calculating measures
        flow_nmae = (np.abs(eta_gen_error[:,0,...][seg==1]).sum()/np.abs(eta_gnd[:,0,...][seg==1]).sum())
        delay_nmae = (np.abs(eta_gen_error[:,1,...][seg==1]).sum()/np.abs(eta_gnd[:,1,...][seg==1]).sum())
        decay_nmae = (np.abs(eta_gen_error[:,2,...][seg==1]).sum()/np.abs(eta_gnd[:,2,...][seg==1]).sum())
        flow_calib_error = np.sqrt(((prob_flow_gnd_in_CI-prob_flow_gen_in_CI)**2).sum())
        delay_calib_error = np.sqrt(((prob_delay_gnd_in_CI-prob_delay_gen_in_CI)**2).sum())
        decay_calib_error = np.sqrt(((prob_decay_gnd_in_CI-prob_decay_gen_in_CI)**2).sum())
        
        # Recording in a text file       
        with open(Path.joinpath(Path(save_path), 'eval_metrics.txt'),'a') as file:
            file.write(label_list[i]+"\n")
            for ast in len(label_list[i])*['*']:
                file.write(ast)
            file.write("\n")
            file.write("%s = %f\n" % ("Flow NMAE", flow_nmae))
            file.write("%s = %f\n" % ("Delay NMAE", delay_nmae))
            file.write("%s = %f\n" % ("Decay NMAE", decay_nmae))
            file.write("%s = %f\n" % ("Flow Calib. Error", flow_calib_error))
            file.write("%s = %f\n" % ("Delay Calib. Error", delay_calib_error))
            file.write("%s = %f\n" % ("Decay Calib. Error", decay_calib_error))
            # file.write("%s = %f\n" % (uq_time_str, x_uq_sec_per_smpl))
            file.write("\n")
            
        # Recording values
        eval_metrics_dic['Variable'].append(label_list[i])
        eval_metrics_dic['Flow NMAE'].append(flow_nmae)
        eval_metrics_dic['Delay NMAE'].append(delay_nmae)
        eval_metrics_dic['Decay NMAE'].append(decay_nmae)
        eval_metrics_dic['Flow Calib. Error'].append(flow_calib_error)
        eval_metrics_dic['Delay Calib. Error'].append(delay_calib_error)
        eval_metrics_dic['Decay Calib. Error'].append(decay_calib_error)
        
    np.save(Path.joinpath(Path(save_path), "eval_metrics"), eval_metrics_dic)

if __name__ == "__main__":
    main()
                            