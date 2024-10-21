import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main() -> None:
    
    # Folders to be read
    complete_read_path = ['/data/brahma01/DCEPerfusionUQ/Simulation/Experiments/Test_snr_15/']
    
    # Settings for saving the plots
    view_flag = True    
    save_path = '/data/brahma01/DCEPerfusionUQ/Simulation/Experiments/Test_snr_15/'
    
    # Generating plot
    if view_flag==True:
        matplotlib.use('TkAgg')
    fig = plt.figure(figsize=(6,6))
    plt.title("Calibration Plot")
    for i, path in enumerate(complete_read_path):        
        prob_flow_gnd_in_CI = np.load(Path.joinpath(Path(path),'prob_flow_gnd_in_CI.npy'))
        prob_flow_gen_in_CI = np.load(Path.joinpath(Path(path),'prob_flow_gen_in_CI.npy'))        
        prob_delay_gnd_in_CI = np.load(Path.joinpath(Path(path),'prob_delay_gnd_in_CI.npy'))
        prob_delay_gen_in_CI = np.load(Path.joinpath(Path(path),'prob_delay_gen_in_CI.npy'))        
        prob_decay_gnd_in_CI = np.load(Path.joinpath(Path(path),'prob_decay_gnd_in_CI.npy'))
        prob_decay_gen_in_CI = np.load(Path.joinpath(Path(path),'prob_decay_gen_in_CI.npy'))
        if i==0:
            plt.plot(prob_flow_gnd_in_CI, prob_flow_gnd_in_CI, linewidth=1, linestyle="dashed", label='perfect model', color='black')
        plt.plot(prob_flow_gen_in_CI, prob_flow_gnd_in_CI, linewidth=1, label='flow', color='red')
        plt.plot(prob_delay_gen_in_CI, prob_delay_gnd_in_CI, linewidth=1, label='delay', color='green')
        plt.plot(prob_decay_gen_in_CI, prob_decay_gnd_in_CI, linewidth=1, label='decay', color='blue')
    plt.legend(loc="upper right")
    if view_flag==True:
        plt.show()
    Path(save_path).mkdir(parents=True, exist_ok=True)    
    fig.savefig(Path.joinpath(Path(save_path), 'calib_plot'), dpi=500)

if __name__ == "__main__":
    main()