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
    view_map_flag = True
    save_path = '/data/brahma01/DCEPerfusionUQ/Simulation/Experiments/Test_snr_15/'
    fontsize = 12
    
    # Generating plot
    if view_flag==True:
        matplotlib.use('TkAgg')
    fig = plt.figure(figsize=(6,6))
    plt.title("Calibration Plot")
    for _, path in enumerate(complete_read_path):
        
        # Loading arrays
        eta_gen_error =  np.load(Path.joinpath(Path(path),'eta_gen_error.npy'))
        eta_gen_uncertainty =  np.load(Path.joinpath(Path(path),'eta_gen_uncertainty.npy'))
        seg =  np.load(Path.joinpath(Path(path),'seg.npy'))
        
        # Remove background
        eta_gen_error[np.repeat(seg[:,None,...],3,axis=1)==0]=0
        eta_gen_uncertainty[np.repeat(seg[:,None,...],3,axis=1)==0]=0
        
        # Visualize loaded maps
        i = 8
        if view_map_flag==True:
            matplotlib.use('TkAgg')
            plt.figure(figsize=(6,6))
            # plt.imshow(eta_gen_error[i,1,...])
            # plt.imshow(eta_gen_uncertainty[i,1,...])
            # plt.imshow(seg[i,...])
            # plt.hist(eta_gen_error[i,1,...][seg[i]==1].flatten(), bins=100, alpha=0.5, label='Target', color='red', density=True)
            # plt.hist(eta_gen_uncertainty[i,1,...][seg[i]==1].flatten(), bins=100, alpha=0.5, label='Target', color='blue', density=True)
            plt.hist(eta_gen_error[:,1,...][seg==1].flatten(), bins=100, alpha=0.5, label='Target', color='red', density=True)
            plt.hist(eta_gen_uncertainty[:,1,...][seg==1].flatten(), bins=100, alpha=0.5, label='Target', color='blue', density=True) 
            plt.show()
        
        
        # Remove background
        # flow_error = eta_gen_error[i:i+1,0,...][seg[i:i+1]==1]
        # flow_uncertainty = eta_gen_uncertainty[i:i+1,0,...][seg[i:i+1]==1]
        # delay_error = eta_gen_error[i:i+1,1,...][seg[i:i+1]==1]
        # delay_uncertainty = eta_gen_uncertainty[i:i+1,1,...][seg[i:i+1]==1]
        # decay_error = eta_gen_error[i:i+1,2,...][seg[i:i+1]==1]
        # decay_uncertainty = eta_gen_uncertainty[i:i+1,2,...][seg[i:i+1]==1]
        flow_error = eta_gen_error[:,0,...][seg==1]
        flow_uncertainty = eta_gen_uncertainty[:,0,...][seg==1]
        delay_error = eta_gen_error[:,1,...][seg==1]
        delay_uncertainty = eta_gen_uncertainty[:,1,...][seg==1]
        decay_error = eta_gen_error[:,2,...][seg==1]
        decay_uncertainty = eta_gen_uncertainty[:,2,...][seg==1]
        
        # Construct mask to sample pixels
        threshold = 3
        flow_rad = np.sqrt(flow_error**2+flow_uncertainty**2)
        flow_rad_MAD = np.median(np.abs(flow_rad-np.median(flow_rad)))
        flow_thres = np.median(flow_rad)+threshold*flow_rad_MAD
        delay_rad = np.sqrt(delay_error**2+delay_uncertainty**2)
        delay_rad_MAD = np.median(np.abs(delay_rad-np.median(delay_rad)))
        delay_thres = np.median(delay_rad)+threshold*delay_rad_MAD
        decay_rad = np.sqrt(decay_error**2+decay_uncertainty**2)
        decay_rad_MAD = np.median(np.abs(decay_rad-np.median(decay_rad)))
        decay_thres = np.median(decay_rad)+threshold*decay_rad_MAD
        
        # Sample points
        flow_error, flow_uncertainty = flow_error[flow_rad<flow_thres], flow_uncertainty[flow_rad<flow_thres]
        delay_error, delay_uncertainty = delay_error[delay_rad<delay_thres], delay_uncertainty[delay_rad<delay_thres]
        decay_error, decay_uncertainty = decay_error[decay_rad<decay_thres], decay_uncertainty[decay_rad<decay_thres]
        
        # Calculate normalizing constant
        flow_C = (flow_error[np.argmax(flow_error)], 
                  flow_error[np.argmax(flow_error)]
                  ) if flow_error.max()>=flow_uncertainty.max() else (
                      flow_uncertainty[np.argmax(flow_uncertainty)], 
                      flow_uncertainty[np.argmax(flow_uncertainty)])
        delay_C = (delay_error[np.argmax(delay_error)], 
                  delay_error[np.argmax(delay_error)]
                  ) if delay_error.max()>=delay_uncertainty.max() else (
                      delay_uncertainty[np.argmax(delay_uncertainty)], 
                      delay_uncertainty[np.argmax(delay_uncertainty)])
        decay_C = (decay_error[np.argmax(decay_error)], 
                  decay_error[np.argmax(decay_error)]
                  ) if decay_error.max()>=decay_uncertainty.max() else (
                      decay_uncertainty[np.argmax(decay_uncertainty)], 
                      decay_uncertainty[np.argmax(decay_uncertainty)])
                  
        # Normalize
        flow_error, flow_uncertainty = flow_error/flow_C[0], flow_uncertainty/flow_C[1]
        delay_error, delay_uncertainty = delay_error/delay_C[0], delay_uncertainty/delay_C[1]
        decay_error, decay_uncertainty = decay_error/decay_C[0], decay_uncertainty/decay_C[1]
        
        # Scatter plot
        ideal = np.linspace(0, 1, 200000)
        # plt.plot(ideal, ideal, linewidth=1, linestyle="dashed", label='ideal', color='black')
        # plt.scatter(flow_error, flow_uncertainty, s=10, label='flow', color='red', alpha=0.05)
        # plt.scatter(delay_error, delay_uncertainty, s=10, label='delay', color='green', alpha=0.05)
        # plt.scatter(decay_error, decay_uncertainty, s=10, label='decay', color='blue', alpha=0.05)
        
        plt.hist(flow_uncertainty.flatten()/flow_error.flatten(), bins=100, alpha=0.5, label='flow', color='red', density=True)
        plt.hist(delay_uncertainty.flatten()/delay_error.flatten(), bins=100, alpha=0.5, label='delay', color='green', density=True)
        plt.hist(decay_uncertainty.flatten()/decay_error.flatten(), bins=100, alpha=0.5, label='decay', color='blue', density=True)
        
        # plt.hist2d(flow_error, flow_uncertainty, bins=100)
        # plt.hist2d(delay_error, delay_uncertainty, bins=100)
        # plt.hist2d(decay_error, decay_uncertainty, bins=100)
    plt.legend(loc="upper right")
    plt.ylabel("Uncertainty", fontsize=fontsize)
    plt.xlabel("RMSE", fontsize=fontsize)
    # plt.xlim(0,1)
    # plt.ylim(0,1)
    if view_flag==True:
        plt.show()
    Path(save_path).mkdir(parents=True, exist_ok=True)    
    fig.savefig(Path.joinpath(Path(save_path), 'calib_plot'), dpi=500)

if __name__ == "__main__":
    main()