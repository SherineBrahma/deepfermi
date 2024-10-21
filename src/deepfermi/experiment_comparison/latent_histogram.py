import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main() -> None:
        
    # Folders to be read    
    hist_name_list = ['latent_histogram']
    complete_read_path = ['/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_540_test_540/']
        
    # Save settings
    samp_points = 10000
    bins = 100
    fontsize = 12
    view_flag = True
    use_all_samples = True
    save_path = '/data/brahma01/WAEUQ/Experiments/Test_cWAE/'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Generating plot
    for i, path in enumerate(complete_read_path):
        
        # Loading arrays
        qz =  np.load(Path.joinpath(Path(path),'qz.npy'))
        
        # Sample latent space marginal
        if use_all_samples==True:
            qz_i = qz
        else:            
            # Given xI, draw qz_samp. We only have access to the considered expected value q(z|xI,xf).
            N = qz.shape[0]
            rand_indx = np.random.randint(0, (N - 1), size=1)
            qz_i = qz[rand_indx,...]
        
        # Calculate statistics
        qz_i_mean = qz_i.mean()
        qz_i_std = qz_i.std()
        
        # Construct mask to sample pixels
        mask = np.arange(0,qz_i.size)
        np.random.shuffle(mask)
        mask = mask[0:samp_points]
        
        # Sample pixels from qz. Also, sample from gaussian noise
        qz_i_samp = qz_i.flatten()[mask]
        pz_samp = np.random.normal(0,1,qz_i_samp.size)

        # Histogram Visualization
        figsize=(8, 6)
        suptitle = 'Latent Space Histogram'
        generated_label = 'Mean: ' + str(round(qz_i_mean, 3)) + ', Std: '  + str(round(qz_i_std, 3))         
        if view_flag==True:
            matplotlib.use('TkAgg')     
        histplot = plt.figure(figsize=figsize)
        histplot.suptitle(suptitle)
        plt.hist(pz_samp, bins=bins, alpha=0.5, label='Target', color='red', density=True) 
        plt.hist(qz_i_samp, bins=bins, alpha=0.5, label=generated_label, color='blue', density=True)
        plt.xlabel('Latent Pixel Value', fontsize=fontsize)
        plt.ylabel('Normalized Frequency', fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.legend(loc="upper right", fontsize=fontsize)
        plt.axis('on')
        if view_flag==True:
            plt.show()
        histplot.savefig(Path.joinpath(Path(save_path), hist_name_list[i]), dpi=500)

if __name__ == "__main__":
    main()