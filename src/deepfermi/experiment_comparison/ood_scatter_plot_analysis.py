import numpy as np
from pathlib import Path

from skimage import measure

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main() -> None:
    
    # Folders to be read    
    label_list = ['R≈1', 'R≈7', None, 'R≈10', None, 'R≈14']
    color_list = ['red', 'blue', 'red', 'blue', 'red', 'blue']
    xaxis_name = 'Training Dataset Acceleration Factor'
    xaxis_values = [7, 10, 14]
    xaxis_labels = ['7', '10', '14']
    
    # OOD dictionary contains scenarios to be tested
    ood_dic = {}
    ood_dic['R'] = {}
    ood_dic['R']['7']=['1', '7']
    ood_dic['R']['10']=['1', '10']
    ood_dic['R']['14']=['1', '14']
    
    c_stdz = [(14.1360, 0.3618), (4.0778, 0.1955), (8.7593, 0.2404)]
    
    # Data array paths    
    complete_read_path = ['/data/brahma01/WAEUQ/Experiments/Failed_Test_nspoke_read_1125_test_full/', 
                          '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_1125_test_1125/',
                          '/data/brahma01/WAEUQ/Experiments/Failed_Test_nspoke_read_750_test_full/',
                          '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_750_test_750/',
                          '/data/brahma01/WAEUQ/Experiments/Failed_Test_nspoke_read_540_test_full/',
                          '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_540_test_540/']
    
    # Folders to be read    
    label_list = ['R≈1', 'R≈7', 'R≈10', 'R≈14', None, None, None, None, None, None, None, None]
    color_list = ['#000000', '#4E4187', '#97DB4F', '#5DA9E9', '#000000', '#4E4187', '#97DB4F', '#5DA9E9', '#000000', '#4E4187', '#97DB4F', '#5DA9E9']
    xaxis_name = 'Training Dataset Acceleration Factor'
    xaxis_labels = ['7', '10', '14']
    
    # OOD dictionary contains scenarios to be tested
    ood_dic = {}
    ood_dic['R'] = {}
    ood_dic['R']['7']=['1', '7', '10', '14']
    ood_dic['R']['10']=['1', '7', '10', '14']
    ood_dic['R']['14']=['1', '7', '10', '14']

    # c_stdz = [(14.1360, 0.3618), (4.0778, 0.1955), (8.7593, 0.2404)]
    
    c_stdz = [(0, 1), (0, 1), (0, 1)]

    # Data array paths    
    complete_read_path = ['/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_1125_test_full/', 
                            '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_1125_test_1125/',
                            '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_1125_test_750/',
                            '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_1125_test_540/',
                            '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_750_test_full/', 
                            '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_750_test_1125/',
                            '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_750_test_750/',
                            '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_750_test_540/',
                            '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_540_test_full/', 
                            '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_540_test_1125/',
                            '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_540_test_750/',
                            '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_540_test_540/']
    
    # Save settings
    # view_flag = True
    fontsize=15
    save_path = '/data/brahma01/mc_dropout_cine_mri/Experiments/Test_cWAE/'
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # Plot scatter-plot
    metric_list = ['OODScore', 'CalibError']
    lim_list = [(0.04, 0.15), (48, 58), (0.95, 1.0), (None, None)]
    for i, metric in enumerate(metric_list):
        
        # Load and plot metrics
        path_count = 0
        for j, inv_var in enumerate(ood_dic.keys()):
            
            figsize = (8,6)
            matplotlib.use('TkAgg')
            scatter_plot = plt.figure(figsize=figsize)            
            for k, train_var in enumerate(ood_dic[inv_var].keys()):
                
                mu = c_stdz[k][0]
                sigma = c_stdz[k][1]
                
                for l, test_var in enumerate(ood_dic[inv_var][train_var]):
                    
                    # Load dictionary
                    eval_metrics = np.load(Path.joinpath(Path(complete_read_path[path_count]),'eval_metrics.npy'), allow_pickle=True).item()                    
                    wdist = np.load(Path.joinpath(Path(complete_read_path[path_count]),'wdist.npy'), allow_pickle=True)
                    
                    if metric=='OODScore':
                        metric_plot = np.abs(((wdist-mu)/sigma).mean())
                        print(complete_read_path[path_count])               
                        width = 0.2
                        test_var_space = (width/(ood_dic[inv_var][train_var].__len__()-1))*l - width/2                    
                        yaxis = (np.random.normal(0,0.01,metric_plot.shape) + test_var_space + (k+1)).mean()
                    else:
                        metric_plot = np.abs(np.array(eval_metrics[metric]))
                        yaxis = np.random.normal(0,0.02,metric_plot.shape) + test_var_space + (k+1)
                    plt.scatter(yaxis, metric_plot, label=label_list[path_count], s=100, color=color_list[path_count])
                    plt.legend(loc="upper right", fontsize=fontsize)                    
                    plt.axis('on')
                    path_count+=1
            plt.xticks(np.arange(1,ood_dic[inv_var].__len__()+1,1), xaxis_labels, fontsize=fontsize)
            plt.ylabel(metric, fontsize=fontsize)
            plt.xlabel(xaxis_name, fontsize=fontsize)
            if metric=='OODScore':
                plt.ylim(bottom=None, top=None)
            plt.xlim(left=0, right=ood_dic[inv_var].__len__()+1)
            plt.show()
        
    a = 1
        
    
    
    
    # # Print metrics
    # metric_list = ['NRMSE', 'PSNR', 'SSIM', 'CalibError']
    # lim_list = [(0.04, 0.15), (48, 58), (0.95, 1.0), (None, None)]
    # for i, metric in enumerate(metric_list):
        
    #     # Load and plot metrics
    #     fig = plt.figure()
    #     max_metric = -np.inf
    #     min_metric = np.inf
    #     for j, path in enumerate(complete_read_path):
            
    #         # Load dictionary
    #         eval_metrics = np.load(Path.joinpath(Path(path),'eval_metrics.npy'), allow_pickle=True).item()            
    #         # Plot Metric
    #         fig.suptitle(metric)
    #         plt.plot(xaxis_values, eval_metrics[metric], label=label_list[j], linewidth=1, color=color_list[j], marker='o')
    #         plt.xticks(xaxis_values, xaxis_labels, fontsize=fontsize)
    #         plt.xlabel(xaxis_name)
    #         plt.ylabel(metric)
    #         plt.legend(loc="upper right", fontsize=fontsize)
    #         plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    #         # Calculating meta information
    #         max_metric = max(max(np.array(eval_metrics[metric])), max_metric)
    #         min_metric = min(min(np.array(eval_metrics[metric])), min_metric)
    #     # plt.yticks(np.round(np.linspace(0.95*min_metric, 1.2*max_metric, 4),3), fontsize=fontsize)
    #     plt.ylim(bottom=lim_list[i][0], top=lim_list[i][1])        
    #     plt.close()
    #     fig.savefig(Path.joinpath(Path(save_path), metric), dpi=500)

if __name__ == "__main__":
    main()
    

                        
                        
# # OOD dictionary contains scenarios to be tested
# ood_dic = {}
# ood_dic['R'] = {}
# ood_dic['R']['7']=['1', '7', '10', '14']
# ood_dic['R']['10']=['1', '7', '10', '14']
# ood_dic['R']['14']=['1', '7', '10', '14']

# c_stdz = [(14.1360, 0.3618), (4.0778, 0.1955), (8.7593, 0.2404)]

# # Data array paths    
# complete_read_path = ['/data/brahma01/WAEUQ/Experiments/Failed_Test_nspoke_read_1125_test_full/', 
#                         '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_1125_test_1125/',
#                         '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_1125_test_750/',
#                         '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_1125_test_540/',
#                         '/data/brahma01/WAEUQ/Experiments/Failed_Test_nspoke_read_750_test_full/', 
#                         '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_750_test_1125/',
#                         '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_750_test_750/',
#                         '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_750_test_540/',
#                         '/data/brahma01/WAEUQ/Experiments/Failed_Test_nspoke_read_540_test_full/', 
#                         '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_540_test_1125/',
#                         '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_540_test_750/',
#                         '/data/brahma01/WAEUQ/Experiments/Test_nspoke_read_540_test_540/']