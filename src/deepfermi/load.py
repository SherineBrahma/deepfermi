from tqdm import tqdm
import h5py
import numpy as np
import torch


def load_h5py(dataset_path, dsplit):
    with h5py.File(dataset_path, "r") as f:
        data_dic = {}
        for ds in tqdm(range(list(f.keys()).__len__())):
            ds_key = list(f.keys())[ds]
            for info in range(list(f[ds_key].keys()).__len__()):
                info_key = list(f[ds_key].keys())[info]
                data_dic[info_key] = f[ds_key][info_key][()][ 0:dsplit[ds_key], ...]
                # data_dic[info_key] = f[ds_key][info_key][()][ 3:4, ...] # [ 2:3, ...] # 
    return data_dic

def load_npz(dataset_path, dsplit):
    data = np.load(dataset_path, allow_pickle=True)
    data_dic = {}   
    for i in data.keys():
        for j in data[i].item().keys():
            data_dic[j] = data[i].item()[j][0:dsplit[i]]
    return data_dic

def five_fold_cross_val_split(data_dic, val_fold=0):
    
    fold_0 = ['46', '38', '76', '102', '44']
    fold_1 = ['56', '75', '109', '62']
    fold_2 = ['45', '101', '10', '54']
    fold_3 = ['42', '47', '107', '61']
    fold_4 = ['39', '83', '37', '58', '71']
    
    # Hard-coded fold number
    if val_fold==0:
        fold_val = fold_0
        fold_train = fold_1 + fold_2 + fold_3 + fold_4
    elif val_fold==1:
        fold_val = fold_1
        fold_train = fold_0 + fold_2 + fold_3 + fold_4
    elif val_fold==2:
        fold_val = fold_2
        fold_train = fold_0 + fold_1 + fold_3 + fold_4
    elif val_fold==3:
        fold_val = fold_3
        fold_train = fold_0 + fold_1 + fold_2 + fold_4
    elif val_fold==4:
        fold_val = fold_4
        fold_train = fold_0 + fold_1 + fold_2 + fold_3   
    
    # Construct hierarchy of data by patient number
    cross_val_dic = {}
    for i, pid in enumerate(data_dic['pat_train']):
        cross_val_dic[str(pid.item())]={}
        cross_val_dic[str(pid.item())]['pid'] = data_dic['pat_train'][i]
        cross_val_dic[str(pid.item())]['im_sig'] = data_dic['im_sig_train'][i]
        cross_val_dic[str(pid.item())]['ctc'] = data_dic['ctc_train'][i]
        cross_val_dic[str(pid.item())]['aif'] = data_dic['aif_train'][i]
        cross_val_dic[str(pid.item())]['seg'] = data_dic['seg_train'][i]
        cross_val_dic[str(pid.item())]['time'] = data_dic['time_train'][i]
        cross_val_dic[str(pid.item())]['wlen'] = data_dic['wlen_train'][i]
    for i, pid in enumerate(data_dic['pat_val']):
        cross_val_dic[str(pid.item())]={}
        cross_val_dic[str(pid.item())]['pid'] = data_dic['pat_val'][i]
        cross_val_dic[str(pid.item())]['im_sig'] = data_dic['im_sig_val'][i]
        cross_val_dic[str(pid.item())]['ctc'] = data_dic['ctc_val'][i]
        cross_val_dic[str(pid.item())]['aif'] = data_dic['aif_val'][i]
        cross_val_dic[str(pid.item())]['seg'] = data_dic['seg_val'][i]
        cross_val_dic[str(pid.item())]['time'] = data_dic['time_val'][i]
        cross_val_dic[str(pid.item())]['wlen'] = data_dic['wlen_val'][i]
        
    # Constructing Training Set        
    pat_train = []
    im_sig_train = []
    ctc_train = []
    aif_train = []
    seg_train = []
    time_train = []
    wlen_train = []
    for pid in fold_train:
        pat_train.append(cross_val_dic[pid]['pid'])
        im_sig_train.append(cross_val_dic[pid]['im_sig'])
        ctc_train.append(cross_val_dic[pid]['ctc'])
        aif_train.append(cross_val_dic[pid]['aif'])
        seg_train.append(cross_val_dic[pid]['seg'])
        time_train.append(cross_val_dic[pid]['time'])
        wlen_train.append(cross_val_dic[pid]['wlen'])
    data_dic['pat_train'] = pat_train
    data_dic['im_sig_train'] = torch.stack(im_sig_train)
    data_dic['ctc_train'] = torch.stack(ctc_train)
    data_dic['aif_train'] = torch.stack(aif_train)
    data_dic['seg_train'] = torch.stack(seg_train)
    data_dic['time_train'] = torch.stack(time_train)
    data_dic['wlen_train'] = torch.stack(wlen_train)
    
    # Constructing Validation Set        
    pat_val = []
    im_sig_val = []
    ctc_val = []
    aif_val = []
    seg_val = []
    time_val = []
    wlen_val = []
    for pid in fold_val:
        pat_val.append(cross_val_dic[pid]['pid'])
        im_sig_val.append(cross_val_dic[pid]['im_sig'])
        ctc_val.append(cross_val_dic[pid]['ctc'])
        aif_val.append(cross_val_dic[pid]['aif'])
        seg_val.append(cross_val_dic[pid]['seg'])
        time_val.append(cross_val_dic[pid]['time'])
        wlen_val.append(cross_val_dic[pid]['wlen'])
    data_dic['pat_val'] = pat_val
    data_dic['im_sig_val'] = torch.stack(im_sig_val)
    data_dic['ctc_val'] = torch.stack(ctc_val)
    data_dic['aif_val'] = torch.stack(aif_val)
    data_dic['seg_val'] = torch.stack(seg_val)
    data_dic['time_val'] = torch.stack(time_val)
    data_dic['wlen_val'] = torch.stack(wlen_val)
    
    return data_dic

