
import copy
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset
import datautils.seq_transforms as transforms
from params.seedutils import seed_everything_update

def dataset_transform(data, labels, input_shape, device, aug_num, trial_seed, normalize_type = 'z-score'):
    # -- transformation compositions
    # aug_num: None, 'getone', 'gettwo'

    # initial transform
    transform = transforms.Compose([
        transforms.Retype(),
        transforms.Normalize(normalize_type),
        transforms.ToTensor(device, input_shape)
    ])
    data, labels = shuffle_datasets(data, labels)
    x = transform(data)
    y = torch.tensor(labels).view(-1).to(device).long()

    if aug_num >0:
        transform = transforms.Compose([
            transforms.Retype(), # np.float32
            transforms.RandomStretch(),
            transforms.RandomCrop(),
            transforms.RandomAddGaussian(),
            transforms.Normalize(normalize_type),
            transforms.ToTensor(device, input_shape)
        ])

        x, y = augmentation(x, y, data, labels, transform, trial_seed, aug_num = aug_num)

    return TensorDataset(x, y)

def augmentation(x_init, y_init, data, labels,transform, trial_seed, aug_num = 2):
    multi_views = [x_init]
    y_extend = [y_init]
    device = y_init.device
    for i in range(aug_num):
        seed_everything_update(seed=trial_seed, remark='aug_idx'+str(i))
        multi_views.append(transform(data))
        y_extend.append(torch.tensor(labels).view(-1).to(device).long())
    # set seed back
    seed_everything_update(seed=trial_seed)
    return torch.cat(multi_views), torch.cat(y_extend)

def shuffle_datasets(data, labels):
    rng_state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(rng_state)
    np.random.shuffle(labels)
    return data, labels

def sig_segmentation(data, label, seg_len, start=0, stop=None):
    '''
    This function is mainly used to segment the raw 1-d signal into samples and labels
    using the sliding window to split the data
    '''
    data_seg = []
    lab_seg = []
    start_temp, stop_temp, stop = start, seg_len, stop if stop is not None else len(data)
    while stop_temp <= stop:
        sig = data[start_temp:stop_temp]
        sig = sig.reshape(-1, 1)
        data_seg.append( sig ) # z-score normalization
        lab_seg.append(label)
        start_temp += seg_len
        stop_temp += seg_len
    return data_seg, lab_seg