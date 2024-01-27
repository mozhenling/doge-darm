"""
Domainbed Datasets
"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
import numpy as np
from scipy.io import loadmat
import pandas as pd
from datautils.seq_process import sig_segmentation, dataset_transform

DATASETS = [
    'CU_Actuator',
    'CWRU_Bearing',

    'PHM_Gear',
    'UBFC_Motor',
]

def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        """
        __getitem__() is a magic method in Python, which when used in a class,
        allows its instances to use the [] (indexer) operators. Say x is an
        instance of this class, then x[i] is roughly equivalent to type(x).__getitem__(x, i).
        """
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)

class CU_Actuator(MultipleDomainDataset):
    ENVIRONMENTS = ['20kg', '40kg', 'neg40kg']

    def __init__(self, args, hparams):
        super().__init__()
        if args.data_dir is None:
            raise ValueError('Data directory not specified!')
        # -----------------------------------------------------------
        # self.filename = {'p1': None,      # part 1: class type
        #                  'p2': ('sin', 'trap'), # part 2: motion profile
        #                  'p3': ('1st', '2nd'),  # part 3: severity for back and lub faults
        #                  'p4': ('3rd', '4th'), # part 4: severity for point fault
        #                  'p5': (str(i+1) for i in range(10))} # part 5: repeat
        # -----------------------------------------------------------
        # -- sample points
        self.seg_len = 1000   # len of each sample
        self.len_total = 4000
        self.instance_size = 160  # per class
        self.class_name_dict = {'back':{'Backlash1.mat':'1st', 'Backlash2.mat':'2nd'},
                                'lub':{'LackLubrication1.mat':'1st', 'LackLubrication2.mat':'2nd'},
                                'point':{'Spalling3.mat':'3rd', 'Spalling4.mat':'4th'}}

        self.class_list = [i for i in range(len(self.class_name_dict))]
        self.environments = ['20kg', '40kg', 'neg40kg']
        # -----------------------------------------------------------
        self.input_shape = (1, self.seg_len)
        self.num_classes = len(self.class_list)
        self.datasets = []

        # -----------------------------------------------------------
        for env_id, env_name in enumerate(self.environments):
            aug_num = 0 if env_id in args.test_envs else args.aug_num

            data, labels = self.get_samples(args.data_dir, args.dataset,  env_name)
            self.datasets.append(dataset_transform(np.array(data), labels, self.input_shape, args.device,
                                                   aug_num, args.trial_seed))

    def get_samples(self, root,dataset, env_name):
        data_seg_all = []
        label_seg_all = []
        file_idxs = [str(i+1) for i in range(10)]
        for cls, lab in zip(self.class_name_dict, self.class_list):
            for cls_file in self.class_name_dict[cls]:
                file_path = os.path.join(root,dataset, cls_file)
                data_dict = loadmat(file_path)
                for motion in ['sin', 'trap']:
                    severity = self.class_name_dict[cls][cls_file]
                    for file_id in file_idxs:
                        file_str = cls + motion+ severity + env_name + file_id
                        if file_str == 'backtrap1st40kg2': # cope with missing values
                            data_dict['backtrap1st40kg2'] = (data_dict['backtrap1st40kg1'] + data_dict['backtrap1st40kg3']) / 2
                        data = np.concatenate((data_dict[file_str][:,1], data_dict[file_str][:,2]), axis=0)
                        data_temp, label_temp = sig_segmentation(data, label=lab, seg_len = self.seg_len,
                                                                 start = 0, stop=self.len_total)
                        data_seg_all.extend(data_temp)
                        label_seg_all.extend(label_temp)
        return data_seg_all, label_seg_all


class CWRU_Bearing(MultipleDomainDataset):
    ENVIRONMENTS = ['0hp_1797rpm', '2hp_1750rpm', '3hp_1730rpm']

    def __init__(self, args, hparams):
        super().__init__()

        if args.data_dir is None:
            raise ValueError('Data directory not specified!')

        # -- sample points
        self.seg_len = 1000
        self.instance_size = 150

        self.class_name_list = ['normal',#0
                                'ball',  #1
                                'inner', #2
                                'outer'] #3

        self.class_list = [i for i in range(len(self.class_name_list))]
        self.environments = ['0hp_1797rpm', '2hp_1750rpm', '3hp_1730rpm']
        # -----------------------------------------------------------
        self.input_shape = (1, self.seg_len)
        self.num_classes = len(self.class_list)
        self.datasets = []

        # -----------------------------------------------------------
        for env_id, env_name in enumerate(self.environments):
            aug_num = 0 if env_id in args.test_envs else args.aug_num
            file_path = os.path.join(args.data_dir, args.dataset, 'CWRU_DE_'+env_name+'_seg.mat')
            data_dict = loadmat(file_path)
            data, labels = data_dict['data'], data_dict['labels'].squeeze()
            self.datasets.append(dataset_transform(data, labels, self.input_shape, args.device,
                                                   aug_num, args.trial_seed))




class UBFC_Motor(MultipleDomainDataset):
    ENVIRONMENTS =['0%', '25%', '50%']

    def __init__(self, args, hparams):
        super().__init__()
        if args.data_dir is None:
            raise ValueError('Data directory not specified!')

        # -- sample points
        self.seg_len = 1024
        self.instance_size = 270 # per class
        self.sig_type = 'current'

        self.class_name_list = ['normal',#0
                                'UBS10%',  #1 umbalanced supply
                                'UBS20%'  #2 umbalanced supply
                                    ] #3

        self.class_list = [i for i in range(len(self.class_name_list))]
        self.environments = ['0%', '25%', '50%']
        # -----------------------------------------------------------
        self.input_shape = (1, self.seg_len)
        self.num_classes = len(self.class_list)
        self.datasets = []

        # -----------------------------------------------------------
        for env_id, env_name in enumerate(self.environments):
            aug_num = 0 if env_id in args.test_envs else args.aug_num
            file_path = os.path.join(args.data_dir,  args.dataset, 'stator_'+self.sig_type+
                                     '_load'+env_name+'_3cls.mat')
            data_dict = loadmat(file_path)
            data, labels = data_dict['data'], data_dict['labels'].squeeze()
            self.datasets.append(dataset_transform(data, labels, self.input_shape, args.device,
                                                   aug_num, args.trial_seed))



class PHM_Gear(MultipleDomainDataset):
    ENVIRONMENTS = ['30hz_Low_1', '35hz_Low_1', '40hz_Low_1']

    def __init__(self, args, hparams):
        super().__init__()

        if args.data_dir is None:
            raise ValueError('Data directory not specified!')

        # -- sample points
        self.seg_len = 1024
        self.instance_size = 520 # per class
        self.sig_type = 'vibration'
        self.obj_type = 'gear'

        self.class_name_list = ['s1',#0
                                's2',  #1
                                's3' ] #2

        self.class_list = [i for i in range(len(self.class_name_list))]
        self.environments = ['30hz_Low_1', '35hz_Low_1', '40hz_Low_1']
        # -----------------------------------------------------------
        self.input_shape = (1, self.seg_len)
        self.num_classes = len(self.class_list)
        self.datasets = []

        # -----------------------------------------------------------
        for env_id, env_name in enumerate(self.environments):
            aug_num = 0 if env_id in args.test_envs else args.aug_num
            file_path = os.path.join(args.data_dir, args.dataset,  self.obj_type+'_'+self.sig_type+
                                     '_speed_'+env_name+'_3cls.mat')
            data_dict = loadmat(file_path)
            data, labels = data_dict['data'], data_dict['labels'].squeeze()
            self.datasets.append(dataset_transform(data, labels, self.input_shape, args.device,
                                                   aug_num, args.trial_seed))



if __name__ == '__main__':

    root = r'..\datasets\CWRU'
    device = 'cuda'
    test_env_ids = [1]

