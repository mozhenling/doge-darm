#!/bin/bash

echo '------- Launch Algorithm --------'

python -m main_sweep\
       --command r\
       --command_launcher plain\
       --n_trials 1\
       --n_hparams_from 0\
       --n_hparams 1\
       --datasets PU_Bearing\
       --data_dir=./datasets/\
       --algorithms DARM\
       --sweep_test_envs "[[0]]"\
       --steps 10\
       --checkpoint_freq 1\
       --hparams_search_mode g\
       --avg_std t\
       --hparams_grid_bases "[{'name':'loss_pp_weight', 'start':-1, 'stop':3, 'num':5, 'base':10},
                            {'name':'loss_ii_weight', 'start':-1, 'stop':3, 'num':5, 'base':10}]"\
       --skip_model_save

echo '------- Algorithm Finished --------'
