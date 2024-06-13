#!/bin/bash

echo '------- Launch Algorithm --------'

python -m main_sweep\
       --command r\
       --command_launcher plain\
       --n_trials 1\
       --n_hparams 1\
       --datasets UBFC_Motor\
       --data_dir=./datasets/\
       --algorithms DARM\
       --skip_model_save

echo '------- Algorithm Finished --------'
