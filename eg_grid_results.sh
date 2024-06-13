#!/bin/bash

python -m main_hparams\
       --input_dir '.\outputs\sweep_outs'\
       --dataset PU_Bearing\
       --algorithm DARM\
       --test_env 0\

