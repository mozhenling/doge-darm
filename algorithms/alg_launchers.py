# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""
import subprocess
import time
import torch
import numpy as np
from algorithms.trainer import train
from datautils import data_process, diag_datasets as datasets
import copy
import tqdm
import shlex
import hashlib
import json
import os
from oututils import os_utils
import shutil

def plain_launcher(args_list, sweep_start_time, zip_output_dir, output_dir, zip_output_time):
    """Launch training via train function"""
    is_time_out = False
    for i, args in enumerate(args_list):
        print(f'About to launch job #{i+1}')
        is_time_out = train(args, sweep_start_time, is_time_out, zip_output_time)
        if is_time_out:
            print(f'Job #{i+1} was incomplete due to time out!')
            print(f'({len(args_list) - i} jobs incomplete in total)')
            break
        else:
            print(f'Job #{i + 1} was completed' + f' ({len(args_list) - i - 1} jobs left)')

    # os.makedirs(sweep_args.zip_output_dir, exist_ok=True)
    if is_time_out:
        output_name = os.path.join(zip_output_dir, f'0-time_out_sweeps')
        os_utils.tozip(output_name, output_dir)
    else:
        output_name = os.path.join(zip_output_dir, '0-job_out_sweeps')
        os_utils.tozip(output_name, output_dir)

def local_launcher(commands):
    """Launch commands serially on the local machine."""
    print('start local_launcher')
    for i, cmd in enumerate(commands) :
        subprocess.run(cmd, shell=True)

def dummy_launcher(commands):
    """
    Doesn't run anything; instead, prints each command.
    Useful for testing.
    """
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')

def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')
    try:
        # Get list of GPUs from env, split by ',' and remove empty string ''
        # To handle the case when there is one extra comma: `CUDA_VISIBLE_DEVICES=0,1,2,3, python3 ...`
        available_gpus = [x for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x != '']
    except Exception:
        # If the env variable is not set, we use all GPUs
        available_gpus = [str(x) for x in range(torch.cuda.device_count())]
    n_gpus = len(available_gpus)
    procs_by_gpu = [None]*n_gpus

    while len(commands) > 0:
        for idx, gpu_idx in enumerate(available_gpus):
            proc = procs_by_gpu[idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()

REGISTRY = {
    'plain': plain_launcher,
    'local': local_launcher,
    'dummy': dummy_launcher,
    'multi_gpu': multi_gpu_launcher
}

class Job:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args, sweep_output_dir, is_cmd_launcher = False):
        ################### sweep folder encoder ##################
        identifier_keys = ['dataset', 'algorithm', 'test_envs',
                           'holdout_fraction', 'hparams_seed',
                           'task', 'trial_seed']
        # seed is dependent on dataset,algorithm, test_envs, hparams_seed, trial_seed
        sweep_folder_identifier = {key: train_args[key] for key in identifier_keys}
        args_str = json.dumps(sweep_folder_identifier, sort_keys=True)
        ###########################################################
        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
        self.output_dir = os.path.join(sweep_output_dir, args_hash)
        self.train_args = copy.deepcopy(train_args)
        self.train_args['output_dir'] = self.output_dir
        self.is_cmd_launcher = is_cmd_launcher
        if self.is_cmd_launcher:
            command = ['python', '-m', 'main_train']
            for k, v in sorted(self.train_args.items()):
                if isinstance(v, list):
                    v = ' '.join([str(v_) for v_ in v])
                elif isinstance(v, str):
                    v = shlex.quote(v)
                if k in ['skip_model_save']:
                    command.append(f'--{k}')
                else:
                    command.append(f'--{k} {v}')
            self.command_str = ' '.join(command)
            # print(self.command_str)

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = Job.DONE
        elif os.path.exists(self.output_dir):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['test_envs'],
            self.train_args['hparams_seed'])
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn, is_cmd_launcher, sweep_start_time, zip_output_dir, output_dir, zip_output_time):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        if is_cmd_launcher:
            commands = [job.command_str for job in jobs]
            launcher_fn(commands)
        else:
            args_list = [job.train_args for job in jobs]
            launcher_fn(args_list, sweep_start_time, zip_output_dir, output_dir, zip_output_time)
        print(f'Tried {len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} existed jobs!')


def all_test_env_combinations(n):
    """
    For a dataset with n >= 3 envs, return all combinations of 1 and 2 test
    envs.
    """
    assert(n >= 3)
    for i in range(n):
        yield [i]
        for j in range(i+1, n):
            yield [i, j]

def make_args_list(n_trials, dataset_names, algorithms, n_hparams_from, n_hparams, avg_std, steps, checkpoint_freq,
    data_dir, task, holdout_fraction, nets_base, aug_num, skip_model_save, sweep_test_envs, single_test_envs, hparams):
    args_list = []
    for trial_seed_id in range(n_trials):
        for dataset in dataset_names:
            for algorithm in algorithms:
                if sweep_test_envs is not None:
                    all_test_envs = sweep_test_envs
                else:
                    if single_test_envs:
                        all_test_envs = [
                            [i] for i in range(datasets.num_environments(dataset))]
                    else:
                        all_test_envs = all_test_env_combinations(
                            datasets.num_environments(dataset))
                for test_envs in all_test_envs:
                    for hparams_seed_id in range(n_hparams_from, n_hparams):
                        train_args = {}
                        train_args['dataset'] = dataset
                        train_args['algorithm'] = algorithm
                        train_args['test_envs'] = test_envs
                        train_args['holdout_fraction'] = holdout_fraction
                        train_args['data_dir'] = data_dir
                        train_args['task'] = task
                        train_args['nets_base'] = nets_base
                        train_args['aug_num'] = aug_num
                        train_args['skip_model_save'] = skip_model_save
                        train_args['n_trials'] = n_trials
                        train_args['n_hparams'] = n_hparams
                        train_args['avg_std'] = avg_std
                        train_args['hparams_seed_id'] = hparams_seed_id
                        train_args['hparams_seed'] = data_process.seed_hash(hparams_seed_id,  hparams)
                        train_args['trial_seed'] = trial_seed_id
                        train_args['seed'] = data_process.seed_hash(dataset, algorithm, test_envs, train_args['hparams_seed'], trial_seed_id)
                        if steps is not None:
                            train_args['steps'] = steps
                        if checkpoint_freq is not None:
                            train_args['checkpoint_freq']=checkpoint_freq
                        if hparams is not None:
                            train_args['hparams'] = hparams
                        args_list.append(train_args)
    return args_list

def ask_for_confirmation():
    response = input('Are you sure? (y/n) ')
    if not response.lower().strip()[:1] == "y":
        print('Nevermind!')
        exit(0)