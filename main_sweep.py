
import time
from algorithms import  alg_launchers
from params.sweep_params import get_args
import itertools
import numpy as np
import math
import os
import shutil

def combine_dicts(*dict_lists):
    # Generate all possible combinations of dictionaries
    combinations = itertools.product(*dict_lists)
    # Convert combinations to a list of dictionaries
    result = [dict(itertools.chain.from_iterable(d.items() for d in combination)) for combination in combinations]
    return result

def hparam_single_grid(name, start, stop, num=None, step=None, base=None):
    if num is None and step is None:
        raise ValueError(" 'num' and 'step' should not be None at the same time! ")

    if num is not None and step is None:
        step = (stop - start) / num

    if base is not None:
        return [{name:math.pow(base, value)} for value in np.arange(start, stop, step, dtype=float)]
    else:
        return [{name:value} for value in np.arange(start, stop, step, dtype=float)]

def hparam_grids(param_bases):
    param_list = [hparam_single_grid(**param_info) for param_info in param_bases]
    return combine_dicts(*param_list)

def get_args_list(args, hparams):
    args_list = alg_launchers.make_args_list(
            n_trials=args.n_trials,
            dataset_names=args.datasets,
            algorithms=args.algorithms,
            n_hparams_from=args.n_hparams_from,
            n_hparams=args.n_hparams,
            avg_std = args.avg_std,
            steps=args.steps,
            checkpoint_freq = args.checkpoint_freq,
            data_dir=args.data_dir,
            task=args.task,
            skip_model_save=args.skip_model_save,
            holdout_fraction=args.holdout_fraction,
            nets_base=args.nets_base,
            aug_num=args.aug_num,
            sweep_test_envs=args.sweep_test_envs,
            single_test_envs=args.single_test_envs,
            hparams=hparams
        )
    return args_list
if __name__ == "__main__":
    sweep_start_time = time.time()
    args = get_args()

    if args.hparams_search_mode in ['g','grid'] and args.hparams_grid_bases is not None:
        if len(args.hparams_grid_bases)==0:
            raise ValueError('args.hparams_grid_bases is empty. It should be a list of dictionaries! Or set None to choose other search methods.')
        hparams_list_of_dict = hparam_grids(args.hparams_grid_bases)
        args_list = []
        for hparams in hparams_list_of_dict:
            if args.hparams is not None:
                # args.hparams can always rewrite the hparams being used
                hparams.update(args.hparams)
            args_list.extend(get_args_list(args, hparams))

    else:
        args_list = get_args_list(args, args.hparams)

    is_cmd_launcher = False if args.command_launcher in ['plain'] else True

    if args.command in ['n', 'new', 'new_all']:
        # -- start all regardless what have been created
        if os.path.exists(args.output_dir):
            print('Deleted previous sweep_outs!')
            shutil.rmtree(args.output_dir)

    jobs = [alg_launchers.Job(train_args, args.output_dir, is_cmd_launcher) for train_args in args_list]

    for job in jobs:
        print(job)
    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == alg_launchers.Job.DONE]),
        len([j for j in jobs if j.state == alg_launchers.Job.INCOMPLETE]),
        len([j for j in jobs if j.state == alg_launchers.Job.NOT_LAUNCHED]))
    )

    if args.command in ['r', 'run', 'launch', 'n', 'new', 'new_all']:
        to_delete = [j for j in jobs if j.state == alg_launchers.Job.INCOMPLETE]
        if len(to_delete) > 0:
            print(f'About to delete {len(to_delete)} incomplete jobs to restart.')
            alg_launchers.Job.delete(to_delete)
        to_launch = [j for j in jobs if j.state == alg_launchers.Job.NOT_LAUNCHED] + to_delete
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            alg_launchers.ask_for_confirmation()
        launcher_fn = alg_launchers.REGISTRY[args.command_launcher]
        zip_output_dir, output_dir, zip_output_time = args.zip_output_dir, args.output_dir, args.zip_output_time
        alg_launchers.Job.launch(to_launch, launcher_fn, is_cmd_launcher, sweep_start_time, zip_output_dir, output_dir, zip_output_time)

    elif args.command in ['c', 'clear', 'delete_incomplete']:
        to_delete = [j for j in jobs if j.state == alg_launchers.Job.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            alg_launchers.ask_for_confirmation()
        alg_launchers.Job.delete(to_delete)

    else:
        raise ValueError('args.command is not matched!')


    sweep_stop_time = time.time()
    print('#' * 10, ' total_time = {:.2f} min '.format((sweep_stop_time - sweep_start_time) / 60), '#' * 10)