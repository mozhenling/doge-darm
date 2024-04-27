
import time
from algorithms import  alg_launchers
from params.sweep_params import get_args

if __name__ == "__main__":
    sweep_start_time = time.time()
    args = get_args()

    ###########################################################################
    # ---------------------------- test
    # -- run
    args.command = 'r'
    args.command_launcher= 'plain'
    args.n_trails = 1
    args.n_hparams = 1
    args.datasets = ['CU_Actuator']
    args.data_dir = './datasets'
    args.algorithms = ['DARM']
    args.skip_model_save = True
    args.zip_output_time = 10 # seconds

    ###########################################################################

    args_list = alg_launchers.make_args_list(
        n_trials=args.n_trials,
        dataset_names=args.datasets,
        algorithms=args.algorithms,
        n_hparams_from=args.n_hparams_from,
        n_hparams=args.n_hparams,
        steps=args.steps,
        data_dir=args.data_dir,
        task=args.task,
        skip_model_save = args.skip_model_save,
        holdout_fraction=args.holdout_fraction,
        nets_base= args.nets_base,
        aug_num = args.aug_num,
        single_test_envs=args.single_test_envs,
        hparams=args.hparams
    )
    is_cmd_launcher = False if args.command_launcher in ['plain'] else True
    jobs = [alg_launchers.Job(train_args, args.output_dir, is_cmd_launcher) for train_args in args_list]

    for job in jobs:
        print(job)
    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == alg_launchers.Job.DONE]),
        len([j for j in jobs if j.state == alg_launchers.Job.INCOMPLETE]),
        len([j for j in jobs if j.state == alg_launchers.Job.NOT_LAUNCHED]))
    )

    if args.command in ['r', 'run', 'launch']:
        to_delete = [j for j in jobs if j.state == alg_launchers.Job.INCOMPLETE]
        if len(to_delete) > 0:
            print(f'About to delete {len(to_delete)} incomplete jobs to restart them.')
            alg_launchers.Job.delete(to_delete)
        to_launch = [j for j in jobs if j.state == alg_launchers.Job.NOT_LAUNCHED] + to_delete
        print(f'About to launch {len(to_launch)} jobs.')
        if not args.skip_confirmation:
            alg_launchers.ask_for_confirmation()
        launcher_fn = alg_launchers.REGISTRY[args.command_launcher]
        alg_launchers.Job.launch(to_launch, launcher_fn, sweep_start_time, is_cmd_launcher, args)

    elif args.command in ['c', 'clear', 'delete_incomplete']:
        to_delete = [j for j in jobs if j.state == alg_launchers.Job.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        if not args.skip_confirmation:
            alg_launchers.ask_for_confirmation()
        alg_launchers.Job.delete(to_delete)

    sweep_stop_time = time.time()
    print('#' * 10, ' total_time = {:.2f} min '.format((sweep_stop_time - sweep_start_time) / 60), '#' * 10)