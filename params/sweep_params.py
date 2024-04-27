
import argparse
from datautils import diag_datasets as datasets
from algorithms import alg_selector
DATASETS = [d for d in datasets.DATASETS if "Debug" not in d]

def get_args():
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('--command', choices=['r','run','c', 'clear','launch', 'delete_incomplete'], default='r')
    parser.add_argument('--command_launcher', type=str, default='plain')
    parser.add_argument('--datasets', nargs='+', type=str, default=DATASETS)
    parser.add_argument('--algorithms', nargs='+', type=str, default=alg_selector.ALGORITHMS)
    parser.add_argument('--task', type=str, default="domain_generalization")
    parser.add_argument('--n_hparams_from', type=int, default=0)

    parser.add_argument('--nets_base', type=str, default="diag_nets",
                        help='networks for featurizer and classifier')
    parser.add_argument('--aug_num', type=int, default=0)

    parser.add_argument('--output_dir', type=str, default=r'./outputs/sweep_outs')
    parser.add_argument('--zip_output_time', type=float, default=42900,
                        help='the time (in seconds) to stop training and zip the output. Default: 11h55m')
    parser.add_argument('--zip_output_dir', type=str, default=r'./0-zips')

    parser.add_argument('--data_dir', type=str, default=r'./datasets')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_trials', type=int, default=3)
    parser.add_argument('--n_hparams', type=int, default=20)

    parser.add_argument('--steps', type=int, default=None)
    parser.add_argument('--hparams', type=str, default=None)
    parser.add_argument('--holdout_fraction', type=float, default=0.2)

    parser.add_argument('--erm_losses', nargs='+', type=str, default='CELoss')

    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--single_test_envs', type=bool, default=True)
    parser.add_argument('--skip_confirmation', type=bool, default=True)
    args = parser.parse_args()
    return args