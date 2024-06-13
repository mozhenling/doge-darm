
import argparse
import numpy as np
from oututils import  model_selection, print_outs

if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = argparse.ArgumentParser(
        description="Domain generalization testbed")
    parser.add_argument("--input_dir", type=str, default=r'./outputs/sweep_outs')
    parser.add_argument("--save_dir", type=str, default=r'./outputs/hparams_outs')

    parser.add_argument('--dataset', type=str, default="ColoredMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--sub_algorithm', type=str, default=None)
    parser.add_argument('--erm_loss', type=str, default='CELoss')
    parser.add_argument('--test_env', type=int, default=0)

    args = parser.parse_args()

    SELECTION_METHODS = [
        model_selection.IIDAccuracySelectionMethod,
        # model_selection.LeaveOneOutSelectionMethod, # show results when args.single_test_envs = False (costly)
        # model_selection.OracleSelectionMethod,
    ]
    for selection_method in SELECTION_METHODS:
        print_outs.print_hparams_grid_results(args, selection_method)