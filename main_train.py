import os
import time
import torch
from params.train_params import get_args
from algorithms.trainer import train

if __name__ == "__main__":
    train_start_time = time.time()

    args = get_args()
    if torch.cuda.is_available():
        device = "cuda"

    else:
        device = "cpu"

    args.device = device


    args.output_dir = os.path.join(args.output_dir,
                                    args.dataset + '_'+ args.algorithm+ '_test_ids_' + str(args.test_envs))

    # Convert the namespace to a dictionary
    args_dict = vars(args)
    train_outs = train(args_dict=args_dict)