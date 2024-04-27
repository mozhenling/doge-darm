import os
import sys
import PIL
import time
import json
import torch
import random
import torchvision
import collections
import numpy as np
from oututils import os_utils
from params.train_params import get_args
from params.alg_params import default_hparams, random_hparams
from algorithms import alg_selector, optimization
from datautils import data_process, bed_dataloaders as dataloader, dataset_selector
from params.seedutils import seed_everything_update
########################################################################################################################
#------------------------------------------------------------ Main
########################################################################################################################
#------------------------------------------------------------ re-set imporatant params for each algorithm (for runs of no commind lines)
def train(args_dict, sweep_start_time, is_time_out, sweep_args):
    # ------------------------------------------------------------ Versions of your packages
    print("Versions:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))
    # ------------------------------------------------------------ args

    args = get_args()

    #  cpu or gpu
    if torch.cuda.is_available():
        device = "cuda"

    else:
        device = "cpu"

    args.device = device
    print('Args:')

    vars(args).update(args_dict)
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    # #------------------------------------------------------------ for checkpointing
    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None
    #------------------------------------------------------------ save outputs and errors into the files
    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = os_utils.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = os_utils.Tee(os.path.join(args.output_dir, 'err.txt'))

    #------------------------------------------------------------ Hyper parameters
    if args.hparams_seed == 0:
        hparams = default_hparams(args)
    else:
        hparams = random_hparams(args)
    if args.hparams:
        hparams.update(args.hparams)

    # --------------------------------------------------------------
    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))
    # -- update the augmentation number for small datasets
    # args.aug_num = hparams['aug_num']
#------------------------------------------------------------ Limit randomness
    """
    You can configure PyTorch to avoid using nondeterministic
    algorithms for some operations, so that multiple calls to 
    those operations, given the same inputs, will produce the
    same result.
    Ref.: https://pytorch.org/docs/stable/notes/randomness.html
    """
    seed_everything_update(args.seed)

    #------------------------------------------------------------ get dataset
    """
    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.
    """
    dataset = dataset_selector.get_dataset_object(args, hparams)
    # print('Dataset copmleted')
    # the in-split data, a list of environment data after the data split
    in_splits  = []
    # the out_split data, a list of environment data after the data split
    out_splits = []
    for env_i, env in enumerate(dataset):
        # print('Process environment '+str(env_i) + ' ...')
        # split the data of each environment into an in-split and an out-split
        out, in_ = data_process.split_dataset(env, int(len(env)*args.holdout_fraction),
                                              data_process.seed_hash(args.trial_seed, env_i))
        # balance the labels of the split
        if hparams['class_balanced']:
            in_weights = data_process.make_weights_for_balanced_classes(in_)
            out_weights = data_process.make_weights_for_balanced_classes(out)

        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        # print('Environment '+str(env_i) + ' completed')
    #------------------------------------------------------------ training data loader (takes too much time)
    train_loaders = [dataloader.InfiniteDataLoader(
                                dataset=env,
                                weights=env_weights,
                                batch_size=hparams['batch_size'],
                                num_workers=0 # for small data set, 0 is just fine enough; Otherwise, it may even cost time
                                    )  for i, (env, env_weights) in enumerate(in_splits) if i not in args.test_envs
                    ]
    # print('Train_loader compeleted')
    #------------------------------------------------------------ evaluation data loader (takes too much time)
    eval_loaders = [dataloader.FastDataLoader(
                        dataset=env,
                        batch_size=hparams['batch_size'],
                        num_workers=0 # for small data set, 0 is just fine enough; Otherwise, it may even cost time
                            ) for env, _ in (in_splits + out_splits)
                    ]
    # print('Eval_loader compeleted')
    #------------------------------------------------------------ set the algorithm
    eval_weights = [None for _, weights in (in_splits + out_splits)]
    eval_loader_names = ['env{}_in'.format(i) for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i) for i in range(len(out_splits))]

    algorithm_class = alg_selector.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(input_shape = dataset.input_shape,
                                num_classes = dataset.num_classes,
                                num_domains = len(in_splits) - len(args.test_envs), # the number of training domains
                                hparams =hparams,
                                args=args)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)
    #------------------------------------------------------------ prepare data
    # a list of environment data loaders except for the test environment
    train_minibatches_iterator = zip(*train_loaders)
    # create a dict with default values as an empty list. The keys are non-existent and will be given later.
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env, _ in in_splits])

    n_steps = args.steps if args.steps is not None else hparams['steps']

    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ
    #------------------------------------------------------------ prepare save
    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))
    #------------------------------------------------------------ start optimziation
    last_results_keys = None
    for step in range(start_step, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device)) for x, y in next(train_minibatches_iterator)]
        # -------------------- update the model by one batch of data ---------------------------
        # the batch data is generated by the infinite data loader (random batch with replacement)
        step_vals = algorithm.update(minibatches_device)
        # --------------------------------------------------------------------------------------
        checkpoint_vals['step_time'].append(time.time() - step_start_time)
        time_monitor = time.time() - sweep_start_time
        if time_monitor >sweep_args.zip_output_time and sweep_args.zip_output_time>0:
            is_time_out = True
            break # break the training loop
        # --------------------------------------------------------------------------------------

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            if step == n_steps - 1:
                args.is_end = True
            else:
                args.is_end = False
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }
            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val) # losses averaged over the checkpoint frequency
            # get in-splits and out-splits accuracies of all environments
            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            for name, loader, weights in evals:
                acc = optimization.accuracy(algorithm, loader, weights, device, name, args)
                results[name+'_acc'] = acc
            # Returns the maximum GPU memory occupied by tensors in bytes for a given device
            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.) # in Gb

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                os_utils.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            os_utils.print_row([results[key] for key in results_keys], colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')
    #------------------------------------------------------------ save the model at the last step
    save_checkpoint('model.pkl')

    if not is_time_out:
        with open(os.path.join(args.output_dir, 'done'), 'w') as f:
            f.write('done')

    return is_time_out