
import random
import numpy as np
import os
import torch
import hashlib

def seed_everything_update(seed=42, remark = 'default'):
    """
    Seed everything.
     Completely reproducible results are not guaranteed across
     PyTorch releases, individual commits, or different platforms.
     Furthermore, results may not be reproducible between CPU and
     GPU executions, even when using identical seeds.

     However, there are some steps you can take to limit the number
     of sources of nondeterministic behavior for a specific platform,
     device, and PyTorch release.

    Ref.:
    https://pytorch.org/docs/stable/notes/randomness.html
    https://pytorch.org/docs/stable/notes/randomness.html#reproducibility
    """
    # -- the actual seed, encoded by *args of seed_hash()
    seed = seed_hash(seed, remark)

    # For custom operators, you might need to set python seed as well
    random.seed(seed)

    # os.environ['PYTHONHASHSEED'] = str(seed)

    # If you or any of the libraries you are using rely on NumPy, you can seed the global NumPy RNG with
    np.random.seed(seed)

    # PyTorch random number generator (RNG)
    # You can use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Avoiding nondeterministic algorithms
    torch.backends.cudnn.deterministic = True

    # The cuDNN library, used by CUDA convolution operations,
    # can be a source of nondeterminism across multiple executions of an application.
    # Disabling the benchmarking feature with torch.backends.cudnn.benchmark = False
    # causes cuDNN to deterministically select an algorithm (instead of benchmarking
    # to find the fastest one)
    torch.backends.cudnn.benchmark = False

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)