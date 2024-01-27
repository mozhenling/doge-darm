import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
"""
References:
[1] A. A. Goshtasby, “Similarity and Dissimilarity Measures,” in Image Registration: Principles,
    Tools and Methods, A. A. Goshtasby, Ed., in Advances in Computer Vision and Pattern Recognition. ,
    London: Springer, 2012, pp. 7–66. doi: 10.1007/978-1-4471-2458-0_2.

"""

Criteria = [
    'l2_dist',
    'dot_dist'
]

def get_dist_criterion(criterion_name):
    if criterion_name not in globals():
        raise NotImplementedError(
            "criterion not found: {}".format(criterion_name))
    return globals()[criterion_name]

def flatten(x1, x2):
    return x1.view(x1.size(0), -1), x2.view(x2.size(0), -1)


def l2_dist(x1, x2):
    """
    L2 norm: value down, similarity up
        x1: feature    (batch, fea_dim)
        x2: prototypes (num_cls, fea_dim)
        return: (batch, num_cls)
    Other measures will do the broadcasting similarly
    The final will be of size (batch, num_cls)
    Ref.: https://pytorch.org/docs/stable/generated/torch.cdist.html
    """
    return torch.cdist(x1, x2, p=2)

def dot_dist(x1, x2):
    # inner product/ dot product
    return -torch.matmul(x1, x2.T)


