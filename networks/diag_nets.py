
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models

from networks import wide_resnet
import copy
import numpy as np
from numpy.random import beta

def Featurizer(input_shape, hparams, args):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 2:
        return LeNet_1d_Featurizer()#LeNet_1d_Featurizer() # hparams can be added if needed
    else:
        raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)

# -----------------------input size>=32---------------------------------
class LeNet_1d_Featurizer(nn.Module):
    def __init__(self, in_channel=1):
        super(LeNet_1d_Featurizer, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channel, 6, 5),
            nn.BatchNorm1d(6),  # 64
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(6, 16, 5),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(25)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 25, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.n_outputs =84

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x



