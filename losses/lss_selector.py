
from losses.rbst_losses import *

Losses = [
    'JSDLoss',
    'GLSLoss',
    'CELoss',
    'SCELoss',
    'RCELoss',
    'NRCELoss',
    'NCELoss',
    'MAELoss',
    'NMAE',
    'GCELoss',
    'NGCELoss',
    'AGCELoss',
    'AUELoss',
    'ANormLoss',
    'AExpLoss',
    'NCEandRCE',
    'NCEandMAE',
    'NLNL',
    'FocalLoss',
    'NFocalLoss',
    'NFLandRCE',
    'NFLandMAE',
    'NCEandAGCE',
    'NCEandAUE',
    'NCEandAEL',
    'NFLandAGCE',
    'NFLandAUE',
    'NFLandAEL',
    'ANLandRCE',
    'NCEandANL'
]

def get_loss_class(loss_name):
    if loss_name not in globals():
        raise NotImplementedError(
            "loss not found: {}".format(loss_name))
    return globals()[loss_name]
