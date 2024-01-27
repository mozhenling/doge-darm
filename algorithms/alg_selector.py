# coding=utf-8


from algorithms.classes.DARM import DARM
from algorithms.classes.DARM_ip import DARM_ip
from algorithms.classes.DARM_ippp import DARM_ippp
from algorithms.classes.DARM_piii import DARM_piii


from algorithms.classes.ERM import ERM
from algorithms.classes.Mixup import Mixup

from algorithms.classes.AbstractCausIRL import CausIRL_MMD
from algorithms.classes.IB_IRM import IB_IRM
from algorithms.classes.IRM import IRM

from algorithms.classes.ANDMask import ANDMask
from algorithms.classes.RSC import RSC
from algorithms.classes.SANDMask import SANDMask

from algorithms.classes.AbstractCAD import  CondCAD
from algorithms.classes.SelfReg import SelfReg


ALGORITHMS = [
    'DARM', 'DARM_ip','DARM_ippp','DARM_piii',
    'ERM',
    'Mixup',

    'CausIRL_MMD',
    'IB_IRM',
    'IRM',

    'ANDMask',
    'RSC',
    'SANDMask',

    'CondCAD',
    'SelfReg'
]

def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

