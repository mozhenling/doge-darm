# coding=utf-8
"""
Optimization related implementations
"""
import sys
import torch
import numpy as np
#--------------------------------------------- Optimizer
def get_optimizer(params, hparams, args):
    if args.optimizer in ['sgd','SGD']:
        optimizer = torch.optim.SGD(
            params, lr=hparams["lr"], momentum=hparams["momentum"], weight_decay=hparams['weight_decay'], nesterov=True
        )
    else:
        optimizer = torch.optim.Adam(
            params, lr= hparams["lr"], weight_decay= hparams['weight_decay']
        )
    return optimizer

#--------------------------------------------- learning scheduler
def get_scheduler(optimizer, args):
    if args.scheduler is None:
        return None
    elif args.scheduler == 'cos':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, args.max_epoch * args.steps_per_epoch)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay))
    return scheduler

#--------------------------------------------- others
def get_params(alg, args, inner=False, alias=True, isteacher=False):
    if args.schuse:
        if args.schusech == 'cos':
            initlr = args.lr
        else:
            initlr = 1.0
    else:
        if inner:
            initlr = args.inner_lr
        else:
            initlr = args.lr
    if isteacher:
        params = [
            {'params': alg[0].parameters(), 'lr': args.lr_decay1 * initlr},
            {'params': alg[1].parameters(), 'lr': args.lr_decay2 * initlr},
            {'params': alg[2].parameters(), 'lr': args.lr_decay2 * initlr}
        ]
        return params
    if inner:
        params = [
            {'params': alg[0].parameters(), 'lr': args.lr_decay1 *
             initlr},
            {'params': alg[1].parameters(), 'lr': args.lr_decay2 *
             initlr}
        ]
    elif alias:
        params = [
            {'params': alg.featurizer.parameters(), 'lr': args.lr_decay1 * initlr},
            {'params': alg.classifier.parameters(), 'lr': args.lr_decay2 * initlr}
        ]
    else:
        params = [
            {'params': alg[0].parameters(), 'lr': args.lr_decay1 * initlr},
            {'params': alg[1].parameters(), 'lr': args.lr_decay2 * initlr}
        ]
    if ('DANN' in args.algorithm) or ('CDANN' in args.algorithm):
        params.append({'params': alg.discriminator.parameters(),
                       'lr': args.lr_decay2 * initlr})
    if ('CDANN' in args.algorithm):
        params.append({'params': alg.class_embeddings.parameters(),
                       'lr': args.lr_decay2 * initlr})
    return params

#--------------------------------------------- accuracy in the optimization
def accuracy(network, loader, weights, device, name, args):

    acc_dict = {}

    network.eval()
    with torch.no_grad():
        acc_dict['default'] = get_acc(network, loader, weights, device, name, args)
    network.train()

    return max(list(acc_dict.values()))

def get_acc(network, loader, weights, device, name, args):
    correct = 0
    total = 0
    weights_offset = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        p = network.predict(x)
        if weights is None:
            batch_weights = torch.ones(len(x))
        else:
            batch_weights = weights[weights_offset: weights_offset + len(x)]
            weights_offset += len(x)
        batch_weights = batch_weights.to(device)

        if p.size(1) == 1:
            correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
        else:
            correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
        total += batch_weights.sum().item()

    return correct / total

