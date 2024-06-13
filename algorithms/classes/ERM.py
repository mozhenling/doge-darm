# coding=utf-8
import torch
import torch.nn as nn
# import torch.nn.functional as F
from losses.lss_selector import get_loss_class
from networks.net_selector import get_nets
from algorithms.classes.Algorithm import Algorithm
from algorithms.optimization import get_optimizer, get_scheduler

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, args):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams,args)

        self.featurizer, self.classifier = get_nets(input_shape, num_classes, num_domains, hparams, args)

        self.network = nn.Sequential(self.featurizer, self.classifier)

        self.optimizer = get_optimizer(params =self.network.parameters(), hparams=self.hparams, args=self.args)
        self.scheduler = get_scheduler(optimizer = self.optimizer, args= self.args)

        if args.erm_loss in ['GLSLoss']:
            self.erm_loss = get_loss_class(args.erm_loss)(num_classes=num_classes, smooth_rate = hparams['smooth_rate'])
        elif args.erm_loss in ['JSDLoss']:
            self.erm_loss = get_loss_class(args.erm_loss)(num_classes=num_classes, d_weight=hparams['d_weight'])
        else:
             self.erm_loss = get_loss_class(args.erm_loss)(num_classes=num_classes)

        self.device = args.device
    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, y in minibatches])
        all_logits = self.predict(all_x)
        nll = 0.
        all_logits_idx = 0
        for i, (x, y) in enumerate(minibatches):
            logits = all_logits[all_logits_idx:all_logits_idx + x.shape[0]]
            all_logits_idx += x.shape[0]
            nll += self.erm_loss(logits, y)
        nll /= len(minibatches)

        self.optimizer.zero_grad()
        nll.backward()
        self.optimizer.step()
        # -- if learning schedule is not none, adjust learning rate based on sch
        if self.args.scheduler:
            self.scheduler.step()

        return {'loss': nll.item()}

    def predict(self, x):
        return self.network(x)

