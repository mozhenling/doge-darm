
import numpy as np
import torch.nn.functional as F
from algorithms.classes.ERM import ERM
from datautils.data_process import random_pairs_of_minibatches

class Mixup(ERM):
    """
    Mixup of minibatches from different domains
    https://arxiv.org/pdf/2001.00677.pdf
    https://arxiv.org/pdf/1912.01805.pdf
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, args):
        super(Mixup, self).__init__(input_shape, num_classes, num_domains, hparams, args)
        # self.erm_loss = get_loss_class(args.erm_loss)(num_classes=num_classes)
    def update(self, minibatches, unlabeled=None):
        objective = 0

        for (xi, yi), (xj, yj) in random_pairs_of_minibatches(minibatches):
            lam = np.random.beta(self.hparams["mixup_alpha"],
                                 self.hparams["mixup_alpha"])

            x = lam * xi + (1 - lam) * xj

            predictions = self.predict(x)
            # ------------------------------------------------------------------------
            objective += lam * F.cross_entropy(predictions, yi)      # F.cross_entropy(predictions, yi)
            objective += (1 - lam) *F.cross_entropy(predictions, yj) # F.cross_entropy(predictions, yj)

        objective /= len(minibatches)

        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()
        if self.args.scheduler:
            self.scheduler.step()

        return {'loss': objective.item()}