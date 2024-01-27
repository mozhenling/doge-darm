import torch
from algorithms.classes.DARM import DARM

class DARM_ip(DARM):
    """
    Distance Aware Risk Minimization (DARM) based on ip distances
    We try to evaluate the following "distances" for domain generalization:
        D1: Instance-to-Prototype Distance (the base, ip) with the distance measure as a hyperparameter

    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, args):
        super(DARM_ip, self).__init__(input_shape, num_classes, num_domains, hparams, args)

    def update(self, minibatches, unlabeled=None):
        # -- pool all domains of data/labels
        x = torch.cat([x for x, y in minibatches])
        y = torch.cat([y for x, y in minibatches])

        # -- prepare features
        z = self.featurizer(x)

        #-- prepare different losses
        loss_dict = {}
        # -- instance-to-prototype loss
        sim_logits = self.classifier.forward(z)
        loss_dict['loss_ip'] = self.classifier.loss_ip(sim_logits, y)

        # -- sum up all losses with weights
        loss_sum = torch.sum(torch.stack([self.hparams[key + '_weight'] * loss_dict[key] for key in loss_dict.keys()], dim=0))


        self.optimizer.zero_grad()
        loss_sum.backward()
        self.optimizer.step()
        # -- if learning schedule is not none, adjust learning rate based on sch
        if self.args.scheduler:
            self.scheduler.step()

        # ------------------------- return results: each loss and final weighted loss
        output_dict = {}
        for i, (key, value) in enumerate(loss_dict.items()):
            output_dict[ key] = value.item()
        output_dict['loss_sum'] = loss_sum.item()
        self.update_count += 1
        return output_dict

    def predict(self, x):
        sim_logits = self.network(x)
        return sim_logits

