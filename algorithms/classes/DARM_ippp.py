import torch
from algorithms.classes.DARM import DARM

class DARM_ippp(DARM):
    """
    Distance Aware Risk Minimization (DARM) based on ip and pp distances
    We try to evaluate the following "distances" for domain generalization:
        D1: Instance-to-Prototype Distance (the base, ip)
        D3: Prototype-to-Prototype Distance (pp)
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, args):
        super(DARM_ippp, self).__init__(input_shape, num_classes, num_domains, hparams, args)


    def update(self, minibatches, unlabeled=None):
        # -- pool all domains of data/labels
        x = torch.cat([x for x, y in minibatches])
        y = torch.cat([y for x, y in minibatches])

        self.hparams['loss_pp_weight'] = (self.loss_pp_weight_orignal if self.update_count
                                                          >= self.hparams['warm_up_pp'] else 1.0)

        # -- prepare features
        z = self.featurizer(x)

        #-- prepare different losses
        loss_dict = {}
        # -- instance-to-prototype loss
        sim_logits = self.classifier.forward(z)
        loss_dict['loss_ip'] = self.classifier.loss_ip(sim_logits, y)

        # -- prototype-to-prototype loss
        loss_dict['loss_pp'] = self.classifier.loss_pp()

        # -- sum up all losses with weights
        loss_sum = torch.sum(torch.stack([self.hparams[key + '_weight'] * loss_dict[key] for key in loss_dict.keys()], dim=0))

        if  self.update_count == self.hparams['warm_up_pp']:
            # Reset Adam, because it doesn't like the sharp jump in gradient
            # magnitudes that happens at this step.
            self.optimizer = torch.optim.Adam(
                list(self.featurizer.parameters()) + list(self.classifier.parameters()),
                lr=self.hparams["lr"],
                weight_decay=self.hparams['weight_decay'])

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

