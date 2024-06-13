import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithms.classes.Algorithm import Algorithm
from algorithms.optimization import get_optimizer, get_scheduler
from networks.net_selector import get_nets
from losses.dist_criteria import get_dist_criterion


class DARM(Algorithm):
    """
    Distance Aware Risk Minimization (DARM)
    We try to consider the following "distances" for domain generalization:
        D1: Instance-to-Prototype Distance (the base)
        D2: Instance-to-Instance Distance
        D3: Prototype-to-Prototype Distance
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams, args):
        super(DARM, self).__init__(input_shape, num_classes, num_domains, hparams, args)
        self.register_buffer('update_count', torch.tensor([0]))
        # -- build the model
        self.featurizer, _ = get_nets(input_shape, num_classes, num_domains, hparams, args)
        self.dist_criterion = get_dist_criterion(hparams['dist_criterion'])

        self.classifier = PrototypeClassifier(feat_dim=self.featurizer.n_outputs,
                                            num_classes=num_classes, dist_criterion = self.dist_criterion,
                                            pair_reserve_rate = hparams['pair_reserve_rate'])
        self.network = nn.Sequential(self.featurizer, self.classifier)

        # -- set the optimizer
        self.optimizer = get_optimizer(params=self.network.parameters(), hparams=self.hparams, args=self.args)
        self.scheduler = get_scheduler(optimizer=self.optimizer, args=self.args)

        self.loss_ii_weight_orignal = hparams['loss_ii_weight']
        self.loss_pp_weight_orignal = hparams['loss_pp_weight']

    def update(self, minibatches, unlabeled=None):
        # -- pool all domains of data/labels
        x = torch.cat([x for x, y in minibatches])
        y = torch.cat([y for x, y in minibatches])
        y_domain = torch.cat([torch.full((x.shape[0],), i ) for i, (x, y) in enumerate(minibatches)])

        self.hparams['loss_ii_weight'] = (self.loss_ii_weight_orignal if self.update_count
                                                            >= self.hparams['warm_up_ii'] else 1.0)
        self.hparams['loss_pp_weight'] = (self.loss_pp_weight_orignal if self.update_count
                                                          >= self.hparams['warm_up_pp'] else 1.0)

        # -- prepare features
        z = self.featurizer(x)

        #-- prepare different losses
        loss_dict = {}
        # -- instance-to-prototype loss
        sim_logits = self.classifier.forward(z)
        loss_dict['loss_ip'] = self.classifier.loss_ip(sim_logits, y)

        # -- instance-to-instance loss
        loss_dict['loss_ii'] = self.classifier.loss_ii(z, y, y_domain)

        # -- prototype-to-prototype loss
        loss_dict['loss_pp'] = self.classifier.loss_pp()

        # -- sum up all losses with weights
        loss_sum = torch.sum(torch.stack([self.hparams[key + '_weight'] * loss_dict[key] for key in loss_dict.keys()], dim=0))


        if self.update_count == self.hparams['warm_up_ii'] or self.update_count == self.hparams['warm_up_pp']:
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

class PrototypeClassifier(torch.nn.Module):
    def __init__(self, feat_dim, num_classes, dist_criterion,
                 temperature = 1, base_temperature=1, init_weight=True, pair_reserve_rate=1):
        super(PrototypeClassifier, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.dist_criterion = dist_criterion
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.pair_reserve_rate = pair_reserve_rate
        # -- each class has a prototype
        self.prototypes = nn.Parameter(torch.randn(self.num_classes,self.feat_dim).cuda(),
                                      requires_grad=True)
        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        nn.init.kaiming_normal_(self.prototypes)

    def forward(self, x):
        # replace the standard logits (z*w, the inner product) by negative distances between z and w
        return -self.dist_criterion(x, self.prototypes)

    def loss_ip(self, sim_logits, labels):
        # reduction='mean'
        return F.cross_entropy(sim_logits, labels)

    def loss_pp(self):
        # -- prototype-to-prototype loss: minimizing loss_pp <=> maximizing dists between pi and pj

        mask_no_self = (~torch.eye(self.num_classes).bool()).float().to(self.prototypes.device)
        # mask_is_self = (torch.eye(self.num_classes).bool()).float().to(self.prototypes.device)

        sim_logits = -self.dist_criterion(self.prototypes, self.prototypes)
        # with temperature parameters to control concentrations
        logits = sim_logits / self.temperature

        # shift the origin for numerical stability (normally, the largest negative distance is from the self comparison)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # drop self comparisons
        exp_logits = torch.exp(logits) * mask_no_self

        return torch.log(1 + exp_logits.sum(dim=1).mean(dim=0) )

    def loss_ii(self, z, y, y_domain):
        """

        The implementation is inspired and modified based the followings.
        Ref.[1] Y. Ruan, Y. Dubois, and C. J. Maddison, “Optimal Representations for Covariate
                Shift.” arXiv, Mar. 14, 2022. Accessed: Nov. 14, 2023. [Online]. Available: http://arxiv.org/abs/2201.00057

        Ref.[2] P. Khosla, et al., in “Supervised Contrastive Learning“.
                 Modified from  https://github.com/HobbitLong/SupContrast/blob/8d0963a7dbb1cd28accb067f5144d61f18a77588/losses.py#L11

        """
        device = z.device
        batch_size = z.shape[0]

        y = y.contiguous().view(-1, 1)
        y_domain = y_domain.contiguous().view(-1, 1)
        
        # same labels are 1s, otherwise 0s
        mask_y_in_cls = torch.eq(y, y.T).to(device)
        mask_y_out_cls = ~mask_y_in_cls
        
        # different domains are 1s, otherwise 0s
        mask_y_ood = torch.ne(y_domain, y_domain.T).to(device)
        mask_y_iid = ~mask_y_ood
        
        # drop the "current"/"self" example: diagonal entries are zeros
        mask_no_self = ~torch.eye(batch_size).bool().to(device)  

        # for multiplications
        mask_y_in_cls, mask_y_out_cls, mask_no_self, mask_y_ood, mask_y_iid = \
        mask_y_in_cls.float(), mask_y_out_cls.float(), mask_no_self.float(), mask_y_ood.float(), mask_y_iid.float()

        # compute similarity-based logits
        # similarity = negative distance
        # comparisons among features
        sim_logits = -self.dist_criterion(z, z)
        # with temperature parameters to control concentrations
        logits = sim_logits / self.temperature
        # drop self comparisons
        logits = logits * mask_no_self

        # shift the origin for numerical stability (normally, the largest negative distance is from the self comparison)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # reserve counterpart pairs based on self.pair_reserve_rate
        reserve_counterpart = (torch.rand_like(logits) < self.pair_reserve_rate).float().to(device)
        # get positive set mask
        mask_y_ood_with_reserved_iid =  torch.where(mask_y_ood == 1., mask_y_ood, reserve_counterpart)
        # get negative set mask
        # mask_y_iid_with_reserved_ood = torch.where(mask_y_iid == 1., mask_y_iid, reserve_counterpart)

        # Positive sets: S
        mask_S = mask_no_self * mask_y_in_cls * mask_y_ood_with_reserved_iid
        # Negative sets: R
        mask_R = mask_no_self #* mask_y_in_cls * mask_y_iid_with_reserved_ood

        # apply R
        sum_exp_logits = (torch.exp(logits) * mask_R).sum(1, keepdim=True)

        log_prob = torch.log(sum_exp_logits) - logits

        # apply S with valid values
        valid_idx = (mask_S.sum(1) > 0)
        mask_S = mask_S[valid_idx]
        log_prob = log_prob[valid_idx]
        mean_log_prob_pos = (mask_S * log_prob).sum(1) / mask_S.sum(1)

        # apply temperature
        loss = (self.temperature / self.base_temperature) * mean_log_prob_pos

        # select valid values and take the average over batch I
        def finite_mean(x):
                # only 1D for now
                num_finite = (torch.isfinite(x).float()).sum()
                mean = torch.where(torch.isfinite(x), x, torch.tensor(0.0).to(x)).sum()
                if num_finite != 0:
                    mean = mean / num_finite
                else:
                    return torch.tensor(0.0).to(x)
                return mean

        mean_out = finite_mean(loss)

        return mean_out