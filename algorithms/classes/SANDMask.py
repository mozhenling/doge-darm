
import torch
import torch.nn.functional as F
from algorithms.classes.ERM import ERM
import torch.autograd as autograd

class SANDMask(ERM):
    """
    SAND-mask: An Enhanced Gradient Masking Strategy for the Discovery of Invariances in Domain Generalization
    <https://arxiv.org/abs/2106.02266>
    """

    def __init__(self,input_shape, num_classes, num_domains, hparams, args):
        super(SANDMask, self).__init__(input_shape, num_classes, num_domains, hparams, args)

        self.tau = hparams["tau"]
        self.k = hparams["k"]


        self.register_buffer('update_count', torch.tensor([0]))

    def update(self, minibatches, unlabeled=None, doyojo=None):

        mean_loss = 0
        if doyojo is None:
            param_gradients = [[] for _ in self.network.parameters()]
        else:
            param_gradients = [[] for _ in doyojo.subalg_net_parameters()]
        for i, (x, y) in enumerate(minibatches):
            logits = self.network(x) if doyojo is None else doyojo.subalg_env_logits(x)
            env_loss = F.cross_entropy(logits, y)
            mean_loss += env_loss.item() / len(minibatches)
            if doyojo is None:
                env_grads = autograd.grad(env_loss, self.network.parameters(), retain_graph=True, allow_unused=True)
            else:
                env_grads = autograd.grad(env_loss, doyojo.subalg_net_parameters(), retain_graph=True, allow_unused=True)
            for grads, env_grad in zip(param_gradients, env_grads):
                if env_grad is None:
                    continue
                grads.append(env_grad)

        if doyojo is not None:
            self.param_gradients = param_gradients
            return {}
            # ------------------------------------------------------------------------
        else:
            self.optimizer.zero_grad()
            # gradient masking applied here
            self.mask_grads(param_gradients, self.network.parameters())
            self.optimizer.step()
            if self.args.scheduler:
                self.scheduler.step()
            self.update_count += 1

            return {'loss': mean_loss}

    # ------------------------------------------------------------------------
    # ------------------------- DoYoJo sub-algorithm -------------------------
    def update_alpha_nets(self, doyojo):
        self.mask_grads( self.param_gradients, doyojo.subalg_net_parameters())
    # ------------------------------------------------------------------------

    def mask_grads(self, gradients, params):
        '''
        Here a mask with continuous values in the range [0,1] is formed to control the amount of update for each
        parameter based on the agreement of gradients coming from different environments.
        '''
        for param, grads in zip(params, gradients):
            if grads != []:
                grads = torch.stack(grads, dim=0)
                avg_grad = torch.mean(grads, dim=0)
                grad_signs = torch.sign(grads)
                gamma = torch.tensor(1.0).to(self.args.device)
                grads_var = grads.var(dim=0)
                grads_var[torch.isnan(grads_var)] = 1e-17
                lam = (gamma * grads_var).pow(-1)
                #-------------- checked !
                mask = torch.tanh(self.k * lam * (torch.abs(grad_signs.mean(dim=0)) - self.tau))
                mask = torch.max(mask, torch.zeros_like(mask))
                #-------------
                mask[torch.isnan(mask)] = 1e-17
                mask_t = (mask.sum() / mask.numel())
                param.grad = mask * avg_grad
                param.grad *= (1. / (1e-10 + mask_t))
        return 0

