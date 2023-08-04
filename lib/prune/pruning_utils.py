import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from .phew import phew_utils
# from layers import Conv2d
# from layers import Linear

__all__ = [
    'masked_parameters',
    'SynFlow',
    'Mag',
    'Taylor1ScorerAbs',
    'Rand',
    'SNIP',
    'GraSP',
    'check_sparsity',
    'check_sparsity_dict',
    'prune_model_identity',
    'prune_model_custom',
    'extract_mask',
    'ERK',
    'PHEW',
]


def masks(module):
    r"""Returns an iterator over modules masks, yielding the mask.
    """
    for name, buf in module.named_buffers():
        if "mask" in name:
            yield buf


def masked_parameters(model):
    r"""Returns an iterator over models prunable parameters, yielding both the
    mask and parameter tensors.
    """
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            mask = torch.ones_like(module.weight)
            prune.CustomFromMask.apply(module, 'weight', mask)
            yield module.weight_mask, module.weight_orig


class Pruner:

    def __init__(self, masked_parameters):
        self.masked_parameters = list(masked_parameters)
        self.scores = {}

    def score(self, model, loss, dataloader, device):
        raise NotImplementedError

    def _global_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level globally.
        """
        # # Set score for masked parameters to -inf
        # for mask, param in self.masked_parameters:
        #     score = self.scores[id(param)]
        #     score[mask == 0.0] = -np.inf

        # Threshold scores
        global_scores = torch.cat(
            [torch.flatten(v) for v in self.scores.values()])
        k = int((1 - sparsity) * global_scores.numel())
        if not k < 1:
            threshold = global_scores.topk(k)[0][-1]
            for mask, param in self.masked_parameters:
                score = self.scores[id(param)]
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def _local_mask(self, sparsity):
        r"""Updates masks of model with scores by sparsity level parameter-wise.
        """
        for mask, param in self.masked_parameters:
            score = self.scores[id(param)]
            k = int((1 - sparsity) * score.numel())
            if not k < 1:
                threshold, _ = torch.kthvalue(torch.flatten(score), k)
                zero = torch.tensor([0.]).to(mask.device)
                one = torch.tensor([1.]).to(mask.device)
                mask.copy_(torch.where(score <= threshold, zero, one))

    def mask(self, sparsity, scope):
        r"""Updates masks of model with scores by sparsity according to scope.
        """
        if scope == 'global':
            self._global_mask(sparsity)
        if scope == 'local':
            self._local_mask(sparsity)

    @torch.no_grad()
    def apply_mask(self):
        r"""Applies mask to prunable parameters.
        """
        for mask, param in self.masked_parameters:
            param.mul_(mask)

    def alpha_mask(self, alpha):
        r"""Set all masks to alpha in model.
        """
        for mask, _ in self.masked_parameters:
            mask.fill_(alpha)

    # Based on https://github.com/facebookresearch/open_lth/blob/master/utils/tensor_utils.py#L43
    def shuffle(self):
        for mask, param in self.masked_parameters:
            shape = mask.shape
            perm = torch.randperm(mask.nelement())
            mask = mask.reshape(-1)[perm].reshape(shape)

    def invert(self):
        for v in self.scores.values():
            v.div_(v**2)

    def stats(self):
        r"""Returns remaining and total number of prunable parameters.
        """
        remaining_params, total_params = 0, 0
        for mask, _ in self.masked_parameters:
            remaining_params += mask.detach().cpu().numpy().sum()
            total_params += mask.numel()
        return remaining_params, total_params


class SynFlow(Pruner):

    def __init__(self, masked_parameters):
        super(SynFlow, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):

        @torch.no_grad()
        def linearize(model):
            # model.double()
            signs = {}
            for name, param in model.state_dict().items():
                signs[name] = torch.sign(param)
                param.abs_()
            return signs

        @torch.no_grad()
        def nonlinearize(model, signs):
            # model.float()
            for name, param in model.state_dict().items():
                param.mul_(signs[name])

        signs = linearize(model)

        (data, _) = next(iter(dataloader))
        input_dim = list(data[0, :].shape)
        input = torch.ones([1] + input_dim).to(
            device)  #, dtype=torch.float64).to(device)
        output = model(input)
        torch.sum(output).backward()

        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()

        nonlinearize(model, signs)


class Mag(Pruner):

    def __init__(self, masked_parameters):
        super(Mag, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.data).detach().abs_()


class Rand(Pruner):

    def __init__(self, masked_parameters):
        super(Rand, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.randn_like(p)


# Based on https://github.com/mi-lad/snip/blob/master/snip.py#L18
class SNIP(Pruner):

    def __init__(self, masked_parameters):
        super(SNIP, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):

        # allow masks to have gradient
        for m, _ in self.masked_parameters:
            m.requires_grad = True

        # compute gradient
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss(output, target).backward()

        # calculate score |g * theta|
        for m, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(m.grad).detach().abs_()
            p.grad.data.zero_()
            m.grad.data.zero_()
            m.requires_grad = False

        # normalize score
        all_scores = torch.cat(
            [torch.flatten(v) for v in self.scores.values()])
        norm = torch.sum(all_scores)
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


# Based on https://github.com/alecwangcq/GraSP/blob/master/pruner/GraSP.py#L49
class GraSP(Pruner):

    def __init__(self, masked_parameters):
        super(GraSP, self).__init__(masked_parameters)
        self.temp = 200
        self.eps = 1e-10

    def score(self, model, loss, dataloader, device):

        # first gradient vector without computational graph
        stopped_grads = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = torch.autograd.grad(
                L, [p for (_, p) in self.masked_parameters],
                create_graph=False)
            flatten_grads = torch.cat(
                [g.reshape(-1) for g in grads if g is not None])
            stopped_grads += flatten_grads

        # second gradient vector with computational graph
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data) / self.temp
            L = loss(output, target)

            grads = torch.autograd.grad(
                L, [p for (_, p) in self.masked_parameters], create_graph=True)
            flatten_grads = torch.cat(
                [g.reshape(-1) for g in grads if g is not None])

            gnorm = (stopped_grads * flatten_grads).sum()
            gnorm.backward()

        # calculate score Hg * theta (negate to remove top percent)
        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p.data).detach()
            p.grad.data.zero_()

        # normalize score
        all_scores = torch.cat(
            [torch.flatten(v) for v in self.scores.values()])
        norm = torch.abs(torch.sum(all_scores)) + self.eps
        for _, p in self.masked_parameters:
            self.scores[id(p)].div_(norm)


class Taylor1ScorerAbs(Pruner):

    def __init__(self, masked_parameters):
        super(Taylor1ScorerAbs, self).__init__(masked_parameters)

    def score(self, model, loss, dataloader, device):

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss(output, target).backward()

        for _, p in self.masked_parameters:
            self.scores[id(p)] = torch.clone(p.grad * p).detach().abs_()
            p.grad.data.zero_()


class ERK(Pruner):
    """ERK pruning method

    code is copy and modified from snippet in FreeTicket github
    """

    def __init__(self, masked_parameters):
        """TODO: to be defined.

        :masked_parameters: TODO

        """
        Pruner.__init__(self, masked_parameters)

    def score(self, *args, **kwargs):
        pass

    def mask(self, sparsity, scope=None):
        r"""Updates masks of model with scores by sparsity according to scope.
        """
        total_params = 0
        for mask, weight in self.masked_parameters:
            total_params += weight.numel()
        is_epsilon_valid = False
        erk_power_scale = 1.0
        dense_layers = set()
        density = 1 - sparsity
        while not is_epsilon_valid:
            divisor = 0
            rhs = 0
            raw_probabilities = {}
            for name, (mask, params) in enumerate(self.masked_parameters):
                n_param = np.prod(params.shape)
                n_zeros = n_param * (1 - density)
                n_ones = n_param * density

                if name in dense_layers:
                    # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                    rhs -= n_zeros

                else:
                    # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                    # equation above.
                    rhs += n_ones
                    # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                    raw_probabilities[name] = (
                        np.sum(mask.shape) /
                        np.prod(mask.shape))**erk_power_scale
                    # Note that raw_probabilities[mask] * n_param gives the individual
                    # elements of the divisor.
                    divisor += raw_probabilities[name] * n_param
            # By multipliying individual probabilites with epsilon, we should get the
            # number of parameters per layer correctly.
            epsilon = rhs / divisor
            # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
            # mask to 0., so they become part of dense_layers sets.
            max_prob = np.max(list(raw_probabilities.values()))
            max_prob_one = max_prob * epsilon
            if max_prob_one > 1:
                is_epsilon_valid = False
                for mask_name, mask_raw_prob in raw_probabilities.items():
                    if mask_raw_prob == max_prob:
                        print(
                            f"Sparsity of var:{mask_name} had to be set to 0.")
                        dense_layers.add(mask_name)
            else:
                is_epsilon_valid = True
        self.density_dict = {}
        total_nonzero = 0.0
        # With the valid epsilon, we can set sparsities of the remaning layers.
        for name, (mask, params) in enumerate(self.masked_parameters):
            n_param = np.prod(mask.shape)
            if name in dense_layers:
                self.density_dict[name] = 1.0
            else:
                probability_one = epsilon * raw_probabilities[name]
                self.density_dict[name] = probability_one
            mask.data.copy_(
                (torch.rand(mask.shape) < self.density_dict[name]).float())
            total_nonzero += self.density_dict[name] * mask.numel()


class PHEW(Pruner):
    """Docstring for PHEW. """

    def __init__(self, masked_parameters):
        """TODO: to be defined.

        :masked_parameters: TODO

        """
        Pruner.__init__(self, masked_parameters)

    def score(self, *args, **kwargs):
        pass

    def mask(self, sparsity, scope=None):
        parameters = [mask[1] for mask in self.masked_parameters]
        prob, reverse_prob, kernel_prob = phew_utils.generate_probability(
            parameters)

        weight_masks, bias_masks = phew_utils.generate_masks(
            [torch.zeros_like(p) for p in parameters])

        prune_perc = sparsity * 100
        weight_masks, bias_masks = phew_utils.phew_masks(parameters,
                                                         prune_perc,
                                                         prob,
                                                         reverse_prob,
                                                         kernel_prob,
                                                         weight_masks,
                                                         bias_masks,
                                                         verbose=True)
        for i, (m, _) in enumerate(self.masked_parameters):
            m.data.copy_(weight_masks[i].data)


def check_sparsity(model):

    sum_list = 0
    zero_sum = 0

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            sum_list = sum_list + float(m.weight.nelement())
            zero_sum = zero_sum + float(torch.sum(m.weight_mask == 0))
    print('* remain weight = ', 100 * (1 - zero_sum / sum_list), '%')

    return 100 * (1 - zero_sum / sum_list)


def check_sparsity_dict(model_dict):

    sum_list = 0
    zero_sum = 0

    for key in model_dict.keys():
        if 'mask' in key:
            sum_list = sum_list + float(model_dict[key].nelement())
            zero_sum = zero_sum + float(torch.sum(model_dict[key] == 0))
    print('* remain weight = ', 100 * (1 - zero_sum / sum_list), '%')

    return 100 * (1 - zero_sum / sum_list)


def prune_model_identity(model):

    print('start pruning with identity mask')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            print('identity pruning layer {}'.format(name))
            prune.Identity.apply(m, 'weight')


def prune_model_custom(model, mask_dict):

    print('start pruning with custom mask')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            print('custom pruning layer {}'.format(name))
            prune.CustomFromMask.apply(m,
                                       'weight',
                                       mask=mask_dict[name + '.weight_mask'])


def extract_mask(model_dict):

    new_dict = {}

    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = copy.deepcopy(model_dict[key])

    return new_dict
