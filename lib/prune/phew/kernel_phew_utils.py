import random
import copy
import torch
import numpy as np
from Prune import Utils

def get_norm(p):
    p2 = abs(p.detach().cpu().numpy())
    p1 = np.zeros((p2.shape[0],p.shape[1]))
    for i in range(p2.shape[0]):
        for j in range(p2.shape[1]):
            p1[i][j] = np.sum(np.absolute(p2[i][j]))
    return p1

def generate_probability(network, verbose = True):
    net = copy.deepcopy(network)
    prob = []
    reverse_prob = []
    kernel_prob = []
    layer_no = 0
    for p in net.parameters():
        if len(p.data.size()) != 1:
            prob.append([])
            reverse_prob.append([])
            kernel_prob.append([])
            p2 = abs(p.detach().cpu().numpy())
            p1 = get_norm(p)
            p1 = np.array(p1)
            for i in range(p1.shape[0]):
                par = p1[i]
                if verbose == True:
                    print(f'i value : {i},' \
                          + f'\tTarget Value: {p1.shape[0]}', end="\r", flush=True)
                if sum(par) != 0:
                    pvals = [float(float(o) / float(sum(par))) for o in par]
                else:
                    pvals = [float(float(1) / float(par.shape[0])) for o in par]
                reverse_prob[layer_no].append(pvals)
            p1 = np.transpose(np.array(p1))
            for i in range(p1.shape[0]):
                par = p1[i]
                if verbose == True:
                    print(f'i value : {i},' \
                          + f'\tTarget Value: {p1.shape[0]}', end="\r", flush=True)
                if sum(par) != 0:
                    pvals = [float(float(o) / float(sum(par))) for o in par]
                else:
                    pvals = [float(float(1)/float(par.shape[0])) for o in par]
                prob[layer_no].append(pvals)
            layer_no = layer_no + 1
    return prob, reverse_prob

def generate_masks(model):
    weight_masks = []
    for p in model.parameters():
        if len(p.data.size()) != 1:
            retained_inds = p.data.abs() > 0
            weight_masks.append(retained_inds.float())

    bias_masks = []
    for i in range(len(weight_masks)):
        mask = torch.ones(len(weight_masks[i]))
        for j in range(len(weight_masks[i])):
            if torch.sum(weight_masks[i][j]) == 0:
                mask[j] = torch.tensor(0.0)
        bias_masks.append(mask)
    return weight_masks, bias_masks

def sum_masks(mask):
    num = 0
    for i in range(len(mask)):
        num = num + torch.sum(mask[i])
    return num

def select_seed_unit(mask, robin_unit = -1, forward = True, round_robin = False):
    seed_unit = None
    if forward:
        length = mask[0].shape[1]
    elif not forward:
        length = mask[-1].shape[0]
    if not round_robin:
        seed_unit = random.choice(list(range(length)))
    elif round_robin:
        if robin_unit < length - 1:
            seed_unit = robin_unit + 1
        else:
            seed_unit = 0
    return seed_unit

def select_seed_unit_counter(mask, counter, forward = True):
    seed_unit = None
    if forward:
        length = mask[0].shape[1]
    elif not forward:
        length = mask[-1].shape[0]
    seed_unit = np.argmin(counter)
    return seed_unit

def get_param_options(mask, prev_unit, prob, forward):
    if forward:
        idx1 = int(np.random.choice(list(range(mask.shape[0])), 1, p=prob))
        idx2 = int(prev_unit)
        return idx1, idx2, mask.shape[0]
    elif not forward:
        idx1 = int(prev_unit)
        idx2 = int(np.random.choice(list(range(mask.shape[1])), 1, p=prob))
        return idx1, idx2

def get_unit_options(mask, prev_unit, prob, forward):
    if forward:
        idx_options = list(range(mask.shape[0]))
        return np.random.choice(idx_options, 1, p=prob),prev_unit
    elif not forward:
        idx_options = list(range(mask.shape[1]))
        return prev_unit, np.random.choice(idx_options, 1, p=prob), mask.shape[1]

def conv_to_linear_unit(mask, prev_unit, conv_length, linear_length, forward):
    if forward:
        idx = random.choice(list(range(int(mask.shape[1]/conv_length))))
        idx = idx + int(mask.shape[1]/conv_length)*prev_unit
        return idx
    elif not forward:
        factor = int(linear_length/conv_length)
        idx = int(prev_unit/factor)
        return idx

def phew_masks(network, prune_perc,  prob, reverse_prob, weight_masks,bias_masks, verbose = True, kernel_conserved = True):
    net = copy.deepcopy(network)
    num = 0
    for p in net.parameters():
        p.data.fill_(0)
        if len(p.data.size()) != 1:
            num = num + 1
    #weight_masks, bias_masks = generate_masks(net)
    num_weights = Utils.count_weights(network)
    num_weights = int((1 - prune_perc / 100.0) * num_weights)
    input_robin_unit = -1
    input_counter = np.zeros(weight_masks[0].shape[1])
    output_robin_unit = -1
    output_counter = np.zeros(weight_masks[-1].shape[0])
    i = 0

    while sum_masks(weight_masks) < num_weights:

        if i%2 == 0:

            conv_length = 0
            #prev_unit = select_seed_unit_counter(weight_masks, input_counter, forward=True)
            prev_unit = select_seed_unit(weight_masks, input_robin_unit, forward=True, round_robin=True)
            input_robin_unit = prev_unit
            input_counter[prev_unit] = input_counter[prev_unit] + 1
            ctol_flag = 0

            for k in range(len(weight_masks)):
                if len(weight_masks[k].shape) == 4:
                    if weight_masks[k].shape[2] == 1:
                        weight_masks[k].fill_(1)
                    else:
                        idx1, idx2, conv_length = get_param_options(weight_masks[k], prev_unit,prob[k][int(prev_unit)],forward=True)
                        weight_masks[k][idx1, idx2].fill_(1)
                        bias_masks[k][idx1] = 1
                        prev_unit = idx1
                        ctol_flag = 1
                elif len(weight_masks[k].shape) == 2:
                    if ctol_flag == 1:
                        prev_unit = conv_to_linear_unit(weight_masks[k], prev_unit, conv_length, 0, forward=True)
                        ctol_flag = 0
                    idx1, idx2 = get_unit_options(weight_masks[k], prev_unit, prob[k][int(prev_unit)], forward=True)
                    weight_masks[k][idx1, idx2] = 1
                    bias_masks[k][idx1] = 1
                    prev_unit = idx1
                    if k == num - 1:
                        output_counter[prev_unit] = output_counter[prev_unit] + 1
        else:
            # prev_unit = select_seed_unit(weight_masks, output_counter, forward=False)
            prev_unit = select_seed_unit(weight_masks, output_robin_unit, forward=False, round_robin=True)
            output_robin_unit = prev_unit
            output_counter[prev_unit] = output_counter[prev_unit] + 1
            ltoc_flag = 0
            linear_length = 0
            for k in range(len(weight_masks)):
                if len(weight_masks[num - k - 1].shape) == 2:
                    idx1, idx2, linear_length = get_unit_options(weight_masks[num - k - 1], prev_unit,
                                                                 reverse_prob[num - k - 1][int(prev_unit)],
                                                                 forward=False)
                    weight_masks[num - k - 1][idx1, idx2] = 1
                    bias_masks[num - k - 1][idx1] = 1
                    prev_unit = idx2
                    ltoc_flag = 1
                if len(weight_masks[num - k - 1].shape) == 4:
                    if ltoc_flag == 1:
                        prev_unit = conv_to_linear_unit(weight_masks[num - k - 1], prev_unit, conv_length,
                                                        linear_length, forward=False)
                        ltoc_flag = 0
                    if weight_masks[num - k - 1].shape[2] == 1:
                        weight_masks[num - k - 1].fill_(1)
                    else:
                        idx1, idx2 = get_param_options(weight_masks[num - k - 1], prev_unit,
                                                                   reverse_prob[num - k - 1][int(prev_unit)],forward=False)
                        # weight_masks[num-k-1][idx1,idx2,idx3,idx4] = 1
                        weight_masks[num - k - 1][idx1, idx2].fill_(1)
                        bias_masks[num - k - 1][idx1] = 1
                        prev_unit = idx2
                    if k == num - 1:
                        input_counter[prev_unit] = input_counter[prev_unit] + 1
        i = i + 1
        if verbose:
            print(f'Enabled Weights: {sum_masks(weight_masks)},' \
                  + f'\tTarget Weights: {num_weights}', end="\r", flush=True)
    Utils.ratio(network, weight_masks)
    return weight_masks, bias_masks