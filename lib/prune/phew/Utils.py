import copy

import numpy as np
import torch
from torch.autograd import Variable


def to_var(x, requires_grad=False, volatile=False):
    if torch.cuda.is_available():
        x = x.to(torch.device("cuda"))
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def count_weights(parameters):
    num = 0
    for p in parameters:
        if len(p.data.size()) != 1:
            num = num + p.numel()
    return num


def count(weight_masks):
    n = 0
    for i in range(len(weight_masks)):
        n = n + torch.sum(weight_masks[i])
    return n


def ratio(parameters, weight_masks):
    print(1.0 - (count(weight_masks)) / (count_weights(parameters)))


def path_kernel_trace(network, weight_mask, bias_mask, dataloader, dev):
    (data, _) = next(iter(dataloader))
    input_dim = list(data[0, :].shape)
    input = torch.ones([1] + input_dim).to(dev)

    net = copy.deepcopy(network)
    for p in net.parameters():
        p.data = p.data.abs_() * p.data.abs_()

    net.eval()
    net.to(dev)
    net.set_masks(weight_mask, bias_mask)
    output = net(input)
    torch.sum(output).backward()

    pk_trace = torch.tensor(0.0)
    for p in net.parameters():
        if len(p.data.size()) != 1:
            pk_trace = pk_trace + torch.sum(p.grad.data)

    return pk_trace


def layerwise_randomshuffle(net, weight_masks, fraction, dev):

    scores = copy.deepcopy(weight_masks)
    wm = []
    for i in range(len(weight_masks)):
        scores[i] = torch.zeros(scores[i].size())
        ratio1 = 100.0 - 100.0 * torch.sum(weight_masks[i]) / torch.numel(
            weight_masks[i])
        width = 0
        for j in range(len(weight_masks[i])):
            if torch.sum(weight_masks[i][j]) > 0:
                width = width + 1
        width = int((len(weight_masks[i]) - width) * fraction + width)
        all_weights = []
        for j in range(len(scores[i])):
            if j < width:
                scores[i][j] = 100.0 * torch.abs(
                    torch.randn(scores[i][j].size())
                )  #+ 100.0 * torch.ones(scores[i][j].size())
                all_weights += list(
                    scores[i][j].cpu().detach().data.abs().numpy().flatten())
            else:
                scores[i][j] = torch.zeros(scores[i][j].size())
                all_weights += list(
                    scores[i][j].cpu().detach().data.abs().numpy().flatten())
        threshold = np.percentile(np.array(all_weights),
                                  ratio1.cpu().detach().numpy())
        pruned_inds = scores[i] > threshold
        wm.append(pruned_inds.float())

    bm = []
    for i in range(len(wm)):
        mask = torch.ones(len(wm[i]))
        for j in range(len(wm[i])):
            if torch.sum(wm[i][j]) == 0:
                mask[j] = torch.tensor(0.0)
        mask.to(dev)
        bm.append(mask)
    ratio(net, wm)
    return wm, bm
