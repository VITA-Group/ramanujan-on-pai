import torch
import copy
import numpy as np
from Prune import Utils

def snip_prune_masks(network, prune_perc, dataloader, loss, dev):
    net = copy.deepcopy(network)
    net.to(dev)
    net.train()

    for batch_idx, (data,target) in enumerate(dataloader):
        data, target = data.to(dev), target.to(dev)
        L = loss(net(data), target)
        L.backward()

    scores = []
    for p in net.parameters():
        if len(p.data.size()) != 1:
            scores.append(p.grad.abs_() * p.data.abs_())

    all_weights = []
    for i in range(len(scores)):
        all_weights += list(scores[i].cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(all_weights), prune_perc)

    weight_masks = []
    for i in range(len(scores)):
        pruned_inds = scores[i] > threshold
        weight_masks.append(pruned_inds.float())

    bias_masks = []
    for i in range(len(weight_masks)):
        mask = torch.ones(len(weight_masks[i]))
        for j in range(len(weight_masks[i])):
            if torch.sum(weight_masks[i][j]) == 0:
                mask[j] = torch.tensor(0.0)
        mask.to(dev)
        bias_masks.append(mask)
    Utils.ratio(net,weight_masks)
    del net
    return weight_masks, bias_masks