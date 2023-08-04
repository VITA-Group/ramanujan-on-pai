import torch
import copy
import numpy as np
from Prune import Utils

def grasp_prune_masks(network, prune_perc, dataloader, loss, dev):
    net = copy.deepcopy(network)
    net.to(dev)
    net.train()

    temp = 200
    eps = 1e-10

    stopped_grads = 0
    for batch_idx, (data,target) in enumerate(dataloader):
        data, target = data.to(dev), target.to(dev)
        output = net(data) / temp
        L = loss(output, target)

        grads = torch.autograd.grad(L, [p for p in net.parameters()], create_graph=False)
        flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])
        stopped_grads += flatten_grads

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(dev), target.to(dev)
        output = net(data) / temp
        L = loss(output, target)

        grads = torch.autograd.grad(L, [p for p in net.parameters()], create_graph=True)
        flatten_grads = torch.cat([g.reshape(-1) for g in grads if g is not None])

        gnorm = (stopped_grads * flatten_grads).sum()
        gnorm.backward()

    scores = []
    norm = 0
    for p in net.parameters():
        if len(p.data.size()) != 1:
            scores.append(p.grad * p.data)
            norm = norm+torch.sum(p.grad * p.data)
    norm = torch.abs(norm) + eps

    all_weights = []
    for i in range(len(scores)):
        scores[i]=scores[i]/norm
        all_weights += list(scores[i].cpu().data.numpy().flatten())
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
    Utils.ratio(net, weight_masks)
    del net
    return weight_masks, bias_masks