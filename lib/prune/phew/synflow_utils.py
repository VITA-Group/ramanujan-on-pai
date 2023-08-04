import torch
import copy
import numpy as np
from Prune import Utils

def synflow_prune_masks(network, prune_perc, dataloader,  iterations, dev):

    (data, _) = next(iter(dataloader))
    input_dim = list(data[0, :].shape)
    input = torch.ones([1] + input_dim).to(dev)
    rho = float(float(1) / float(1 - prune_perc / 100.0))

    net = copy.deepcopy(network)
    for p in net.parameters():
        p.data = p.data.abs_()

    for i in range(iterations):
        net1 = copy.deepcopy(net)
        net1.to(dev)
        net1.eval()

        if i>0:
            net1.set_masks(weight_masks,bias_masks)
        output = net1(input)
        torch.sum(output).backward()

        scores = []
        for p in net1.parameters():
            if len(p.data.size()) != 1:
                scores.append(p.grad.abs_() * p.data.abs_())

        ratio = float(1.0 / (rho ** ((i + 1) / iterations)))
        ratio = (1 - ratio) * 100.0
        print(ratio)

        all_weights = []
        for k in range(len(scores)):
            all_weights += list(scores[k].cpu().data.abs().numpy().flatten())
        threshold = np.percentile(np.array(all_weights), ratio)

        weight_masks = []
        for k in range(len(scores)):
            pruned_inds = scores[k] > threshold
            weight_masks.append(pruned_inds.float())
        Utils.ratio(network, weight_masks)

        bias_masks = []
        for k in range(len(weight_masks)):
            mask = torch.ones(len(weight_masks[k]))
            for j in range(len(weight_masks[k])):
                if torch.sum(weight_masks[k][j]) == 0:
                    mask[j] = torch.tensor(0.0)
            mask.to(dev)
            bias_masks.append(mask)
        del net1
    Utils.ratio(net,weight_masks)
    return weight_masks, bias_masks