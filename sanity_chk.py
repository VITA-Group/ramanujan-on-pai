import torch
import torch.nn.utils.prune as prune
from torch.nn import Conv2d

from lib.common_models import models
from lib.prune.pruning_utils import masked_parameters

if __name__ == "__main__":
    path = './results/density_0.01/cifar10/resnet34/SNIP/1/latest/model.pth'
    model = models['resnet34'](num_classes=10, seed=1)
    list(masked_parameters(model))
    model.load_state_dict(torch.load(path, map_location='cpu')['state_dict'])
    print(model.conv1.weight_mask.sum() / model.conv1.weight_mask.numel())
    print((model.conv1.weight_orig != 0.0).sum() / model.conv1.weight.numel())
