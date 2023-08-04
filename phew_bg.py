import copy

from lib.common_models.models import models
from lib.prune.pruning_utils import check_sparsity
from lib.prune.pruning_utils import masked_parameters
from lib.prune.pruning_utils import PHEW

if __name__ == "__main__":
    model = models['vgg16'](num_classes=10, seed=1)
    pruner = PHEW(masked_parameters(model))
    pruner.mask(0.99)
    check_sparsity(model)
