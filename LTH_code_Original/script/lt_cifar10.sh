python -u main_imp.py \
	--data ../data \
	--dataset cifar10 \
	--arch resnet20s \
	--seed 1 \
	--prune_type lt \
	--rewind_epoch 0 \
    --pruning_times 60 \
	--save_dir LT_cifar10_resnet20s_seed1 