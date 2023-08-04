python -u main_imp.py \
	--data ../data \
	--dataset cifar10 \
	--arch resnet20s \
	--seed 1 \
	--prune_type rewind_lt \
	--rewind_epoch 3 \
    --pruning_times 60 \
	--save_dir LT_rewind3_cifar10_resnet20s_seed1 