CUDA_VISIBLE_DEVICES=$3 nohup python -u main_imp.py \
	--data ../data \
	--dataset cifar10 \
	--arch resnet20s \
	--seed 1 \
	--prune_type rewind_lt \
	--rewind_epoch $1 \
    --epochs $2 \
	--save_dir LT_rewind$1_epochs$2_cifar10_resnet20s_seed1 > log_IMP_LT_rewind$1_epochs$2_cifar10_resnet20s_seed1.out 2>&1 &