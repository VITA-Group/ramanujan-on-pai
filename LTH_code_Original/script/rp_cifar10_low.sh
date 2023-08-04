CUDA_VISIBLE_DEVICES=$2 nohup python -u main_train_rp.py \
    --data ../data \
    --dataset cifar10 \
    --arch resnet20s \
    --seed 1 \
    --workers 2 \
    --save_dir rp_cifar10_low/$1 \
    --lr 0.01 \
    --state $1 \
    --tickets_init LT_low_cifar10_resnet20s_seed1/0checkpoint.pth.tar > log_rp_cifar10_low_$1.out 2>&1 &




