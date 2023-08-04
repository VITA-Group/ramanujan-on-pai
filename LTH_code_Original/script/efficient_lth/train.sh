for ((i=$1; i<=$2; i=i+1))
do  
    CUDA_VISIBLE_DEVICES=$3 python -u main_train.py \
        --data ../data \
        --dataset cifar10 \
        --arch resnet20s \
        --seed 1 \
        --tickets_mask LT_rewind$4_epochs$5_cifar10_resnet20s_seed1/${i}checkpoint.pth.tar \
        --tickets_init LT_rewind$4_epochs$5_cifar10_resnet20s_seed1/${i}checkpoint.pth.tar \
        --save_dir LT_rewind$4_epochs$5_cifar10_resnet20s_seed1/full-data-training/${i} 
done
