#!/bin/bash
read -p "Cuda: " cuda
read -p "Dataset: " d
read -p "Model: " model
read -p "seed: " seed

save_dir="results" #_no_skip"
prunes="SNIP GraSP SynFlow ERK Rand iterSNIP" #PHEW"
density="0.01 0.05 0.1 0.2 0.4"
skip_if_exist=true # if directory exited, skip if true otherwise generate another instance within

for dense in $density; do
for prune in $prunes; do
    if [ ! -d "$save_dir/density_$dense/$d/$model/$prune/$seed" ] || [ $skip_if_exist != true ] ;
    then
	echo "$save_dir/density_$dense/$d/$model/$prune/$seed"
	CUDA_VISIBLE_DEVICES=$cuda python3 main.py --model $model --data $d \
	--decay-schedule constant \
	--seed $seed \
	--optimizer sgd\
	--prune $prune\
	--sparse \
	--density $dense \
	--save_dir $save_dir
    fi
done
done
