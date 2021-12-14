#!/bin/bash

set -eux

save_dir=/tmp/ibc_logs/mlp_ebm_langevin/langevin/20211213-221138

python3 ibc/data/policy_eval.py -- \
 --alsologtostderr \
 --num_episodes=1 \
 --task=PARTICLE \
 --use_image_obs=False \
 --dataset_path=/tmp/ibc_tmp/data \
 --output_path=/tmp/ibc_tmp/vid \
 --video \
 --saved_model_path ${save_dir}/policies/policy \
 --checkpoint_path ${save_dir}/policies/policy

 # --policy=particle_green_then_blue \
 # --replicas=2  \
