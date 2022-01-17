#!/bin/bash
set -eux

rm -rf ibc/data/particle/*.tfrecord
python3 ibc/data/policy_eval.py -- \
 --alsologtostderr \
 --policy=particle_green_then_blue \
 --task=PARTICLE \
 --dataset_path=ibc/data/particle/2d_oracle_particle.tfrecord \
 --num_episodes=200 \
 --replicas=10  \
 --use_image_obs=False

 # --num_episodes=1 \
