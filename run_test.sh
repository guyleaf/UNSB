#!/usr/bin/env bash
set -e

for epoch in {5..400..5}
do
    python test.py \
    --dataroot ./datasets/horse2zebra \
    --name h2z_SB_2 \
    --checkpoints_dir ./checkpoints \
    --mode sb \
    --eval \
    --phase test \
    --num_test 1000 \
    --epoch "$epoch" \
    --gpu_ids 0
done

# python test.py \
#     --dataroot ./datasets/horse2zebra \
#     --name h2z_SB \
#     --checkpoints_dir ./checkpoints \
#     --mode sb \
#     --eval \
#     --phase test \
#     --num_test 120 \
#     --epoch 265 \
#     --gpu_ids 0
