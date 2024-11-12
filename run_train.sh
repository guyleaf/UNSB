#!/usr/bin/env bash
set -e

# python train.py --dataroot ./datasets/horse2zebra --name h2z_SB_2 --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 0 --display_env h2z_SB_2

python train.py --dataroot ./datasets/maps --name maps_SB --mode sb --lambda_SB 1.0 --lambda_NCE 1.0 --gpu_ids 1 --direction BtoA --display_env maps_SB
