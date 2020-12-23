#!/bin/bash

# Sunglass
python MLFinalProject.py --poisoned_data data/sunglasses_poisoned_data.h5 --model models/output/sunglass_fixed_net.h5 --action infer --batch_size 512  | tee models/output/log/sunglass_infer.log

# MTMT
python MLFinalProject.py --poisoned_data data/Multi-trigger\ Multi-target/eyebrows_poisoned_data.h5 --batch_size 512 --action infer --model models/output/multi_trigger_multi_target_fixed_net | tee models/output/log/multi_trigger_multi_target_infer_eyebrows.log

python MLFinalProject.py --poisoned_data data/Multi-trigger\ Multi-target/lipstick_poisoned_data.h5 --batch_size 512 --action infer --model models/output/multi_trigger_multi_target_fixed_net | tee models/output/log/multi_trigger_multi_target_infer_lipstick.log

python MLFinalProject.py --poisoned_data data/Multi-trigger\ Multi-target/sunglasses_poisoned_data.h5 --batch_size 512 --action infer --model models/output/multi_trigger_multi_target_fixed_net | tee models/output/log/multi_trigger_multi_target_infer_sunglass.log

# Anonymous 1
python MLFinalProject.py --poisoned_data data/anonymous_1_poisoned_data.h5 --batch_size 512 --model models/output/anonymous_1_fixed_net -action infer | tee models/output/log/anonymous_1_infer.log
