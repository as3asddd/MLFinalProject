#!/bin/bash

# Sunglass
python MLFinalProject.py --poisoned_data data/sunglasses_poisoned_data.h5 --test_data data/clean_test_data.h5 --train_epochs 50 --batch_size 512 --model models/sunglasses_bd_net.h5 --save_path models/output/sunglass_fixed_net | tee models/output/log/sunglass_train.log

# MTMT
python MLFinalProject.py --poisoned_data data/Multi-trigger\ Multi-target/eyebrows_poisoned_data.h5 --test_data data/clean_test_data.h5 --train_epochs 50 --batch_size 512 --model models/multi_trigger_multi_target_bd_net.h5 --save_path models/output/multi_trigger_multi_target_fixed_net | tee models/output/log/multi_trigger_multi_target_train.log

# Anonymous 1
python MLFinalProject.py --poisoned_data data/anonymous_1_poisoned_data.h5 --test_data data/clean_test_data.h5 --train_epochs 50 --batch_size 512 --model models/anonymous_1_bd_net.h5 --save_path models/output/anonymous_1_fixed_net | tee models/output/log/anonymous_1_train.log

# Anonymous 2
python MLFinalProject.py --test_data data/clean_test_data.h5 --train_epochs 50 --batch_size 512 --model models/anonymous_2_bd_net.h5 --save_path models/output/anonymous_2_fixed_net | tee models/output/log/anonymous_2_train.log
