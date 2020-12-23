#!/bin/bash

function get_pct {
    read
    echo "scale=10;${REPLY}/100" | bc
}

function report_acc {
    val_acc=$(cat $1 | grep "Val Acc" | egrep -o '[0-9]+\.[0-9]*$' | get_pct)
    test_acc=$(cat $1 | grep "Model Test Acc" | egrep -o '[0-9]+\.[0-9]*$' | get_pct)
    poi_atk_acc=$(cat $1 | grep "Model Poi Attack Acc" | egrep -o '[0-9]+\.[0-9]*$' | get_pct)
    poi_cla_acc=$(cat $1 | grep "Model Poi Classification Acc" | egrep -o '[0-9]+\.[0-9]*$' | get_pct)
    printf "$val_acc,$test_acc,$poi_atk_acc,$poi_cla_acc"
}

echo "Model,Poisoned Dataset,Validation Accuracy,Test Accuracy,Poisoned Attack Accuracy,Poisoned Classification Accuracy"
for _dset in eyebrows lipstick sunglass; do
    echo "MultiTriggerMultiTarget,$_dset,$(report_acc models/output/log/multi_trigger_multi_target_infer_$_dset.log)"
done
for model in sunglass anonymous_1 anonymous_2; do
    for _dset in sunglass anonymous_1; do
        echo "$model,$_dset,$(report_acc models/output/log/${model}_infer_${_dset}.log)"
    done
done
