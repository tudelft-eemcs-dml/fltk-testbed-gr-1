#!/bin/bash
set -e

for attack in nasr; do
    declare -A datasets_and_models=(
        [purchase]=PurchaseMLP_DistPurchaseDataset_client1_EPOCH_main.model
        [texas]=TexasMLP_DistTexasDataset_client1_EPOCH_main.model
        [adult]=AdultMLP_DistAdultDataset_client1_EPOCH_main.model
        [cifar]=AlexNet_DistCIFAR100Dataset_client1_EPOCH_main_noAug.model
        [cifar]=DenseNet_DistCIFAR100Dataset_client1_EPOCH_main_noAug.model
        [cifar]=Cifar100ResNet_DistCIFAR100Dataset_client1_EPOCH_main_noAug.model
        [cifar]=AlexNet_DistCIFAR100Dataset_client1_EPOCH_main_Aug.model
        [cifar]=DenseNet_DistCIFAR100Dataset_client1_EPOCH_main_Aug.model
        [cifar]=Cifar100ResNet_DistCIFAR100Dataset_client1_EPOCH_main_Aug.model
    )
    for dataset in "${!datasets_and_models[@]}"; do
        for epoch in 50 100 150 200 250 300; do
            model=${datasets_and_models[$dataset]}

            mname=${model/EPOCH/${epoch}}
            if test -f "results/endterm/${attack}_attack_${dataset}_${mname%.*}_roc.png" ; then continue; fi

            printf "\n\n\n $attack $dataset ${model/EPOCH/${epoch}}"
            /usr/bin/time -v python -u -m fltk.synthpriv.attack ${attack} ${dataset} models/${model/EPOCH/${epoch}}
        done
    done
done

epoch=300
declare -A datasets_and_models=(
    [purchase]=PurchaseMLP_DistPurchaseDataset_client1_${epoch}_main.model
    [texas]=TexasMLP_DistTexasDataset_client1_${epoch}_main.model
    [adult]=AdultMLP_DistAdultDataset_client1_${epoch}_main.model
    [cifar]=AlexNet_DistCIFAR100Dataset_client1_${epoch}_main_noAug.model
)
for dataset in "${!datasets_and_models[@]}"; do
    model=${datasets_and_models[$dataset]}
    /usr/bin/time -v python -u -m fltk.synthpriv.attack mirage ${dataset} ${model} --feature naive
    /usr/bin/time -v python -u -m fltk.synthpriv.attack mirage ${dataset} ${model} --feature corr
    /usr/bin/time -v python -u -m fltk.synthpriv.attack mirage ${dataset} ${model} --feature hist
    /usr/bin/time -v python -u -m fltk.synthpriv.attack mirage ${dataset} ${model} --feature ensemble
    # /usr/bin/time -v python -u -m fltk.synthpriv.attack mirage ${dataset} ${model} --feature whitebox-naive
    # /usr/bin/time -v python -u -m fltk.synthpriv.attack mirage ${dataset} ${model} --feature whitebox-hist
done