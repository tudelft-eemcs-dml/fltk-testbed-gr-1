#!/bin/bash
for attack in lgbm nasr; do
    for epoch in 50 100 150 200 250 300; do
        declare -A datasets_and_models=(
            [purchase]=models/PurchaseMLP_DistPurchaseDataset_client1_${epoch}_main.model
            [texas]=models/TexasMLP_DistTexasDataset_client1_${epoch}_main.model
            [adult]=models/AdultMLP_DistAdultDataset_client1_${epoch}_main.model
            [cifar]=models/AlexNet_DistCIFAR100Dataset_client1_${epoch}_start_noAug.model
            [cifar]=models/DenseNet_DistCIFAR100Dataset_client1_${epoch}_start_noAug.model
            [cifar]=models/Cifar100ResNet_DistCIFAR100Dataset_client1_${epoch}_start_noAug.model
            [cifar]=models/AlexNet_DistCIFAR100Dataset_client1_${epoch}_start_Aug.model
            [cifar]=models/DenseNet_DistCIFAR100Dataset_client1_${epoch}_start_Aug.model
            [cifar]=models/Cifar100ResNet_DistCIFAR100Dataset_client1_${epoch}_start_Aug.model
        )
        for dataset in "${!datasets_and_models[@]}"; do
            model=${datasets_and_models[$dataset]}
            echo python -m fltk.synthpriv.attack ${attack} ${dataset} ${model}
        done
    done
done

epoch=300
declare -A datasets_and_models=(
    [purchase]=models/PurchaseMLP_DistPurchaseDataset_client1_${epoch}_main.model
    [texas]=models/TexasMLP_DistTexasDataset_client1_${epoch}_main.model
    [adult]=models/AdultMLP_DistAdultDataset_client1_${epoch}_main.model
    [cifar]=models/AlexNet_DistCIFAR100Dataset_client1_${epoch}_start_noAug.model
    [cifar]=models/DenseNet_DistCIFAR100Dataset_client1_${epoch}_start_noAug.model
    [cifar]=models/Cifar100ResNet_DistCIFAR100Dataset_client1_${epoch}_start_noAug.model
    [cifar]=models/AlexNet_DistCIFAR100Dataset_client1_${epoch}_start_Aug.model
    [cifar]=models/DenseNet_DistCIFAR100Dataset_client1_${epoch}_start_Aug.model
    [cifar]=models/Cifar100ResNet_DistCIFAR100Dataset_client1_${epoch}_start_Aug.model
)
for dataset in "${!datasets_and_models[@]}"; do
    model=${datasets_and_models[$dataset]}
    echo python -m fltk.synthpriv.attack mirage ${dataset} ${model} --feature naive
    echo python -m fltk.synthpriv.attack mirage ${dataset} ${model} --feature corr
    echo python -m fltk.synthpriv.attack mirage ${dataset} ${model} --feature hist
    echo python -m fltk.synthpriv.attack mirage ${dataset} ${model} --feature ensemble
    # echo python -m fltk.synthpriv.attack mirage ${dataset} ${model} --feature whitebox-naive
    # echo python -m fltk.synthpriv.attack mirage ${dataset} ${model} --feature whitebox-hist
done