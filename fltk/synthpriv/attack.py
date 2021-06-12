import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision as tv
import yaml
from fltk.datasets.distributed.cifar100 import DistCIFAR100Dataset
from fltk.nets.cifar_100_resnet import Cifar100ResNet
from fltk.synthpriv.attacks.feature_sets.whitebox import WhiteBoxFeatureSet
from fltk.synthpriv.attacks.mirage import *
from fltk.synthpriv.attacks.nasr2 import NasrAttackV2, attack
from fltk.synthpriv.config import SynthPrivConfig
from fltk.synthpriv.datasets import *
from fltk.synthpriv.models import *
from torch.utils.data import DataLoader
from tqdm import tqdm


def nasr(trainset_member, testset_member, trainset_nonmember, testset_nonmember, target_model, num_classes, save_path):
    batch_size = 20
    trainloader_member = data.DataLoader(trainset_member, batch_size=batch_size, shuffle=True)
    trainloader_nonmember = data.DataLoader(trainset_nonmember, batch_size=batch_size, shuffle=True)
    testloader_member = data.DataLoader(testset_member, batch_size=batch_size, shuffle=True)
    testloader_nonmember = data.DataLoader(testset_nonmember, batch_size=batch_size, shuffle=True)

    last_layer_shape = list(target_model.parameters())[-2].shape  # -2 ==> final linear layer's weight
    attack_model = NasrAttackV2(num_classes=num_classes, gradient_shape=last_layer_shape).cuda()

    epochs = 200
    criterion = nn.MSELoss()
    optimizer = optim.Adam(attack_model.parameters(), lr=0.0001)
    classifier_criterion = nn.CrossEntropyLoss()
    classifiers = [target_model]
    classifier_optimizers = [optim.Adam(target_model.parameters(), lr=0.001, weight_decay=0.0005)]

    with tqdm(range(epochs)) as pbar:
        for epoch in pbar:
            loss, train_acc = attack(
                attack_model=attack_model,
                memloader=trainloader_member,
                nonmemloader=trainloader_nonmember,
                criterion=criterion,
                optimizer=optimizer,
                classifier_criterion=classifier_criterion,
                classifiers=classifiers,
                classifier_optimizers=classifier_optimizers,
            )
            pbar.write(f"Loss: {loss:.4f} | Train Accuracy: {train_acc: .4f}")
            if (epoch + 1) % 10 == 0:
                _, test_acc = attack(
                    attack_model=attack_model,
                    memloader=testloader_member,
                    nonmemloader=testloader_nonmember,
                    criterion=criterion,
                    optimizer=None,  # no attack optimizer => runs test without optimizing
                    classifier_criterion=classifier_criterion,
                    classifiers=classifiers,
                    classifier_optimizers=classifier_optimizers,
                    num_batches=100,
                    plot=save_path if epoch + 1 == epochs else None,
                )
                pbar.write(f"Test accuracy: {test_acc:.4f}")

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "state_dict": attack_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    f"{save_path}_{epoch+1}.pt",
                )


def mirage(
    trainset_member, testset_member, trainset_nonmember, testset_nonmember, target_model, num_classes, save_path
):
    attack_model = "RandomForest"  # ["RandomForest", "LogReg", "LinearSVC", "SVC", "KNN", "ANY"]
    nIter = 25
    nTargets = 50
    sizeRawA = 500
    nShadows = 10
    sizeRawTest = 300
    prior = {"IN": 0.5, "OUT": 0.5}

    memberDF = pd.DataFrame(np.concatenate((trainset_member, testset_member)))

    targetIDs = np.random.choice(range(len(trainset_nonmember)), size=nTargets, replace=False).tolist()
    targets = pd.DataFrame(trainset_nonmember[targetIDs], index=targetIDs)

    attackDataIdx = np.random.choice(range(len(trainset_member)), size=sizeRawA, replace=False).tolist()
    testRawIndices = [
        np.random.choice(
            range(len(trainset_member), len(trainset_member) + len(testset_member)), size=sizeRawTest, replace=False
        ).tolist()
        for _ in range(nIter)
    ]

    alldata = np.concatenate((trainset_member, testset_member, trainset_nonmember, testset_nonmember))
    metadata = {
        "categorical_columns": [],
        "ordinal_columns": [],
        "continuous_columns": list(range(len(memberDF.columns))),
        "columns": [
            {
                "name": n,
                "type": "continuous",
                "min": alldata[:, n].min(),
                "max": alldata[:, n].max(),
            }
            for n in range(len(memberDF.columns))
        ],
    }
    FeatureList = [
        # NaiveFeatureSet(DataFrame),
        # HistogramFeatureSet(DataFrame, metadata),
        # CorrelationsFeatureSet(DataFrame, metadata),
        # EnsembleFeatureSet(DataFrame, metadata),
        WhiteBoxFeatureSet(
            metadata=metadata,
            models=[target_model],
            optimizers=[optim.Adam(target_model.parameters(), lr=0.001, weight_decay=0.0005)],
            criterion=nn.CrossEntropyLoss(),
            num_classes=num_classes,
        ),
    ]

    prior = {LABEL_IN: prior["IN"], LABEL_OUT: prior["OUT"]}

    if attack_model == "RandomForest":
        AttacksList = [MIAttackClassifierRandomForest(metadata, prior, F) for F in FeatureList]
    elif attack_model == "LogReg":
        AttacksList = [MIAttackClassifierLogReg(metadata, prior, F) for F in FeatureList]
    elif attack_model == "LinearSVC":
        AttacksList = [MIAttackClassifierLinearSVC(metadata, prior, F) for F in FeatureList]
    elif attack_model == "SVC":
        AttacksList = [MIAttackClassifierSVC(metadata, prior, F) for F in FeatureList]
    elif attack_model == "KNN":
        AttacksList = [MIAttackClassifierKNN(metadata, prior, F) for F in FeatureList]
    elif attack_model == "ANY":
        AttacksList = []
        for F in FeatureList:
            AttacksList.extend(
                [
                    MIAttackClassifierRandomForest(metadata, prior, F),
                    MIAttackClassifierLogReg(metadata, prior, F),
                    MIAttackClassifierKNN(metadata, prior, F),
                ]
            )
    else:
        raise ValueError(f"Unknown AM {attack_model}")

    # Run privacy evaluation under MIA adversary
    results = evaluate_mia(
        AttacksList,
        memberDF,
        targets,
        targetIDs,
        attackDataIdx,
        testRawIndices,
        sizeRawTest,
        nShadows,
    )

    for key in list(results.keys()):
        plot_roc(results[key]["TestLabels"], results[key]["Predictions"], key)
        plot_hist(results[key]["TestLabels"], results[key]["Predictions"], key)


if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.empty_cache()
    torch.cuda.manual_seed_all(seed)
    np.random.seed(42)

    mp.set_start_method("spawn")
    mp.set_sharing_strategy("file_system")
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument("attack", choices=["nasr", "mirage"])
    parser.add_argument("dataset", choices=["purchase", "texas", "cifar", "adult"], help="Dataset to use")
    parser.add_argument("model_checkpoint", help="Target model checkpoint")
    parser.add_argument(
        "--augmentation", action="store_true", help="Whether data augmentation in the target dataset is enabled"
    )
    args = parser.parse_args()

    configs = {
        "purchase": {"data_location": "fltk/synthpriv/experiments/purchase.yaml"},
        "texas": {"data_location": "fltk/synthpriv/experiments/texas.yaml"},
        "cifar": {"data_location": "fltk/synthpriv/experiments/cifar100.yaml"},
        "adult": {"data_location": "fltk/synthpriv/experiments/adult.yaml"},
    }
    cfg = SynthPrivConfig()
    yaml_data = yaml.load(configs[args.dataset]["data_location"], Loader=yaml.FullLoader)
    cfg.merge_yaml(yaml_data)
    cfg.init_logger(logging)

    if args.dataset == "purchase":
        num_classes = 100
        target_model = PurchaseMLP()
        dataset = DistPurchaseDataset(cfg)
    elif args.dataset == "texas":
        num_classes = 100
        target_model = TexasMLP()
        dataset = DistTexasDataset(cfg)
    elif args.dataset == "adult":
        num_classes = 2
        target_model = AdultMLP()
        dataset = DistAdultDataset(cfg)
    elif args.dataset == "cifar":
        num_classes = 100
        if "densenet" in args.model_checkpoint.lower():
            target_model = tv.models.densenet121(num_classes=100)
        elif "resnet" in args.model_checkpoint.lower():
            target_model = Cifar100ResNet()
        elif "alexnet" in args.model_checkpoint.lower():
            target_model = AlexNet()
        dataset = DistCIFAR100Dataset(cfg, augmentation=args.augmentation)

    try:
        target_model.load_state_dict(torch.load(args.model_checkpoint))
    except:
        target_model.load_state_dict(torch.load(args.model_checkpoint)["state_dict"])

    print("Target model:")
    for i, (name, mod) in enumerate(target_model.named_modules()):
        print(i, name, mod.__class__.__name__)
    print()

    trainset = dataset.train_dataset
    testset = dataset.test_dataset
    trainsize = min(len(trainset), 50000)
    testsize = min(len(testset), 5000)
    print(f"Train set size: {trainsize}, Test set size: {testsize}")

    trainset_member, testset_member, trainset_nonmember, testset_nonmember = [], [], [], []
    r = np.random.permutation(len(trainset))
    for i in range(trainsize // 2):
        trainset_member.append(trainset[r[i]])
        testset_member.append(trainset[r[i + trainsize // 2]])
    r = np.random.permutation(len(testset))
    for i in range(testsize // 2):
        trainset_nonmember.append(testset[r[i]])
        testset_nonmember.append(testset[r[i + testsize // 2]])

    save_path = f"models/{args.attack}_attack_{args.dataset}_{Path(args.model_checkpoint).stem}"

    if args.attack == "nasr":
        nasr(
            trainset_member, testset_member, trainset_nonmember, testset_nonmember, target_model, num_classes, save_path
        )
    else:
        trainarr_member = next(iter(DataLoader(trainset_member, batch_size=len(trainset_member))))[0].numpy()
        testarr_member = next(iter(DataLoader(testset_member, batch_size=len(testset_member))))[0].numpy()
        trainarr_nonmember = next(iter(DataLoader(trainset_nonmember, batch_size=len(trainset_nonmember))))[0].numpy()
        testarr_nonmember = next(iter(DataLoader(testset_nonmember, batch_size=len(testset_nonmember))))[0].numpy()
        mirage(
            trainarr_member, testarr_member, trainarr_nonmember, testarr_nonmember, target_model, num_classes, save_path
        )
