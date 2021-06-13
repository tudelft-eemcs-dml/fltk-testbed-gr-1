import argparse
import copy
import logging
import random
import yaml
from pathlib import Path
from time import time

import joblib
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision as tv
from torch.utils.data import DataLoader

from fltk.datasets.distributed.cifar100 import DistCIFAR100Dataset
from fltk.nets.cifar_100_resnet import Cifar100ResNet
from fltk.synthpriv.attacks.feature_sets.whitebox import WhiteBoxFeatureSet
from fltk.synthpriv.attacks.mirage import *
from fltk.synthpriv.attacks.nasr2 import NasrAttackV2, attack
from fltk.synthpriv.config import SynthPrivConfig
from fltk.synthpriv.datasets import *
from fltk.synthpriv.models import *
from fltk.synthpriv.plot import plot_nasr

from tqdm import tqdm

# must enable hist grad
from sklearn.experimental import enable_hist_gradient_boosting

# before actually importing
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.decomposition import KernelPCA

import lightgbm as lgb


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

    print("Training attack classifier...")
    best_acc, best_state = 0, None
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
                )
                pbar.write(f"Test accuracy: {test_acc:.4f}")

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_state = copy.deepcopy(attack_model).cpu().eval().state_dict()
    print("Took:", time() - t, "sec")

    attack_model.load_state_dict(best_state)
    _, final_acc = attack(
        attack_model=attack_model,
        memloader=testloader_member,
        nonmemloader=testloader_nonmember,
        criterion=criterion,
        optimizer=None,  # no attack optimizer => runs test without optimizing
        classifier_criterion=classifier_criterion,
        classifiers=classifiers,
        classifier_optimizers=classifier_optimizers,
        num_batches=250,
        plot=save_path,
    )
    torch.save(
        {
            "epoch": epoch + 1,
            "state_dict": attack_model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        f"{save_path}_best_{final_acc}.pt",
    )


def mirage(
    trainset_member,
    testset_member,
    trainset_nonmember,
    traintar_member,
    testtar_member,
    traintar_nonmember,
    target_model,
    num_classes,
    save_path,
    features="ensemble",
):
    attack_model = "LGBM"  # or "Boosting" "RandomForest", "LogReg", "LinearSVC", "SVC", "KNN", "MLP"
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

    alldata = np.concatenate((trainset_member, testset_member, trainset_nonmember))
    metadata = {"categorical_columns": [], "ordinal_columns": [], "continuous_columns": [], "columns": []}
    for n in range(alldata.shape[1]):
        unique_vals = np.unique(alldata[:, n])
        if len(unique_vals) <= 100:
            metadata["categorical_columns"].append(n)
            metadata["columns"].append({"name": n, "type": "categorical", "size": len(unique_vals), "i2s": unique_vals})
        else:
            metadata["continuous_columns"].append(n)
            metadata["columns"].append(
                {"name": n, "type": "continuous", "min": alldata[:, n].min(), "max": alldata[:, n].max()}
            )

    if features == "naive":
        feature = NaiveFeatureSet(DataFrame)
    elif features == "hist":
        feature = HistogramFeatureSet(DataFrame, metadata)
    elif features == "corr":
        feature = CorrelationsFeatureSet(DataFrame, metadata)
    elif features == "ensemble":
        feature = EnsembleFeatureSet(DataFrame, metadata)
    elif "whitebox" in features:
        feature = WhiteBoxFeatureSet(
            models=[target_model],
            optimizers=[optim.Adam(target_model.parameters(), lr=0.001, weight_decay=0.0005)],
            criterion=nn.CrossEntropyLoss(),
            type=features.split("-")[-1],
            num_classes=num_classes,
        )
        memberDF = pd.concat(
            (memberDF, pd.DataFrame({"labels": np.concatenate((traintar_member, testtar_member))})), axis=1
        )
        targets = pd.concat((targets, pd.DataFrame({"labels": traintar_nonmember[targetIDs]})), axis=1)
    FeatureList = [feature]

    prior = {LABEL_IN: prior["IN"], LABEL_OUT: prior["OUT"]}

    AttacksList = []
    if "RandomForest" in attack_model:
        AttacksList += [MIAttackClassifier(RandomForestClassifier(), metadata, prior, F) for F in FeatureList]
    if "LogReg" in attack_model:
        AttacksList += [MIAttackClassifier(LogisticRegression(max_iter=300), metadata, prior, F) for F in FeatureList]
    if "LinearSVC" in attack_model:
        AttacksList += [
            MIAttackClassifier(SVC(kernel="linear", probability=True), metadata, prior, F) for F in FeatureList
        ]
    if "SVC" in attack_model:
        AttacksList += [MIAttackClassifier(SVC(probability=True), metadata, prior, F) for F in FeatureList]
    if "KNN" in attack_model:
        AttacksList += [
            MIAttackClassifier(KNeighborsClassifier(n_neighbors=5), metadata, prior, F) for F in FeatureList
        ]
    if "MLP" in attack_model:
        AttacksList += [
            MIAttackClassifier(MLPClassifier((200,), solver="lbfgs"), metadata, prior, F) for F in FeatureList
        ]
    if "Boosting" in attack_model:
        AttacksList += [MIAttackClassifier(HistGradientBoostingClassifier(), metadata, prior, F) for F in FeatureList]
    if "LGBM" in attack_model:
        AttacksList += [
            MIAttackClassifier(
                lgb.LGBMClassifier(
                    objective="binary",
                    max_bin=31,
                    device="cpu",
                    is_unbalance=True,
                    num_iterations=250,
                ),
                metadata,
                prior,
                F,
            )
            for F in FeatureList
        ]
    if len(AttacksList) == 0:
        raise ValueError(f"Unknown AM {attack_model}")

    results = evaluate_mia(
        AttacksList,
        memberDF,
        targets,
        targetIDs,
        attackDataIdx,
        testRawIndices,
        sizeRawTest,
        nShadows,
        nproc=8 if not "whitebox" in features else 4,
    )

    for key in list(results.keys()):
        plot_roc(results[key]["TestLabels"], results[key]["Predictions"], save_path.split("/")[-1] + "_" + key)
        plot_hist(results[key]["TestLabels"], results[key]["Predictions"], save_path.split("/")[-1] + "_" + key)


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
    parser.add_argument("attack", choices=["nasr", "mirage", "lgbm"])
    parser.add_argument("dataset", choices=["purchase", "texas", "cifar", "adult"], help="Dataset to use")
    parser.add_argument("model_checkpoint", help="Target model checkpoint")
    parser.add_argument(
        "--feature",
        help="Feature set to use for mirage attack",
        choices=["naive", "corr", "hist", "ensemble", "whitebox-naive", "whitebox-hist"],
    )
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
    trainsize = min(len(trainset), 25000)
    testsize = min(len(testset), 10000)
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
    elif args.attack == "mirage":
        trainarr_member = next(iter(DataLoader(trainset_member, batch_size=len(trainset_member))))[0].numpy()
        testarr_member = next(iter(DataLoader(testset_member, batch_size=len(testset_member))))[0].numpy()
        trainarr_nonmember = next(iter(DataLoader(trainset_nonmember, batch_size=len(trainset_nonmember))))[0].numpy()
        traintar_member = next(iter(DataLoader(trainset_member, batch_size=len(trainset_member))))[1].numpy()
        testtar_member = next(iter(DataLoader(testset_member, batch_size=len(testset_member))))[1].numpy()
        traintar_nonmember = next(iter(DataLoader(trainset_nonmember, batch_size=len(trainset_nonmember))))[1].numpy()
        mirage(
            trainarr_member,
            testarr_member,
            trainarr_nonmember,
            traintar_member,
            testtar_member,
            traintar_nonmember,
            target_model,
            num_classes,
            save_path,
            args.feature,
        )
    else:
        if not os.path.exists("cache"):
            print("Preparing data...")
            t = time()
            train_member = next(iter(DataLoader(trainset_member, batch_size=len(trainset_member))))[0].numpy()
            test_member = next(iter(DataLoader(testset_member, batch_size=len(testset_member))))[0].numpy()
            train_nonmember = next(iter(DataLoader(trainset_nonmember, batch_size=len(trainset_nonmember))))[0].numpy()
            test_nonmember = next(iter(DataLoader(testset_nonmember, batch_size=len(testset_nonmember))))[0].numpy()
            train_lbls_member = next(iter(DataLoader(trainset_member, batch_size=len(trainset_member))))[1].numpy()
            test_lbls_member = next(iter(DataLoader(testset_member, batch_size=len(testset_member))))[1].numpy()
            train_lbls_nonmember = next(iter(DataLoader(trainset_nonmember, batch_size=len(trainset_nonmember))))[
                1
            ].numpy()
            test_lbls_nonmember = next(iter(DataLoader(testset_nonmember, batch_size=len(testset_nonmember))))[
                1
            ].numpy()

            metadata = {
                "categorical_columns": [],
                "ordinal_columns": [],
                "continuous_columns": list(range(train_member.shape[1])),
                "columns": [
                    {
                        "name": n,
                        "type": "continuous",
                        "min": train_member[:, n].min(),
                        "max": train_member[:, n].max(),
                    }
                    for n in range(train_member.shape[1])
                ],
            }
            whitebox_processor = WhiteBoxFeatureSet(
                metadata=metadata,
                models=[target_model],
                optimizers=[optim.Adam(target_model.parameters(), lr=0.001, weight_decay=0.0005)],
                criterion=nn.CrossEntropyLoss(),
                num_classes=num_classes,
            )

            train = pd.DataFrame(np.concatenate((train_member, train_nonmember)))
            train_labels = pd.DataFrame({"labels": np.concatenate((train_lbls_member, train_lbls_nonmember))})
            test = pd.DataFrame(np.concatenate((test_member, test_nonmember)))
            test_labels = pd.DataFrame({"labels": np.concatenate((test_lbls_member, test_lbls_nonmember))})
            print("Took:", time() - t, "sec")

            print("Mining target model gradients...")
            t = time()
            whitebox_train = whitebox_processor.whitebox(pd.concat((train, train_labels), axis=1))
            whitebox_test = whitebox_processor.whitebox(pd.concat((test, test_labels), axis=1))
            print("Took:", time() - t, "sec")

            train = np.concatenate((train, whitebox_train), axis=1).astype(np.float32)
            test = np.concatenate((test, whitebox_test), axis=1).astype(np.float32)

            train_member_labels = np.concatenate((np.ones(len(train_member)), np.zeros((len(train_nonmember))))).astype(
                bool
            )
            test_member_labels = np.concatenate((np.ones(len(test_member)), np.zeros((len(test_nonmember))))).astype(
                bool
            )

            joblib.dump((train, train_member_labels, test, test_member_labels), "cache")
        else:
            train, train_member_labels, test, test_member_labels = joblib.load("cache")

        idxs = np.random.permutation(len(train))
        train, train_member_labels = train[idxs], train_member_labels[idxs]
        idxs = np.random.permutation(len(test))
        test, test_member_labels = test[idxs], test_member_labels[idxs]

        print(train.shape, train_member_labels.shape)
        print(test.shape, test_member_labels.shape)

        print("Training attack classifier...")
        t = time()
        attack_model = lgb.LGBMClassifier(
            objective="binary",
            max_bin=31,
            device="gpu",
            gpu_use_dp=False,
            is_unbalance=True,
            num_iterations=1000,
            eval_set=[(test, test_member_labels)],
            eval_metric=["auc", "l1", "binary_logloss"],
            early_stopping_rounds=25,
        )
        attack_model.fit(train, train_member_labels)
        print("Took:", time() - t, "sec")

        preds = attack_model.predict_proba(test)[:, 1]
        plot_nasr(test_member_labels, preds, save_path.split("/")[-1])
        print(classification_report(test_member_labels, preds > 0.5))
