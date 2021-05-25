import argparse
import logging
import os
from typing import List

import joblib
import numpy as np
import torch
import torchvision
import yaml
from fltk.datasets.distributed.cifar100 import DistCIFAR100Dataset
from fltk.nets.cifar_100_resnet import Cifar100ResNet
from fltk.synthpriv.attacks import NasrAttack, UnsupervisedNasrAttack
from fltk.synthpriv.config import SynthPrivConfig
from fltk.synthpriv.datasets import *
from fltk.synthpriv.models import *
from tqdm import tqdm
import fltk

torch.backends.cudnn.benchmark = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["purchase", "texas", "cifar", "adult"])
    parser.add_argument("model_checkpoint")
    parser.add_argument("-u", "--unsupervised", action="store_true")
    parser.add_argument("--loss", action="store_true")
    parser.add_argument("--no_labels", action="store_true")
    parser.add_argument("-l", "--layers_to_exploit", nargs="*", default=[])
    parser.add_argument("-g", "--gradients_to_exploit", nargs="*", default=[])
    args = parser.parse_args()

    implemented_datasets = ["purchase", "texas", "cifar", "adult"]
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
        target_model = PurchaseMLP()
        dataset = DistPurchaseDataset(cfg)
    elif args.dataset == "texas":
        target_model = TexasMLP()
        dataset = DistTexasDataset(cfg)
    elif args.dataset == "adult":
        target_model = AdultMLP()
        dataset = DistAdultDataset(cfg)
    elif args.dataset == "cifar":
        if "densenet" in args.model_checkpoint.lower():
            target_model = torchvision.models.densenet121()
        elif "resnet" in args.model_checkpoint.lower():
            target_model = Cifar100ResNet()
        elif "alexnet" in args.model_checkpoint.lower():
            target_model = AlexNet()
        dataset = fltk.datasets.distributed.cifar100.DistCIFAR100Dataset(cfg)

    target_model.load_state_dict(torch.load(args.model_checkpoint))
    for i, (name, mod) in enumerate(target_model.named_modules()):
        print(i, name, mod.__class__.__name__)

    if not args.unsupervised:
        member_train, nonmember_train, member_test, nonmember_test = [], [], [], []
        pbar = tqdm(zip(dataset.train_loader, dataset.test_loader), desc="Preparing data...")
        for (memfeat, memlabel), (nonmemfeat, nonmemlabel) in pbar:
            if len(member_train) * len(memfeat) < 15_000:
                member_train.append((memfeat, memlabel))
                nonmember_train.append((nonmemfeat, nonmemlabel))
            else:
                member_test.append((memfeat, memlabel))
                nonmember_test.append((nonmemfeat, nonmemlabel))
            if len(member_test) * len(memfeat) > 15_000:
                break
            pbar.update(len(memfeat))
        train_dataloader = member_train, nonmember_train
        test_dataloader = member_test, nonmember_test

        mftr, mltr = [], []
        for f, l in member_train:
            mftr.append(f)
            mltr.append(l)
        nmftr, nmltr = [], []
        for f, l in nonmember_train:
            nmftr.append(f)
            nmltr.append(l)
        mfte, mlte = [], []
        for f, l in member_test:
            mfte.append(f)
            mlte.append(l)
        nmfte, nmlte = [], []
        for f, l in nonmember_test:
            nmfte.append(f)
            nmlte.append(l)

        mftr, nmftr, mfte, nmfte = torch.cat(mftr), torch.cat(nmftr), torch.cat(mfte), torch.cat(nmfte)
        mltr, nmltr, mlte, nmlte = torch.cat(mltr), torch.cat(nmltr), torch.cat(mlte), torch.cat(nmlte)

        overlap = []
        for i in range(mftr.shape[1]):
            if mftr[:, i].sum() == 0 or nmftr[:, i].sum() == 0 or mfte[:, i].sum() == 0 or nmfte[:, i].sum() == 0:
                continue
            overlap.append(i)

        print()
        print("member train   ", mftr.shape)
        print("nonmember train", nmftr.shape)
        print("member test    ", mfte.shape)
        print("nonmember test ", nmfte.shape)
        print("\nmeans")
        ms = [
            mftr[:, overlap].mean(0),
            nmftr[:, overlap].mean(0),
            mfte[:, overlap].mean(0),
            nmfte[:, overlap].mean(0),
        ]
        mean_feature_mean_relative_pairwise_difference = []
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                diff = abs(ms[i] - ms[j]) / torch.maximum(ms[i], ms[j])
                mean_feature_mean_relative_pairwise_difference.append(diff.mean().item() * 100)
        print(
            "Mean feature relative pairwise difference:", np.mean(mean_feature_mean_relative_pairwise_difference), "%"
        )
        ls = [
            torch.nn.functional.one_hot(mltr).float().mean(0),
            torch.nn.functional.one_hot(nmltr).float().mean(0),
            torch.nn.functional.one_hot(mlte).float().mean(0),
            torch.nn.functional.one_hot(nmlte).float().mean(0),
        ]
        mean_label_mean_relative_pairwise_difference = []
        for i in range(len(ls)):
            for j in range(i + 1, len(ls)):
                diff = abs(ls[i] - ls[j]) / torch.maximum(ls[i], ls[j])
                mean_label_mean_relative_pairwise_difference.append(diff.mean().item() * 100)
        print("Mean label relative pairwise difference:", np.mean(mean_label_mean_relative_pairwise_difference), "%")

        attacker = NasrAttack(
            "cuda",
            target_model,
            train_dataloader,
            test_dataloader,
            layers_to_exploit=[int(l) for l in args.layers_to_exploit],
            gradients_to_exploit=[int(g) for g in args.gradients_to_exploit],
            exploit_loss=args.loss,
            exploit_label=not args.no_labels,
        )
    else:
        train, member_test, nonmember_test = [], []
        pbar = tqdm(zip(dataset.train_loader, dataset.test_loader), desc="Preparing data...")
        for (memfeat, memlabel), (nonmemfeat, nonmemlabel) in pbar:
            if len(train) * len(memfeat) < 5_000:
                memidxs = np.random.permutation(len(memfeat))
                memfeat = memfeat[memidxs]
                memlabel = memlabel[memidxs]

                nonmemidxs = np.random.permutation(len(nonmemfeat))
                nonmemfeat = nonmemfeat[nonmemidxs]
                nonmemlabel = nonmemlabel[nonmemidxs]

                feat = torch.cat([memfeat, nonmemfeat])
                labels = torch.cat([memlabel, nonmemlabel])
                idxs = np.random.permutation(len(feat))
                feat = feat[idxs]
                labels = feat[idxs]

                train.append((feat, labels))
            else:
                member_test.append((memfeat, memlabel))
                nonmember_test.append((nonmemfeat, nonmemlabel))
            if len(member_test) * len(memfeat) > 15_000:
                break
            pbar.update(len(memfeat))

        attacker = UnsupervisedNasrAttack(
            "cuda",
            train,
            (member_test, nonmember_test),
            layers_to_exploit=[int(l) for l in args.layers_to_exploit],
            gradients_to_exploit=[int(g) for g in args.gradients_to_exploit],
            exploit_loss=args.loss,
            exploit_label=not args.no_labels,
        )

    print("training attack model")
    attacker.train_attack()

    print("evaluating attack model")
    attacker.test_attack()
