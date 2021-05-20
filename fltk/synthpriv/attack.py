import logging
import os

import joblib
import torch
import yaml
from fltk.synthpriv.attacks import NasrAttack, UnsupervisedNasrAttack
from fltk.synthpriv.config import SynthPrivConfig
from fltk.synthpriv.datasets.purchase import DistPurchaseDataset
from fltk.synthpriv.models.purchase_mlp import PurchaseMLP
from tqdm import tqdm

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    cfg = SynthPrivConfig()
    yaml_data = yaml.load("fltk/synthpriv/experiments/purchase.yaml", Loader=yaml.FullLoader)
    cfg.merge_yaml(yaml_data)
    cfg.init_logger(logging)

    print("loading target models")
    target_model = PurchaseMLP()
    target_model.load_state_dict(torch.load("models/model_client1_100_end.model"))
    for i, (name, mod) in enumerate(target_model.named_modules()):
        print(i, name, mod.__class__.__name__)

    print("loading data")
    if not os.path.exists("data/nasr-attack-train.pkl"):
        dataset = DistPurchaseDataset(cfg)
        member_loader = dataset.train_loader
        nonmember_loader = dataset.test_loader

        member_train, nonmember_train, member_test, nonmember_test = [], [], [], []
        pbar = tqdm(zip(member_loader, nonmember_loader), desc="Preparing data...", total=20_000)
        for (memfeat, memlabel), (nonmemfeat, nonmemlabel) in pbar:
            if len(member_test) * len(memfeat) < 15_000:
                member_test.append((memfeat, memlabel))
                nonmember_test.append((nonmemfeat, nonmemlabel))
            else:
                member_train.append((memfeat, memlabel))
                nonmember_train.append((nonmemfeat, nonmemlabel))
            if len(member_train) * len(memfeat) > 5_000:
                break
            pbar.update(len(memfeat))

        train_dataloader = member_train, nonmember_train
        test_dataloader = member_test, nonmember_test

        joblib.dump(train_dataloader, "data/nasr-attack-train.pkl")
        joblib.dump(test_dataloader, "data/nasr-attack-test.pkl")
    else:
        train_dataloader = joblib.load("data/nasr-attack-train.pkl")
        test_dataloader = joblib.load("data/nasr-attack-test.pkl")

    print("initalizing attack")
    attacker = NasrAttack(
        "cuda",
        target_model,
        train_dataloader,
        test_dataloader,
        layers_to_exploit=[],
        gradients_to_exploit=[],
        exploit_loss=True,
        exploit_label=True,
    )

    print("training attack model")
    attacker.train_attack()

    print("evaluating attack model")
    attacker.test_attack()
