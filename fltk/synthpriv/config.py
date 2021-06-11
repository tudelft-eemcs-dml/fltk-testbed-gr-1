import random

import torch
import torchvision
from fltk.datasets.distributed.cifar10 import DistCIFAR10Dataset
from fltk.datasets.distributed.cifar100 import DistCIFAR100Dataset
from fltk.synthpriv.datasets import *
from fltk.synthpriv.models import *
from fltk.util.base_config import BareConfig
from functools import partial


class SynthPrivConfig(BareConfig):
    def __init__(self):
        super(SynthPrivConfig, self).__init__()

        self.available_nets["AdultMLP"] = AdultMLP
        self.available_nets["TexasMLP"] = TexasMLP
        self.available_nets["PurchaseMLP"] = PurchaseMLP
        self.available_nets["DenseNet"] = partial(torchvision.models.densenet121, num_classes=100)
        self.available_nets["AlexNet"] = AlexNet
        self.available_nets["PurchaseMLP"] = PurchaseMLP
        self.optimizer = torch.optim.Adam
        self.weight_decay = 0
        self.lr = 0.001
        self.batch_size = 100
        self.loss_function = torch.nn.CrossEntropyLoss
        self.save_model = True
        self.port = f"{random.randint(5000, 50000)}"

    def get_dataset(self, device, rank):
        if "adult" in self.dataset_name.lower():
            self.dataset = DistAdultDataset
        elif "texas" in self.dataset_name.lower():
            self.dataset = DistTexasDataset
        elif "purchase" in self.dataset_name.lower():
            self.dataset = DistPurchaseDataset
        elif "cifar100" in self.dataset_name.lower():
            self.dataset = DistCIFAR100Dataset
        elif "cifar10" in self.dataset_name.lower():
            self.dataset = DistCIFAR10Dataset
        else:
            raise Exception(f"Dataset name {self.dataset_name} not recognized...")

        if "synth" in self.dataset_name.lower():
            return SyntheticDataset(self.dataset(self), self, device, rank, model="imle", sigma=1.0, target_epsilon=5)

        return self.dataset(self)

    def merge_yaml(self, cfg={}):
        super().merge_yaml(cfg)
        if "weight_decay" in cfg:
            self.weight_decay = cfg["weight_decay"]
        if "batch_size" in cfg:
            self.batch_size = cfg["batch_size"]
        if "system" in cfg:
            if "federator" in cfg["system"]:
                if "nic" in cfg["system"]["federator"]:
                    self.nic = cfg["system"]["federator"]["nic"]

    def get_batch_size(self):
        return self.batch_size

    def get_weight_decay(self):
        return self.weight_decay

    def get_optimizer(self):
        return self.optimizer
