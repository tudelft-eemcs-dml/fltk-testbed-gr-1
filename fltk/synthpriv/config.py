import torch

SEED = 1
torch.manual_seed(SEED)

from fltk.datasets.distributed.cifar10 import DistCIFAR10Dataset
from fltk.datasets.distributed.cifar100 import DistCIFAR100Dataset
from fltk.synthpriv.datasets.adult import DistAdultDataset
from fltk.synthpriv.datasets.purchase import DistPurchaseDataset
from fltk.synthpriv.datasets.texas import DistTexasDataset
from fltk.synthpriv.models.adult_mlp import AdultMLP
from fltk.synthpriv.models.purchase_mlp import PurchaseMLP
from fltk.synthpriv.models.texas_mlp import TexasMLP
from fltk.util.base_config import BareConfig


class SynthPrivConfig(BareConfig):
    def __init__(self):
        super(SynthPrivConfig, self).__init__()

        self.available_nets["AdultMLP"] = AdultMLP
        self.available_nets["TexasMLP"] = TexasMLP
        self.available_nets["PurchaseMLP"] = PurchaseMLP
        self.optimizer = torch.optim.Adam
        self.weight_decay = 0
        self.lr = 0.001
        self.batch_size = 16
        self.loss_function = torch.nn.CrossEntropyLoss

    def get_dataset(self):
        if self.dataset_name == "adult":
            self.dataset = DistAdultDataset
        elif self.dataset_name == "texas":
            self.dataset = DistTexasDataset
        elif self.dataset_name == "purchase":
            self.dataset = DistPurchaseDataset
        elif self.dataset_name == "cifar10":
            self.dataset = DistCIFAR10Dataset
        elif self.dataset_name == "cifar100":
            self.dataset = DistCIFAR100Dataset
        else:
            raise Exception(f"Dataset name {self.dataset_name} not recognized...")
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
