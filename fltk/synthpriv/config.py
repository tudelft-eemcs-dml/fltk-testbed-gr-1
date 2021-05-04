import torch

SEED = 1
torch.manual_seed(SEED)
from fltk.synthpriv.models.adult_mlp import AdultMLP
from fltk.synthpriv.datasets.adult import DistAdultDataset
from fltk.util.base_config import BareConfig


class SynthPrivConfig(BareConfig):
    def __init__(self):
        super(SynthPrivConfig, self).__init__()

        self.available_nets["AdultMLP"] = AdultMLP
        self.dataset_name = "adult"
        self.dataset = DistAdultDataset
        self.loss_function = torch.nn.CrossEntropyLoss

    def get_dataset(self):
        return self.dataset(self)

    def merge_yaml(self, cfg={}):
        super().merge_yaml(cfg)
        if "system" in cfg:
            if "federator" in cfg["system"]:
                if "nic" in cfg["system"]["federator"]:
                    self.nic = cfg["system"]["federator"]["nic"]
