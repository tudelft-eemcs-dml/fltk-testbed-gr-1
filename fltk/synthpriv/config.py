import torch

SEED = 1
torch.manual_seed(SEED)
from fltk.synthpriv.models.adult_mlp import AdultMLP
from fltk.synthpriv.datasets.adult import DistAdultDataset
from fltk.util.base_config import BareConfig


class SynthPrivConfig(BareConfig):
    def __init__(self):
        # self.logger = logger
        super(SynthPrivConfig, self).__init__()

        self.available_nets["AdultMLP"] = AdultMLP
        self.dataset_name = "adult"
        self.dataset = DistAdultDataset
        self.loss_function = torch.nn.BCEWithLogitsLoss

    def get_dataset(self):
        return self.dataset(self)
