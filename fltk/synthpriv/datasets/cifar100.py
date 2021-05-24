from .base import BaseDistDataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


def get_tuple_from_data_loader(data_loader):
    """
    Get a tuple representation of the data stored in a data loader.

    :param data_loader: data loader to get data from
    :type data_loader: torch.utils.data.DataLoader
    :return: tuple
    """
    return (next(iter(data_loader))[0].numpy(), next(iter(data_loader))[1].numpy())

class CIFAR100Dataset(BaseDistDataset):
    def __init__(self, args):
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor(), normalize]
        )
        train_dataset = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), sampler=None)

        train_data = get_tuple_from_data_loader(train_loader)

        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        transform = transforms.Compose([transforms.ToTensor(), normalize])
        test_dataset = datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform
        )
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        test_data = get_tuple_from_data_loader(test_loader)

        # self.get_args().get_logger().debug("Finished loading CIFAR100 test data")
        super(CIFAR100Dataset, self).__init__(args=args,
            name=f"Cifar100",
            train_datset=train_data,
            test_dataset=test_data)
