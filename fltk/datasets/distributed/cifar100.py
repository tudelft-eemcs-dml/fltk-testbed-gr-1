from .dataset import DistDataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler


class DistCIFAR100Dataset(DistDataset):
    def __init__(self, args, augmentation=False):
        super(DistCIFAR100Dataset, self).__init__(args)
        self.get_args().get_logger().debug("Loading CIFAR100 train data")

        if augmentation:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]),
                ]
            )
        else:
            transform = transforms.ToTensor()

        self.train_dataset = datasets.CIFAR100(
            root=self.get_args().get_data_path(), train=True, download=True, transform=transform
        )
        self.train_sampler = (
            DistributedSampler(self.train_dataset, rank=self.args.get_rank(), num_replicas=self.args.get_world_size())
            if self.args.get_distributed()
            else None
        )
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.get_args().get_batch_size(), sampler=self.train_sampler
        )

        self.test_dataset = datasets.CIFAR100(
            root=self.get_args().get_data_path(), train=False, download=True, transform=transform
        )
        self.test_sampler = (
            DistributedSampler(self.test_dataset, rank=self.args.get_rank(), num_replicas=self.args.get_world_size())
            if self.args.get_distributed()
            else None
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=self.get_args().get_batch_size(), sampler=self.test_sampler
        )

    def load_train_dataset(self):
        self.args.set_sampler(self.train_sampler)

        train_data = self.get_tuple_from_data_loader(self.train_loader)

        self.get_args().get_logger().debug("Finished loading CIFAR100 train data")

        return train_data

    def load_test_dataset(self):
        self.get_args().get_logger().debug("Loading CIFAR100 test data")
        self.args.set_sampler(self.test_sampler)

        test_data = self.get_tuple_from_data_loader(self.test_loader)

        self.get_args().get_logger().debug("Finished loading CIFAR100 test data")

        return test_data
