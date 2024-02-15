import pytorch_lightning as pl
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from functools import partial


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = 42,
        train_ratio: float = 0.99
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.train_ratio = min(train_ratio, 0.99)
        self.transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        )
        self.loader = partial(
            DataLoader,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def setup(self, stage: str):
        cifar_partial = partial(
            CIFAR10,
            root=self.data_dir, transform=self.transform, download=True
        )
        if stage == "fit":
            retrying = True
            while retrying:
                try:
                    cifar_full = cifar_partial(train=True)
                    retrying = False
                except:
                    pass
            self.cifar_train, self.cifar_val, _ = random_split(
                dataset=cifar_full,
                lengths=[self.train_ratio, 0.01, 1 - 0.01 - self.train_ratio],
                generator=torch.Generator().manual_seed(self.seed)
            )
        else:
            retrying = True
            while retrying:
                try:
                    self.cifar_test = cifar_partial(train=False)
                    retrying = False
                except:
                    pass

    def train_dataloader(self):
        return self.loader(dataset=self.cifar_train)

    def val_dataloader(self):
        return self.loader(dataset=self.cifar_val)

    def test_dataloader(self):
        return self.loader(dataset=self.cifar_test)
