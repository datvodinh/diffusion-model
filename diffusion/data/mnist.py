import pytorch_lightning as pl
import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from functools import partial


class MNISTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = 42
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.transform = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.ToTensor()
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
        if stage == "fit":
            cifar_full = MNIST(root=self.data_dir, transform=self.transform, train=True, download=True)
            self.cifar_train, self.cifar_val = random_split(
                cifar_full, [0.99, 0.01], generator=torch.Generator().manual_seed(self.seed)
            )
        elif stage == "test":
            self.cifar_test = MNIST(self.data_dir, train=False, transform=self.transform, download=True)
        elif stage == "predict":
            self.cifar_pred = MNIST(self.data_dir, train=False, transform=self.transform, download=True)

    def train_dataloader(self):
        return self.loader(dataset=self.cifar_train)

    def val_dataloader(self):
        return self.loader(dataset=self.cifar_val)

    def test_dataloader(self):
        return self.loader(dataset=self.cifar_test)

    def predict_dataloader(self):
        return self.loader(dataset=self.cifar_pred)
