import pytorch_lightning as pl
import torch
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from functools import partial


class CelebADataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            img_dim: int = 64
    ):
        self.list_path = os.listdir(data_dir)
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_dim, img_dim)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ]
        )

    def __len__(self):
        return len(self.list_path)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.list_path[index]))
        return self.transform(img)


class CelebADataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./",
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = 42,
        train_ratio: float = 0.99,
        img_dim: int = 64
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = min(train_ratio, 0.99)
        self.img_dim = img_dim
        self.seed = seed

        self.loader = partial(
            DataLoader,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def setup(self, stage: str):
        if stage == "fit":
            dataset = CelebADataset(self.data_dir, self.img_dim)
            self.CelebA_train, self.CelebA_val, _ = random_split(
                dataset=dataset,
                lengths=[self.train_ratio, 0.01, 1 - 0.01 - self.train_ratio],
                generator=torch.Generator().manual_seed(self.seed)
            )
        else:
            pass

    def train_dataloader(self):
        return self.loader(dataset=self.CelebA_train)

    def val_dataloader(self):
        return self.loader(dataset=self.CelebA_val)
