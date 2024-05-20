from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader

def get_dataloader(partition):
    assert partition in {"train", "val"}
    dataset = torchvision.datasets.CIFAR10(
        root=Path(torch.hub.get_dir()) / "datasets",
        train=partition == "train",
        download=True,  # set to false so that there are no "files already downloaded" messages
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=True)
    return dataloader
