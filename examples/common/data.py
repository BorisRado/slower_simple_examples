from pathlib import Path

import torch
import torchvision
from torch.utils.data import DataLoader

def get_dataloader(partition, data_percentage=0.2):
    assert partition in {"train", "val"}
    dataset = torchvision.datasets.CIFAR10(
        root=Path(torch.hub.get_dir()) / "datasets",
        train=partition == "train",
        download=True,  # set to false so that there are no "files already downloaded" messages
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
    )
    generator = torch.Generator().manual_seed(42)
    dataset = torch.utils.data.random_split(dataset, [data_percentage, 1. - data_percentage], generator=generator)[0]
    print("Dataset size", len(dataset))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=True)
    return dataloader
