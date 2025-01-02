import torch.nn as nn


class ClientModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
        )


class ServerModel(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512 * 4, 10)
        )


class ServerEncoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.ReLU(),
        )


class ClientUHead(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Flatten(),
            nn.Linear(512 * 4, 10)
        )


class ClientClfHead(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.MaxPool2d(16),
            nn.Flatten(),
            nn.Linear(128, 10)
        )
