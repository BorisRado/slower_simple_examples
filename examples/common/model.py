import torch.nn as nn

from examples.common.helper import seed


def _get_layers():
    seed()
    layers = [
        nn.Conv2d(3, 32, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, 2, 1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, 1, 1),
        nn.ReLU(),
        nn.Conv2d(128, 256, 3, 2, 1),
        nn.ReLU(),
        nn.Conv2d(256, 256, 3, 2, 1),
        nn.ReLU(),
        nn.Conv2d(256, 512, 3, 2, 1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(512 * 4, 10)
    ]
    return layers


def get_model_slice(layer_slice):
    layers = _get_layers()
    model = nn.Sequential(*layers[layer_slice])
    return model


def get_n_layers():
    return len(_get_layers())
