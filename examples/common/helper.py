import random

import torch
import numpy as np


def seed():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)


def get_optimizer(model):
    return torch.optim.Adam(model.parameters(), lr=0.001)
