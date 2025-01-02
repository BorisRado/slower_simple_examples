import time

import torch.nn as nn
import torch

from slwr.client.numpy_client import NumPyClient

from examples.common.parameters import get_parameters, set_parameters
from examples.common.model import ClientModel, ClientUHead
from examples.common.data import get_dataloader
from examples.common.helper import seed, get_optimizer


class UClient(NumPyClient):

    def __init__(self, cid, n_client_layers):
        super().__init__()
        self.cid = cid
        self.n_client_layers = n_client_layers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = nn.ModuleDict({
            "encoder": ClientModel(),
            "clf_head": ClientUHead()
        })
        print("training model on", self.device)
        seed()
        self.model.to(self.device)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        self.server_model_proxy.torch()

        set_parameters(self.model, parameters)

        self.model.train()
        optimizer = get_optimizer(self.model)
        criterion = nn.CrossEntropyLoss()
        trainloader = get_dataloader("train")

        start_time = time.time()
        for images, labels in trainloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # get client embeddings
            client_embeddings = self.model["encoder"](images)

            # get server embeddings
            server_embeddings = self.server_model_proxy.u_forward(embeddings=client_embeddings)
            server_embeddings.to(self.device).requires_grad_(True)

            # final predictions
            preds = self.model["clf_head"](server_embeddings)

            optimizer.zero_grad()

            # start backpropagating
            loss = criterion(preds, labels)
            loss.backward()

            # get gradient from the server
            server_gradient = self.server_model_proxy.u_backward(
                gradient=server_embeddings.grad
            )
            # backpropagate gradient received by the server
            client_embeddings.backward(server_gradient.to(self.device))
            optimizer.step()

        return (
            get_parameters(self.model),
            len(trainloader.dataset),
            {"train_time": time.time() - start_time}
        )

    def evaluate(self, parameters, config):
        self.server_model_proxy.torch()
        self.model.eval()
        set_parameters(self.model, parameters)

        valloader = get_dataloader("val")

        correct, processed = 0, 0
        for images, labels in valloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                embeddings = self.model["encoder"](images)

            embeddings = self.server_model_proxy.predict(embeddings=embeddings)
            server_embeddings = embeddings.to(self.device)

            preds = self.model["clf_head"](server_embeddings)
            preds = torch.argmax(preds, axis=1)

            processed += labels.shape[0]
            correct += (preds == labels).sum()

        accuracy = float(correct / processed)
        print(f"accuracy {accuracy} {correct} {processed}")
        return accuracy, len(valloader.dataset), {}
