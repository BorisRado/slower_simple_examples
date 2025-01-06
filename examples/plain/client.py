import time

import torch

from slwr.client.numpy_client import NumPyClient

from examples.common.parameters import get_parameters, set_parameters
from examples.common.model import ClientModel
from examples.common.data import get_dataloader
from examples.common.helper import seed, get_optimizer


class PlainClient(NumPyClient):

    def __init__(self, cid, use_streams, data_percentage):
        super().__init__()
        self.cid = cid
        print("Initialized client with ID ", cid)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        seed()
        self.model = ClientModel()
        self.model.to(self.device)
        self.use_streams = use_streams
        self.data_percentage = data_percentage

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        self.server_model_proxy.torch()
        set_parameters(self.model, parameters)

        self.model.train()
        optimizer = get_optimizer(self.model)

        start_time = time.time()

        seed()
        trainloader = get_dataloader("train", self.data_percentage)
        for images, labels in trainloader:
            images = images.to(self.device)

            # get embeddings
            embeddings = self.model(images)

            # get gradient from the server
            gradient = self.server_model_proxy.serve_grad_request(
                embeddings=embeddings,
                labels=labels,
                _streams_=self.use_streams,
            )

            assert gradient.shape == embeddings.shape

            # backpropagate gradient received by the server
            optimizer.zero_grad()
            embeddings.backward(gradient.to(self.device))
            optimizer.step()
        print("Train time: ", time.time() - start_time)
        return get_parameters(self.model), len(trainloader.dataset), {"train_time": time.time() - start_time}

    def evaluate(self, parameters, config):
        self.server_model_proxy.numpy()
        self.model.eval()
        set_parameters(self.model, parameters)

        valloader = get_dataloader("val")

        correct, processed = 0, 0
        for images, labels in valloader:
            images = images.to(self.device)

            with torch.no_grad():
                embeddings = self.model(images)

            preds = self.server_model_proxy.predict(
                embeddings=embeddings.cpu().numpy(),
                _streams_=self.use_streams,
            )

            processed += labels.shape[0]
            correct += (preds == labels.numpy()).sum()

        accuracy = float(correct / processed)
        print(f"accuracy {accuracy} {correct} {processed}")
        return accuracy, len(valloader.dataset), {}
