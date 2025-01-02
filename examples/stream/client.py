import time

import torch

from slwr.client.numpy_client import NumPyClient
from slwr.common import RequestType

from examples.common.parameters import get_parameters, set_parameters
from examples.common.model import ClientModel, ClientClfHead
from examples.common.data import get_dataloader
from examples.common.helper  import seed, get_optimizer


class StreamClient(NumPyClient):

    def __init__(self, cid):
        super().__init__()
        self.cid = cid
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = torch.nn.ModuleDict({
            "encoder": ClientModel(),
            "head": ClientClfHead(),
        })
        self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        seed()

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        self.server_model_proxy.torch()

        set_parameters(self.model, parameters)

        self.model.train()
        trainloader = get_dataloader("train")

        optimizer = get_optimizer(self.model)
        start_time = time.time()

        for images, labels in trainloader:
            images = images.to(self.device)

            # get embeddings
            embeddings = self.model["encoder"](images)

            # get gradient from the server
            self.server_model_proxy.update_server_model(
                embeddings=embeddings,
                labels=labels,
                _type_=RequestType.STREAM,
            )

            client_preds = self.model["head"](embeddings)
            loss = self.criterion(client_preds, labels.to(self.device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        number_processed = self.server_model_proxy.get_number_processed()
        print(f"Just testing that the server returns the right thing... {number_processed}")

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

            with torch.no_grad():
                embeddings = self.model["encoder"](images)

            preds = self.server_model_proxy.predict(
                embeddings = embeddings
            )

            processed += labels.shape[0]
            correct += (preds == labels.numpy()).sum()

        accuracy = float(correct / processed)
        print(f"accuracy {accuracy} {correct} {processed}")
        return accuracy, len(valloader.dataset), {}
