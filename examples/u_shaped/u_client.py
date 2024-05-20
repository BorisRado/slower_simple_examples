import time

import torch.nn as nn
import torch

from slower.client.numpy_client import NumPyClient

from examples.common.parameters import get_parameters, set_parameters
from examples.common.model import get_model_slice, get_n_layers
from examples.common.data import get_dataloader
from examples.common.helper import seed


class UClient(NumPyClient):

    def __init__(self, cid, n_client_layers):
        super().__init__()
        self.cid = cid
        self.n_client_layers = n_client_layers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        tot_layers = get_n_layers()
        self.model = nn.ModuleDict({
            "encoder": get_model_slice(slice(0, n_client_layers * 2)),
            "clf_head": get_model_slice(slice(tot_layers - 2, tot_layers))
        })
        print("training model on", self.device)
        seed()
        self.model.to(self.device)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def to_torch(self, array):
        return torch.from_numpy(array).to(self.device)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05)
        criterion = nn.CrossEntropyLoss()
        trainloader = get_dataloader("train")

        start_time = time.time()
        for images, labels in trainloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # get client embeddings
            client_embeddings = self.model["encoder"](images)

            # get server embeddings
            client_fwd_batch_data = {"embeddings": client_embeddings.detach().cpu().numpy()}
            res = self.server_model_proxy.numpy_u_forward(
                batch_data=client_fwd_batch_data
            )
            server_embeddings = self.to_torch(res["embeddings"])
            server_embeddings.requires_grad_(True)

            # final predictions
            preds = self.model["clf_head"](server_embeddings)

            optimizer.zero_grad()

            # start backpropagating
            loss = criterion(preds, labels)
            loss.backward()

            # get gradient from the server
            client_bck_batch_data = {"gradient": server_embeddings.grad.detach().cpu().numpy()}
            res = self.server_model_proxy.numpy_u_backward(
                batch_data=client_bck_batch_data
            )
            # backpropagate gradient received by the server
            server_gradient = self.to_torch(res["gradient"])
            client_embeddings.backward(server_gradient)

            optimizer.step()

        return get_parameters(self.model), len(trainloader.dataset), {"train_time": time.time() - start_time}

    def evaluate(self, parameters, config):
        self.model.eval()
        set_parameters(self.model, parameters)

        valloader = get_dataloader("val")

        correct, processed = 0, 0
        for images, labels in valloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.no_grad():
                embeddings = self.model["encoder"](images)

            batch_data = {"embeddings": embeddings.cpu().numpy()}
            res = self.server_model_proxy.numpy_serve_prediction_request(
                batch_data=batch_data
            )
            server_embeddings = self.to_torch(res["embeddings"])

            preds = self.model["clf_head"](server_embeddings)
            preds = torch.argmax(preds, axis=1)

            processed += labels.shape[0]
            correct += (preds == labels).sum()

        accuracy = float(correct / processed)
        print(f"accuracy {accuracy} {correct} {processed}")
        return accuracy, len(valloader.dataset), {}
