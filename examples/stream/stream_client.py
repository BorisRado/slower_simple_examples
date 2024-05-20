import time

import torch

from slower.client.numpy_client import NumPyClient

from examples.common.parameters import get_parameters, set_parameters
from examples.common.model import get_model_slice
from examples.common.data import get_dataloader
from examples.common.helper  import seed


class StreamClient(NumPyClient):

    def __init__(self, cid, n_client_layers):
        super().__init__()
        self.cid = cid
        self.n_client_layers = n_client_layers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = get_model_slice(slice(0, n_client_layers * 2))  # * 2 cuz every second layer is ReLU
        self.model.to(self.device)
        seed()

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        self.model.train()
        trainloader = get_dataloader("train")

        start_time = time.time()

        for idx, (images, labels) in enumerate(trainloader):
            images = images.to(self.device)

            # get embeddings
            embeddings = self.model(images)

            # get gradient from the server
            batch_data = {
                "embeddings": embeddings.detach().cpu().numpy(),
                "labels": labels.numpy()
            }
            self.server_model_proxy.numpy_update_server_model(
                batch_data=batch_data
            )
            if self.server_model_proxy.get_pending_batches_count() > 100:
                print("sleeping...")
                time.sleep(1)
        server_res = self.server_model_proxy.numpy_close_stream()
        print(f"Just testing that the server returns the right thing... {server_res}")
        return get_parameters(self.model), len(trainloader.dataset), {"train_time": time.time() - start_time}

    def evaluate(self, parameters, config):
        self.model.eval()
        set_parameters(self.model, parameters)

        valloader = get_dataloader("val")

        correct, processed = 0, 0
        for images, labels in valloader:
            images = images.to(self.device)

            with torch.no_grad():
                embeddings = self.model(images)

            batch_data = {
                "embeddings": embeddings.cpu().numpy()
            }
            preds = self.server_model_proxy.numpy_serve_prediction_request(
                batch_data=batch_data
            )["predictions"]

            processed += labels.shape[0]
            correct += (preds == labels.numpy()).sum()

        accuracy = float(correct / processed)
        print(f"accuracy {accuracy} {correct} {processed}")
        return accuracy, len(valloader.dataset), {}
