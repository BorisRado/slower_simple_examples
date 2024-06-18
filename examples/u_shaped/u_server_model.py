from numpy import ndarray
import torch
from flwr.common import GetParametersRes

from slower.server.server_model.numpy_server_model import NumPyServerModel

from examples.common.model import get_model_slice, get_n_layers
from examples.common.parameters import get_parameters, set_parameters


class UServerModel(NumPyServerModel):

    def __init__(self, n_client_layers) -> None:
        super().__init__()

        self.model = get_model_slice(slice(n_client_layers * 2, get_n_layers() - 2))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("training server model on", self.device)
        self.model = self.model.to(self.device)

    def predict(self, embeddings):
        embeddings = torch.from_numpy(embeddings).to(self.device)
        with torch.no_grad():
            preds = self.model(embeddings)
        return preds.cpu().numpy()

    def u_forward(self, embeddings):
        self.client_embeddings = self.to_torch(embeddings)
        self.client_embeddings.requires_grad_(True)
        self.server_embeddings = self.model(self.client_embeddings)
        return self.server_embeddings.detach().cpu().numpy()

    def u_backward(self, gradient):
        gradient = self.to_torch(gradient)
        self.optimizer.zero_grad()
        self.server_embeddings.backward(gradient)
        self.optimizer.step()
        return self.client_embeddings.grad.detach().cpu().numpy()

    def get_parameters(self) -> GetParametersRes:
        return get_parameters(self.model)

    def configure_fit(self, parameters, config):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05)
        self.model.train()
        set_parameters(self.model, parameters)

    def configure_evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()

    def to_torch(self, array: ndarray):
        return torch.from_numpy(array).to(self.device)
