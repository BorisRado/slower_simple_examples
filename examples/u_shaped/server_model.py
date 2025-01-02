import torch
from flwr.common import GetParametersRes

from slwr.server.server_model.numpy_server_model import NumPyServerModel
from slwr.server.server_model.utils import pytorch_format

from examples.common.model import ServerEncoder
from examples.common.parameters import get_parameters, set_parameters
from examples.common.helper import get_optimizer


class UServerModel(NumPyServerModel):

    def __init__(self, n_client_layers) -> None:
        super().__init__()

        self.model = ServerEncoder()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("training server model on", self.device)
        self.model = self.model.to(self.device)

    @pytorch_format
    def predict(self, embeddings):
        embeddings = embeddings.to(self.device)
        with torch.no_grad():
            preds = self.model(embeddings)
        return preds

    @pytorch_format
    def u_forward(self, embeddings):
        self.client_embeddings = embeddings.to(self.device)
        self.client_embeddings.requires_grad_(True)
        self.server_embeddings = self.model(self.client_embeddings)
        return self.server_embeddings

    @pytorch_format
    def u_backward(self, gradient):
        self.optimizer.zero_grad()
        self.server_embeddings.backward(gradient.to(self.device))
        self.optimizer.step()
        return self.client_embeddings.grad

    def get_parameters(self) -> GetParametersRes:
        return get_parameters(self.model)

    def configure_fit(self, parameters, config):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = get_optimizer(self.model)
        self.model.train()
        set_parameters(self.model, parameters)

    def configure_evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()
