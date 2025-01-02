import torch
from flwr.common import GetParametersRes

from slwr.server.server_model.numpy_server_model import NumPyServerModel
from slwr.server.server_model.utils import pytorch_format

from examples.common.model import ServerModel
from examples.common.parameters import get_parameters, set_parameters
from examples.common.helper import get_optimizer, seed


class PlainServerModel(NumPyServerModel):

    def __init__(self) -> None:
        super().__init__()

        seed()
        self.model = ServerModel()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    @pytorch_format
    def predict(self, embeddings):
        with torch.no_grad():
            preds = self.model(embeddings.to(self.device))
        return torch.argmax(preds, axis=1)

    @pytorch_format
    def serve_grad_request(self, embeddings, labels):
        embeddings.requires_grad_(True)
        preds = self.model(embeddings.to(self.device))
        loss = self.criterion(preds, labels.to(self.device))

        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()

        return embeddings.grad

    def get_parameters(self) -> GetParametersRes:
        return get_parameters(self.model)

    def get_fit_result(self):
        print("Obtaining new weights...")
        return self.get_parameters(), {}

    def configure_fit(self, parameters, config):
        print("Configuring fit")
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = get_optimizer(self.model)
        set_parameters(self.model, parameters)
        self.model.train()

    def configure_evaluate(self, parameters, config):
        print("Configuring evaluate")
        set_parameters(self.model, parameters)
        self.model.eval()
