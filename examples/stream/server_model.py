import torch
from flwr.common import GetParametersRes

from slwr.server.server_model.numpy_server_model import NumPyServerModel
from slwr.server.server_model.utils import pytorch_format

from examples.common.model import ServerModel
from examples.common.parameters import get_parameters, set_parameters
from examples.common.helper import get_optimizer


class StreamServerModel(NumPyServerModel):

    def __init__(self) -> None:
        super().__init__()

        self.model = ServerModel()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    @pytorch_format
    def predict(self, embeddings):
        embeddings = embeddings.to(self.device)
        with torch.no_grad():
            preds = self.model(embeddings)
        preds = torch.argmax(preds, axis=1)
        return preds

    @pytorch_format
    def update_server_model(self, embeddings, labels):
        self.n += 1
        embeddings, labels = embeddings.to(self.device), labels.to(self.device)

        preds = self.model(embeddings)
        loss = self.criterion(preds, labels)

        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_parameters(self) -> GetParametersRes:
        return get_parameters(self.model)

    def configure_fit(self, parameters, config):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = get_optimizer(self.model)
        self.model.train()
        self.n = 0
        set_parameters(self.model, parameters)

    def configure_evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()

    def get_number_processed(self):
        return [{"num": torch.tensor([self.n,]).numpy()}]
