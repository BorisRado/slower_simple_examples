import torch
from flwr.common import GetParametersRes

from slower.server.server_model.numpy_server_model import NumPyServerModel

from examples.common.model import get_model_slice, get_n_layers
from examples.common.parameters import get_parameters, set_parameters


class StreamServerModel(NumPyServerModel):

    def __init__(self, n_client_layers) -> None:
        super().__init__()

        self.model = get_model_slice(slice(n_client_layers * 2, get_n_layers()))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def predict(self, embeddings):
        embeddings = torch.from_numpy(embeddings).to(self.device)
        with torch.no_grad():
            preds = self.model(embeddings)
        preds = torch.argmax(preds, axis=1)
        return preds.cpu().numpy()

    def update_server_model(self, embeddings, labels):
        self.n += 1
        embeddings = torch.from_numpy(embeddings).to(self.device)
        labels = torch.from_numpy(labels).to(self.device)

        preds = self.model(embeddings)
        loss = self.criterion(preds, labels)

        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {}

    def get_parameters(self) -> GetParametersRes:
        return get_parameters(self.model)

    def configure_fit(self, parameters, config):
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.05)
        self.model.train()
        self.n = 0
        set_parameters(self.model, parameters)

    def configure_evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self.model.eval()

    def get_synchronization_result(self):
        return {"n_processed": torch.tensor([self.n,]).numpy()}
