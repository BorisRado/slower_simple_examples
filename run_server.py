import hydra
from hydra.utils import instantiate

from flwr.server import ServerConfig

from slwr.server.app import start_server
from slwr.server.strategy import PlainSlStrategy


def average(metrics):
    # Multiply accuracy of each client by number of examples used
    vals = [m["train_time"] for _, m in metrics]
    return {
        "train_time": sum(vals) / len(metrics)
    }


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg):
    server_model_fn = instantiate(cfg.configuration.server_model, _partial_=True)

    n_clients = cfg.num_clients
    strategy = PlainSlStrategy(
        init_server_model_fn=server_model_fn,
        min_available_clients=n_clients,
        min_fit_clients=n_clients,
        common_server_model=cfg.common_server_model,
        process_clients_as_batch=cfg.process_clients_as_batch,
        fit_metrics_aggregation_fn=average
    )
    history = start_server(
        server_address="[::]:50051",
        strategy=strategy,
        config=ServerConfig(num_rounds=4),
    )
    train_time = [v[1] for v in history.metrics_distributed_fit["train_time"]]
    print(f"Average train time: {sum(train_time) / len(train_time)} {cfg.common_server_model} {cfg.process_clients_as_batch} {cfg.configuration.client._target_.split('.')[1]}")



if __name__ == "__main__":
    run()
