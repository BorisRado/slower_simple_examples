import hydra
from hydra.utils import instantiate

from slower.simulation.app import start_simulation
from slower.server.strategy.plain_sl_strategy import PlainSlStrategy


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg):
    client_fn = instantiate(cfg.configuration.client, _partial_=True)
    server_model_fn = instantiate(cfg.configuration.server_model, _partial_=True)
    print(client_fn)
    print(server_model_fn)

    strategy = PlainSlStrategy(common_server=False, init_server_model_fn=server_model_fn)
    start_simulation(
        client_fn=client_fn, num_clients=1, client_resources={"num_gpus": 0, "num_cpus": 8}, strategy=strategy
    )

if __name__ == "__main__":
    run()
