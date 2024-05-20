import hydra
from hydra.utils import instantiate

from slower.server.app import start_server
from slower.server.strategy.plain_sl_strategy import PlainSlStrategy


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg):
    server_model_fn = instantiate(cfg.configuration.server_model, _partial_=True)

    strategy = PlainSlStrategy(common_server=False, init_server_model_fn=server_model_fn)
    start_server(server_address="0.0.0.0:9091", strategy=strategy)


if __name__ == "__main__":
    run()
