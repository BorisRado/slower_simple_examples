import hydra
from hydra.utils import instantiate

from slower.client.app import start_client


@hydra.main(version_base=None, config_path="conf", config_name="config")
def run(cfg):
    client_fn = instantiate(cfg.configuration.client, _partial_=True)

    start_client(server_address="0.0.0.0:9091", client_fn=client_fn)

if __name__ == "__main__":
    run()
