import hydra
import omegaconf
from omegaconf import OmegaConf
import mlflow

from mlflow_practice.experiment.pytorch import PyTorchExperiment
from mlflow_practice.optimizer import get_optimizer

@hydra.main(
    version_base=None,
    config_path="../cfg-dl",
    config_name="config",
)
def main(cfg: omegaconf.DictConfig) -> None:
    print(omegaconf.OmegaConf.to_container(cfg))

    cfg = OmegaConf.to_container(cfg, resolve=True)

    mlflow.set_experiment(cfg["experiment_name"])

    model = hydra.utils.instantiate(cfg["model"])
    data = hydra.utils.instantiate(cfg["data"])

    opt_class = get_optimizer(cfg['optimizer']['name'])
    optimizer = opt_class(params = model.parameters(), **cfg['optimizer']['params'])

    experiment = PyTorchExperiment(model = model, data = data, optimizer = optimizer, epochs = cfg['epochs'])

    experiment.run()

if __name__ == "__main__":
    main()