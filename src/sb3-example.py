import mlflow
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
import hydra
from omegaconf import OmegaConf
import numpy as np
import sys
import tempfile

class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
        self,
        key_values,
        key_excluded,
        step = 0
    ) -> None:

        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):

            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)

@hydra.main(version_base=None, config_path="../cfg-rl", config_name="config")
def main(cfg):
    loggers = Logger(
        folder=None,
        output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()]
    )

    cfg = OmegaConf.to_container(cfg, resolve=True)
    mlflow.set_experiment(cfg["experiment_name"])

    model = hydra.utils.instantiate(cfg["model"])
    
    model.set_logger(loggers)
    model.learn(total_timesteps=10_000)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save(tmpdir + "/model.zip")
        mlflow.log_artifact(tmpdir)

    model.env.close()

if __name__ == "__main__":
    main()