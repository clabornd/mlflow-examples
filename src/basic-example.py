import mlflow
from sklearn.metrics import mean_squared_error, roc_auc_score
import pickle
import tempfile

from omegaconf import OmegaConf
import hydra

import logging
from mlflow_practice.experiment.sklearn import SkLearnExperiment

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@hydra.main(version_base=None, config_path="../cfg", config_name="config")
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    mlflow.set_experiment(cfg["experiment_name"])
    # mlflow.sklearn.autolog()

    with mlflow.start_run():
        mlflow.log_params(cfg)
        # Load the diabetes dataset.

        data_obj = hydra.utils.instantiate(cfg["data"])
        model = hydra.utils.instantiate(cfg["model"])

        exp = SkLearnExperiment(
            model = model, 
            data = data_obj,
            task = cfg["task"],
            model_target = cfg["model"]["_target_"]
        )

        exp.run()

        # Save the model as an artifact in a temporary directory.
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = tmpdir + "/model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(exp.model, f)

            mlflow.log_artifact(model_path)

if __name__ == "__main__":
    main()
