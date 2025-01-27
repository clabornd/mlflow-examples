import mlflow
from sklearn.metrics import mean_squared_error, roc_auc_score
import pickle
import tempfile

from omegaconf import OmegaConf
import hydra

import logging

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

        # for simple sklearn datasets, we would implement a separate get_train and get_test method
        # so we can keep the same api.
        X_train, X_test, y_train, y_test = data_obj.get_train_test_splits()

        # Create and train models.
        model = hydra.utils.instantiate(cfg["model"])

        model.fit(X_train, y_train)

        # Save the model as an artifact in a temporary directory.
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = tmpdir + "/model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            mlflow.log_artifact(model_path)

        # log test set performance depending on task type
        if cfg.get("task"):
            if cfg['task'] == "classification":
                probs = model.predict_proba(X_test)
                performance = roc_auc_score(y_test, probs, multi_class="ovr")
                mlflow.log_metrics({"roc_auc_score": performance})
            elif cfg['task'] == 'regression':
                preds = model.predict(X_test)
                performance = mean_squared_error(y_test, preds)
                mlflow.log_metrics({"mean_squared_error": performance})
        else:
            logger.warning("No task specified, attempting to infer task by target type")

            # actually attempt to infer type here....this is why it seems like all of this should be wrapped in an 'experiment' class.
            
if __name__ == "__main__":
    main()
