import mlflow
from sklearn.metrics import mean_squared_error
import pickle
import tempfile

from omegaconf import OmegaConf
import hydra


@hydra.main(version_base=None, config_path="../cfg", config_name="config")
def main(cfg):
    cfg = OmegaConf.to_container(cfg, resolve=True)
    mlflow.set_experiment(cfg["experiment_name"])
    # mlflow.autolog()

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

        preds = model.predict(X_test)
        mlflow.log_metrics({"mse": mean_squared_error(y_test, preds)})


if __name__ == "__main__":
    main()
