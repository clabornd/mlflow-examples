from .base import BaseExperiment
import mlflow
from sklearn.metrics import mean_squared_error, roc_auc_score
import numpy as np

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class SkLearnExperiment(BaseExperiment):
    def __init__(self, model, data, task, model_target=None, **kwargs):
        super().__init__(model, data, **kwargs)

        self.task = task
        self.model_target = model_target

    def evaluate(self, X_test, y_test):
        # log test set performance depending on task type
        if self.task:
            if self.task == "classification":
                probs = self.model.predict_proba(X_test)
                performance = roc_auc_score(y_test, probs, multi_class="ovr")
                mlflow.log_metrics({"roc_auc_score": performance})
            elif self.task == 'regression':
                preds = self.model.predict(X_test)
                performance = mean_squared_error(y_test, preds)
                mlflow.log_metrics({"mean_squared_error": performance})
        else:
            logger.warning("No task specified, attempting to infer task by target type")

    def preprocess(self, X_train, X_test, y_train, y_test):
        if self.model_target == "xgboost.XGBClassifier":
            from sklearn.preprocessing import LabelEncoder
            enc = LabelEncoder()
            y_train = enc.fit_transform(y_train)
            y_test = enc.transform(y_test)

        return X_train, X_test, y_train, y_test

    def run(self):
        splits = self.data.get_train_test_splits()

        X_train, X_test, y_train, y_test = self.preprocess(*splits)

        # check if they have chosen the right algorithm
        if self.task == "classification":
            train_classes = np.unique(y_train)
            test_classes = np.unique(y_test)
            intx_classes = np.intersect1d(train_classes, test_classes)

            if not len(train_classes) == len(test_classes) == len(intx_classes):
                logger.warning("You selected classification but your train and test target variables do not have the same classes, problems may arise.  Check that your data is a classification task or specify `model=<algo>/regression` if you meant to perform regression.")
        
        if self.task == "regression":
            if not np.issubdtype(y_train.dtype, np.number) or not np.issubdtype(y_test.dtype, np.number):
                logger.warning("You specified a regression algorithm but your train and/or test target variables are non-numeric, problems may arise.  Check that your data specifies a regression task or specify model=<algo>/classification` if you meant to perform classification.")

        self.model.fit(X_train, y_train)

        self.evaluate(X_test, y_test)
