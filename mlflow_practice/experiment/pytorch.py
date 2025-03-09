from .base import BaseExperiment
from tqdm import tqdm
import mlflow
import torch

class PyTorchExperiment(BaseExperiment):
    def __init__(self, optimizer, epochs, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.optimizer = optimizer
        self.epochs = epochs
        self.loss = torch.nn.CrossEntropyLoss()

    def evaluate(self):
        raise NotImplementedError
    
    def train(self):
        pbar = tqdm(range(self.epochs))        

        for epoch in pbar:
            self.model.train()

            for batch in self.data.train:
                self.optimizer.zero_grad()

                X, y = batch
                y_pred = self.model(X)
                loss = self.loss(y_pred, y)
                loss.backward()

                mlflow.log_metric("loss", loss.item())

                self.optimizer.step()

                pbar.set_description(f"Epoch: {epoch}, loss: {loss.item()}")     

    def test(self):
        self.model.eval()
        with torch.inference_mode():
            for batch in self.data.test:
                X, y = batch
                y_pred = self.model(X)
                loss = self.loss(y_pred, y)
                mlflow.log_metric("test_loss", loss.item()) 

    def run(self):      
        self.train()
        self.test()
