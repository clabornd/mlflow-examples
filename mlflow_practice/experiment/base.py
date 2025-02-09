class BaseExperiment:
    def __init__(self, model, data, **kwargs):
        self.model = model
        self.data = data

    def evaluate(self):
        raise NotImplementedError
    
    def run(self):
        raise NotImplementedError