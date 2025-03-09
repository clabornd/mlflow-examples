import importlib

def get_optimizer(name: str = "Adam"):
    module = importlib.import_module("torch.optim")
    optimizer = getattr(module, name)

    return optimizer