from torchvision import transforms
import torch

class MNISTDefault(transforms.Compose):
    def __init__(self, flatten = True):
        trf_list = []
        trf_list.append(transforms.ToTensor())
        trf_list.append(transforms.Normalize((0.1307), (0.3081)))

        if flatten:
            trf_list.append(transforms.Lambda(lambda x: torch.flatten(x)))

        super().__init__(trf_list)
