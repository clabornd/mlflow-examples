import sklearn.datasets as ds
from sklearn.model_selection import train_test_split

class SklData:
    def __init__(self, dataset_name = "diabetes"):
        call = "load_" + dataset_name
        self.data = eval("ds." + call + "()")

    def get_X(self):
        return self.data.data

    def get_y(self):
        return self.data.target
    
    def get_train_test_splits(self, **kwargs):
        return train_test_split(self.data.data, self.data.target, **kwargs)