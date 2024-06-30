import sklearn.datasets as ds

class SklData:
    def __init__(self, dataset_name = "diabetes"):
        call = "load_" + dataset_name
        self.data = eval("ds." + call + "()")

    def get_X(self):
        return self.data.data

    def get_y(self):
        return self.data.target