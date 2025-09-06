import pandas as pd

class DataLoader:
    def __init__(self, path):
        self.path = path

    def load(self):
        try:
            data = pd.read_csv(self.path)
        except Exception as e:
            raise IOError(f"Failed to load data from {self.path}: {str(e)}")
        if data.empty:
            raise ValueError(f"Loaded data from {self.path} is empty.")
        return data