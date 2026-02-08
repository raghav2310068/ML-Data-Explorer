import pandas as pd

class DataLoader:
    def __init__(self):
        self.df = None

    def load_csv(self, file):
        self.df = pd.read_csv(file)
        return self.df
