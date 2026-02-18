from sklearn.preprocessing import LabelEncoder
import pandas as pd

class Encoder:
    def __init__(self, df):
        self.df = df

    def label_encode(self, column):
        le = LabelEncoder()
        self.df[column] = le.fit_transform(self.df[column].astype(str))
        return self.df

    def one_hot_encode(self, column):
        self.df = pd.get_dummies(self.df, columns=[column])
        return self.df
