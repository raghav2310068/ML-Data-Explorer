import pandas as pd

class DataOverview:
    def __init__(self, df):
        self.df = df

    def basic_info(self):
        return {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "missing": self.df.isnull().sum().sum()
        }

    def column_summary(self):
        return pd.DataFrame({
            "Column": self.df.columns,
            "Type": self.df.dtypes.astype(str),
            "Missing": self.df.isnull().sum()
        })

    def stats(self):
        return self.df.describe(include="all")
