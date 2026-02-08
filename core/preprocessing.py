import pandas as pd

class Preprocessing:
    def __init__(self, df):
        self.df = df

    def fill_na(self, column, method):
        col_dtype = self.df[column].dtype

        # Numeric column
        if pd.api.types.is_numeric_dtype(col_dtype):
            if method == "mean":
                value = self.df[column].mean()
            elif method == "median":
                value = self.df[column].median()
            elif method == "mode":
                value = self.df[column].mode()[0]
            else:
                raise ValueError("Invalid method for numeric column")

        # Categorical / Text column
        else:
            if method != "mode":
                raise TypeError(
                    f"Method '{method}' not allowed for non-numeric column '{column}'"
                )
            value = self.df[column].mode()[0]

        self.df[column] = self.df[column].fillna(value)
        return self.df
