from scipy.stats import ttest_ind, chi2_contingency
import pandas as pd
class HypothesisTesting:
    def __init__(self, df):
        self.df = df

    def t_test(self, col1, col2):
        stat, p = ttest_ind(self.df[col1].dropna(), self.df[col2].dropna())
        return stat, p

    def chi_square(self, col1, col2):
        table = pd.crosstab(self.df[col1], self.df[col2])
        stat, p, _, _ = chi2_contingency(table)
        return stat, p
