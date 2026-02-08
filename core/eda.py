import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class EDA:
    def __init__(self, df):
        self.df = df

    # ---------- UNIVARIATE ----------
    def univariate_numeric(self, col):
        fig, ax = plt.subplots()
        sns.histplot(self.df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        return fig

    def univariate_categorical(self, col):
        fig, ax = plt.subplots()
        self.df[col].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"Counts of {col}")
        return fig

    # ---------- BIVARIATE ----------
    def numeric_vs_numeric(self, x, y):
        fig, ax = plt.subplots()
        sns.scatterplot(data=self.df, x=x, y=y, ax=ax)
        ax.set_title(f"{x} vs {y}")
        return fig

    def categorical_vs_numeric(self, cat, num):
        fig, ax = plt.subplots()
        sns.boxplot(data=self.df, x=cat, y=num, ax=ax)
        ax.set_title(f"{num} by {cat}")
        return fig

    # ---------- MULTIVARIATE ----------
    def correlation_heatmap(self):
        fig, ax = plt.subplots(figsize=(8, 6))
        corr = self.df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Heatmap")
        return fig

    def numeric_vs_numeric_plot(self, x, y, plot_type):
        fig, ax = plt.subplots()

        if plot_type == "Scatter":
            sns.scatterplot(data=self.df, x=x, y=y, ax=ax)

        elif plot_type == "Line":
            sns.lineplot(data=self.df, x=x, y=y, ax=ax)

        elif plot_type == "Regression":
            sns.regplot(data=self.df, x=x, y=y, ax=ax)

        ax.set_title(f"{plot_type}: {x} vs {y}")
        return fig

    def categorical_vs_numeric_plot(self, cat, num, plot_type):
        fig, ax = plt.subplots()

        if plot_type == "Box":
            sns.boxplot(data=self.df, x=cat, y=num, ax=ax)

        elif plot_type == "Violin":
            sns.violinplot(data=self.df, x=cat, y=num, ax=ax)

        elif plot_type == "Bar (Mean)":
            sns.barplot(data=self.df, x=cat, y=num, ax=ax, estimator="mean")

        ax.set_title(f"{plot_type}: {num} by {cat}")
        return fig
