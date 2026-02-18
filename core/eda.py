import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class EDA:
  def __init__(self, df):
    self.df = df

  
  def univariate_numeric(self, col):
    figure, axis = plt.subplots()
    sns.histplot(self.df[col], kde=True, axis=axis)
    axis.set_title(f"Distribution of {col}")
    return figure

  def univariate_categorical(self, col):
    figure, axis = plt.subplots()
    self.df[col].value_counts().plot(kind="bar", axis=axis)
    axis.set_title(f"Counts of {col}")
    return figure

  def numeric_vs_numeric(self, x, y):
    figure, axis = plt.subplots()
    sns.scatterplot(data=self.df, x=x, y=y, axis=axis)
    axis.set_title(f"{x} vs {y}")
    return figure

  def categorical_vs_numeric(self, cat, num):
    figure, axis = plt.subplots()
    sns.boxplot(data=self.df, x=cat, y=num, axis=axis)
    axis.set_title(f"{num} by {cat}")
    return figure

  def correlation_heatmap(self):
    figure, axis = plt.subplots(figuresize=(8, 6))
    corr = self.df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", axis=axis)
    axis.set_title("Correlation Heatmap")
    return figure

  def numeric_vs_numeric_plot(self, x, y, plot_type):
    figure, axis = plt.subplots()

    if plot_type == "Scatter":
      sns.scatterplot(data=self.df, x=x, y=y, axis=axis)

    elif plot_type == "Line":
      sns.lineplot(data=self.df, x=x, y=y, axis=axis)

    elif plot_type == "Regression":
      sns.regplot(data=self.df, x=x, y=y, axis=axis)

    axis.set_title(f"{plot_type}: {x} vs {y}")
    return figure

  def categorical_vs_numeric_plot(self, cat, num, plot_type):
    figure, axis = plt.subplots()

    if plot_type == "Box":
      sns.boxplot(data=self.df, x=cat, y=num, axis=axis)

    elif plot_type == "Violin":
      sns.violinplot(data=self.df, x=cat, y=num, axis=axis)

    elif plot_type == "Bar (Mean)":
      sns.barplot(data=self.df, x=cat, y=num, axis=axis, estimator="mean")

    axis.set_title(f"{plot_type}: {num} by {cat}")
    return figure

