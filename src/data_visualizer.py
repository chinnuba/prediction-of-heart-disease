import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import _clean_data

class DataVisualizer:
    def __init__(self, df):
        self.df = df

    def histogram_boxplot(self):
        """Visualize each numerical column with histograms and box plots."""
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64']).columns
        num_plots = len(numerical_cols)

        fig, axes = plt.subplots(num_plots, 2, figsize=(15, num_plots * 5))
        fig.tight_layout(pad=5.0)

        for i, col in enumerate(numerical_cols):
            sns.histplot(self.df[col], kde=True, ax=axes[i, 0])
            axes[i, 0].set_title(f'Histogram of {col}')

            sns.boxplot(x=self.df[col], ax=axes[i, 1])
            axes[i, 1].set_title(f'Box plot of {col}')

        plt.show()

    def pair_plot(self):
            """Visualize pairwise relationships in the dataset."""
            sns.pairplot(self.df)
            plt.title('Pair Plot')
            plt.show()

    def scatter_plot(self, col1, col2):
            """Visualize the scatter plot between two numerical columns."""
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=self.df[col1], y=self.df[col2])
            plt.title(f'Scatter Plot between {col1} and {col2}')
            plt.show()

    def count_plot(self, col):
            """Visualize the count plot of a categorical column."""
            plt.figure(figsize=(10, 6))
            sns.countplot(x=self.df[col])
            plt.title(f'Count Plot of {col}')
            plt.show()

    def violin_plot(self, numerical_col, categorical_col):
            """Visualize the distribution of a numerical column across categories."""
            plt.figure(figsize=(10, 6))
            sns.violinplot(x=self.df[categorical_col], y=self.df[numerical_col])
            plt.title(f'Violin Plot of {numerical_col} by {categorical_col}')
            plt.show()

    def distribution_plot(self, col):
            """Visualize the distribution of a numerical column."""
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df[col], kde=True)
            plt.title(f'Distribution Plot of {col}')
            plt.show()

    def correlation_heatmap(self):
            """Visualize the correlation matrix as a heatmap."""
            processed_df = _clean_data(self.df)
            plt.figure(figsize=(12, 10))
            correlation_matrix = processed_df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
            plt.title('Correlation Matrix Heatmap')
            plt.show()