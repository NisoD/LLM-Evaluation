import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class LowPerformanceVisualizer:
    def __init__(self, base_results_dir: str = "../app/results_local"):
        """
        Initialize the visualizer with the base directory containing results.

        Args:
            base_results_dir: Path to the directory containing the results
        """
        self.base_dir = Path(base_results_dir)
        self.colors = {
            'Meta-Llama-3-8B-Instruct': '#2ca02c',
            'Llama-3.2-3B-Instruct': '#d62728',
            'Mistral-7B-Instruct-v0.3': '#9467bd',
            'Llama-3.2-1B-Instruct': '#17becf',
            'OLMoE-1B-7B-0924-Instruct': '#ff7f0e'
        }

    def load_all_results(self) -> pd.DataFrame:
        """
        Load and combine all parquet files from the results directory.

        Returns:
            pd.DataFrame: Combined dataset of all results
        """
        all_data = []

        for shots_dir in self.base_dir.glob("Shots_*"):
            shots = int(shots_dir.name.split('_')[1])

            for model_dir in shots_dir.glob("*"):
                model_name = model_dir.name

                for dataset_dir in model_dir.glob("*"):
                    dataset = dataset_dir.name
                    parquet_file = dataset_dir / 'low_performance_questions.parquet'

                    if parquet_file.exists():
                        df = pd.read_parquet(parquet_file)
                        df['shots'] = shots
                        df['model_name'] = model_name
                        df['dataset'] = dataset
                        all_data.append(df)

        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def calculate_position_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistics about answer position frequencies.

        Args:
            df: DataFrame containing the results

        Returns:
            pd.DataFrame: Processed statistics about position frequencies
        """
        position_stats = (
            df.groupby(['model_name', 'dataset', 'sample_index', 'answer_position'])
            .size()
            .reset_index(name='count')
        )

        # Calculate percentage for each position within each question
        position_stats['total'] = position_stats.groupby(
            ['model_name', 'dataset', 'sample_index']
        )['count'].transform('sum')

        position_stats['percentage'] = (position_stats['count'] / position_stats['total'] * 100)

        # Get the most chosen position for each question
        most_chosen = position_stats.loc[
            position_stats.groupby(['model_name', 'dataset', 'sample_index'])['count']
            .idxmax()
        ]

        return most_chosen

    def create_scatter_plot(self, stats_df: pd.DataFrame) -> go.Figure:
        """
        Create scatter plot showing most chosen answer positions.

        Args:
            stats_df: DataFrame containing position statistics

        Returns:
            go.Figure: Plotly figure object
        """
        fig = px.scatter(
            stats_df,
            x='dataset',
            y='percentage',
            color='model_name',
            color_discrete_map=self.colors,
            hover_data=['sample_index', 'answer_position'],
            title='Most Frequently Chosen Answer Positions for Low Performance Questions'
        )

        fig.update_layout(
            xaxis_title="Dataset",
            yaxis_title="Percentage of Answers (%)",
            xaxis={'tickangle': 45},
            showlegend=True,
            height=600,
            plot_bgcolor='white'
        )

        return fig

    def create_box_plot(self, stats_df: pd.DataFrame) -> go.Figure:
        """
        Create box plot showing distribution of answer position percentages.

        Args:
            stats_df: DataFrame containing position statistics

        Returns:
            go.Figure: Plotly figure object
        """
        fig = px.box(
            stats_df,
            x='dataset',
            y='percentage',
            color='model_name',
            color_discrete_map=self.colors,
            title='Distribution of Answer Position Frequencies for Low Performance Questions'
        )

        fig.update_layout(
            xaxis_title="Dataset",
            yaxis_title="Percentage of Answers (%)",
            xaxis={'tickangle': 45},
            showlegend=True,
            height=600,
            plot_bgcolor='white'
        )

        return fig

    def create_combined_plots(self, stats_df: pd.DataFrame, datasets: list[str],
                              suffix: str = "") -> tuple[go.Figure, go.Figure]:
        """
        Create scatter and box plots for specified datasets with detailed x-axis labels.

        Args:
            stats_df: DataFrame containing position statistics
            datasets: List of datasets to include in visualization
            suffix: Optional suffix for plot titles

        Returns:
            tuple[go.Figure, go.Figure]: Scatter plot and box plot figures
        """
        filtered_stats = stats_df[stats_df['dataset'].isin(datasets)].copy()

        # Create detailed x-axis labels combining dataset and question index
        filtered_stats['x_label'] = filtered_stats.apply(
            lambda row: f"{row['dataset']}_{row['sample_index']}",
            axis=1
        )

        scatter_fig = px.scatter(
            filtered_stats,
            x='x_label',
            y='percentage',
            color='model_name',
            color_discrete_map=self.colors,
            hover_data=['dataset', 'sample_index', 'answer_position'],
            title=f'Most Frequently Chosen Answer Positions{suffix}'
        )

        box_fig = px.box(
            filtered_stats,
            x='x_label',
            y='percentage',
            color='model_name',
            color_discrete_map=self.colors,
            title=f'Distribution of Answer Position Frequencies{suffix}'
        )

        for fig in [scatter_fig, box_fig]:
            fig.update_layout(
                xaxis_title="Dataset_QuestionIndex",
                yaxis_title="Percentage of Answers (%)",
                xaxis={
                    'tickangle': 45,
                    'tickmode': 'array',
                    'ticktext': filtered_stats['x_label'].unique(),
                    'tickvals': list(range(len(filtered_stats['x_label'].unique()))),
                    'showticklabels': True
                },
                showlegend=True,
                height=800,
                width=1200,
                margin=dict(b=150),  # Increase bottom margin for labels
                plot_bgcolor='white'
            )

        return scatter_fig, box_fig

    def save_visualizations(self, output_dir: str = "visualization_results"):
        """
        Generate and save all visualizations, including both complete and random dataset plots.

        Args:
            output_dir: Directory to save the visualization files
        """
        os.makedirs(output_dir, exist_ok=True)

        # Load and process data
        print("Loading data...")
        df = self.load_all_results()

        if df.empty:
            print("No data found in the results directory.")
            return

        print("Calculating position statistics...")
        position_stats = self.calculate_position_stats(df)

        # Create and save visualizations
        print("Creating scatter plot...")
        scatter_fig = self.create_scatter_plot(position_stats)
        scatter_fig.write_html(os.path.join(output_dir, "position_scatter.html"))

        print("Creating box plot...")
        box_fig = self.create_box_plot(position_stats)
        box_fig.write_html(os.path.join(output_dir, "position_box.html"))

        # Save processed statistics
        print("Saving processed statistics...")
        position_stats.to_parquet(os.path.join(output_dir, "position_statistics.parquet"))

        # Create visualizations for random datasets
        unique_datasets = position_stats['dataset'].unique()
        if len(unique_datasets) >= 3:
            random_datasets = np.random.choice(unique_datasets, size=3, replace=False)
            print(f"\nCreating visualizations for random datasets: {random_datasets}")

            random_scatter_fig, random_box_fig = self.create_combined_plots(
                position_stats,
                random_datasets,
                suffix=" (Random Sample)"
            )

            random_scatter_fig.write_html(os.path.join(output_dir, "position_scatter_random.html"))
            random_box_fig.write_html(os.path.join(output_dir, "position_box_random.html"))

            # Save the random dataset selection for reference
            with open(os.path.join(output_dir, "random_datasets.txt"), "w") as f:
                f.write("\n".join(random_datasets))

        print(f"\nAll visualizations saved to {output_dir}")


if __name__ == "__main__":
    visualizer = LowPerformanceVisualizer()
    visualizer.save_visualizations()