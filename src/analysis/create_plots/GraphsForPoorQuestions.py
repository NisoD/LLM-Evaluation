import pandas as pd
import plotly.express as px
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class VisualizationConfig:
    base_dir: Path
    output_dir: Path
    shots: str = "Shots_0"


class PositionVisualizer:
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Define model/dataset mappings based on directory structure
        self.models = [
            'allenai_OLMoE-1B-7B-0924-Instruct',
            'meta-llama_Meta-Llama-3-8B-Instruct'
        ]

        self.datasets = [
            'ai2_arc.arc_challenge',
            'ai2_arc.arc_easy',
            'hellaswag',
            'openbook_qa',
            'social_iqa',
            'mmlu.global_facts',
            'mmlu.sociology',
            'mmlu.econometrics',
            'mmlu.high_school_geography',
        ]

    def get_distribution_path(self, model: str, dataset: str) -> Path:
        """Constructs path to position distribution parquet file."""
        return (self.config.base_dir / self.config.shots / model / dataset /
                "enumerator_analysis" / "position_distribution.parquet")

    def read_distribution_data(self, model: str, dataset: str) -> Optional[pd.DataFrame]:
        """Reads and enriches position distribution data with dataset-specific question IDs."""
        file_path = self.get_distribution_path(model, dataset)
        if not file_path.exists():
            return None

        df = pd.read_parquet(file_path)
        df['question_id'] = f"{dataset}_q" + df['sample_index'].astype(str)
        df['model'] = model
        df['dataset'] = dataset
        return df

    def create_visualization(self, data: pd.DataFrame, color_by: str,
                             title: str, output_name: str) -> None:
        """Creates and saves an interactive scatter plot visualization."""
        fig = px.scatter(
            data,
            x='question_id',  # Changed from sample_index to question_id
            y='percentage',
            color=color_by,
            title=title,
            labels={
                'question_id': 'Dataset Question ID',
                'percentage': 'Answer Position Frequency (%)'
            },
            custom_data=['position', 'frequency', 'total_count', 'dataset']
        )

        # Configure hover information
        fig.update_traces(
            marker=dict(size=10),
            hovertemplate=(
                "Dataset: %{customdata[3]}<br>"
                "Question: %{x}<br>"
                "Answer Position: %{customdata[0]}<br>"
                "Frequency: %{customdata[1]} / %{customdata[2]}<br>"
                "Percentage: %{y:.1f}%"
                "<extra></extra>"
            )
        )

        # Style the plot
        fig.update_layout(
            xaxis=dict(
                title_text="Dataset Question ID",
                gridcolor='lightgrey',
                showgrid=True,
                zeroline=False,
                tickangle=45  # Angled labels for better readability
            ),
            yaxis=dict(
                title_text="Answer Position Frequency (%)",
                range=[0, 100],
                gridcolor='lightgrey',
                showgrid=True,
                zeroline=False
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)'
            ),
            margin=dict(b=100)  # Added margin for angled labels
        )

        output_path = self.config.output_dir / f"{output_name}.html"
        fig.write_html(str(output_path))

    def visualize_model_comparison(self, dataset: str) -> None:
        """Creates comparison visualization of all models for a given dataset."""
        distributions = []
        for model in self.models:
            df = self.read_distribution_data(model, dataset)
            if df is not None:
                distributions.append(df)

        if distributions:
            plot_data = pd.concat(distributions, ignore_index=True)
            self.create_visualization(
                data=plot_data,
                color_by='model',
                title=f'Model Comparison - {dataset}',
                output_name=f'model_comparison_{dataset.replace(".", "_")}'
            )

    def visualize_dataset_comparison(self, model: str) -> None:
        """Creates comparison visualization of all datasets for a given model."""
        distributions = []
        for dataset in self.datasets:
            df = self.read_distribution_data(model, dataset)
            if df is not None:
                distributions.append(df)

        if distributions:
            plot_data = pd.concat(distributions, ignore_index=True)
            self.create_visualization(
                data=plot_data,
                color_by='dataset',
                title=f'Dataset Comparison - {model}',
                output_name=f'dataset_comparison_{model}'
            )


def main():
    config = VisualizationConfig(
        base_dir=Path('/home/daniel/PycharmProjects/LLM-Evaluation/src/analysis/app/results_local'),
        output_dir=Path('/home/daniel/PycharmProjects/LLM-Evaluation/src/analysis/app/visualizations')
    )

    visualizer = PositionVisualizer(config)

    # Create model comparisons for each dataset
    for dataset in visualizer.datasets:
        visualizer.visualize_model_comparison(dataset)

    # Create dataset comparisons for each model
    for model in visualizer.models:
        visualizer.visualize_dataset_comparison(model)


if __name__ == "__main__":
    main()