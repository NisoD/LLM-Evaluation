import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import random


class DataPreprocessor:
    def __init__(self, base_results_dir: str = "../app/results_local"):
        """Initialize preprocessor with directory path and color scheme."""
        self.base_dir = Path(base_results_dir)
        self.colors = {
            'Meta-Llama-3-8B-Instruct': '#2ca02c',
            'Llama-3.2-3B-Instruct': '#d62728',
            'Mistral-7B-Instruct-v0.3': '#9467bd',
            'Llama-3.2-1B-Instruct': '#17becf',
            'OLMoE-1B-7B-0924-Instruct': '#ff7f0e'
        }

    def process_data(self, output_dir: str = "processed_data"):
        """Process and save visualizations and preprocessed data."""
        os.makedirs(output_dir, exist_ok=True)

        print("Loading full dataset...")
        full_df = self._load_results()

        if full_df.empty:
            print("No data found in results directory.")
            return

        print("Generating main visualizations...")
        self._create_main_visualizations(full_df, output_dir)

        print("Creating sample datasets...")
        self._create_sample_datasets(full_df, output_dir)

    def _load_results(self) -> pd.DataFrame:
        """Load and combine all parquet files from results directory."""
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

    def _calculate_position_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate position statistics with essential information."""
        stats = (
            df.groupby(['model_name', 'dataset', 'sample_index', 'closest_answer'])
            .agg({
                'generated_text': 'first',
                'ground_truth': 'first',
                'score': 'first'
            })
            .reset_index()
        )

        # Calculate counts and percentages
        counts = df.groupby(
            ['model_name', 'dataset', 'sample_index', 'closest_answer']
        ).size().reset_index(name='count')

        stats = stats.merge(counts, on=['model_name', 'dataset', 'sample_index', 'closest_answer'])
        stats['total'] = stats.groupby(['model_name', 'dataset', 'sample_index'])['count'].transform('sum')
        stats['percentage'] = (stats['count'] / stats['total'] * 100)

        return stats

    def _create_model_plot(self, stats_df: pd.DataFrame, model_name: str) -> go.Figure:
        """Create visualization for a specific model."""
        medians = stats_df.groupby('dataset')['percentage'].median().sort_values()
        sorted_datasets = medians.index.tolist()

        fig = go.Figure()
        fig.add_trace(go.Box(
            y=stats_df['dataset'],
            x=stats_df['percentage'],
            name=model_name,
            boxpoints='all',
            jitter=0.8,
            pointpos=0,
            marker=dict(
                color=self.colors.get(model_name, '#000000'),
                size=4,
                opacity=0.7
            ),
            line=dict(width=1)
        ))

        fig.update_layout(
            title=f"Answer Distribution for {model_name}",
            xaxis_title="Percentage of Answers (%)",
            yaxis_title="Dataset",
            yaxis=dict(
                categoryorder='array',
                categoryarray=sorted_datasets,
                tickangle=0
            ),
            height=max(800, len(sorted_datasets) * 30),
            width=1000,
            margin=dict(l=200, r=50, t=100, b=50),
            plot_bgcolor='white'
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        fig.update_yaxes(showgrid=False)

        return fig

    def _create_main_visualizations(self, df: pd.DataFrame, output_dir: str):
        """Create and save main visualizations for all data."""
        stats = self._calculate_position_stats(df)

        for model in df['model_name'].unique():
            model_stats = stats[stats['model_name'] == model]
            fig = self._create_model_plot(model_stats, model)
            fig.write_html(os.path.join(output_dir, f"{model.replace('/', '_')}_full.html"))

        # Save overall statistics
        stats.to_parquet(os.path.join(output_dir, "full_statistics.parquet"))

    def _create_sample_datasets(self, df: pd.DataFrame, output_dir: str):
        """Create and save sample datasets for interactive analysis."""
        samples_dir = os.path.join(output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)

        required_columns = [
            'model_name', 'dataset', 'sample_index', 'closest_answer',
            'generated_text', 'ground_truth', 'score', 'percentage'
        ]

        for i in range(5):
            sample_datasets = random.sample(list(df['dataset'].unique()), 5)
            sample_df = df[df['dataset'].isin(sample_datasets)].copy()

            sample_stats = self._calculate_position_stats(sample_df)

            # Save minimal required data
            interactive_data = sample_stats[required_columns].copy()
            interactive_data.to_parquet(
                os.path.join(samples_dir, f"sample_{i}.parquet"),
                compression='snappy'
            )


def main():
    preprocessor = DataPreprocessor()
    preprocessor.process_data()


if __name__ == "__main__":
    main()