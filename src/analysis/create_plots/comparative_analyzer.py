#!/usr/bin/env python3
"""
Analyze and visualize position bias patterns in LLM responses across different models and datasets.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


@dataclass(frozen=True)
class AnalysisConfig:
    shots: List[int]
    models: List[str]
    datasets: List[str]
    base_dir: str = "../app/results_local"

    @classmethod
    def create_default(cls) -> 'AnalysisConfig':
        return cls(
            shots=[0, 5],
            models=[
                'meta-llama/Llama-3.2-1B-Instruct',
                'allenai/OLMoE-1B-7B-0924-Instruct',
                'meta-llama/Meta-Llama-3-8B-Instruct',
                'meta-llama/Llama-3.2-3B-Instruct',
                'mistralai/Mistral-7B-Instruct-v0.3',
            ],
            datasets=[
                "ai2_arc.arc_challenge",
                "ai2_arc.arc_easy",
                "hellaswag",
                "openbook_qa",
                "social_iqa",
                "mmlu.global_facts",
                "mmlu.sociology",
                "mmlu.econometrics",
                "mmlu.high_school_geography",
            ]
        )


@dataclass
class ModelAnalysis:
    name: str
    positions: pd.DataFrame
    distribution: Optional[pd.DataFrame] = None


class PositionBiasAnalyzer:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.base_dir = Path(config.base_dir)
        self.logger = logging.getLogger(__name__)

    def load_model_data(self, model_name: str, dataset: str, shots: int) -> Optional[ModelAnalysis]:
        model_dir = self.base_dir / f"Shots_{shots}" / model_name.replace('/', '_')
        dataset_dir = model_dir / dataset.replace('/', '_') / 'enumerator_analysis'

        position_path = dataset_dir / 'position_distribution.parquet'
        distribution_path = dataset_dir / 'answer_distribution_matrix.parquet'

        if not position_path.exists():
            self.logger.warning(f"No position data found for {model_name} on {dataset}")
            return None

        try:
            positions_df = pd.read_parquet(position_path)
            positions_df['model'] = model_name
            positions_df['dataset'] = dataset
            positions_df['shots'] = shots

            distribution_df = None
            if distribution_path.exists():
                distribution_df = pd.read_parquet(distribution_path)

            return ModelAnalysis(
                name=model_name,
                positions=positions_df,
                distribution=distribution_df
            )
        except Exception as e:
            self.logger.error(f"Error loading data for {model_name} on {dataset}: {e}")
            return None

    def generate_model_dataset_comparison(
            self,
            model_name: str,
            datasets: List[str],
            shots: int,
            output_dir: Path
    ) -> None:
        self.logger.info(f"Generating dataset comparison for {model_name}")
        analyses = [
            self.load_model_data(model_name, dataset, shots)
            for dataset in datasets
        ]
        valid_analyses = [a for a in analyses if a is not None]

        if not valid_analyses:
            self.logger.warning(f"No valid data found for model {model_name}")
            return

        combined_df = pd.concat([a.positions for a in valid_analyses], ignore_index=True)

        fig = self._create_position_scatter(
            df=combined_df,
            color_column='dataset',
            title=f"Position Bias Analysis: {model_name} ({shots} shots)",
            legend_title="Dataset"
        )

        output_path = output_dir / f"{model_name.replace('/', '_')}_datasets_comparison.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        self.logger.info(f"Saved comparison plot to {output_path}")

    def generate_dataset_model_comparison(
            self,
            dataset: str,
            models: List[str],
            shots: int,
            output_dir: Path
    ) -> None:
        self.logger.info(f"Generating model comparison for {dataset}")
        analyses = [
            self.load_model_data(model, dataset, shots)
            for model in models
        ]
        valid_analyses = [a for a in analyses if a is not None]

        if not valid_analyses:
            self.logger.warning(f"No valid data found for dataset {dataset}")
            return

        combined_df = pd.concat([a.positions for a in valid_analyses], ignore_index=True)

        fig = self._create_position_scatter(
            df=combined_df,
            color_column='model',
            title=f"Position Bias Analysis: {dataset} ({shots} shots)",
            legend_title="Model"
        )

        output_path = output_dir / f"{dataset.replace('/', '_')}_models_comparison.html"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        self.logger.info(f"Saved comparison plot to {output_path}")

    def generate_focused_comparisons(self, output_dir: Path) -> None:
        """Generate a set of focused comparisons for specific models and datasets."""
        # Compare Mistral across key datasets
        mistral_datasets = [
            "ai2_arc.arc_challenge",
            "ai2_arc.arc_easy",
            "mmlu.global_facts",
            "mmlu.sociology"
        ]

        self.generate_model_dataset_comparison(
            model_name='mistralai/Mistral-7B-Instruct-v0.3',
            datasets=mistral_datasets,
            shots=0,
            output_dir=output_dir / 'mistral_comparisons'
        )

        # Compare models on ARC Challenge
        arc_models = [
            'meta-llama/Llama-3.2-1B-Instruct',
            'mistralai/Mistral-7B-Instruct-v0.3',
            'meta-llama/Meta-Llama-3-8B-Instruct'
        ]

        self.generate_dataset_model_comparison(
            dataset="ai2_arc.arc_challenge",
            models=arc_models,
            shots=0,
            output_dir=output_dir / 'arc_comparisons'
        )

        # Compare MMLU subjects across models
        mmlu_datasets = [
            "mmlu.global_facts",
            "mmlu.sociology",
            "mmlu.high_school_geography"
        ]

        self.generate_model_dataset_comparison(
            model_name='mistralai/Mistral-7B-Instruct-v0.3',
            datasets=mmlu_datasets,
            shots=5,
            output_dir=output_dir / 'mmlu_comparisons'
        )

    def _create_position_scatter(
            self,
            df: pd.DataFrame,
            color_column: str,
            title: str,
            legend_title: str
    ) -> go.Figure:
        fig = px.scatter(
            df,
            x='position',
            y='percentage',
            color=color_column,
            title=title,
            labels={
                'position': 'Answer Position',
                'percentage': 'Selection Frequency (%)',
                color_column: legend_title
            },
            hover_data=['accuracy', 'sample_index']
        )

        fig.update_layout(
            xaxis=dict(tickmode='linear', dtick=1),
            yaxis=dict(range=[0, 100]),
            plot_bgcolor='white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            width=1000,
            height=600
        )

        return fig


def setup_logging(level: int = logging.INFO) -> None:
    """Configure logging settings."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze position bias in LLM responses across models and datasets."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="../app/results_local",
        help="Base directory containing analysis results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="position_bias_plots",
        help="Directory to save visualization outputs"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_args()
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger(__name__)

    try:
        config = AnalysisConfig.create_default()
        output_dir = Path(args.output_dir)

        analyzer = PositionBiasAnalyzer(config)
        analyzer.generate_focused_comparisons(output_dir)

        logger.info("Analysis complete. Results saved to %s", output_dir)

    except Exception as e:
        logger.error("Error during analysis: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()