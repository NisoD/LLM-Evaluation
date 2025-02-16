import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import os
from pathlib import Path


class ModelPerformanceCollector:
    """Collects template performance data and generates histograms per model."""

    def __init__(self, base_results_dir: str):
        """Initialize collector with output directory."""
        self.results_dir = Path(base_results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.model_data = {}  # Simple dict instead of defaultdict

    def add_batch_data(self, df: pd.DataFrame, model_name: str) -> None:
        """Add performance data from a batch of evaluations."""
        if df.empty:
            return

        template_scores = df.groupby('template')['score'].mean() * 100

        if model_name not in self.model_data:
            self.model_data[model_name] = {
                'scores': list(template_scores.values),
                'count': len(df)
            }
        else:
            self.model_data[model_name]['scores'].extend(template_scores.values)
            self.model_data[model_name]['count'] += len(df)

    def generate_histograms(self) -> None:
        """Create histogram for each model's performance data."""
        for model_name, data in self.model_data.items():
            plt.figure(figsize=(12, 6))

            counts, edges, patches = plt.hist(
                data['scores'],
                bins=np.linspace(0, 100, 21),
                edgecolor='black',
                color='lightblue',
                alpha=0.7
            )

            # Add count labels
            for count, patch in zip(counts, patches):
                if count > 0:
                    height = patch.get_height()
                    center = patch.get_x() + patch.get_width() / 2
                    plt.text(center, height, f'{int(count)}',
                             ha='center', va='bottom', color='blue')

            plt.grid(True, alpha=0.3)
            plt.xlabel('Percentage of Templates with Correct Predictions')
            plt.ylabel('Number of Examples')

            model_display_name = model_name.split('/')[-1]
            plt.title(f'Template Performance Distribution - {model_display_name}\n'
                      f'Total Examples: {data["count"]:,}')

            # Save to current directory
            plt.savefig(f'{model_display_name}_histogram.png',
                        bbox_inches='tight', dpi=300)
            plt.close()

            print(f"📊 Generated histogram for {model_display_name}")