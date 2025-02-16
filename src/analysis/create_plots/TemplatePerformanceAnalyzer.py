import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List
import os


class TemplatePerformanceAnalyzer:
    def __init__(self, base_results_dir: str):
        self.base_results_dir = base_results_dir
        self.bin_edges = np.linspace(0, 100, 21)  # 20 bins from 0 to 100

    def calculate_template_performance(self, df: pd.DataFrame) -> pd.Series:
        return df.groupby('template')['score'].mean() * 100

    def create_histogram(
            self,
            df: pd.DataFrame,
            model_name: Optional[str] = None,
            shots_selected: Optional[int] = None
    ) -> None:
        template_performance = self.calculate_template_performance(df)

        plt.figure(figsize=(12, 6))
        counts, edges, patches = plt.hist(
            template_performance,
            bins=self.bin_edges,
            edgecolor='black',
            color='lightblue',
            alpha=0.7
        )

        # Add count labels above bars
        for count, patch in zip(counts, patches):
            if count > 0:
                height = patch.get_height()
                center = patch.get_x() + patch.get_width() / 2
                plt.text(center, height, f'{int(count)}',
                         ha='center', va='bottom', color='blue')

        plt.grid(True, alpha=0.3)
        plt.xlabel('Percentage of Templates with Correct Predictions')
        plt.ylabel('Number of Examples')
        plt.title('Template Performance Distribution')

        if model_name and shots_selected is not None:
            save_dir = os.path.join(self.base_results_dir, 'template_analysis')
            os.makedirs(save_dir, exist_ok=True)
            filename = f'template_performance_{model_name.replace("/", "_")}_{shots_selected}shots.png'
            plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=300)

        plt.close()

    def analyze_performance(
            self,
            df: pd.DataFrame,
            model_name: Optional[str] = None,
            shots_selected: Optional[int] = None,
            dataset_filter: Optional[List[str]] = None
    ) -> dict:
        if dataset_filter:
            df = df[df['dataset'].isin(dataset_filter)]

        if df.empty:
            raise ValueError("No data remains after filtering")

        template_performance = self.calculate_template_performance(df)

        self.create_histogram(df, model_name, shots_selected)

        return {
            'mean_performance': template_performance.mean(),
            'median_performance': template_performance.median(),
            'std_performance': template_performance.std(),
            'num_templates': len(template_performance),
            'total_examples': len(df)
        }