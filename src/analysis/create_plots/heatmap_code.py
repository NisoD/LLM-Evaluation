from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class HeatmapConfig:
    min_height_per_row: int = 20
    base_width: int = 1000
    colorscale: str = 'RdYlBu_r'


class HeatmapVisualizer:
    def __init__(self, results_dir: str = "../app/results_local"):
        self.results_dir = Path(results_dir)
        self.config = HeatmapConfig()

    def _extract_choice_pattern(self, row: pd.Series) -> str:
        """Creates a sortable pattern string from choice percentages."""
        choice_cols = [col for col in row.index if col.startswith('Choice')]
        choices = [(row[col], i + 1) for i, col in enumerate(choice_cols)]
        sorted_choices = sorted(choices, reverse=True)
        return '_'.join(f"{pct:.1f}_{pos}" for pct, pos in sorted_choices)

    def _aggregate_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Groups questions by their choice distribution pattern and sorts by frequency."""
        choice_cols = [col for col in df.columns if col.startswith('Choice')]

        # Create pattern for each row and count occurrences
        df['pattern'] = df.apply(self._extract_choice_pattern, axis=1)
        pattern_counts = df['pattern'].value_counts()

        # Add frequency information and sort
        df['pattern_frequency'] = df['pattern'].map(pattern_counts)
        return df.sort_values(['pattern_frequency', 'pattern'], ascending=[False, True])

    def create_heatmap(self, shots: int = None) -> Dict[str, go.Figure]:
        """Creates frequency-sorted heatmaps for each model."""
        shot_pattern = f"Shots_{shots}" if shots is not None else "Shots_*"
        heatmaps = {}

        for parquet_file in self.results_dir.glob(f"{shot_pattern}/**/low_performance_questions.parquet"):
            model = parquet_file.parents[1].name
            dataset = parquet_file.parents[0].name
            df = pd.read_parquet(parquet_file)

            # Prepare data for heatmap
            choice_cols = [col for col in df.columns if col.startswith('choice_') and col.endswith('_pct')]
            heatmap_data = df[['sample_index'] + choice_cols].copy()

            # Rename columns and create question identifiers
            heatmap_data.columns = ['sample_index'] + [f'Choice {i + 1}' for i in range(len(choice_cols))]
            heatmap_data['question_id'] = f"{dataset}_{heatmap_data['sample_index'].astype(str)}"

            # Sort by pattern frequency
            sorted_data = self._aggregate_patterns(heatmap_data)

            # Create heatmap
            fig = go.Figure(data=go.Heatmap(
                z=sorted_data[[col for col in sorted_data.columns if col.startswith('Choice')]].values,
                x=[f'Choice {i + 1}' for i in range(len(choice_cols))],
                y=sorted_data['question_id'],
                colorscale=self.config.colorscale,
                zmin=0,
                zmax=100,
                hovertemplate=(
                    '<b>Question:</b> %{y}<br>'
                    '<b>%{x}:</b> %{z:.1f}%<br>'
                    '<b>Pattern Frequency:</b> %{customdata}<br>'
                    '<extra></extra>'
                ),
                customdata=sorted_data['pattern_frequency'].values[:, None].repeat(len(choice_cols), axis=1)
            ))

            title_prefix = f"{shots}-Shot: " if shots is not None else "Combined: "
            fig.update_layout(
                title=f'{title_prefix}Choice Distribution Heatmap - {model}',
                xaxis_title='Answer Choice',
                yaxis_title='Question ID',
                height=max(800, len(sorted_data) * self.config.min_height_per_row),
                width=self.config.base_width,
                yaxis=dict(
                    tickfont=dict(size=10),
                    showgrid=False
                ),
                xaxis=dict(
                    tickfont=dict(size=12),
                    showgrid=False
                )
            )

            heatmaps[model] = fig

        return heatmaps

    def save_visualizations(self, output_dir: str = "heatmaps"):
        """Generates and saves heatmaps for all configurations."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        for shots in [0, 5, None]:
            shot_label = 'combined' if shots is None else f'{shots}_shot'
            heatmaps = self.create_heatmap(shots)

            for model, fig in heatmaps.items():
                output_file = output_path / f"heatmap_{model}_{shot_label}.html"
                fig.write_html(output_file)

        print(f"Heatmaps saved to {output_dir}")


if __name__ == "__main__":
    visualizer = HeatmapVisualizer()
    visualizer.save_visualizations()