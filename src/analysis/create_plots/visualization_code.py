# Graphs.py
import os

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# from main import THRESHOLD

THRESHOLD = 0.1
class SimplifiedVisualizer:
    def __init__(self, base_results_dir: str = "../app/results_local"):
        """Initialize the visualizer with paths and colors for each model."""
        self.base_dir = Path(base_results_dir)
        self.colors = {
            'meta-llama_Meta-Llama-3-8B-Instruct': '#2ca02c',
            'allenai_OLMoE-1B-7B-0924-Instruct': '#ff7f0e'
        }
        self.threshold = 0.1  # Using the imported threshold

    def load_data(self) -> pd.DataFrame:
        """Load and combine all parquet files with the specific threshold."""
        all_data = []

        # Walk through directory structure: shots -> model -> dataset
        for shots_dir in self.base_dir.glob("Shots_*"):
            shots = int(shots_dir.name.split('_')[1])

            for model_dir in shots_dir.glob("*"):
                model_name = model_dir.name

                for dataset_dir in model_dir.glob("*"):
                    dataset = dataset_dir.name

                    # Look for parquet file with threshold in name
                    parquet_file = dataset_dir / f'low_performance_questions_{THRESHOLD}.parquet'
                    if parquet_file.exists():
                        # Read the file and add metadata columns
                        df = pd.read_parquet(parquet_file)
                        df['shots'] = shots
                        df['model_name'] = model_name
                        df['dataset'] = dataset
                        all_data.append(df)

        # Combine all dataframes or return empty dataframe if none found
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def calculate_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate choice distributions and percentages."""
        # Group by relevant columns and count occurrences
        stats = (
            df.groupby(['shots', 'model_name', 'dataset', 'sample_index', 'chosen_position'])
            .size()
            .reset_index(name='count')
        )

        # Calculate total responses for each question
        stats['total_responses'] = stats.groupby(
            ['shots', 'model_name', 'dataset', 'sample_index']
        )['count'].transform('sum')

        # Calculate percentage for each choice
        stats['percentage'] = (stats['count'] / stats['total_responses'] * 100).round(2)

        return stats

    def create_side_by_side_heatmap(self, stats_df: pd.DataFrame, shots: int = None) -> go.Figure:
        """Create a side-by-side heatmap with choice columns sorted by percentage for each question."""
        # Filter by shots if specified
        if shots is not None:
            stats_df = stats_df[stats_df['shots'] == shots]

        # Get unique models
        models = sorted(stats_df['model_name'].unique())

        # Group and sort by dataset and sample_index
        all_questions = []
        for dataset in sorted(stats_df['dataset'].unique()):
            for idx in sorted(stats_df[stats_df['dataset'] == dataset]['sample_index'].unique()):
                all_questions.append((dataset, idx))

        # Create row labels
        row_labels = [f"{ds}_{idx}" for ds, idx in all_questions]

        # Get all possible choices
        all_choices = sorted(stats_df['chosen_position'].unique())

        # Create data matrices for each model
        model_matrices = {}
        for model in models:
            # Initialize with zeros
            matrix = np.zeros((len(all_questions), len(all_choices)))

            # Fill with actual percentages
            for i, (dataset, idx) in enumerate(all_questions):
                model_data = stats_df[(stats_df['model_name'] == model) &
                                      (stats_df['dataset'] == dataset) &
                                      (stats_df['sample_index'] == idx)]

                for j, choice in enumerate(all_choices):
                    choice_data = model_data[model_data['chosen_position'] == choice]
                    if not choice_data.empty:
                        matrix[i, j] = choice_data['percentage'].values[0]

            model_matrices[model] = matrix

        # Create figure with 2 columns for the models
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[f"Model: {models[0].split('_')[-1]}",
                            f"Model: {models[1].split('_')[-1]}"],
            horizontal_spacing=0.01
        )

        # Add heatmap for each model
        for i, model in enumerate(models):
            # For each question, we need:
            # 1. The original choice positions
            # 2. Sorted percentages
            # 3. Mapping from original to sorted

            # Prepare data for sorted heatmap display
            sorted_matrix = np.zeros_like(model_matrices[model])
            col_labels_by_row = []
            hover_text = []

            # Process each question (row)
            for row_idx, (ds, question_idx) in enumerate(all_questions):
                row_data = model_matrices[model][row_idx]

                # Create list of (original_choice_idx, percentage) tuples
                choice_percentage_pairs = [(j, row_data[j]) for j in range(len(all_choices))]

                # Sort by percentage descending
                sorted_pairs = sorted(choice_percentage_pairs, key=lambda x: -x[1])

                # Extract sorted percentages and original choice indices
                sorted_percentages = [pair[1] for pair in sorted_pairs]
                original_choice_indices = [pair[0] for pair in sorted_pairs]

                # Store the sorted percentages in our result matrix
                sorted_matrix[row_idx] = sorted_percentages

                # Create hover text with correct percentages and choice mappings
                row_hover = []
                for orig_idx in original_choice_indices:
                    choice = all_choices[orig_idx]
                    percentage = row_data[orig_idx]
                    row_hover.append(
                        f"Model: {model.split('_')[-1]}<br>" +
                        f"Dataset: {ds}<br>" +
                        f"Question: {question_idx}<br>" +
                        f"Original Choice: {choice}<br>" +
                        f"Percentage: {percentage:.1f}%"
                    )
                hover_text.append(row_hover)

                # Store the choice labels in sorted order for this row
                # (these won't be used for display but for reference)
                sorted_choice_labels = [all_choices[idx] for idx in original_choice_indices]
                col_labels_by_row.append(sorted_choice_labels)

            # Create column labels showing position in sorted order
            sorted_col_labels = [f'Rank {i + 1}' for i in range(len(all_choices))]

            # Add the heatmap
            fig.add_trace(
                go.Heatmap(
                    z=sorted_matrix,
                    x=sorted_col_labels,
                    y=row_labels,
                    text=np.round(sorted_matrix, 1),
                    texttemplate="%{text}%",
                    hoverinfo="text",
                    hovertext=hover_text,
                    colorscale='Blues',
                    showscale=i == len(models) - 1,
                    zmin=0,
                    zmax=100,
                ),
                row=1, col=i + 1
            )

        # Update layout
        title_suffix = f" ({shots}-shot)" if shots is not None else ""
        fig.update_layout(
            title=f'Answer Choice Distribution{title_suffix} - Threshold: {THRESHOLD}',
            height=max(800, len(row_labels) * 25),  # Scale height based on number of rows
            width=1600
        )

        # Update axes
        for col in range(1, 3):
            fig.update_xaxes(title_text="Choice Position (By Rank)", row=1, col=col)
            # Only show y-axis labels for the first heatmap
            if col > 1:
                fig.update_yaxes(showticklabels=False, row=1, col=col)

        return fig

    def create_individual_model_heatmaps(self, stats_df: pd.DataFrame, shots: int = None) -> dict:
        """
        Create individual heatmaps for each model, showing only non-zero rows.
        Returns a dictionary of model names to figure objects.
        """
        # Filter by shots if specified
        if shots is not None:
            stats_df = stats_df[stats_df['shots'] == shots]

        # Get unique models
        models = sorted(stats_df['model_name'].unique())

        # Get all possible choices
        all_choices = sorted(stats_df['chosen_position'].unique())

        # Dictionary to store figures for each model
        model_figures = {}

        # Process each model separately
        for model in models:
            model_data = stats_df[stats_df['model_name'] == model]

            # Find all dataset-question pairs for this model
            model_questions = []
            for dataset in sorted(model_data['dataset'].unique()):
                for idx in sorted(model_data[model_data['dataset'] == dataset]['sample_index'].unique()):
                    model_questions.append((dataset, idx))

            # Skip if no questions for this model
            if not model_questions:
                continue

            # Create matrix for this model
            matrix = np.zeros((len(model_questions), len(all_choices)))

            # Fill with actual percentages
            for i, (dataset, idx) in enumerate(model_questions):
                q_data = model_data[(model_data['dataset'] == dataset) &
                                    (model_data['sample_index'] == idx)]

                for j, choice in enumerate(all_choices):
                    choice_data = q_data[q_data['chosen_position'] == choice]
                    if not choice_data.empty:
                        matrix[i, j] = choice_data['percentage'].values[0]

            # Create row labels
            row_labels = [f"{ds}_{idx}" for ds, idx in model_questions]

            # Identify non-zero rows (rows with at least one non-zero value)
            non_zero_rows = [i for i, row in enumerate(matrix) if np.any(row > 0)]

            # Skip if no non-zero rows
            if not non_zero_rows:
                continue

            # Filter matrix and labels to only include non-zero rows
            filtered_matrix = matrix[non_zero_rows]
            filtered_labels = [row_labels[i] for i in non_zero_rows]
            filtered_questions = [model_questions[i] for i in non_zero_rows]

            # Create sorted data for heatmap
            sorted_matrix = np.zeros_like(filtered_matrix)
            hover_text = []

            # Process each question (row)
            for row_idx, (ds, question_idx) in enumerate(filtered_questions):
                row_data = filtered_matrix[row_idx]

                # Create list of (original_choice_idx, percentage) tuples
                choice_percentage_pairs = [(j, row_data[j]) for j in range(len(all_choices))]

                # Sort by percentage descending
                sorted_pairs = sorted(choice_percentage_pairs, key=lambda x: -x[1])

                # Extract sorted percentages and original choice indices
                sorted_percentages = [pair[1] for pair in sorted_pairs]
                original_choice_indices = [pair[0] for pair in sorted_pairs]

                # Store the sorted percentages in our result matrix
                sorted_matrix[row_idx] = sorted_percentages

                # Create hover text with correct percentages and choice mappings
                row_hover = []
                for orig_idx in original_choice_indices:
                    choice = all_choices[orig_idx]
                    percentage = row_data[orig_idx]
                    row_hover.append(
                        f"Model: {model.split('_')[-1]}<br>" +
                        f"Dataset: {ds}<br>" +
                        f"Question: {question_idx}<br>" +
                        f"Original Choice: {choice}<br>" +
                        f"Percentage: {percentage:.1f}%"
                    )
                hover_text.append(row_hover)

            # Create column labels showing position in sorted order
            sorted_col_labels = [f'Rank {i + 1}' for i in range(len(all_choices))]

            # Create the figure
            fig = go.Figure()

            # Add the heatmap
            fig.add_trace(
                go.Heatmap(
                    z=sorted_matrix,
                    x=sorted_col_labels,
                    y=filtered_labels,
                    text=np.round(sorted_matrix, 1),
                    texttemplate="%{text}%",
                    hoverinfo="text",
                    hovertext=hover_text,
                    colorscale='Blues',
                    zmin=0,
                    zmax=100,
                )
            )

            # Update layout
            model_display_name = model.split('_')[-1]
            title_suffix = f" ({shots}-shot)" if shots is not None else ""
            fig.update_layout(
                title=f'Model: {model_display_name}{title_suffix} - Threshold: {THRESHOLD} (Sorted by Percentage)',
                xaxis_title="Choice Position (By Rank)",
                yaxis_title="Question (Dataset_ID)",
                height=max(400, len(filtered_labels) * 20),  # Scale height based on number of rows
                width=800
            )

            # Store the figure
            model_figures[model] = fig

        return model_figures

    def save_visualizations(self, output_dir: str = "simplified_visualizations"):
        """Generate and save visualizations."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        print("Loading data...")
        df = self.load_data()

        if df.empty:
            print("No data found in the results directory.")
            return

        print("Calculating statistics...")
        stats_df = self.calculate_stats(df)

        # Save statistics to parquet
        stats_file = os.path.join(output_dir, f"stats_{THRESHOLD}.parquet")
        stats_df.to_parquet(stats_file)
        print(f"Saved statistics to {stats_file}")

        # Create visualizations for specific shot configurations
        for shots in sorted(stats_df['shots'].unique()):
            shot_dir = os.path.join(output_dir, f"{shots}-shot")
            os.makedirs(shot_dir, exist_ok=True)
            print(f"Creating visualizations for {shots}-shot configuration...")

            # Create and save side-by-side heatmap
            heatmap_fig = self.create_side_by_side_heatmap(stats_df, shots)
            output_file = os.path.join(shot_dir, f"model_comparison_heatmap_{THRESHOLD}.html")
            heatmap_fig.write_html(output_file)
            print(f"Saved side-by-side heatmap to {output_file}")

            # Create and save individual model heatmaps
            individual_heatmaps = self.create_individual_model_heatmaps(stats_df, shots)
            for model, fig in individual_heatmaps.items():
                model_name = model.split('_')[-1]
                model_file = os.path.join(shot_dir, f"{model_name}_heatmap_{THRESHOLD}.html")
                fig.write_html(model_file)
                print(f"Saved individual heatmap for {model_name} to {model_file}")

            # Create and save model comparison plot
            comparison_fig = self.create_model_comparison_plot(stats_df, shots)
            comparison_file = os.path.join(shot_dir, f"most_frequent_choices_{THRESHOLD}.html")
            comparison_fig.write_html(comparison_file)
            print(f"Saved comparison plot to {comparison_file}")

        # Create combined visualizations with all shots
        print("Creating combined visualizations...")

        combined_heatmap = self.create_side_by_side_heatmap(stats_df)
        combined_heatmap_file = os.path.join(output_dir, f"combined_heatmap_{THRESHOLD}.html")
        combined_heatmap.write_html(combined_heatmap_file)
        print(f"Saved combined heatmap to {combined_heatmap_file}")

        # Create and save combined individual model heatmaps
        combined_individual_heatmaps = self.create_individual_model_heatmaps(stats_df)
        for model, fig in combined_individual_heatmaps.items():
            model_name = model.split('_')[-1]
            model_file = os.path.join(output_dir, f"{model_name}_combined_heatmap_{THRESHOLD}.html")
            fig.write_html(model_file)
            print(f"Saved combined individual heatmap for {model_name} to {model_file}")

        combined_comparison = self.create_model_comparison_plot(stats_df)
        combined_comparison_file = os.path.join(output_dir, f"combined_most_frequent_{THRESHOLD}.html")
        combined_comparison.write_html(combined_comparison_file)
        print(f"Saved combined comparison plot to {combined_comparison_file}")

        print(f"All visualizations saved to {output_dir}")
        return stats_df
    def create_model_comparison_plot(self, stats_df: pd.DataFrame, shots: int = None) -> go.Figure:
        """Create a comparison plot showing distribution of most frequent choices across configurations."""
        if shots is not None:
            stats_df = stats_df[stats_df['shots'] == shots]

        # Identify most frequent choice for each question/model/dataset
        most_frequent_choices = (
            stats_df.groupby(['sample_index', 'dataset', 'model_name', 'chosen_position'])
            ['percentage'].mean()  # Average percentage across configurations
            .reset_index()
            .sort_values('percentage', ascending=False)
            .groupby(['sample_index', 'dataset', 'model_name'])
            .first()  # Keep only the highest percentage choice
            .reset_index()
        )

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Distribution by Dataset (Box Plot)', 'Distribution by Dataset (Scatter)')
        )

        # Box plot
        for model in most_frequent_choices['model_name'].unique():
            model_data = most_frequent_choices[most_frequent_choices['model_name'] == model]
            fig.add_trace(
                go.Box(
                    y=model_data['percentage'],
                    x=model_data['dataset'],
                    name=model,
                    marker_color=self.colors[model]
                ),
                row=1, col=1
            )

        # Scatter plot with jitter
        for model in most_frequent_choices['model_name'].unique():
            model_data = most_frequent_choices[most_frequent_choices['model_name'] == model]
            for dataset in model_data['dataset'].unique():
                dataset_data = model_data[model_data['dataset'] == dataset]

                # Add random jitter to x-position
                jitter = np.random.normal(0, 0.1, size=len(dataset_data))
                dataset_index = list(model_data['dataset'].unique()).index(dataset)

                fig.add_trace(
                    go.Scatter(
                        x=[dataset_index + jitter[i] for i in range(len(dataset_data))],
                        y=dataset_data['percentage'],
                        mode='markers',
                        name=model,
                        showlegend=dataset == model_data['dataset'].iloc[0],  # Show legend only once per model
                        marker=dict(
                            color=self.colors[model],
                            size=8,
                            opacity=0.7
                        ),
                        text=dataset_data['sample_index'],  # Assign question indices
                        hovertemplate=(
                            "Dataset: %{x}<br>"
                            "Percentage: %{y:.2f}%<br>"
                            "Question Index: %{text}"
                        )
                    ),
                    row=1, col=2
                )

        # Add threshold line
        for col in [1, 2]:
            y_threshold = THRESHOLD * 100 if stats_df['percentage'].max() > 1 else THRESHOLD  # Ensure correct scaling

            fig.add_shape(
                type="line",
                x0=-0.5, y0=y_threshold,
                x1=len(most_frequent_choices['dataset'].unique()) - 0.5, y1=y_threshold,
                line=dict(color="red", width=2, dash="dash"),
                row=1, col=col
            )

        # Update layout
        title_suffix = f" ({shots}-shot)" if shots is not None else ""
        fig.update_layout(
            title=f'Distribution of Most Frequent Choices{title_suffix} - Threshold: {THRESHOLD}',
            height=600,
            width=1600,
            showlegend=True,
            xaxis_tickangle=45,
            xaxis2_tickangle=45
        )

        # Update axes
        fig.update_xaxes(title_text="Dataset", row=1, col=1)
        fig.update_xaxes(title_text="Dataset", row=1, col=2,
                         ticktext=most_frequent_choices['dataset'].unique(),
                         tickvals=list(range(len(most_frequent_choices['dataset'].unique()))))
        fig.update_yaxes(title_text="Percentage", row=1, col=1)
        fig.update_yaxes(title_text="Percentage", row=1, col=2)

        return fig
    def highlight_dataset_groups(self, fig):
        """Add alternating background colors to highlight different datasets."""
        # This function can be called after creating the basic heatmap
        # to add visual separation between datasets
        pass  # Implement if needed

    def save_visualizations(self, output_dir: str = "simplified_visualizations"):
        """Generate and save visualizations."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        print("Loading data...")
        df = self.load_data()

        if df.empty:
            print("No data found in the results directory.")
            return

        print("Calculating statistics...")
        stats_df = self.calculate_stats(df)

        # Save statistics to parquet
        stats_file = os.path.join(output_dir, f"stats_{THRESHOLD}.parquet")
        stats_df.to_parquet(stats_file)
        print(f"Saved statistics to {stats_file}")

        # Create visualizations for specific shot configurations
        for shots in sorted(stats_df['shots'].unique()):
            shot_dir = os.path.join(output_dir, f"{shots}-shot")
            os.makedirs(shot_dir, exist_ok=True)
            print(f"Creating visualizations for {shots}-shot configuration...")

            # Create and save side-by-side heatmap
            heatmap_fig = self.create_side_by_side_heatmap(stats_df, shots)
            output_file = os.path.join(shot_dir, f"model_comparison_heatmap_{THRESHOLD}.html")
            heatmap_fig.write_html(output_file)
            print(f"Saved side-by-side heatmap to {output_file}")

            # Create and save individual model heatmaps
            individual_heatmaps = self.create_individual_model_heatmaps(stats_df, shots)
            for model, fig in individual_heatmaps.items():
                model_name = model.split('_')[-1]
                model_file = os.path.join(shot_dir, f"{model_name}_heatmap_{THRESHOLD}.html")
                fig.write_html(model_file)
                print(f"Saved individual heatmap for {model_name} to {model_file}")

            # Create and save model comparison plot
            comparison_fig = self.create_model_comparison_plot(stats_df, shots)
            comparison_file = os.path.join(shot_dir, f"most_frequent_choices_{THRESHOLD}.html")
            comparison_fig.write_html(comparison_file)
            print(f"Saved comparison plot to {comparison_file}")

        # Create combined visualizations with all shots
        print("Creating combined visualizations...")

        combined_heatmap = self.create_side_by_side_heatmap(stats_df)
        combined_heatmap_file = os.path.join(output_dir, f"combined_heatmap_{THRESHOLD}.html")
        combined_heatmap.write_html(combined_heatmap_file)
        print(f"Saved combined heatmap to {combined_heatmap_file}")

        # Create and save combined individual model heatmaps
        combined_individual_heatmaps = self.create_individual_model_heatmaps(stats_df)
        for model, fig in combined_individual_heatmaps.items():
            model_name = model.split('_')[-1]
            model_file = os.path.join(output_dir, f"{model_name}_combined_heatmap_{THRESHOLD}.html")
            fig.write_html(model_file)
            print(f"Saved combined individual heatmap for {model_name} to {model_file}")

        combined_comparison = self.create_model_comparison_plot(stats_df)
        combined_comparison_file = os.path.join(output_dir, f"combined_most_frequent_{THRESHOLD}.html")
        combined_comparison.write_html(combined_comparison_file)
        print(f"Saved combined comparison plot to {combined_comparison_file}")

        print(f"All visualizations saved to {output_dir}")
        return stats_df


if __name__ == "__main__":
    visualizer = SimplifiedVisualizer()
    stats = visualizer.save_visualizations()