import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

class ModelPerformanceVisualizer:
    def __init__(self, base_results_dir: str = "../app/results_local"):
        self.base_dir = Path(base_results_dir)
        self.colors = {
            'Meta-Llama-3-8B-Instruct': '#2ca02c',
            'Llama-3.2-3B-Instruct': '#d62728',
            'Mistral-7B-Instruct-v0.3': '#9467bd',
            'Llama-3.2-1B-Instruct': '#17becf',
            'OLMoE-1B-7B-0924-Instruct': '#ff7f0e'
        }
        self.output_dir = Path("processed_data")
        self.samples_dir = self.output_dir / "samples"

    def preprocess_data(self) -> None:
        """Process dataset and create samples for interactive analysis."""
        os.makedirs(self.samples_dir, exist_ok=True)

        model_stats = []
        for shots_dir in self.base_dir.glob("Shots_*"):
            shots = int(shots_dir.name.split('_')[1])

            for model_dir in shots_dir.glob("*"):
                model_data = self._load_model_data(model_dir, shots)
                if not model_data.empty:
                    # Print column names for debugging
                    print(f"Available columns: {model_data.columns.tolist()}")
                    model_stats.append(model_data)

        if model_stats:
            combined_stats = pd.concat(model_stats, ignore_index=True)
            self._create_sample_datasets(combined_stats)

    def _load_model_data(self, model_dir: Path, shots: int) -> pd.DataFrame:
        """Load and process data for a single model."""
        model_name = model_dir.name
        model_data = []

        for dataset_dir in model_dir.glob("*"):
            parquet_file = dataset_dir / 'low_performance_questions.parquet'
            if parquet_file.exists():
                try:
                    df = pd.read_parquet(parquet_file)
                    df['shots'] = shots
                    df['model_name'] = model_name
                    df['dataset'] = dataset_dir.name

                    # Calculate answer position statistics
                    if 'answer_position' not in df.columns and 'closest_answer' in df.columns:
                        df['answer_position'] = df.apply(
                            lambda row: self._map_answer_position(
                                row['closest_answer'],
                                row.get('enumerator', 'numbers')
                            ),
                            axis=1
                        )
                    model_data.append(df)
                except Exception as e:
                    print(f"Error loading {parquet_file}: {str(e)}")

        return pd.concat(model_data, ignore_index=True) if model_data else pd.DataFrame()

    def _map_answer_position(self, answer: str, enumerator: str) -> int:
        """Maps an answer to its position based on the enumerator type."""
        position_mappings = {
            'greek': "αβγδεζηθικ",
            'keyboard': "!@#$%^₪*)(",
            'capitals': "ABCDEFGHIJ",
            'lowercase': "abcdefghij",
            'numbers': [str(i + 1) for i in range(10)],
            'roman': ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
        }

        try:
            prefix = answer.split('.')[0].strip()
            mapping = position_mappings.get(enumerator, position_mappings['numbers'])
            return mapping.index(prefix) + 1 if prefix in mapping else 0
        except (AttributeError, ValueError, IndexError):
            return 0

    def _create_sample_datasets(self, stats_df: pd.DataFrame) -> None:
        """Create random samples for interactive analysis."""
        unique_datasets = stats_df['dataset'].unique()

        # Define columns based on what's available
        essential_columns = ['model_name', 'dataset', 'sample_index', 'closest_answer', 'score']
        if 'answer_position' in stats_df.columns:
            essential_columns.append('answer_position')

        # Add optional columns if they exist
        for col in ['accuracy', 'cumulative_logprob']:
            if col in stats_df.columns:
                essential_columns.append(col)

        for i in range(5):
            sample_datasets = np.random.choice(unique_datasets, size=5, replace=False)
            sample_data = stats_df[
                stats_df['dataset'].isin(sample_datasets)
            ][essential_columns].copy()

            sample_data.to_parquet(
                self.samples_dir / f"sample_{i}.parquet",
                compression='snappy'
            )

    def create_interactive_plot(self, data: pd.DataFrame, model_name: str) -> go.Figure:
        """Create interactive visualization for model performance analysis."""
        position_column = 'answer_position' if 'answer_position' in data.columns else 'closest_answer'

        # Calculate positional statistics
        position_stats = (
            data.groupby(['dataset', position_column])
            .size()
            .reset_index(name='count')
        )
        position_stats['total'] = position_stats.groupby('dataset')['count'].transform('sum')
        position_stats['percentage'] = (position_stats['count'] / position_stats['total'] * 100)

        # Sort datasets by median percentage
        dataset_order = (
            position_stats.groupby('dataset')['percentage']
            .median()
            .sort_values(ascending=True)
            .index
            .tolist()
        )

        fig = go.Figure()
        fig.add_trace(go.Box(
            y=position_stats['dataset'],
            x=position_stats['percentage'],
            name=model_name,
            boxpoints='all',
            jitter=0.8,
            pointpos=0,
            marker=dict(
                color=self.colors.get(model_name, '#000000'),
                size=4,
                opacity=0.7
            ),
            line=dict(width=1),
            hovertemplate=(
                "<b>Dataset:</b> %{y}<br>" +
                "<b>Percentage:</b> %{x:.1f}%<br>" +
                "<extra></extra>"
            )
        ))

        fig.update_layout(
            title=f"Answer Distribution Analysis - {model_name}",
            xaxis_title="Position Frequency (%)",
            yaxis_title="Dataset",
            yaxis=dict(
                categoryorder='array',
                categoryarray=dataset_order
            ),
            height=600,
            width=1000,
            plot_bgcolor='white',
            showlegend=False
        )

        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
        fig.update_yaxes(showgrid=False)

        return fig

def main():
    st.set_page_config(page_title="Model Analysis", layout="wide")
    st.title("Model Performance Pattern Analysis")

    visualizer = ModelPerformanceVisualizer()

    if 'preprocessed' not in st.session_state:
        with st.spinner("Processing data..."):
            visualizer.preprocess_data()
            st.session_state.preprocessed = True

    try:
        sample_index = st.sidebar.selectbox(
            "Sample Set",
            range(5),
            format_func=lambda x: f"Sample {x + 1}"
        )

        sample_data = pd.read_parquet(
            visualizer.samples_dir / f"sample_{sample_index}.parquet"
        )

        selected_model = st.sidebar.selectbox(
            "Model",
            options=sample_data['model_name'].unique()
        )

        fig = visualizer.create_interactive_plot(
            sample_data[sample_data['model_name'] == selected_model],
            selected_model
        )

        st.plotly_chart(fig, use_container_width=True)

        if st.session_state.get('selected_point'):
            point_data = sample_data.iloc[st.session_state.selected_point]

            st.subheader("Selected Question Details")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Selected Answer")
                st.write(point_data['closest_answer'])
                if 'score' in point_data:
                    st.markdown("#### Score")
                    st.write(f"Score: {point_data['score']:.2f}")

            with col2:
                if 'accuracy' in point_data:
                    st.markdown("#### Accuracy")
                    st.write(f"Accuracy: {point_data['accuracy']:.2%}")
                if 'cumulative_logprob' in point_data:
                    st.markdown("#### Log Probability")
                    st.write(f"Log Prob: {point_data['cumulative_logprob']:.2f}")

    except FileNotFoundError:
        st.error("Please ensure data preprocessing completed successfully.")
        st.write("Try running the preprocessing step first.")

if __name__ == "__main__":
    main()