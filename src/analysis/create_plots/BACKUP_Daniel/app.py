import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random
from typing import Optional

def load_visualization_html(filepath: Path) -> Optional[str]:
    """Load HTML content from file if it exists."""
    try:
        return filepath.read_text(encoding='utf-8') if filepath.exists() else None
    except Exception as e:
        st.error(f"Failed to load visualization: {e}")
        return None

def create_failed_scatter_plot(stats_df: pd.DataFrame, datasets: list[str], model_colors: dict[str, str]) -> go.Figure:
    """Generate scatter plot for cases where both models failed."""
    filtered_stats = stats_df[stats_df['dataset'].isin(datasets)].copy()
    failed_cases = filtered_stats.groupby(['dataset', 'sample_index'])['percentage'].count()
    failed_cases = failed_cases[failed_cases == 2].reset_index()[['dataset', 'sample_index']]
    merged_failed = pd.merge(filtered_stats, failed_cases, on=['dataset', 'sample_index'])

    merged_failed['x_label'] = merged_failed.apply(lambda row: f"{row['dataset']}_{row['sample_index']}", axis=1)

    scatter_fig = px.scatter(
        merged_failed,
        x='x_label',
        y='percentage',
        color='model_name',
        color_discrete_map=model_colors,
        hover_data=['dataset', 'sample_index', 'answer_position'],
        title='Scatter Plot of Questions Both Models Failed'
    )

    scatter_fig.update_layout(
        xaxis_title="Dataset_Question_Index",
        yaxis_title="Percentage of Answers (%)",
        xaxis={
            'tickangle': 45,
            'tickmode': 'array',
            'ticktext': merged_failed['x_label'].unique(),
            'tickvals': list(range(len(merged_failed['x_label'].unique()))),
            'showticklabels': True
        },
        height=800,
        width=1400,
        margin=dict(b=200),
        plot_bgcolor='white'
    )
    return scatter_fig

def main():
    st.set_page_config(page_title="Model Performance Analysis", layout="wide")
    st.title("Model Performance Analysis Dashboard")
    results_dir = Path("visualization_results")
    position_stats_path = results_dir / "position_statistics.parquet"

    st.header("Overall Performance Analysis")
    with st.container():
        scatter_html = load_visualization_html(results_dir / "position_scatter.html")
        if scatter_html:
            st.subheader("Answer Position Distribution")
            components.html(scatter_html, height=700)
        else:
            st.warning("Scatter plot visualization not found")

    with st.container():
        box_html = load_visualization_html(results_dir / "position_box.html")
        if box_html:
            st.subheader("Answer Position Frequency Distribution")
            components.html(box_html, height=700)
        else:
            st.warning("Box plot visualization not found")

    if position_stats_path.exists():
        stats_df = pd.read_parquet(position_stats_path)
        st.header("Sample Analysis")
        st.info("Select datasets to generate detailed visualizations")

        available_datasets = sorted(stats_df['dataset'].unique())
        random_sample_size = st.slider("Select number of random datasets:", min_value=1, max_value=5, value=3)

        selected_datasets = st.multiselect("Select Datasets for Detailed Analysis", options=available_datasets, max_selections=5)

        if not selected_datasets:
            selected_datasets = random.sample(available_datasets, min(random_sample_size, len(available_datasets)))

        if selected_datasets:
            model_colors = {
            'Meta-Llama-3-8B-Instruct': '#2ca02c',
            'meta-llama_Meta-Llama-3-8B-Instruct': '#2ca02c',
            'OLMoE-1B-7B-0924-Instruct': '#ff7f0e',
            'allenai_OLMoE-1B-7B-0924-Instruct': '#ff7f0e'
            }
            failed_scatter_fig = create_failed_scatter_plot(stats_df, selected_datasets, model_colors)
            st.plotly_chart(failed_scatter_fig, use_container_width=True)

if __name__ == "__main__":
    main()
