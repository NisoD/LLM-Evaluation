import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Tuple


def load_visualization_html(filepath: Path) -> Optional[str]:
    """Load HTML content from file if it exists."""
    try:
        return filepath.read_text(encoding='utf-8') if filepath.exists() else None
    except Exception as e:
        st.error(f"Failed to load visualization: {e}")
        return None


def create_sample_plots(
        stats_df: pd.DataFrame,
        datasets: list[str],
        model_colors: dict[str, str]
) -> Tuple[go.Figure, go.Figure]:
    """Generate scatter and box plots for selected datasets."""
    filtered_stats = stats_df[stats_df['dataset'].isin(datasets)].copy()
    filtered_stats['x_label'] = filtered_stats.apply(
        lambda row: f"{row['dataset']}_{row['sample_index']}",
        axis=1
    )

    scatter_fig = px.scatter(
        filtered_stats,
        x='x_label',
        y='percentage',
        color='model_name',
        color_discrete_map=model_colors,
        hover_data=['dataset', 'sample_index', 'answer_position'],
        title='Most Frequently Chosen Answer Positions'
    )

    box_fig = px.box(
        filtered_stats,
        x='x_label',
        y='percentage',
        color='model_name',
        color_discrete_map=model_colors,
        title='Distribution of Answer Position Frequencies'
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
            height=800,
            width=1200,
            margin=dict(b=150),
            plot_bgcolor='white'
        )

    return scatter_fig, box_fig


def main():
    st.set_page_config(page_title="Model Performance Analysis", layout="wide")
    st.title("Model Performance Analysis Dashboard")

    results_dir = Path("visualization_results")
    position_stats_path = results_dir / "position_statistics.parquet"

    # Load and display pre-generated visualizations vertically
    st.header("Overall Performance Analysis")

    # Display scatter plot
    scatter_html = load_visualization_html(results_dir / "position_scatter.html")
    if scatter_html:
        st.subheader("Answer Position Distribution")
        components.html(scatter_html, height=700)
    else:
        st.warning("Scatter plot visualization not found")

    # Display box plot below scatter plot
    box_html = load_visualization_html(results_dir / "position_box.html")
    if box_html:
        st.subheader("Answer Position Frequency Distribution")
        components.html(box_html, height=700)
    else:
        st.warning("Box plot visualization not found")

    # Sample analysis section
    if position_stats_path.exists():
        stats_df = pd.read_parquet(position_stats_path)

        st.header("Sample Analysis")
        st.info("Select datasets to generate detailed visualizations")

        available_datasets = sorted(stats_df['dataset'].unique())
        selected_datasets = st.multiselect(
            "Select Datasets for Detailed Analysis",
            options=available_datasets,
            max_selections=3
        )

        if selected_datasets:
            model_colors = {
                'Meta-Llama-3-8B-Instruct': '#2ca02c',
                'OLMoE-1B-7B-0924-Instruct': '#ff7f0e'
            }

            scatter_fig, box_fig = create_sample_plots(
                stats_df,
                selected_datasets,
                model_colors
            )

            st.plotly_chart(scatter_fig, use_container_width=True)


if __name__ == "__main__":
    main()