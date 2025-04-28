# Add this to your Streamlit app as a new section

# First, define the error_analysis.py module that will parse and process the error annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import re


def parse_error_annotations(raw_text):
    """Parse the raw error annotation text into a structured DataFrame."""

    # Define the error categories
    error_categories = [
        "Wrong Annotation",
        "Wrong Reasoning",
        "Lack of Prior Knowledge",
        "Format Error"
    ]

    # Initialize an empty list to store the parsed data
    parsed_data = []

    # Initialize variables to track current context
    current_model = None
    current_shots = None
    current_dataset = None

    # Regular expressions for parsing
    model_shot_pattern = re.compile(r'###\s*(OLMoE|LLAMA3?)\s*,?\s*(\d+)\s*Shot', re.IGNORECASE)
    dataset_pattern = re.compile(
        r'[-\s]+(AI2_ARC|OpenBook QA|MMLU[-\s]?(College Biology|World Religion|Marketing|Sociology|High School European History))',
        re.IGNORECASE)
    question_error_pattern = re.compile(
        r'[-\s]*(\d+)\s*[-\s]+(Wrong\s*Annotation|Wrong\s*Reasoning|Lack\s*of\s*Prior\s*Knowle[d]?ge|Format\s*Error)',
        re.IGNORECASE)

    # Process each line
    for line in raw_text.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        # Check for model and shots
        model_shot_match = model_shot_pattern.search(line)
        if model_shot_match:
            current_model = model_shot_match.group(1).strip()
            current_shots = int(model_shot_match.group(2).strip())
            # Normalize model names
            if current_model.upper() in ["LLAMA", "LLAMA3"]:
                current_model = "Llama 3"
            continue

        # Check for dataset
        dataset_match = dataset_pattern.search(line)
        if dataset_match:
            raw_dataset = dataset_match.group(1)
            # Handle MMLU subdatasets
            if "MMLU" in raw_dataset:
                if dataset_match.group(2):
                    subdataset = dataset_match.group(2).strip()
                    current_dataset = f"mmlu.{subdataset.lower().replace(' ', '_')}"
            else:
                # Normalize dataset names
                if "AI2_ARC" in raw_dataset:
                    current_dataset = "ai2_arc.arc_challenge"
                elif "OpenBook" in raw_dataset:
                    current_dataset = "openbook_qa"
            continue

        # Check for question error
        question_error_match = question_error_pattern.search(line)
        if question_error_match and current_model and current_shots is not None and current_dataset:
            question_id = int(question_error_match.group(1).strip())
            error_type = question_error_match.group(2).strip()

            # Normalize error types
            for category in error_categories:
                if category.lower() in error_type.lower():
                    error_type = category
                    break

            # Add to parsed data
            parsed_data.append({
                'model': current_model,
                'shots': current_shots,
                'dataset': current_dataset,
                'question_id': question_id,
                'error_type': error_type
            })

    # Convert to DataFrame
    df = pd.DataFrame(parsed_data)
    return df


def create_error_analysis_section(error_annotations_text):
    """Create the error analysis section in the Streamlit app."""

    st.header("Error Analysis")

    # Parse the error annotations
    error_df = parse_error_annotations(error_annotations_text)

    if error_df.empty:
        st.warning("No error annotations were parsed. Please check the format.")
        return

    # Show summary counts
    st.subheader("Error Analysis Summary")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Analyzed Errors", len(error_df))

    with col2:
        unique_questions = error_df['question_id'].nunique()
        st.metric("Unique Questions", unique_questions)

    with col3:
        unique_datasets = error_df['dataset'].nunique()
        st.metric("Datasets Covered", unique_datasets)

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Error Types", "Model Comparison", "Dataset Analysis", "Raw Data"])

    with tab1:
        # Error type distribution
        st.subheader("Distribution of Error Types")

        # Count error types
        error_counts = error_df['error_type'].value_counts().reset_index()
        error_counts.columns = ['Error Type', 'Count']

        # Create pie chart for error types
        fig_pie = px.pie(
            error_counts,
            values='Count',
            names='Error Type',
            title='Distribution of Error Types',
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

        # Error types by model and shots
        st.subheader("Error Types by Model and Shots")

        # Count by model, shots, and error type
        model_shot_errors = error_df.groupby(['model', 'shots', 'error_type']).size().reset_index(name='count')

        # Create grouped bar chart
        fig_bar = px.bar(
            model_shot_errors,
            x='error_type',
            y='count',
            color='model',
            barmode='group',
            facet_col='shots',
            facet_col_wrap=2,
            category_orders={"shots": [0, 5]},
            labels={
                'error_type': 'Error Type',
                'count': 'Number of Errors',
                'model': 'Model',
                'shots': 'Shots'
            },
            title='Error Types by Model and Number of Shots'
        )

        fig_bar.update_layout(
            xaxis_title="Error Type",
            yaxis_title="Count",
            legend_title="Model",
            height=500
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    with tab2:
        # Model comparison
        st.subheader("Model Comparison")

        # Create a heatmap comparing models
        model_error_pivot = error_df.pivot_table(
            index=['model', 'shots'],
            columns='error_type',
            aggfunc='size',
            fill_value=0
        ).reset_index()

        # Add total column
        model_error_pivot['Total'] = model_error_pivot.iloc[:, 2:].sum(axis=1)

        # Display as table
        st.write("Error Counts by Model and Shot Setting")
        model_error_pivot_display = model_error_pivot.copy()
        model_error_pivot_display['Model-Shots'] = model_error_pivot_display['model'] + " (" + \
                                                   model_error_pivot_display['shots'].astype(str) + "-shot)"
        model_error_pivot_display = model_error_pivot_display.drop(columns=['model', 'shots']).set_index('Model-Shots')
        st.dataframe(model_error_pivot_display)

        # Create heatmap for model comparison
        # Prepare data for heatmap
        heatmap_data = []
        for _, row in model_error_pivot.iterrows():
            model = row['model']
            shots = row['shots']
            # Skip the 'model', 'shots' and 'Total' columns
            for col in model_error_pivot.columns[2:-1]:  # Error types only
                heatmap_data.append({
                    'Model': f"{model} ({shots}-shot)",
                    'Error Type': col,
                    'Count': row[col]
                })

        heatmap_df = pd.DataFrame(heatmap_data)

        # Create the heatmap
        fig_heatmap = px.density_heatmap(
            heatmap_df,
            x='Error Type',
            y='Model',
            z='Count',
            color_continuous_scale="Blues",
            text_auto=True,
            title='Error Type Distribution Across Models'
        )

        fig_heatmap.update_layout(
            xaxis_title="Error Type",
            yaxis_title="Model",
            coloraxis_colorbar_title='Count',
            height=500
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Create a summary bar chart of model performance
        model_totals = model_error_pivot.groupby(['model', 'shots'])['Total'].sum().reset_index()
        model_totals['Model-Shots'] = model_totals['model'] + " (" + model_totals['shots'].astype(str) + "-shot)"

        fig_totals = px.bar(
            model_totals,
            x='Model-Shots',
            y='Total',
            color='model',
            labels={
                'Model-Shots': 'Model and Shot Setting',
                'Total': 'Total Errors',
                'model': 'Model'
            },
            title='Total Errors by Model and Shot Setting'
        )

        st.plotly_chart(fig_totals, use_container_width=True)

    with tab3:
        # Dataset analysis
        st.subheader("Dataset Analysis")

        # Count errors by dataset
        dataset_errors = error_df.groupby('dataset').size().reset_index(name='count')
        dataset_errors = dataset_errors.sort_values('count', ascending=False)

        # Create bar chart for datasets
        fig_dataset = px.bar(
            dataset_errors,
            x='dataset',
            y='count',
            color='dataset',
            labels={
                'dataset': 'Dataset',
                'count': 'Number of Errors'
            },
            title='Error Distribution by Dataset'
        )

        fig_dataset.update_layout(
            xaxis_title="Dataset",
            yaxis_title="Count",
            xaxis_tickangle=45,
            height=500,
            showlegend=False
        )

        st.plotly_chart(fig_dataset, use_container_width=True)

        # Error types by dataset
        dataset_error_types = error_df.groupby(['dataset', 'error_type']).size().reset_index(name='count')

        fig_dataset_errors = px.bar(
            dataset_error_types,
            x='dataset',
            y='count',
            color='error_type',
            barmode='stack',
            labels={
                'dataset': 'Dataset',
                'count': 'Number of Errors',
                'error_type': 'Error Type'
            },
            title='Error Types by Dataset'
        )

        fig_dataset_errors.update_layout(
            xaxis_title="Dataset",
            yaxis_title="Count",
            xaxis_tickangle=45,
            height=500,
            legend_title="Error Type"
        )

        st.plotly_chart(fig_dataset_errors, use_container_width=True)

        # Dataset performance across models
        dataset_model = error_df.groupby(['dataset', 'model', 'shots']).size().reset_index(name='count')
        dataset_model['Model-Shots'] = dataset_model['model'] + " (" + dataset_model['shots'].astype(str) + "-shot)"

        fig_dataset_model = px.bar(
            dataset_model,
            x='dataset',
            y='count',
            color='Model-Shots',
            barmode='group',
            labels={
                'dataset': 'Dataset',
                'count': 'Number of Errors',
                'Model-Shots': 'Model and Shots'
            },
            title='Dataset Performance Across Models'
        )

        fig_dataset_model.update_layout(
            xaxis_title="Dataset",
            yaxis_title="Count",
            xaxis_tickangle=45,
            height=500,
            legend_title="Model Configuration"
        )

        st.plotly_chart(fig_dataset_model, use_container_width=True)

    with tab4:
        # Raw data
        st.subheader("Raw Error Data")
        st.dataframe(error_df, use_container_width=True)

        # Download option
        csv = error_df.to_csv(index=False)
        st.download_button(
            label="Download Error Analysis Data",
            data=csv,
            file_name="error_analysis_data.csv",
            mime="text/csv",
        )


# Example usage
# Add this to your Streamlit app:
"""
# Add the error analysis section
st.header("Manual Error Analysis")

# Option 1: Hardcode the error annotations
error_annotations = '''
1. Wrong Annotation
2. Wrong Reasoning
3. Lack of prior Knowledge
4. Format Error
### OLMoE ,  0 Shot
- AI2_ARC
	8 - Wrong reasoning
	12 - Wrong reasoning
	54 - Wrong Annotation 
	67 - Wrong Annotation 
...
'''

# Option 2: Let the user input the error annotations
error_annotations = st.text_area(
    "Enter error annotations",
    height=200,
    value=error_annotations if 'error_annotations' in locals() else ""
)

if error_annotations:
    create_error_analysis_section(error_annotations)
"""