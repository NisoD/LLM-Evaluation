import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np
import re
import random
from datasets import load_dataset

# Set page config
st.set_page_config(
    page_title="Model Performance Analysis Explorer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Function to load dataset and get question content
@st.cache_data
def load_dataset_and_get_question(dataset_name, sample_index):
    try:
        # Convert numpy.int64 to standard Python int
        if hasattr(sample_index, 'item'):
            sample_index = sample_index.item()
        else:
            sample_index = int(sample_index)

        # Logic to load the correct dataset based on name
        if dataset_name.startswith("mmlu."):
            # For MMLU datasets
            subset = dataset_name.split(".", 1)[1]
            ds = load_dataset("cais/mmlu", "all", split="test")
            example = ds[sample_index]

            question = example['question']
            choices = example['choices']
            answer_idx = example['answer']

            formatted_question = f"**Question:** {question}\n\n"
            for i, choice in enumerate(choices):
                formatted_question += f"**{chr(65 + i)}.** {choice}\n"
            formatted_question += f"\n**Correct Answer:** {chr(65 + answer_idx)}"

        elif dataset_name == "ai2_arc.arc_challenge":
            ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
            example = ds[sample_index]

            question = example['question']
            choices = example['choices']['text']
            labels = example['choices']['label']
            answer_key = example['answerKey']

            formatted_question = f"**Question:** {question}\n\n"
            for label, choice in zip(labels, choices):
                formatted_question += f"**{label}.** {choice}\n"
            formatted_question += f"\n**Correct Answer:** {answer_key}"

        elif dataset_name == "ai2_arc.arc_easy":
            ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
            example = ds[sample_index]

            question = example['question']
            choices = example['choices']['text']
            labels = example['choices']['label']
            answer_key = example['answerKey']

            formatted_question = f"**Question:** {question}\n\n"
            for label, choice in zip(labels, choices):
                formatted_question += f"**{label}.** {choice}\n"
            formatted_question += f"\n**Correct Answer:** {answer_key}"

        elif dataset_name == "hellaswag":
            ds = load_dataset("Rowan/hellaswag", split="test")
            example = ds[sample_index]

            context = example['ctx']
            endings = example['endings']
            label = example['label']

            formatted_question = f"**Context:** {context}\n\n**Options:**\n"
            for i, ending in enumerate(endings):
                formatted_question += f"**{i + 1}.** {ending}\n"
            formatted_question += f"\n**Correct Answer:** {int(label) + 1}"

        elif dataset_name == "social_iqa":
            ds = load_dataset("allenai/social_i_qa", split="validation", trust_remote_code=True)
            example = ds[sample_index]

            context = example['context']
            question = example['question']
            options = [example['answerA'], example['answerB'], example['answerC']]
            label = example['label']

            formatted_question = f"**Context:** {context}\n\n**Question:** {question}\n\n"
            for i, option in enumerate(['A', 'B', 'C']):
                formatted_question += f"**{option}.** {options[i]}\n"
            formatted_question += f"\n**Correct Answer:** {label}"

        elif dataset_name == "openbook_qa":
            ds = load_dataset("openbookqa", "main", split="test")
            example = ds[sample_index]

            question = example['question_stem']
            choices = example['choices']['text']
            labels = example['choices']['label']
            answer_key = example['answerKey']

            formatted_question = f"**Question:** {question}\n\n"
            for label, choice in zip(labels, choices):
                formatted_question += f"**{label}.** {choice}\n"
            formatted_question += f"\n**Correct Answer:** {answer_key}"

        else:
            # Generic fallback for unknown datasets
            formatted_question = f"**Dataset:** {dataset_name}\n**Sample Index:** {sample_index}\n\n"
            formatted_question += f"Unable to format question content specifically for this dataset type."

        return formatted_question

    except Exception as e:
        return f"Error retrieving question content: {str(e)}"


# App title and description
st.title("Model Performance Analysis Explorer")
st.markdown("""
Select your parameters in the sidebar to load and visualize a specific parquet file.
""")

# Define parameters for selection
threshold_options = [0.1, 0.45]
shots_options = [0, 5]
model_options = {
    'OLMoE': 'allenai_OLMoE-1B-7B-0924-Instruct',
    'Llama 3': 'meta-llama_Meta-Llama-3-8B-Instruct'
}
mmlu_subtasks = [
    "college_biology",
    "high_school_european_history",
    "marketing",
    "sociology",
    "world_religions"
]
base_datasets = [
    "ai2_arc.arc_challenge",
    "ai2_arc.arc_easy",
    "hellaswag",
    "openbook_qa",
    "social_iqa",
]
all_datasets = base_datasets + [f"mmlu.{task}" for task in mmlu_subtasks]

# Sidebar controls
st.sidebar.header("Select Parameters")

threshold = st.sidebar.selectbox("Threshold", threshold_options)
shots = st.sidebar.selectbox("Shots", shots_options)
model_name_display = st.sidebar.selectbox("Model", list(model_options.keys()))
model_name = model_options[model_name_display]
dataset = st.sidebar.selectbox("Dataset", all_datasets)

# Base path for results
base_path = Path("../app/results_local")

# Generate the file path based on selections
file_path = base_path / f"Shots_{shots}" / model_name / dataset.replace('/',
                                                                        '_') / f'low_performance_questions_{threshold}.parquet'

# Display file path
st.sidebar.subheader("File Path")
st.sidebar.code(str(file_path))


# Function to load and display the data
def load_and_display_data(file_path):
    if not file_path.exists():
        st.error(f"File not found: {file_path}")
        # For testing, provide option to load sample CSV
        use_sample = st.checkbox("Use sample data instead?")
        if use_sample:
            try:
                # Try to load the sample CSV if it exists
                sample_path = "question_12_allenai_OLMoE1B7B0924Instruct_0shots_0.1threshold.csv"
                df = pd.read_csv(sample_path)
                st.success(f"Loaded sample data: {sample_path}")
                return df
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
                return None
        return None

    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading the parquet file: {str(e)}")
        # For testing, provide option to load sample CSV
        use_sample = st.checkbox("Use sample data instead?")
        if use_sample:
            try:
                sample_path = "question_12_allenai_OLMoE1B7B0924Instruct_0shots_0.1threshold.csv"
                df = pd.read_csv(sample_path)
                st.success(f"Loaded sample data: {sample_path}")
                return df
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
        return None


# Load the data
data = load_and_display_data(file_path)

# If data is loaded successfully, display it
if data is not None:
    # Overview statistics
    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", len(data))

    with col2:
        question_count = data['sample_index'].nunique()
        st.metric("Unique Questions", question_count)

    with col3:
        avg_score = data['score'].mean().round(3)
        st.metric("Average Score", f"{avg_score:.3f}")

    # Question performance analysis
    st.header("Question Performance Analysis")

    # Calculate performance per question
    question_performance = data.groupby('sample_index').agg({
        'score': ['mean', 'sum', 'count']
    }).reset_index()

    question_performance.columns = ['sample_index', 'accuracy', 'correct_count', 'total_count']
    question_performance = question_performance.sort_values('accuracy')

    # Plot question performance
    fig = px.scatter(
        question_performance,
        x='sample_index',
        y='accuracy',
        hover_data=['correct_count', 'total_count'],
        labels={'accuracy': 'Accuracy', 'sample_index': 'Question Index'},
        title=f'Question Accuracy (Threshold < {threshold})'
    )

    fig.update_layout(xaxis_title="Question Index", yaxis_title="Accuracy")
    st.plotly_chart(fig, use_container_width=True)
    # This should be added after the first visualization, just before the "Question explorer" section

    # Add a scatter plot comparison across all datasets
    st.header("All Datasets Comparison")

    # Create a button to load all datasets comparison
    if st.button("Load All Datasets Comparison"):
        with st.spinner("Loading data from all models and shots combinations..."):
            # Function to load data for a specific model, shots, threshold combination
            @st.cache_data
            def load_all_combinations(threshold_value):
                all_data = []

                for model_key, model_path in model_options.items():
                    for shot_value in shots_options:
                        for dataset_name in all_datasets:
                            # Generate the file path
                            file_path = base_path / f"Shots_{shot_value}" / model_path / dataset_name.replace('/',
                                                                                                              '_') / f'low_performance_questions_{threshold_value}.parquet'

                            # Try to load the file
                            try:
                                if file_path.exists():
                                    df = pd.read_parquet(file_path)
                                    # Add metadata
                                    df['model_display'] = model_key
                                    df['model_path'] = model_path
                                    df['shots_value'] = shot_value
                                    df['dataset_name'] = dataset_name
                                    all_data.append(df)
                            except Exception as e:
                                st.error(f"Error loading {file_path}: {str(e)}")
                                continue

                # Combine all dataframes
                if all_data:
                    return pd.concat(all_data, ignore_index=True)
                else:
                    return pd.DataFrame()

                # Combine all dataframes
                if all_data:
                    return pd.concat(all_data, ignore_index=True)
                else:
                    return pd.DataFrame()


            # Load all combinations
            combined_data = load_all_combinations(threshold)

            if not combined_data.empty:
                # Group by model, shots, and dataset to get overall performance metrics
                dataset_performance = combined_data.groupby(['model_display', 'shots_value', 'dataset_name']).agg({
                    'score': ['mean', 'sum', 'count'],
                    'sample_index': 'nunique'
                }).reset_index()

                dataset_performance.columns = ['model', 'shots', 'dataset', 'accuracy', 'correct_count', 'total_count',
                                               'unique_questions']


                # Create grouped scatter plot
                fig_dataset = px.scatter(
                    dataset_performance,
                    x='dataset',
                    y='accuracy',
                    color='model',
                    symbol='shots',
                    size='unique_questions',  # Size points by number of questions
                    hover_data=['accuracy', 'unique_questions'],
                    labels={
                        'accuracy': 'Average Accuracy',
                        'dataset': 'Dataset',
                        'unique_questions': 'Number of Questions',
                        'model': 'Model',
                        'shots': 'Shots'
                    },
                    title=f'Dataset Performance Comparison Across All Models and Shots (Threshold: {threshold})',
                )

                # Improve readability
                fig_dataset.update_layout(
                    xaxis_title="Dataset",
                    yaxis_title="Average Accuracy",
                    xaxis_tickangle=45,
                    height=600,
                    legend_title_text="Models and Shots"
                )

                # Add a horizontal line for the threshold
                fig_dataset.add_shape(
                    type="line",
                    x0=-0.5,
                    y0=threshold,
                    x1=len(dataset_performance['dataset'].unique()) - 0.5,
                    y1=threshold,
                    line=dict(color="red", width=2, dash="dash")
                )

                # Add annotation for the threshold line
                fig_dataset.add_annotation(
                    x=dataset_performance['dataset'].unique()[0],
                    y=threshold,
                    xref="x",
                    yref="y",
                    text=f"Threshold: {threshold}",
                    showarrow=True,
                    arrowhead=2,
                    ax=50,
                    ay=-30,
                    bordercolor="#c7c7c7",
                    borderwidth=2,
                    borderpad=4,
                    bgcolor="#ff7f0e",
                    opacity=0.8,
                    font=dict(color="white")
                )

                st.plotly_chart(fig_dataset, use_container_width=True)

                # Add model comparison table
                st.subheader("Model Performance Summary")
                model_summary = dataset_performance.groupby(['model', 'shots']).agg({
                    'accuracy': 'mean',
                    'dataset': 'count',
                    'unique_questions': 'sum'
                }).reset_index()

                model_summary.columns = ['Model', 'Shots', 'Average Accuracy', 'Dataset Count', 'Total Questions']
                model_summary['Average Accuracy'] = model_summary['Average Accuracy'].round(3)
                st.dataframe(model_summary, use_container_width=True)

                # Download option for the full comparison data
                csv = dataset_performance.to_csv(index=False)
                # This should be added to the "All Datasets Comparison" section
                # Place it after the model comparison table but before the download button

                # Add answer choice distribution visualization
                # This should be added to the "All Datasets Comparison" section
                # Place it after the model comparison table but before the download button

                # Add answer choice distribution visualization

                st.subheader("Answer Choice Distribution")

                if 'combined_data' in locals():
                    if not combined_data.empty and 'chosen_position' in combined_data.columns:
                        # Group by model, shots, and chosen position
                        choice_distribution = combined_data.groupby(
                            ['model_display', 'shots_value', 'chosen_position']).size().reset_index(name='count')

                        # Calculate percentages within each model-shots group
                        choice_distribution['total'] = choice_distribution.groupby(['model_display', 'shots_value'])[
                            'count'].transform('sum')
                        choice_distribution['percentage'] = (
                                    choice_distribution['count'] / choice_distribution['total'] * 100).round(1)

                        # Create the bar chart for choice distribution
                        fig_choices = px.bar(
                            choice_distribution,
                            x='chosen_position',
                            y='percentage',
                            color='model_display',
                            barmode='group',
                            facet_col='shots_value',
                            facet_col_wrap=2,
                            labels={
                                'chosen_position': 'Answer Position',
                                'percentage': 'Percentage (%)',
                                'model_display': 'Model',
                                'shots_value': 'Shots'
                            },
                            title='Distribution of Chosen Answer Positions Across Models and Shots',
                            text='percentage'
                        )

                        # Add text labels on the bars
                        fig_choices.update_traces(texttemplate='%{text:.1f}%', textposition='outside')

                        # Improve layout
                        fig_choices.update_layout(
                            xaxis_title="Answer Position",
                            yaxis_title="Percentage of Responses (%)",
                            legend_title="Model",
                            height=500
                        )

                        # Ensure x-axis shows integer positions
                        fig_choices.update_xaxes(type='category')

                        # st.plotly_chart(fig_choices, use_container_width=True)

                        # Add a heatmap showing position distribution
                        st.subheader("Answer Position Heatmap")

                        # Pivot the data for the heatmap
                        heatmap_data = choice_distribution.pivot_table(
                            index=['model_display', 'shots_value'],
                            columns='chosen_position',
                            values='percentage',
                            fill_value=0
                        ).reset_index()

                        # Reshape for plotly heatmap
                        heatmap_long = pd.melt(
                            heatmap_data,
                            id_vars=['model_display', 'shots_value'],
                            var_name='chosen_position',
                            value_name='percentage'
                        )

                        # Create a string column for y-axis labels
                        heatmap_long['model_shots'] = heatmap_long['model_display'] + ' (' + heatmap_long[
                            'shots_value'].astype(str) + ' shots)'

                        # Create the heatmap using go.Heatmap instead of px.density_heatmap
                        # Pivot the data for proper heatmap format
                        pivot_data = heatmap_long.pivot(
                            index='model_shots',
                            columns='chosen_position',
                            values='percentage'
                        )

                        # Create annotations for the percentages
                        annotations = []
                        for i, row in enumerate(pivot_data.index):
                            for j, col in enumerate(pivot_data.columns):
                                value = pivot_data.iloc[i, j]
                                annotations.append(
                                    dict(
                                        x=col,
                                        y=row,
                                        text=f"{value:.1f}%",
                                        showarrow=False,
                                        font=dict(color="black" if value < 50 else "white")
                                    )
                                )

                        # Create the heatmap
                        fig_heatmap = go.Figure(data=go.Heatmap(
                            z=pivot_data.values,
                            x=pivot_data.columns,
                            y=pivot_data.index,
                            colorscale='Blues',
                            hoverongaps=False,
                            colorbar=dict(title='Percentage (%)')
                        ))

                        # Add the annotations
                        fig_heatmap.update_layout(
                            annotations=annotations,
                            title='Heatmap of Answer Position Distribution'
                        )

                        # Improve layout
                        fig_heatmap.update_layout(
                            xaxis_title="Answer Position",
                            yaxis_title="Model & Shots Configuration",
                            coloraxis_colorbar_title='Percentage (%)',
                            height=400,
                            xaxis=dict(
                                tickmode='array',
                                tickvals=[1, 2, 3, 4],  # Specify the exact tick values
                                ticktext=['1', '2', '3', '4']  # Ensure they are displayed as whole numbers
                            )
                        )

                        st.plotly_chart(fig_heatmap, use_container_width=True)
                    else:
                        st.info("Answer choice position data not available in the loaded datasets.")
                # Add these graphs after the model comparison table but before the download button
                # in the "All Datasets Comparison" section

                # 1. Graph to compare number of questions per dataset
                st.subheader("Number of Questions per Dataset")
                fig_question_count = px.bar(
                    dataset_performance,
                    x='dataset',
                    y='unique_questions',
                    color='model',
                    pattern_shape='shots',
                    barmode='group',
                    labels={
                        'unique_questions': 'Number of Questions',
                        'dataset': 'Dataset',
                        'model': 'Model',
                        'shots': 'Shots'
                    },
                    title='Number of Questions per Dataset Across Models and Shots'
                )

                # Improve readability
                fig_question_count.update_layout(
                    xaxis_title="Dataset",
                    yaxis_title="Number of Questions",
                    xaxis_tickangle=45,
                    height=500,
                    legend_title_text="Models and Shots"
                )
                st.plotly_chart(fig_question_count, use_container_width=True)

                # 2. Graph to compare mean accuracy across datasets
                st.subheader("Mean Accuracy Comparison")
                fig_accuracy = px.box(
                    dataset_performance,
                    x='model',
                    y='accuracy',
                    color='shots',
                    points="all",
                    hover_data=['dataset', 'unique_questions', 'correct_count', 'total_count'],
                    labels={
                        'accuracy': 'Mean Accuracy',
                        'model': 'Model',
                        'shots': 'Shots'
                    },
                    title='Mean Accuracy Distribution Across Models and Shots'
                )

                # Add a horizontal line for the threshold
                fig_accuracy.add_shape(
                    type="line",
                    x0=-0.5,
                    y0=threshold,
                    x1=len(dataset_performance['model'].unique()) - 0.5,
                    y1=threshold,
                    line=dict(color="red", width=2, dash="dash")
                )

                # Add annotation for the threshold line
                fig_accuracy.add_annotation(
                    x=0,
                    y=threshold,
                    xref="x",
                    yref="y",
                    text=f"Threshold: {threshold}",
                    showarrow=True,
                    arrowhead=2,
                    ax=50,
                    ay=-30,
                    bordercolor="#c7c7c7",
                    borderwidth=2,
                    borderpad=4,
                    bgcolor="#ff7f0e",
                    opacity=0.8,
                    font=dict(color="white")
                )

                # Improve readability
                fig_accuracy.update_layout(
                    height=500
                )
                st.plotly_chart(fig_accuracy, use_container_width=True)

                # 3. Graph to compare question intersections between shots
                # Replace the existing Question Intersection Analysis section with these four graphs

                # Intersection Analysis section
                st.subheader("Question Intersection Analysis")

                # Calculate intersection data
                intersection_data = []
                model_dataset_questions = {}

                # Process each model and dataset combination
                for model_name in dataset_performance['model'].unique():
                    model_dataset_questions[model_name] = {}

                    for dataset_name in dataset_performance['dataset'].unique():
                        # Filter data for this model and dataset
                        model_dataset_data = combined_data[
                            (combined_data['model_display'] == model_name) &
                            (combined_data['dataset_name'] == dataset_name)
                            ]

                        # Skip if no data or only one shots setting available
                        if model_dataset_data.empty or len(model_dataset_data['shots_value'].unique()) < 2:
                            continue

                        # Get question sets for each shots value
                        questions_by_shots = {}
                        for shot in shots_options:
                            shot_data = model_dataset_data[model_dataset_data['shots_value'] == shot]
                            questions_by_shots[shot] = set(shot_data['sample_index'].unique())

                        # Store questions for later use
                        model_dataset_questions[model_name][dataset_name] = questions_by_shots

                        # Skip if any shot doesn't have questions
                        if any(len(q_set) == 0 for q_set in questions_by_shots.values()):
                            continue

                        # Calculate intersections and unique questions
                        if len(shots_options) == 2:  # For 2 shot options (e.g., 0 and 5)
                            shots_0 = shots_options[0]
                            shots_5 = shots_options[1]

                            # Questions in both shot settings
                            common_questions = questions_by_shots[shots_0].intersection(questions_by_shots[shots_5])

                            # Questions only in 0-shot
                            only_shot_0 = questions_by_shots[shots_0].difference(questions_by_shots[shots_5])

                            # Questions only in 5-shot
                            only_shot_5 = questions_by_shots[shots_5].difference(questions_by_shots[shots_0])

                            # Add to result data
                            intersection_data.append({
                                'model': model_name,
                                'dataset': dataset_name,
                                'category': f'Common to {shots_0} & {shots_5} shots',
                                'count': len(common_questions),
                                'shot_value': 'Both'
                            })

                            intersection_data.append({
                                'model': model_name,
                                'dataset': dataset_name,
                                'category': f'Only in {shots_0} shots',
                                'count': len(only_shot_0),
                                'shot_value': shots_0
                            })

                            intersection_data.append({
                                'model': model_name,
                                'dataset': dataset_name,
                                'category': f'Only in {shots_5} shots',
                                'count': len(only_shot_5),
                                'shot_value': shots_5
                            })

                # Create DataFrame from intersection data
                if intersection_data:
                    intersection_df = pd.DataFrame(intersection_data)

                    # Create tabs for the different intersection graphs
                    # Modify the tab definition from 4 tabs to 6 tabs
                    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                        f"Intersection for {list(model_options.keys())[0]}",
                        f"Intersection for {list(model_options.keys())[1]}",
                        "Model Comparison (0-shot)",
                        "Model Comparison (5-shot)",
                        "Model Intersection (0-shot)",  # New tab for 0-shot model intersection
                        "Model Intersection (5-shot)"  # New tab for 5-shot model intersection
                    ])

                    # 1. Intersection graph for first model
                    with tab1:
                        first_model = list(model_options.keys())[0]
                        model_data = intersection_df[intersection_df['model'] == first_model]

                        if not model_data.empty:
                            fig_model1 = px.bar(
                                model_data,
                                x='dataset',
                                y='count',
                                color='category',
                                barmode='stack',
                                labels={
                                    'count': 'Number of Questions',
                                    'category': 'Question Category',
                                    'dataset': 'Dataset'
                                },
                                title=f'Question Intersection Analysis for {first_model}'
                            )

                            fig_model1.update_layout(
                                xaxis_title="Dataset",
                                yaxis_title="Number of Questions",
                                xaxis_tickangle=45,
                                height=500,
                                legend_title="Question Category"
                            )

                            st.plotly_chart(fig_model1, use_container_width=True)

                            # Show a summary table for this model
                            st.subheader(f"Summary for {first_model}")
                            summary_df = model_data.groupby('category')['count'].sum().reset_index()
                            summary_df['percentage'] = (summary_df['count'] / summary_df['count'].sum() * 100).round(1)
                            summary_df.columns = ['Category', 'Count', 'Percentage (%)']
                            st.dataframe(summary_df, use_container_width=True)
                        else:
                            st.info(f"No intersection data available for {first_model}")

                    # 2. Intersection graph for second model
                    with tab2:
                        second_model = list(model_options.keys())[1]
                        model_data = intersection_df[intersection_df['model'] == second_model]

                        if not model_data.empty:
                            fig_model2 = px.bar(
                                model_data,
                                x='dataset',
                                y='count',
                                color='category',
                                barmode='stack',
                                labels={
                                    'count': 'Number of Questions',
                                    'category': 'Question Category',
                                    'dataset': 'Dataset'
                                },
                                title=f'Question Intersection Analysis for {second_model}'
                            )

                            fig_model2.update_layout(
                                xaxis_title="Dataset",
                                yaxis_title="Number of Questions",
                                xaxis_tickangle=45,
                                height=500,
                                legend_title="Question Category"
                            )

                            st.plotly_chart(fig_model2, use_container_width=True)

                            # Show a summary table for this model
                            st.subheader(f"Summary for {second_model}")
                            summary_df = model_data.groupby('category')['count'].sum().reset_index()
                            summary_df['percentage'] = (summary_df['count'] / summary_df['count'].sum() * 100).round(1)
                            summary_df.columns = ['Category', 'Count', 'Percentage (%)']
                            st.dataframe(summary_df, use_container_width=True)
                        else:
                            st.info(f"No intersection data available for {second_model}")

                    # 3. Model comparison for 0-shot questions
                    with tab3:
                        # Get counts of 0-shot questions per dataset for both models
                        shots_0 = shots_options[0]
                        model_comparison_data = []

                        for dataset_name in dataset_performance['dataset'].unique():
                            for model_name in model_dataset_questions:
                                if dataset_name in model_dataset_questions[model_name]:
                                    if shots_0 in model_dataset_questions[model_name][dataset_name]:
                                        count = len(model_dataset_questions[model_name][dataset_name][shots_0])
                                        model_comparison_data.append({
                                            'model': model_name,
                                            'dataset': dataset_name,
                                            'count': count,
                                            'shot_type': f"{shots_0}-shot"
                                        })

                        if model_comparison_data:
                            model_comparison_df = pd.DataFrame(model_comparison_data)

                            fig_shot0 = px.bar(
                                model_comparison_df,
                                x='dataset',
                                y='count',
                                color='model',
                                barmode='group',
                                labels={
                                    'count': 'Number of Questions',
                                    'model': 'Model',
                                    'dataset': 'Dataset'
                                },
                                title=f'Model Comparison for {shots_0}-shot Questions'
                            )

                            fig_shot0.update_layout(
                                xaxis_title="Dataset",
                                yaxis_title="Number of Questions",
                                xaxis_tickangle=45,
                                height=500
                            )

                            st.plotly_chart(fig_shot0, use_container_width=True)

                            # Add a summary for 0-shot comparison
                            st.subheader(f"Summary for {shots_0}-shot Questions")
                            summary0 = model_comparison_df.groupby('model')['count'].sum().reset_index()
                            summary0.columns = ['Model', 'Total Questions']
                            st.dataframe(summary0, use_container_width=True)
                        else:
                            st.info(f"No data available for {shots_0}-shot comparison")

                    # 4. Model comparison for 5-shot questions
                    with tab4:
                        # Get counts of 5-shot questions per dataset for both models
                        shots_5 = shots_options[1]
                        model_comparison_data = []

                        for dataset_name in dataset_performance['dataset'].unique():
                            for model_name in model_dataset_questions:
                                if dataset_name in model_dataset_questions[model_name]:
                                    if shots_5 in model_dataset_questions[model_name][dataset_name]:
                                        count = len(model_dataset_questions[model_name][dataset_name][shots_5])
                                        model_comparison_data.append({
                                            'model': model_name,
                                            'dataset': dataset_name,
                                            'count': count,
                                            'shot_type': f"{shots_5}-shot"
                                        })

                        if model_comparison_data:
                            model_comparison_df = pd.DataFrame(model_comparison_data)

                            fig_shot5 = px.bar(
                                model_comparison_df,
                                x='dataset',
                                y='count',
                                color='model',
                                barmode='group',
                                labels={
                                    'count': 'Number of Questions',
                                    'model': 'Model',
                                    'dataset': 'Dataset'
                                },
                                title=f'Model Comparison for {shots_5}-shot Questions'
                            )

                            fig_shot5.update_layout(
                                xaxis_title="Dataset",
                                yaxis_title="Number of Questions",
                                xaxis_tickangle=45,
                                height=500
                            )

                            st.plotly_chart(fig_shot5, use_container_width=True)

                            # Add a summary for 5-shot comparison
                            st.subheader(f"Summary for {shots_5}-shot Questions")
                            summary5 = model_comparison_df.groupby('model')['count'].sum().reset_index()
                            summary5.columns = ['Model', 'Total Questions']
                            st.dataframe(summary5, use_container_width=True)
                        else:
                            st.info(f"No data available for {shots_5}-shot comparison")
                        with tab5:
                            # Get the model names for easier reference
                            model1_name = list(model_options.keys())[0]
                            model2_name = list(model_options.keys())[1]
                            shots_0 = shots_options[0]  # 0-shot

                            # Process intersection data between models for 0-shot
                            model_intersection_data = []

                            for dataset_name in dataset_performance['dataset'].unique():
                                # Skip if data for either model is missing
                                if (dataset_name not in model_dataset_questions.get(model1_name, {}) or
                                        dataset_name not in model_dataset_questions.get(model2_name, {})):
                                    continue

                                # Skip if 0-shot data is missing for either model
                                if (shots_0 not in model_dataset_questions[model1_name][dataset_name] or
                                        shots_0 not in model_dataset_questions[model2_name][dataset_name]):
                                    continue

                                # Get question sets for each model
                                model1_questions = model_dataset_questions[model1_name][dataset_name][shots_0]
                                model2_questions = model_dataset_questions[model2_name][dataset_name][shots_0]

                                # Calculate intersections and unique questions
                                common_questions = model1_questions.intersection(model2_questions)
                                only_model1 = model1_questions.difference(model2_questions)
                                only_model2 = model2_questions.difference(model1_questions)

                                # Add to result data
                                model_intersection_data.append({
                                    'dataset': dataset_name,
                                    'category': f'Common to both models',
                                    'count': len(common_questions),
                                    'model': 'Both'
                                })

                                model_intersection_data.append({
                                    'dataset': dataset_name,
                                    'category': f'Only in {model1_name}',
                                    'count': len(only_model1),
                                    'model': model1_name
                                })

                                model_intersection_data.append({
                                    'dataset': dataset_name,
                                    'category': f'Only in {model2_name}',
                                    'count': len(only_model2),
                                    'model': model2_name
                                })

                            # Create visualization for 0-shot model intersection
                            if model_intersection_data:
                                model_intersection_df = pd.DataFrame(model_intersection_data)

                                fig_model_intersection_0 = px.bar(
                                    model_intersection_df,
                                    x='dataset',
                                    y='count',
                                    color='category',
                                    barmode='stack',
                                    labels={
                                        'count': 'Number of Questions',
                                        'category': 'Question Category',
                                        'dataset': 'Dataset'
                                    },
                                    title=f'Question Intersection Analysis Between Models (0-shot)'
                                )

                                fig_model_intersection_0.update_layout(
                                    xaxis_title="Dataset",
                                    yaxis_title="Number of Questions",
                                    xaxis_tickangle=45,
                                    height=500,
                                    legend_title="Question Category"
                                )

                                st.plotly_chart(fig_model_intersection_0, use_container_width=True)

                                # Show a summary table for 0-shot model intersection
                                st.subheader(f"Summary for 0-shot Model Intersection")
                                summary_df = model_intersection_df.groupby('category')['count'].sum().reset_index()
                                summary_df['percentage'] = (
                                            summary_df['count'] / summary_df['count'].sum() * 100).round(1)
                                summary_df.columns = ['Category', 'Count', 'Percentage (%)']
                                st.dataframe(summary_df, use_container_width=True)
                            else:
                                st.info(f"No intersection data available for 0-shot model comparison")

                        # Add the implementation for the 6th tab (Model Intersection for 5-shot)
                        with tab6:
                            # Get the model names for easier reference
                            model1_name = list(model_options.keys())[0]
                            model2_name = list(model_options.keys())[1]
                            shots_5 = shots_options[1]  # 5-shot

                            # Process intersection data between models for 5-shot
                            model_intersection_data = []

                            for dataset_name in dataset_performance['dataset'].unique():
                                # Skip if data for either model is missing
                                if (dataset_name not in model_dataset_questions.get(model1_name, {}) or
                                        dataset_name not in model_dataset_questions.get(model2_name, {})):
                                    continue

                                # Skip if 5-shot data is missing for either model
                                if (shots_5 not in model_dataset_questions[model1_name][dataset_name] or
                                        shots_5 not in model_dataset_questions[model2_name][dataset_name]):
                                    continue

                                # Get question sets for each model
                                model1_questions = model_dataset_questions[model1_name][dataset_name][shots_5]
                                model2_questions = model_dataset_questions[model2_name][dataset_name][shots_5]

                                # Calculate intersections and unique questions
                                common_questions = model1_questions.intersection(model2_questions)
                                only_model1 = model1_questions.difference(model2_questions)
                                only_model2 = model2_questions.difference(model1_questions)

                                # Add to result data
                                model_intersection_data.append({
                                    'dataset': dataset_name,
                                    'category': f'Common to both models',
                                    'count': len(common_questions),
                                    'model': 'Both'
                                })

                                model_intersection_data.append({
                                    'dataset': dataset_name,
                                    'category': f'Only in {model1_name}',
                                    'count': len(only_model1),
                                    'model': model1_name
                                })

                                model_intersection_data.append({
                                    'dataset': dataset_name,
                                    'category': f'Only in {model2_name}',
                                    'count': len(only_model2),
                                    'model': model2_name
                                })

                            # Create visualization for 5-shot model intersection
                            if model_intersection_data:
                                model_intersection_df = pd.DataFrame(model_intersection_data)

                                fig_model_intersection_5 = px.bar(
                                    model_intersection_df,
                                    x='dataset',
                                    y='count',
                                    color='category',
                                    barmode='stack',
                                    labels={
                                        'count': 'Number of Questions',
                                        'category': 'Question Category',
                                        'dataset': 'Dataset'
                                    },
                                    title=f'Question Intersection Analysis Between Models (5-shot)'
                                )

                                fig_model_intersection_5.update_layout(
                                    xaxis_title="Dataset",
                                    yaxis_title="Number of Questions",
                                    xaxis_tickangle=45,
                                    height=500,
                                    legend_title="Question Category"
                                )

                                st.plotly_chart(fig_model_intersection_5, use_container_width=True)

                                # Show a summary table for 5-shot model intersection
                                st.subheader(f"Summary for 5-shot Model Intersection")
                                summary_df = model_intersection_df.groupby('category')['count'].sum().reset_index()
                                summary_df['percentage'] = (
                                            summary_df['count'] / summary_df['count'].sum() * 100).round(1)
                                summary_df.columns = ['Category', 'Count', 'Percentage (%)']
                                st.dataframe(summary_df, use_container_width=True)
                            else:
                                st.info(f"No intersection data available for 5-shot model comparison")

                    # Add a downloadable CSV for the intersection data
                    intersection_csv = intersection_df.to_csv(index=False)
                    st.download_button(
                        label="Download Intersection Analysis Data",
                        data=intersection_csv,
                        file_name=f"question_intersection_analysis_{threshold}.csv",
                        mime="text/csv",
                    )
                else:
                    st.info(
                        "Insufficient data to perform intersection analysis. Need at least two shot settings with questions.")
                st.download_button(
                    label="Download Full Comparison Data",
                    data=csv,
                    file_name=f"all_datasets_comparison_{threshold}.csv",
                    mime="text/csv",
                )
            else:
                st.warning(
                    "No data found for other model and shot combinations. Make sure the files exist in the expected directory structure.")
    # Question explorer
    st.header("Question Explorer")

    # Allow user to select a question
    available_questions = sorted(data['sample_index'].unique())
    # Convert numpy int to Python int for the selectbox
    available_questions = [int(q) for q in available_questions]
    selected_question = st.selectbox("Select a question to explore", available_questions)

    # Filter data for the selected question
    question_data = data[data['sample_index'] == selected_question].copy()

    # Display question details
    st.subheader(f"Question {selected_question} Details")

    # Fetch and display question content
    with st.expander("View Question Content", expanded=True):
        question_content = load_dataset_and_get_question(dataset, selected_question)
        st.markdown(question_content)

    # Show performance metrics for this question
    q_correct = question_data['score'].sum()
    q_total = len(question_data)
    q_accuracy = q_correct / q_total if q_total > 0 else 0

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    with metrics_col1:
        st.metric("Accuracy", f"{q_accuracy:.2%}")
    with metrics_col2:
        st.metric("Correct Answers", f"{q_correct}/{q_total}")
    with metrics_col3:
        st.metric("Configurations", q_total)

    # Response analysis
    if 'chosen_position' in question_data.columns:
        st.subheader("Response Distribution")

        position_count = question_data['chosen_position'].value_counts().reset_index()
        position_count.columns = ['Position', 'Count']

        # Create a bar chart of position distribution
        fig_pos = px.bar(
            position_count,
            x='Position',
            y='Count',
            title='Distribution of Chosen Answer Positions',
            color='Position',
            labels={'Position': 'Answer Position', 'Count': 'Frequency'}
        )
        st.plotly_chart(fig_pos, use_container_width=True)

    # Random sample for analysis
    st.subheader("Random Sample for Analysis")

    sample_size = min(5, len(question_data))
    if st.button(f"Show {sample_size} Random Samples"):
        # Get random sample of rows for this question
        if len(question_data) <= sample_size:
            sampled_data = question_data
        else:
            sampled_data = question_data.sample(sample_size)

        # Display each sample
        for i, (idx, row) in enumerate(sampled_data.iterrows()):
            with st.expander(f"Sample {i + 1} (ID: {row['evaluation_id'][:10]}...)", expanded=True):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Sample Details**")
                    st.markdown(f"**Dataset:** {row['dataset']}")
                    st.markdown(f"**Choices Order:** {row['choices_order']}")
                    st.markdown(f"**Selected Answer:** {row['closest_answer']}")
                    st.markdown(f"**Ground Truth:** {row['ground_truth']}")
                    st.markdown(f"**Score:** {row['score']}")

                with col2:
                    st.markdown("**Generated Text**")
                    st.text_area("Model Output", row['generated_text'], height=200, key=f"text_{i}")

    # Raw data view
    st.subheader("Raw Data View")
    with st.expander("View All Data for This Question"):
        st.dataframe(question_data, use_container_width=True)

    # Download the filtered data
    csv = question_data.to_csv(index=False)
    st.download_button(
        label="Download Question Data as CSV",
        data=csv,
        file_name=f"question_{selected_question}_{model_name}_{shots}shots_{threshold}threshold.csv",
        mime="text/csv",
    )
else:
    st.info("Please select parameters to explore the results.")

# Add footer
st.markdown("---")
st.markdown("**Model Performance Analysis Explorer** | Data from analysis pipeline")