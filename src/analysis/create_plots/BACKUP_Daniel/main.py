import os
import pandas as pd
from multiprocessing import Pool, Manager
from tqdm import tqdm
from dataclasses import dataclass



def map_answer_position(answer: str, enumerator: str) -> int:
    """
    Maps an answer to its position based on the enumerator type.

    Args:
        answer: The answer string to map
        enumerator: The type of enumerator used

    Returns:
        int: The position (1-based) of the answer
    """
    position_mappings = {
        'greek': "αβγδεζηθικ",
        'keyboard': "!@#$%^₪*)(",
        'capitals': "ABCDEFGHIJ",
        'lowercase': "abcdefghij",
        'numbers': [str(i + 1) for i in range(10)],
        'roman': ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    }

    prefix = answer.split('.')[0].strip()
    mapping = position_mappings.get(enumerator, "")
    return mapping.index(prefix) + 1 if prefix in mapping else exit(1)


def analyze_low_performance_questions(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """
    Analyzes questions with low performance across variations.

    Args:
        df: DataFrame containing the model's answers
        threshold: Accuracy threshold for identifying low-performance questions

    Returns:
        pd.DataFrame: Processed data for low-performance questions
    """
    # Calculate accuracy per question
    accuracy_stats = (
        df.groupby('sample_index')
        .agg({
            'score': ['sum', 'count']
        })
        .reset_index()
    )
    accuracy_stats.columns = ['sample_index', 'correct_count', 'total_count']
    accuracy_stats['accuracy'] = (accuracy_stats['correct_count'] / accuracy_stats['total_count']).round(3)

    # Filter low performers
    low_performers = accuracy_stats[accuracy_stats['accuracy'] < threshold]
    # Keep only rows in df where sample_index is in low_performers
    filtered_df = df[df['sample_index'].isin(low_performers['sample_index'])]

    return filtered_df



@dataclass
class Config:
    model_name: str
    shots_selected: int
    dataset: str


def process_configuration(config: Config) -> None:
    """
    Process a single configuration of model and shots count.

    Args:
        config: Config object containing model_name, shots_selected, dataset
    """
    model_name, shots_selected, dataset = config.model_name, config.shots_selected, config.dataset
    print(f"Processing model: {model_name} with {shots_selected} shots for dataset: {dataset}")

    # Load data for current configuration
    data_loader = DataLoader()
    df_partial = data_loader.load_and_process_data(
        model_name=model_name,
        shots=shots_selected,
        datasets=[dataset],
        max_samples=None
    )

    if df_partial.empty:
        return

    # Filter out specific choices orders
    df_partial = df_partial[
        ((df_partial['shots'] != 5) | (~df_partial['choices_order'].isin(["correct_first", "correct_last"])))
    ]

    # Process low-performance questions
    df = analyze_low_performance_questions(df_partial)
    df["closest_answer_position"] = df.apply(lambda row: map_answer_position(row["closest_answer"], row["enumerator"]),
                                            axis=1)
    df["ground_truth_position"] = df.apply(lambda row: map_answer_position(row["ground_truth"], row["enumerator"]),
                                            axis=1)

    if not df.empty:
        # Create output directory
        base_results_dir = os.path.abspath("../app/results_local")
        output_dir = os.path.join(
            base_results_dir,
            f"Shots_{shots_selected}",
            model_name.replace('/', '_'),
            dataset.replace('/', '_')
        )
        os.makedirs(output_dir, exist_ok=True)

        # Save results
        output_path = os.path.join(output_dir, 'low_performance_questions.parquet')
        df.to_parquet(output_path)
        print(f"Saved results to {output_path}")


def run_configuration_analysis(num_processes: int = 1) -> None:
    """
    Run the analysis pipeline for multiple configurations.

    Args:
        num_processes: Number of parallel processes to use
    """
    # Configuration
    shots_to_evaluate = [0, 5]
    models_to_evaluate = ['allenai/OLMoE-1B-7B-0924-Instruct', 'meta-llama/Meta-Llama-3-8B-Instruct']
    mmlu_subtasks = ["college_biology", "high_school_european_history", "marketing", "sociology", "world_religions"]

    datasets = (
            [
                "ai2_arc.arc_challenge",
                "ai2_arc.arc_easy",
                "hellaswag",
                "openbook_qa",
                "social_iqa",
            ] + [f"mmlu.{task}" for task in mmlu_subtasks]
    )

    # Create parameter combinations
    params_list = [
        Config(model_name, shots, dataset)
        for model_name in models_to_evaluate
        for shots in shots_to_evaluate
        for dataset in datasets
    ]

    # Run analysis
    with Manager() as manager:
        with Pool(processes=num_processes) as pool:
            list(tqdm(
                pool.imap_unordered(process_configuration, params_list),
                total=len(params_list),
                desc="Processing configurations"
            ))


if __name__ == "__main__":
    run_configuration_analysis(num_processes=4)
