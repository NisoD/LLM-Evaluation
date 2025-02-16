import json
import os
from multiprocessing import Pool, Manager

from tqdm import tqdm

from src.analysis.create_plots.DataLoader import DataLoader
from src.analysis.create_plots.HammingDistanceClusterAnalyzerAxes import HammingDistanceClusterAnalyzerAxes
from src.analysis.create_plots.ModelPerformanceAnalyzer import ModelPerformanceAnalyzer
from src.analysis.create_plots.PromptConfigurationAnalyzerAxes import PromptConfigurationAnalyzerAxes
from src.analysis.create_plots.PromptQuestionAnalyzer import PromptQuestionAnalyzer

#
# def process_configuration(params):
#     """
#     Process a single configuration of model and shots count
#     """
#     model_name, shots_selected, dataset = params
#     # for Debugging:
#     # model_name = "mistralai/Mistral-7B-Instruct-v0.3"
#     print(f"Processing model: {model_name} with {shots_selected} shots")
#
#     # analyzer = PromptConfigurationAnalyzerAxes()
#     # hamming = HammingDistanceClusterAnalyzerAxes()
#     # prompt_question_analyzer = PromptQuestionAnalyzer()
#     # performance_analyzer = ModelPerformanceAnalyzer()
#     # Load data for current configuration
#     data_loader = DataLoader()
#     df_partial = data_loader.load_and_process_data(model_name=model_name,
#                                                    shots=shots_selected,
#                                                    datasets=[dataset],
#                                                    max_samples=None)
#     # if shots_selected == 5:
#     #     df_partial = df_partial[~df_partial.choices_order.isin(["correct_first", "correct_last"])]
#     # base_results_dir = "../app/results_local"
#     if df_partial.empty:
#         return
#     df_partial = df_partial[~df_partial.choices_order.isin(["correct_first", "correct_last"])]
#     base_results_dir = "../app/results_local"
#     # create global path from the base results directory withou ".."
#     base_results_dir = os.path.abspath(base_results_dir)
#
#     os.makedirs(base_results_dir, exist_ok=True)
#     #
#     # performance_analyzer.generate_model_performance_comparison(
#     #     df=df_partial,
#     #     model_name=model_name,
#     #     shots_selected=shots_selected,
#     #     base_results_dir=base_results_dir
#     # )
#     #
#     # filtered_datasets = analyzer.process_and_visualize_configurations(
#     #     df=df_partial,
#     #     model_name=model_name,
#     #     shots_selected=shots_selected,
#     #     interesting_datasets=[dataset],
#     #     base_results_dir=base_results_dir
#     # )
#
#     # interesting_datasets = list(filtered_datasets)
#     #
#     # hamming.perform_clustering_for_model(
#     #     df=df_partial,
#     #     model_name=model_name,
#     #     shots_selected=shots_selected,
#     #     interesting_datasets=interesting_datasets,
#     #     base_results_dir=base_results_dir
#     # )
#     #
#     # prompt_question_analyzer.process_and_visualize_questions(
#     #     df=df_partial,
#     #     model_name=model_name,
#     #     shots_selected=shots_selected,
#     #     interesting_datasets=interesting_datasets,
#     #     base_results_dir=base_results_dir
#     # )
#
#
# def run_configuration_analysis(num_processes=1) -> None:
#     """
#     Run the main analysis pipeline in parallel for evaluating prompt configurations
#     across different models and shot counts.
#     """
#     # Configuration parameters
#     shots_to_evaluate = [0,5]
#     models_to_evaluate = [
#         # 'meta-llama/Llama-3.2-1B-Instruct',
#         # 'allenai/OLMoE-1B-7B-0924-Instruct',
#         # 'meta-llama/Meta-Llama-3-8B-Instruct',
#         'meta-llama/Llama-3.2-3B-Instruct',
#         # 'mistralai/Mistral-7B-Instruct-v0.3',
#     ]
#     interesting_datasets = [
#         # "ai2_arc.arc_challenge",
#         # "ai2_arc.arc_easy",
#         # "hellaswag",
#         # "openbook_qa",
#         # "social_iqa",
#     ]
#
#     subtasks = [
#         "abstract_algebra",
#         "anatomy",
#         "astronomy",
#         "business_ethics",
#         "clinical_knowledge",
#         "college_biology",
#         "college_chemistry",
#         "college_computer_science",
#         "college_mathematics",
#         "college_medicine",
#         "college_physics",
#         "computer_security",
#         "conceptual_physics",
#         "econometrics",
#         "electrical_engineering",
#         "elementary_mathematics",
#         "formal_logic",
#         "global_facts",
#         "high_school_biology",
#         "high_school_chemistry",
#         "high_school_computer_science",
#         "high_school_european_history",
#         "high_school_geography",
#         "high_school_government_and_politics",
#         "high_school_macroeconomics",
#         "high_school_mathematics",
#         "high_school_microeconomics",
#         "high_school_physics",
#         "high_school_psychology",
#         "high_school_statistics",
#         "high_school_us_history",
#         "high_school_world_history",
#         "human_aging",
#         "human_sexuality",
#         "international_law",
#         "jurisprudence",
#         "logical_fallacies",
#         "machine_learning",
#         "management",
#         "marketing",
#         "medical_genetics",
#         "miscellaneous",
#         "moral_disputes",
#         "moral_scenarios",
#         "nutrition",
#         "philosophy",
#         "prehistory",
#         "professional_accounting",
#             "professional_law",
#         "professional_medicine",
#         "professional_psychology",
#         "public_relations",
#         "security_studies",
#         "sociology",
#         "us_foreign_policy",
#         "virology",
#         "world_religions",
#     ]
#     pro_subtuask = [
#         "history",
#         "law",
#         "health",
#         "physics",
#         "business",
#         "other",
#         "philosophy",
#         "psychology",
#         "economics",
#         "math",
#         "biology",
#         "chemistry",
#         "computer_science",
#         "engineering",
#     ]
#     interesting_datasets.extend(["mmlu."+ name for name in subtasks])
#     interesting_datasets = ["mmlu_pro." + name for name in pro_subtuask]
#
#     # Setup results directory
#
#     # Create parameter combinations for parallel processing
#     # model_name, shots_selected, dataset
#     params_list = [
#         (model_name, shots_selected, dataset)
#         for dataset in interesting_datasets
#         for shots_selected in shots_to_evaluate
#         for model_name in models_to_evaluate
#     ]
#
#     #
#     # interesting_datasets = ['hellaswag']
#     # model_name = "meta-llama/Llama-3.2-3B-Instruct"
#     # models_to_evaluate = [model_name]
#     # shots_selected = 0
#     # templates = [
#     #     "MultipleChoiceTemplatesInstructionsStandard",
#     #     "MultipleChoiceTemplatesInstructionsWithoutTopicHarness",
#     #     "MultipleChoiceTemplatesInstructionsProSACould"
#     # ]
#     # params_list = [(model_name, shots_selected, dataset) for dataset in interesting_datasets]
#     with Manager() as manager:
#         with Pool(processes=num_processes) as pool:
#             for _ in tqdm(
#                     pool.imap_unordered(process_configuration_with_immediate_error, params_list),
#                     total=len(params_list),
#                     desc="Processing configurations"
#             ):
#                 pass
#
#
# import traceback
# from datetime import datetime
#
#
# def immediate_error_callback(error, params):
#     print("\n" + "=" * 50)
#     print(f"Error occurred at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"Parameters that failed: {json.dumps(params, indent=2)}")
#     print(f"Error message: {str(error)}")
#     print("Traceback:")
#     print(traceback.format_exc())
#     print("=" * 50 + "\n")
#
#
# def process_configuration_with_immediate_error(params):
#     try:
#         return process_configuration(params)
#     except Exception as e:
#         immediate_error_callback(e, params)
#         return {'status': 'error', 'params': params, 'error': str(e)}
#
#
# if __name__ == "__main__":
#     run_configuration_analysis(num_processes=4)
import json
import os
from typing import List, Dict
import pandas as pd
from multiprocessing import Pool, Manager
from tqdm import tqdm


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
    return mapping.index(prefix) + 1 if prefix in mapping else 0


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

    # Process answers for low-performing questions
    if not low_performers.empty:
        result_df = (
            df[df['sample_index'].isin(low_performers['sample_index'])]
            .assign(
                answer_position=lambda x: x.apply(
                    lambda row: map_answer_position(row['closest_answer'], row['enumerator']),
                    axis=1
                )
            )
            .merge(
                accuracy_stats[['sample_index', 'accuracy']],
                on='sample_index',
                how='left'
            )
        )
        return result_df

    return pd.DataFrame()


def process_configuration(params: tuple) -> None:
    """
    Process a single configuration of model and shots count.

    Args:
        params: Tuple containing (model_name, shots_selected, dataset)
    """
    model_name, shots_selected, dataset = params
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
    df_partial = df_partial[~df_partial.choices_order.isin(["correct_first", "correct_last"])]

    # Process low-performance questions
    results_df = analyze_low_performance_questions(df_partial)

    if not results_df.empty:
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
        results_df.to_parquet(output_path)
        print(f"Saved results to {output_path}")


def process_with_error_handling(params: tuple) -> Dict:
    """
    Wrapper function for process_configuration with error handling.

    Args:
        params: Configuration parameters

    Returns:
        Dict: Status of the processing
    """
    try:
        return {'status': 'success', 'params': params, 'result': process_configuration(params)}
    except Exception as e:
        print(f"\nError processing {params}:\n{str(e)}")
        return {'status': 'error', 'params': params, 'error': str(e)}


def run_configuration_analysis(num_processes: int = 1) -> None:
    """
    Run the analysis pipeline for MMLU and MMLU Pro datasets.

    Args:
        num_processes: Number of parallel processes to use
    """
    # Configuration
    shots_to_evaluate = [0, 5]
    # models_to_evaluate = [
    #     #         # 'meta-llama/Llama-3.2-1B-Instruct',
        #         # 'allenai/OLMoE-1B-7B-0924-Instruct',
    #     #         # 'meta-llama/Meta-Llama-3-8B-Instruct',
    #     #         'meta-llama/Llama-3.2-3B-Instruct',
    #     #         # 'mistralai/Mistral-7B-Instruct-v0.3',
    #     #     ]
    models_to_evaluate = ['allenai/OLMoE-1B-7B-0924-Instruct']

    # MMLU datasets
    mmlu_subtasks = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]

    # MMLU Pro datasets
    mmlu_pro_subtasks = [
        "history", "law", "health", "physics", "business", "other",
        "philosophy", "psychology", "economics", "math", "biology",
        "chemistry", "computer_science", "engineering"
    ]

    # Combine all datasets
    datasets = (
        [
            "ai2_arc.arc_challenge",
            "ai2_arc.arc_easy",
            "hellaswag",
            "openbook_qa",
            "social_iqa",
        ]+
            [f"mmlu.{task}" for task in mmlu_subtasks] +
            [f"mmlu_pro.{task}" for task in mmlu_pro_subtasks]
    )

    # Create parameter combinations
    params_list = [
        (model_name, shots, dataset)
        for model_name in models_to_evaluate
        for shots in shots_to_evaluate
        for dataset in datasets
    ]

    # Run analysis
    with Manager() as manager:
        with Pool(processes=num_processes) as pool:
            list(tqdm(
                pool.imap_unordered(process_with_error_handling, params_list),
                total=len(params_list),
                desc="Processing configurations"
            ))


if __name__ == "__main__":
    run_configuration_analysis(num_processes=4)