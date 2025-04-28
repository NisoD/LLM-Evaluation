# MAIN.PY
import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Set
from pathlib import Path
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

from analysis.create_plots.DataLoader import DataLoader
import logging

from analysis.create_plots.get_example_from_row import InstanceLoader

THRESHOLD = 0.1
# Define the log file path
log_file = "analysis_log.txt"
# Configure logging to write to both console and file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode='w'),  # Write logs to file
        logging.StreamHandler()  # Also print logs to console
    ]
)


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for analysis pipeline with immutable attributes."""
    model_name: str
    shots_selected: int
    dataset: str

    @property
    def output_path(self) -> Path:
        """Generate standardized output path for results."""
        base_dir = Path("../app/results_local")
        return base_dir / f"Shots_{self.shots_selected}" / \
            self.model_name.replace('/', '_') / \
            self.dataset.replace('/', '_') / \
            f'low_performance_questions_{THRESHOLD}.parquet'


class AnswerMapper:
    """Maps multiple choice answer strings to valid positions (1-4)."""

    POSITION_MAPPINGS = {
        'greek': "αβγδεζηθικ",
        'keyboard': "!@#$%^₪*)(",
        'capitals': "ABCDEFGHIJ",
        'lowercase': "abcdefghij",
        'numbers': [str(i + 1) for i in range(10)],
        'roman': ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    }

    VALID_POSITIONS: Set[int] = {1, 2, 3, 4}

    @classmethod
    def get_position(cls, answer: str) -> Optional[int]:
        """
        Maps an answer to a valid multiple choice position.

        Args:
            answer: The answer string to map (e.g., "A.", "1)", "α.")

        Returns:
            Optional[int]: Position (1-4) if valid, None otherwise
        """
        if not answer or not isinstance(answer, str):
            return None

        try:
            prefix = answer.split('.')[0].strip()
            if not prefix:
                return None

            for mapping in cls.POSITION_MAPPINGS.values():
                if prefix in mapping:
                    position = mapping.index(prefix) + 1
                    return position if position in cls.VALID_POSITIONS else exit(1)

            return None

        except (AttributeError, IndexError):
            return None


class PerformanceAnalyzer:
    """Analyzes model performance on questions."""

    def __init__(self, threshold: float = THRESHOLD):
        self.threshold = threshold

    def find_low_performers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies questions with performance below threshold.

        Args:
            df: DataFrame with model answers and scores

        Returns:
            pd.DataFrame: Filtered data for low-performing questions
        """
        accuracy_stats = (
            df.groupby('sample_index') # Group By Question Index
            .agg({
                'score': ['sum', 'count']
                # Get Sum of score and Count Number of Instances
            })
            .reset_index()
        )

        accuracy_stats.columns = ['sample_index', 'correct_count', 'total_count']

        accuracy_stats['accuracy'] = (
                accuracy_stats['correct_count'] /
                accuracy_stats['total_count']
        ).round(3)

        low_performers = accuracy_stats[accuracy_stats['accuracy'] < self.threshold]

        # Log information about each low-performing sample
        for _, row in low_performers.iterrows():
            sample_index = row['sample_index']
            correct = row['correct_count']
            total = row['total_count']
            accuracy = row['accuracy']

            logging.info(
                f"Low performer detected - Sample {sample_index}: "
                f"Accuracy {accuracy:.1%} ({correct}/{total} correct)"
            )

        return df[df['sample_index'].isin(low_performers['sample_index'])]


def process_configuration(config: AnalysisConfig) -> Dict[str, any]:
    """
    Processes a single configuration of model and dataset.

    Args:
        config: Configuration parameters for analysis

    Returns:
        Dict containing processing status and metrics
    """
    try:
        print(f"Processing {config.model_name} with {config.shots_selected} "
              f"shots for {config.dataset}")

        data_loader = DataLoader()
        df = data_loader.load_and_process_data(
            model_name=config.model_name,
            shots=config.shots_selected,
            datasets=[config.dataset],
            max_samples=None
        )

        if df.empty:
            return {"status": "empty", "config": config}

        # Filter specific choices orders using vectorized operations
        mask = (df['shots'] != 5) | (~df['choices_order'].isin(["correct_first", "correct_last"]))
        df = df[mask]

        # Log total configurations per question
        question_counts = df.groupby("sample_index").size()
        correct_counts = df.groupby("sample_index")["score"].sum()

        for question_id in question_counts.index:
            logging.info(
                f"Question {question_id}: {question_counts[question_id]} configurations, "
                f"{correct_counts.get(question_id, 0)} correct"
            )

        # Analyze performance
        analyzer = PerformanceAnalyzer()
        df = analyzer.find_low_performers(df)

        if df.empty:
            return {"status": "no_low_performers", "config": config}

        # Map answers to positions with validation
        df = df.copy()
        initial_rows = len(df)

        # Map ground truth to position

        # Map model's chosen answer to position
        # Create a copy of df to prevent modifying the original during get_example_from_index
        working_df = df.copy()
        chosen_positions = InstanceLoader.get_example_from_index(config.dataset, working_df)

        # Only keep rows that exist in chosen_positions (some may have been dropped)
        df = df.loc[chosen_positions.index]
        df['chosen_position'] = chosen_positions

        # Log a sample of the results
        logging.info(f"Sample of chosen positions: {df['chosen_position'].sample(min(5, len(df))).to_string()}")

        if df.empty:
            return {
                "status": "invalid_mappings",
                "config": config,
                "initial_rows": initial_rows
            }

        # Save processed results
        output_path = config.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path)

        return {
            "status": "success",
            "config": config,
            "initial_rows": initial_rows,
            "final_rows": len(df)
        }

    except Exception as e:
        logging.exception(f"Error processing {config.model_name} with {config.dataset}")
        return {
            "status": "error",
            "config": config,
            "error": str(e)
        }


def run_analysis_pipeline(num_processes: int = 4) -> None:
    """
    Runs the complete analysis pipeline with parallel processing.

    Args:
        num_processes: Number of worker processes to use
    """
    models = [
        'allenai/OLMoE-1B-7B-0924-Instruct',
        'meta-llama/Meta-Llama-3-8B-Instruct'
    ]
    shots = [0, 5]
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
    datasets = base_datasets + [f"mmlu.{task}" for task in mmlu_subtasks]

    configs = [
        AnalysisConfig(model, shots_val, dataset)
        for model in models
        for shots_val in shots
        for dataset in datasets
    ]

    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_configuration, configs),
            total=len(configs),
            desc="Processing configurations"
        ))




if __name__ == "__main__":
    run_analysis_pipeline()
