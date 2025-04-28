# helpers.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import pandas as pd
import logging
from datetime import datetime
THRESHOLD_FOR_EVALUATION = 0.1
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the analysis pipeline."""
    # Dataset configuration
    mmlu_subtasks: List[str] = field(default_factory=lambda: [
        "college_biology",
        "high_school_european_history",
        "marketing",
        "sociology",
        "world_religions"
    ])

    base_datasets: List[str] = field(default_factory=lambda: [
        "ai2_arc.arc_challenge",
        "ai2_arc.arc_easy",
        "hellaswag",
        "openbook_qa",
        "social_iqa"
    ])

    # Model configuration
    models: List[str] = field(default_factory=lambda: [
        "allenai/OLMoE-1B-7B-0924-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct"
    ])

    shots: List[int] = field(default_factory=lambda: [0, 5])

    # Analysis parameters
    performance_threshold: float = THRESHOLD_FOR_EVALUATION
    num_processes: int = 4
    results_dir: Path = Path("../app/results_local")

    @property
    def all_datasets(self) -> List[str]:
        """Get complete list of datasets including MMLU subtasks."""
        return self.base_datasets + [f"mmlu.{task}" for task in self.mmlu_subtasks]
class AnswerMapper:
    """Handles answer position mapping and validation."""

    POSITION_MAPPINGS = {
        'greek': "αβγδεζηθικ",
        'keyboard': "!@#$%^₪*)(",
        'capitals': "ABCDEFGHIJ",
        'lowercase': "abcdefghij",
        'numbers': "123456789",
        'roman': ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    }

    @classmethod
    def map_position(cls, answer: str, enumerator: str) -> int:
        """Maps an answer to its numerical position."""
        try:
            prefix = answer.split('.')[0].strip()
            mapping = cls.POSITION_MAPPINGS.get(enumerator, "")

            if isinstance(mapping, list):  # Handle Roman numerals
                return mapping.index(prefix) + 1
            return mapping.index(prefix) + 1

        except (ValueError, AttributeError) as e:
            logger.error(f"Invalid answer format: {answer} for enumerator {enumerator}")
            raise ValueError(f"Cannot map position for answer: {answer}") from e


class ModelAnalyzer:
    """Analyzes model performance on questions."""

    def __init__(self, threshold: float = THRESHOLD_FOR_EVALUATION):
        self.threshold = threshold

    def analyze_low_performance(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Analyze questions with performance below threshold."""
        logger.info("Starting low performance analysis...")
        logger.info(f"Initial data shape: {df.shape}")

        # Calculate per-question accuracy
        question_stats = self._calculate_question_stats(df)

        # Identify poor performers
        low_performers = question_stats[question_stats['accuracy'] < self.threshold]
        logger.info(f"Found {len(low_performers)} low performing questions")

        if low_performers.empty:
            return None

        # Analyze poor performing questions
        return self._analyze_poor_performers(df, low_performers, question_stats)

    def _calculate_question_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate accuracy statistics for each question."""
        stats = (
            df.groupby('sample_index')
            .agg({
                'score': ['sum', 'count']
            })
            .reset_index()
        )

        stats.columns = ['sample_index', 'correct_count', 'total_count']
        stats['accuracy'] = stats['correct_count'] / stats['total_count']

        return stats

    def _analyze_poor_performers(
            self,
            df: pd.DataFrame,
            low_performers: pd.DataFrame,
            question_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """Analyze distribution of answers for poorly performing questions."""
        poor_performing_df = df[df['sample_index'].isin(low_performers['sample_index'])].copy()

        # Map answers to positions
        poor_performing_df['answer_position'] = poor_performing_df.apply(
            lambda row: min(AnswerMapper.map_position(row['closest_answer'], row['enumerator']), 4),
            axis=1
        )

        distributions = []
        for idx in poor_performing_df['sample_index'].unique():
            distribution = self._analyze_question_distribution(
                poor_performing_df, idx, question_stats
            )
            distributions.append(distribution)
            self._log_question_analysis(distribution)

        result_df = pd.DataFrame(distributions)
        self._log_overall_distribution(result_df)

        return result_df

    def _analyze_question_distribution(
            self,
            df: pd.DataFrame,
            question_idx: int,
            stats: pd.DataFrame
    ) -> Dict:
        """Analyze answer distribution for a single question."""
        question_data = df[df['sample_index'] == question_idx]
        total_responses = len(question_data)

        distribution = {
            'sample_index': question_idx,
            'total_responses': total_responses,
            'accuracy': stats.loc[
                stats['sample_index'] == question_idx,
                'accuracy'
            ].iloc[0],
            'timestamp': datetime.now().isoformat()
        }

        position_counts = question_data['answer_position'].value_counts()
        for choice in range(1, 5):
            count = position_counts.get(choice, 0)
            percentage = (count / total_responses * 100) if total_responses > 0 else 0
            distribution.update({
                f'choice_{choice}_count': count,
                f'choice_{choice}_pct': round(percentage, 2)
            })

        return distribution

    def _log_question_analysis(self, distribution: Dict) -> None:
        """Log analysis results for a single question."""
        logger.info(f"\nQuestion {distribution['sample_index']} "
                    f"(Accuracy: {distribution['accuracy']:.2f})")

        total_pct = 0
        for choice in range(1, 5):
            pct = distribution[f'choice_{choice}_pct']
            count = distribution[f'choice_{choice}_count']
            logger.info(f"  Choice {choice}: {pct:.2f}% ({count} responses)")
            total_pct += pct
        logger.info(f"  Total: {total_pct:.2f}%")

    def _log_overall_distribution(self, result_df: pd.DataFrame) -> None:
        """Log overall distribution statistics."""
        logger.info("\n=== Overall Answer Distribution ===")
        total_responses = result_df['total_responses'].sum()

        for choice in range(1, 5):
            total_choice = result_df[f'choice_{choice}_count'].sum()
            overall_pct = (total_choice / total_responses * 100)
            logger.info(f"Choice {choice}: {overall_pct:.2f}% "
                        f"(Total: {total_choice} responses)")


def save_results(
        df: pd.DataFrame,
        config: PipelineConfig,
        model_name: str,
        dataset: str,
        shots: int
) -> Path:
    """Save analysis results to parquet file."""
    output_dir = (
            config.results_dir
            / f"Shots_{shots}"
            / model_name.replace('/', '_')
            / dataset.replace('/', '_')
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / 'low_performance_questions.parquet'
    df.to_parquet(output_path)
    logger.info(f"Saved results to {output_path}")

    return output_path