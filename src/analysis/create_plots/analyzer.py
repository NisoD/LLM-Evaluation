# analyzer.py
from typing import Optional
import pandas as pd
from dataclasses import dataclass

from analysis.create_plots.enums import Enumerator
from analysis.create_plots.models import EvaluationResult
from analysis.create_plots.utils import AnswerMapper


@dataclass
class ModelAnalyzer:
    """Analyzes model performance on questions."""

    threshold: float

    def analyze_low_performance(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Analyze questions with performance below threshold.

        Args:
            df: DataFrame with model responses

        Returns:
            DataFrame with analysis results or None if no low performers
        """
        # Calculate per-question accuracy
        question_stats = (
            df.groupby('sample_index')
            .agg({
                'score': ['sum', 'count']
            })
            .reset_index()
        )

        question_stats.columns = ['sample_index', 'correct_count', 'total_count']
        question_stats['accuracy'] = question_stats['correct_count'] / question_stats['total_count']

        # Identify poor performers
        low_performers = question_stats[question_stats['accuracy'] < self.threshold]

        if low_performers.empty:
            return None

        # Analyze distribution for poor performers
        results = []
        for idx in low_performers['sample_index']:
            result = self._analyze_question(df, idx, question_stats)
            results.append(result.to_dict())

        return pd.DataFrame(results)

    def _analyze_question(
            self,
            df: pd.DataFrame,
            question_idx: int,
            stats: pd.DataFrame
    ) -> EvaluationResult:
        """Analyze a single question's performance."""
        question_data = df[df['sample_index'] == question_idx]
        total_responses = len(question_data)

        # Get answer positions
        positions = question_data.apply(
            lambda row: min(
                AnswerMapper.map_position(row['closest_answer'], Enumerator(row['enumerator'])),
                4
            ),
            axis=1
        )

        # Calculate distributions
        distributions = {}
        pos_counts = positions.value_counts()

        for choice in range(1, 5):
            count = pos_counts.get(choice, 0)
            pct = (count / total_responses * 100) if total_responses > 0 else 0
            distributions[choice] = (count, round(pct, 2))

        return EvaluationResult(
            sample_index=question_idx,
            total_responses=total_responses,
            accuracy=stats.loc[stats['sample_index'] == question_idx, 'accuracy'].iloc[0],
            choice_distributions=distributions
        )