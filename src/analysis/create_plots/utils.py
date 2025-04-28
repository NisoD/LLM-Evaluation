# utils.py
from typing import Optional
import pandas as pd
from pathlib import Path


class AnswerMapper:
    """Maps answer strings to numerical positions."""

    @staticmethod
    def map_position(answer: str, enumerator: Enumerator) -> int:
        """
        Maps an answer to its position (1-based).

        Args:
            answer: The answer string
            enumerator: The enumerator type

        Returns:
            The 1-based position of the answer

        Raises:
            ValueError: If the answer format is invalid
        """
        try:
            prefix = answer.split('.')[0].strip()
            mapping = enumerator.mapping
            return mapping.index(prefix) + 1
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid answer format: {answer} for enumerator {enumerator}") from e


class DataFrameUtils:
    """Utility functions for DataFrame operations."""

    @staticmethod
    def save_parquet(
            df: pd.DataFrame,
            base_path: Path,
            model_name: str,
            dataset: str,
            shots: int
    ) -> Path:
        """Save DataFrame as parquet with organized directory structure."""
        output_dir = base_path / f"Shots_{shots}" / model_name.replace('/', '_') / dataset.replace('/', '_')
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / 'low_performance_questions.parquet'
        df.to_parquet(output_path)
        return output_path