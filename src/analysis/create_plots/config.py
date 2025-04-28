# config.py
from dataclasses import dataclass
from typing import List, Literal, Optional
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    """Configuration for dataset evaluation."""
    mmlu_subtasks: List[str] = (
        "college_biology", "high_school_european_history",
        "marketing", "sociology", "world_religions"
    )

    base_datasets: List[str] = (
        "ai2_arc.arc_challenge",
        "ai2_arc.arc_easy",
        "hellaswag",
        "openbook_qa",
        "social_iqa"
    )

    @property
    def all_datasets(self) -> List[str]:
        """Get all datasets including MMLU subtasks."""
        return self.base_datasets + [f"mmlu.{task}" for task in self.mmlu_subtasks]


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for model evaluation."""
    models: List[str] = (
        "allenai/OLMoE-1B-7B-0924-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct"
    )
    shots: List[int] = (0, 5)


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for analysis parameters."""
    performance_threshold: float = 0.1
    num_processes: int = 4
    results_dir: Path = Path("../app/results_local")