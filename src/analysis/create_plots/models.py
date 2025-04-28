# models.py
from dataclasses import dataclass
from datetime import datetime


@dataclass
class EvaluationResult:
    """Represents a single evaluation result."""
    sample_index: int
    total_responses: int
    accuracy: float
    choice_distributions: dict[int, tuple[int, float]]  # {choice: (count, percentage)}
    timestamp: datetime = datetime.now()

    def to_dict(self) -> dict:
        """Convert the result to a dictionary format."""
        result = {
            'sample_index': self.sample_index,
            'total_responses': self.total_responses,
            'accuracy': self.accuracy,
            'timestamp': self.timestamp.isoformat()
        }

        for choice, (count, pct) in self.choice_distributions.items():
            result.update({
                f'choice_{choice}_count': count,
                f'choice_{choice}_pct': pct
            })

        return result
