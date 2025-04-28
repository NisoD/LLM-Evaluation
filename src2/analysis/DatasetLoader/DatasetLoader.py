import operator
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List, Union, Iterator, Any

from datasets import load_dataset, concatenate_datasets, Dataset, IterableDataset


class ComparisonOperator(Enum):
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="
    EQ = "=="
    NE = "!="


@dataclass
class NumericFilter:
    value: float
    operator: ComparisonOperator


@dataclass
class QueryCriteria:
    # File structure criteria (used for initial file selection)
    models: Optional[Union[str, List[str]]] = None
    shots: Optional[Union[int, List[int]]] = None
    datasets: Optional[Union[str, List[str]]] = None

    # Content-based filters (applied after loading)
    template: Optional[Union[str, List[str]]] = None
    separator: Optional[Union[str, List[str]]] = None
    enumerator: Optional[Union[str, List[str]]] = None
    choices_order: Optional[Union[str, List[str]]] = None
    quantization: Optional[Union[str, List[str]]] = None
    cumulative_logprob: Optional[Union[float, NumericFilter, List[NumericFilter]]] = None
    score: Optional[Union[float, NumericFilter, List[NumericFilter]]] = None
    closest_answer: Optional[Union[str, List[str]]] = None
    ground_truth: Optional[Union[str, List[str]]] = None


class DatasetLoader:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self._load_metadata()

    def _load_metadata(self):
        """Load dataset structure metadata"""
        # Example structure: model/num_shots/dataset
        self.metadata = {
            "structure": {
                "models": ["llama2-13b", "gpt-3.5"],
                "shots": [0, 2, 4],
                "datasets": ["mmlu", "longbench"]
            },
            "file_pattern": "{model}/{shots}/{dataset}"
        }

    def load(self,
             criteria: Optional[QueryCriteria] = None,
             chunk_size: int = 1000,
             streaming: bool = False,
             **kwargs) -> Union[Dataset, IterableDataset, Iterator[List[Dict[str, Any]]]]:
        """
        Load and filter dataset based on criteria
        Args:
            criteria: Search criteria for both file selection and content filtering
            chunk_size: Size of chunks for processing
            streaming: Whether to return an iterator
            **kwargs: Additional arguments passed to load_dataset
        """
        if criteria is None:
            criteria = QueryCriteria()

        # Get matching file paths based on structure criteria
        file_paths = self._get_matching_files(criteria)

        if streaming:
            return self._stream_datasets(file_paths, criteria, chunk_size, **kwargs)
        else:
            return self._load_chunked(file_paths, criteria, chunk_size, **kwargs)

    def _get_matching_files(self, criteria: QueryCriteria) -> List[str]:
        """Get files matching the structural criteria (model/shots/dataset)"""
        models = self._to_list(criteria.models) or self.metadata["structure"]["models"]
        shots = self._to_list(criteria.shots) or self.metadata["structure"]["shots"]
        datasets = self._to_list(criteria.datasets) or self.metadata["structure"]["datasets"]

        matching_files = []
        for model in models:
            for shot in shots:
                for dataset in datasets:
                    file_path = self.metadata["file_pattern"].format(
                        model=model,
                        shots=shot,
                        dataset=dataset
                    )
                    matching_files.append(file_path)

        return matching_files

    def _apply_content_filters(self, item: Dict, criteria: QueryCriteria) -> bool:
        """Apply all content-based filters to a single item"""

        def check_numeric(value: float, filter_value: Union[float, NumericFilter, List[NumericFilter]]) -> bool:
            if isinstance(filter_value, (int, float)):
                return value == filter_value
            elif isinstance(filter_value, NumericFilter):
                op = getattr(operator, filter_value.operator.name.lower())
                return op(value, filter_value.value)
            elif isinstance(filter_value, list):
                return any(check_numeric(value, f) for f in filter_value)
            return True

        def check_field(field: str, criterion: Any) -> bool:
            if criterion is None:
                return True
            value = item.get(field)
            if value is None:
                return False
            criterion_list = self._to_list(criterion)
            return value in criterion_list

        # Apply each content-based filter
        if not check_field("template", criteria.template):
            return False
        if not check_field("separator", criteria.separator):
            return False
        if not check_field("enumerator", criteria.enumerator):
            return False
        if not check_field("choices_order", criteria.choices_order):
            return False
        if not check_field("quantization", criteria.quantization):
            return False
        if not check_field("closest_answer", criteria.closest_answer):
            return False
        if not check_field("ground_truth", criteria.ground_truth):
            return False

        if criteria.cumulative_logprob and not check_numeric(
                item.get("cumulative_logprob", 0),
                criteria.cumulative_logprob
        ):
            return False

        if criteria.score and not check_numeric(
                item.get("score", 0),
                criteria.score
        ):
            return False

        return True

    def _stream_datasets(self,
                         file_paths: List[str],
                         criteria: QueryCriteria,
                         chunk_size: int,
                         **kwargs) -> Iterator[List[Dict[str, Any]]]:
        """Stream and filter datasets"""
        for file_path in file_paths:
            try:
                ds = load_dataset(
                    self.dataset_name,
                    data_files=file_path,
                    streaming=True,
                    **kwargs
                )

                chunk = []
                for item in ds:
                    if self._apply_content_filters(item, criteria):
                        chunk.append(item)
                        if len(chunk) >= chunk_size:
                            yield chunk
                            chunk = []

                if chunk:
                    yield chunk

            except Exception as e:
                warnings.warn(f"Error processing {file_path}: {e}")

    def _load_chunked(self,
                      file_paths: List[str],
                      criteria: QueryCriteria,
                      chunk_size: int,
                      **kwargs) -> Dataset:
        """Load and filter datasets in chunks"""
        all_chunks = []

        for file_path in file_paths:
            try:
                ds = load_dataset(
                    self.dataset_name,
                    data_files=file_path,
                    **kwargs
                )

                # Process in chunks to avoid memory issues
                for i in range(0, len(ds), chunk_size):
                    chunk = ds.select(range(i, min(i + chunk_size, len(ds))))

                    # Filter chunk based on content criteria
                    if criteria:
                        chunk = chunk.filter(
                            lambda x: self._apply_content_filters(x, criteria)
                        )

                    if len(chunk) > 0:
                        all_chunks.append(chunk)

            except Exception as e:
                warnings.warn(f"Error processing {file_path}: {e}")

        if not all_chunks:
            raise ValueError("No matching data found for the specified criteria")

        return concatenate_datasets(all_chunks)

    @staticmethod
    def _to_list(value: Any) -> Optional[List]:
        if value is None:
            return None
        if isinstance(value, (str, int, float)):
            return [value]
        return list(value)

    def _process_item(self,
                      item: Dict[str, Any],
                      model: str,
                      shots: str,
                      dataset: str) -> Dict[str, Any]:
        """Process a single item from the dataset, enriching it with metadata"""
        # Create a new dict to avoid modifying the original
        processed = dict(item)

        # Add file path metadata if not already present
        if 'model' not in processed:
            processed['model'] = model
        if 'shots' not in processed:
            processed['shots'] = int(shots)
        if 'dataset' not in processed:
            processed['dataset'] = dataset

        # Add computed fields if needed
        if 'score' in processed and isinstance(processed['score'], str):
            processed['score'] = float(processed['score'])

        if 'cumulative_logprob' in processed and isinstance(processed['cumulative_logprob'], str):
            processed['cumulative_logprob'] = float(processed['cumulative_logprob'])

        # Add any additional computed fields or transformations here

        return processed


# Usage examples:
def example_usage():
    loader = DatasetLoader("your-dataset")

    # Load all MMLU data for a specific model with 2 shots
    basic_criteria = QueryCriteria(
        models="llama2-13b",
        shots=2,
        datasets="mmlu"
    )

    # Complex query combining file structure and content filters
    complex_criteria = QueryCriteria(
        models=["llama2-13b", "gpt-3.5"],
        shots=[2, 4],
        datasets="mmlu",
        template="cot",
        score=NumericFilter(0.8, ComparisonOperator.GT)
    )

    criteria = QueryCriteria(
        models="llama2-13b",
        shots=[2, 4],
        datasets="mmlu",
        score=NumericFilter(0.8, ComparisonOperator.GT)
    )
    for chunk in loader.load(
            criteria=criteria,
            streaming=True,
            chunk_size=100
    ):
        for item in chunk:
            print(f"Model: {item['model']}, Score: {item['score']}")


# 1. Explore using manifest
manifest = load_dataset("dataset-name/manifest")
relevant_files = manifest.filter(lambda x:
    x['model'] == 'llama2-13b' and
    x['avg_score'] > 0.8
)

# 2. Load and process data in chunks
loader = DatasetLoader("dataset-name")
for chunk in loader.load(
    data_files=relevant_files['file_path'],
    streaming=True
):
    pass