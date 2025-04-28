from pathlib import Path
import pandas as pd
from typing import Optional, Dict, List
# data_loader.py
import time
from typing import Dict, List, Any
from typing import Optional

import pandas as pd
import pyarrow.compute as pc
from datasets import load_dataset
from huggingface_hub import HfFileSystem

import sys
from pathlib import Path

import yaml

GLOBAL_CONFIG = Path(__file__).parent / 'global_config.yaml'
LOCAL_CONFIG = Path(__file__).parent / 'local_config.yaml'


class Config:
    def __init__(self):
        self.config_values = self.load_config()

    def load_config(self):
        config = {}
        try:
            with open(GLOBAL_CONFIG, 'r') as file:
                config = yaml.safe_load(file)
        except FileNotFoundError:
            print("Global config not found.", file=sys.stderr)

        try:
            with open(LOCAL_CONFIG, 'r') as file:
                local_config = yaml.safe_load(file)
                if local_config:
                    config.update(local_config)
        except FileNotFoundError:
            print("Local config not found.", file=sys.stderr)

        return config


# from src.analysis.create_plots.check_data import get_parquet_files_from_hf

repo_name = "eliyahabba/llm-evaluation-analysis-split"


class DataLoader:
    def __init__(self, dataset_name=repo_name, split="train", batch_size=10000):
        """
        Initializes the DataLoader with dataset details.

        Args:
            dataset_name (str): Name of the dataset to load from HuggingFace.
            split (str): Dataset split to use.
            batch_size (int): Number of samples per batch.
        """
        self.dataset = None
        self.dataset_name = dataset_name
        self.split = split
        self.batch_size = batch_size

    def load_and_process_data(self, model_name, shots,
                              datasets=None, template=None, separator=None, enumerator=None, choices_order=None,
                              max_samples=None, drop=True):
        start_time = time.time()
        # print(f"Processing model: {model_name}")
        self.load_data_with_filter(max_samples, drop, model_name, shots, datasets)
        if len(self.dataset) == 0:
            print(f"No data found for model {model_name} and shots {shots}")
            return pd.DataFrame()
        full_results = self.extract_data(model_name, shots,
                                         dataset=datasets, template=template, separator=separator,
                                         enumerator=enumerator,
                                         choices_order=choices_order)
        clean_df = self.remove_duplicates(full_results)
        load_time = time.time()
        print(f"The size of the data after removing duplicates is: {len(clean_df)}")
        print(f"Data loading completed in {load_time - start_time:.2f} seconds")
        return clean_df

    def load_data_with_filter_local(self, max_samples=None, drop=True, model_name=None, shots=None, dataset=None):
        """
        Efficiently loads data using the `datasets` and `pyarrow` libraries,
        with filtering during loading for better memory management.
        """
        # Load the dataset if not already loaded
        file = f"{model_name}_shot{shots}_{dataset}.parquet"
        data_files = [file]
        if len(data_files) > 0:
            if self.dataset is None:
                # split = split_with_filter
                self.dataset = load_dataset(self.dataset_name, data_files=data_files, cache_dir=None)
            else:
                self.dataset = load_dataset(self.dataset_name, split=self.split)
        print("The size of the data after filtering is: ", len(self.dataset))
        if drop:
            self.dataset = self.dataset.remove_columns(['family', 'generated_text', 'ground_truth'])

    def load_data_with_filter(self, max_samples=None, drop=True, model_name=None, shots=None, datasets=None):
        """
        Efficiently loads data using the `datasets` and `pyarrow` libraries,
        with filtering during loading for better memory management.
        """
        # Load the dataset if not already loaded
        fs = HfFileSystem()
        existing_files = fs.ls(f"datasets/{repo_name}", detail=False)
        existing_files = [file.split('/')[-1] for file in existing_files if
                          file.endswith('.parquet')]
        # split only the file name that contains the model name and shots and dataset
        if model_name is not None:
            model_name = model_name.split("/")[-1]
            existing_files = [file for file in existing_files if model_name in file]
        if shots is not None:
            shots = "shots" + str(shots)
            existing_files = [file for file in existing_files if shots in file]
        if datasets is not None:
            datasets = [dataset.split("/")[-1] for dataset in datasets]
            existing_files = [file for file in existing_files if any(dataset in file for dataset in datasets)]

        if len(existing_files) > 0:
            if self.dataset is None:
                # split = split_with_filter
                self.dataset = load_dataset(self.dataset_name, data_files=existing_files, split=self.split)
            else:
                self.dataset = load_dataset(self.dataset_name, split=self.split)
        print("The size of the data after filtering is: ", len(self.dataset))
        if drop:
            self.dataset = self.dataset.remove_columns(['cumulative_logprob', 'generated_text', 'ground_truth'])

    def extract_data2(
            self,
            model_name: str,
            shots: int,
            dataset: Optional[str] = None,
            template: Optional[str] = None,
            separator: Optional[str] = None,
            enumerator: Optional[str] = None,
            choices_order: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Efficiently filter a large dataset using batched processing and parallel execution.

        Args:
            model_name: Name of the model to filter by
            shots: Number of shots to filter by
            dataset: Optional dataset name filter
            template: Optional template filter
            separator: Optional separator filter
            enumerator: Optional enumerator filter
            choices_order: Optional choices order filter

        Returns:
            pd.DataFrame: Filtered dataset as a pandas DataFrame
        """
        try:
            print("Starting data filtering process...")

            # Step 1: Define the columns we need to keep
            # Add any additional columns you need in the final output
            required_columns = [
                'model',
                'shots',
                'dataset',
                'template',
                'separator',
                'enumerator',
                'choices_order'
            ]

            # Step 2: Select only necessary columns to reduce memory usage
            dataset_subset = self.dataset.select_columns(required_columns)

            # Step 3: Define the batch filtering function
            def filter_batch(examples: Dict[str, List[Any]]) -> Dict[str, List[bool]]:
                """
                Process a batch of examples and return a boolean mask for filtering.

                Args:
                    examples: Dictionary of column names to lists of values

                Returns:
                    Dictionary with a single 'keep' key containing boolean mask
                """
                batch_size = len(examples['model'])
                # Initialize all rows as True
                mask = [True] * batch_size

                # Apply each filter condition if it exists
                # Required filters
                mask = [m and (mod == model_name) for m, mod in zip(mask, examples['model'])]
                mask = [m and (s == shots) for m, s in zip(mask, examples['shots'])]

                # Optional filters
                if dataset is not None:
                    mask = [m and (d == dataset) for m, d in zip(mask, examples['dataset'])]

                if template is not None:
                    mask = [m and (t == template) for m, t in zip(mask, examples['template'])]

                if separator is not None:
                    mask = [m and (sep == separator) for m, sep in zip(mask, examples['separator'])]

                if enumerator is not None:
                    mask = [m and (e == enumerator) for m, e in zip(mask, examples['enumerator'])]

                if choices_order is not None:
                    mask = [m and (c == choices_order) for m, c in zip(mask, examples['choices_order'])]

                return {'keep': mask}

            # Step 4: Apply the batched filtering
            print("Applying filters...")
            filtered_dataset = dataset_subset.map(
                filter_batch,
                batched=True,
                batch_size=100000,  # Adjust based on available memory
                num_proc=4,
                remove_columns=dataset_subset.column_names,
                load_from_cache_file=True,
                desc="Filtering dataset"
            )

            # Step 5: Convert to pandas DataFrame
            print("Converting to pandas DataFrame...")
            df_filtered = filtered_dataset.to_pandas()

            # Step 6: Print summary statistics
            print("\nFiltering results:")
            print(f"Total rows after filtering: {len(df_filtered)}")
            print("\nModel distribution:")
            print(df_filtered['model'].value_counts())

            return df_filtered

        except Exception as e:
            raise Exception(f"Error during data filtering: {str(e)}")

    def extract_data(self, model_name, shots, dataset=None, template=None, separator=None, enumerator=None,
                     choices_order=None):
        arrow_table = self.dataset.data.table

        conditions = [
            pc.equal(arrow_table['model'], model_name),
            pc.equal(arrow_table['shots'], shots)
        ]

        optional_filters = {
            'template': template,
            'separator': separator,
            'enumerator': enumerator,
            'choices_order': choices_order
        }

        for column, value in optional_filters.items():
            if value is not None:
                conditions.append(pc.equal(arrow_table[column], value))

        combined_condition = conditions[0]
        for condition in conditions[1:]:
            combined_condition = pc.and_(combined_condition, condition)

        try:
            print("Filtering data...")
            filtered_table = arrow_table.filter(combined_condition)
            df_filtered = filtered_table.to_pandas()
            print(df_filtered['model'].value_counts())
            return df_filtered
        except Exception as e:
            raise Exception(f"Error filtering data: {str(e)}")

    def remove_duplicates(self, df):
        """
        Removes duplicates from the DataFrame.
        """
        cols = ['sample_index', 'model', 'dataset', 'template', 'separator', 'enumerator', 'choices_order', 'shots']
        df_unique = df.drop_duplicates(subset=['evaluation_id'])
        return df_unique



class QuestionAnalyzer:
    """Manages analysis of poorly performing questions and their responses."""

    def __init__(self, base_results_dir: str = "visualization_results"):
        self.base_dir = Path(base_results_dir)
        self.responses_dir = self.base_dir / "question_responses"
        self.responses_dir.mkdir(exist_ok=True)
        self.data_loader = DataLoader()

    def get_question_responses(self, model_name: str, shots: int,
                               dataset: str, question_index: int,
                               force_reload: bool = False) -> pd.DataFrame:
        """
        Retrieves model responses for a specific question, either from cache or by loading.

        Args:
            model_name: Name of the model
            shots: Number of shots used
            dataset: Dataset name
            question_index: Index of the question
            force_reload: Whether to force reload from HuggingFace

        Returns:
            DataFrame containing responses with columns:
            [template, generated_text, ground_truth, closest_answer, score]
        """
        cache_path = self._get_cache_path(model_name, dataset, question_index)

        if not force_reload and cache_path.exists():
            return pd.read_parquet(cache_path)

        df = self.data_loader.load_and_process_data(
            model_name=model_name,
            shots=shots,
            datasets=[dataset],
            max_samples=None,
            drop=False  # Keep all columns for analysis
        )

        if df.empty:
            return pd.DataFrame()

        # Filter for specific question
        question_responses = df[df['sample_index'] == question_index].copy()

        # Store in cache
        if not question_responses.empty:
            question_responses.to_parquet(cache_path)

        return question_responses

    def store_error_annotation(self, dataset: str, question_index: int,
                               model_name: str, error_category: str,
                               notes: Optional[str] = None) -> None:
        """
        Stores error annotation for a specific question.

        Args:
            dataset: Dataset name
            question_index: Question index
            model_name: Model name
            error_category: One of: FORMAT_ERROR, WRONG_REASONING,
                          WRONG_ANNOTATION, LACK_OF_KNOWLEDGE
            notes: Optional additional notes
        """
        annotations_path = self.base_dir / "error_annotations.parquet"

        annotation = pd.DataFrame([{
            'dataset': dataset,
            'sample_index': question_index,
            'model_name': model_name,
            'error_category': error_category,
            'notes': notes,
            'timestamp': pd.Timestamp.now()
        }])

        if annotations_path.exists():
            existing = pd.read_parquet(annotations_path)
            mask = ((existing['dataset'] == dataset) &
                    (existing['sample_index'] == question_index) &
                    (existing['model_name'] == model_name))

            if mask.any():
                existing.loc[mask] = annotation.iloc[0]
            else:
                existing = pd.concat([existing, annotation], ignore_index=True)
            existing.to_parquet(annotations_path)
        else:
            annotation.to_parquet(annotations_path)

    def get_error_annotations(self,
                              dataset: Optional[str] = None,
                              model_name: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieves error annotations with optional filtering.

        Args:
            dataset: Optional dataset to filter by
            model_name: Optional model name to filter by

        Returns:
            DataFrame containing error annotations
        """
        annotations_path = self.base_dir / "error_annotations.parquet"
        if not annotations_path.exists():
            return pd.DataFrame()

        annotations = pd.read_parquet(annotations_path)

        if dataset:
            annotations = annotations[annotations['dataset'] == dataset]
        if model_name:
            annotations = annotations[annotations['model_name'] == model_name]

        return annotations

    def _get_cache_path(self, model_name: str, dataset: str,
                        question_index: int) -> Path:
        """Generates cache file path for question responses."""
        sanitized_name = model_name.replace('/', '_')
        return self.responses_dir / f"{dataset}_{question_index}_{sanitized_name}.parquet"