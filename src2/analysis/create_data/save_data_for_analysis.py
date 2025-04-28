import argparse
import hashlib
import json
import logging
import os
import random
import time
from datetime import datetime
from difflib import get_close_matches
from functools import partial
from multiprocessing import Pool, cpu_count, Manager
from pathlib import Path
from typing import Dict, Optional, Iterator, Union, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import HfFileSystem
from tqdm import tqdm

from src.experiments.experiment_preparation.dataset_scheme.conversions.dataset_schema_adapter_ibm import \
    QuantizationPrecision
from src.experiments.experiment_preparation.datasets_configurations.DatasetConfigFactory import DatasetConfigFactory
from src.utils.Constants import Constants


def setup_logging():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [PID %(process)d] - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'processing_log_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


class RunOutputMerger:
    """A class to process and merge run data with associated metadata."""

    def __init__(self, parquet_path: Path, batch_size: int = 100000):
        """
        Initialize the RunDataProcessor.

        Args:
            parquet_path (str): Path to the parquet file
            batch_size (int): Size of batches for processing
        """
        self.parquet_path = parquet_path
        self.batch_size = batch_size
        self.runs_df = self._load_runs_dataset()

    def _load_runs_dataset(self) -> pd.DataFrame:
        """
        Load and process the runs dataset into a dictionary for O(1) lookups.

        Returns:
            Dict[str, Dict[str, Any]]: Processed runs dictionary
        """
        runs_ds = load_dataset("OfirArviv/HujiCollabRun")
        train_runs_ds = runs_ds['run']
        df = train_runs_ds.to_pandas()
        return df

    def _add_run_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add run-related columns to the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with added run columns
        """

        # Map run information to DataFrame
        run_df_unique = self.runs_df.drop_duplicates(subset=['run_id'])
        run_df_unique.columns = ['run_' + col if col != 'run_id' else col for col in run_df_unique.columns]
        keep_columns = ['run_id', 'run_model_args', 'run_unitxt_recipe','run_quantization_bit_count']
        run_df_unique = run_df_unique[keep_columns]

        merged_df = df.merge(
            run_df_unique,
            on='run_id',
            how='left'
        )

        return merged_df

    def process_batches(self) -> Iterator[pd.DataFrame]:
        """
        Process the parquet file in batches, adding run information.

        Yields:
            pd.DataFrame: Processed batch with added run information
        """
        try:
            parquet_file = pq.ParquetFile(self.parquet_path)
            columns_to_keep = ['id', 'run_id', 'scores', 'references', 'generated_text', 'cumulative_logprob',
                               'raw_input']

            # for batch in parquet_file.iter_batches(batch_size=self.batch_size):
            for batch in parquet_file.iter_batches(batch_size=self.batch_size, columns=columns_to_keep):
                df = batch.to_pandas()
                # eval on the raw_input column, and next take only task_data key
                df['question'] = df['raw_input'].apply(lambda x: eval(x)['task_data']).apply(
                    lambda x: eval(x)['question'
                    if 'question' in x else 'context'])
                df['options'] = df['raw_input'].apply(lambda x: eval(x)['task_data']).apply(
                    lambda x: eval(x)['options'])
                # drop the raw_input column
                df.drop(columns=['raw_input'], inplace=True)
                df = self._add_run_columns(df)
                df.dropna(subset=['run_model_args'], inplace=True)
                yield df
        except Exception as e:
            print(f"Error processing parquet file: {str(e)}")
            raise


class Converter:
    def __init__(self, models_metadata_path: Union[str, Path]):
        """Initialize converter with path to models metadata."""
        self.models_metadata_path = Path(models_metadata_path)
        self._load_models_metadata()
        self._template_cache = {}  # Cache for templates
        self._index_map_cache = {}  # Cache for index map

    def _load_models_metadata(self) -> None:
        """Load models metadata from file."""
        with open(self.models_metadata_path) as f:
            self.models_metadata = json.load(f)

    def _parse_config_string(self, config_string: str) -> Dict:
        """Parse configuration string to dictionary."""
        return dict(pair.split('=') for pair in config_string.split(','))

    def _build_model_section(self, row: pd.Series
                             ) -> Tuple[str, Optional[str]]:
        """Build model section of schema."""

        # Get model configuration
        model_args = row['run_model_args']
        model_metadata = self.models_metadata[model_args['model']]
        return model_args['model'], model_metadata.get("ModelInfo", {}).get("family")

    def _build_model_section_without_family(self, row: pd.Series
                                            ) -> str:
        """Build model section of schema."""

        # Get model configuration
        model_args = row['run_model_args']
        return model_args['model']

    def parse_template_params(self, recipe: Dict) -> Tuple[str, str, dict]:
        """Extract template parameters from recipe configuration."""
        config = recipe['template'].split('.')[-1]

        enumerator = config.split('enumerator_')[1].split("_")[0]
        choice_sep = config.split('choicesSeparator_')[1].split("_")[0]
        shuffle = config.split('shuffleChoices_')[1]

        choices_order_map = {"randiom": {
            "method": "random",
            "description": "Randomly shuffles all choices"
        },
            "False":
                {
                    "method": "none",
                    "description": "Preserves the original order of choices as provided"
                },
            "lengthSort":
                {
                    "method": "shortest_to_longest",
                    "description": "Orders choices by length, from shortest to longest text"
                },
            "lengthSortReverse":
                {
                    "method": "longest_to_shortest",
                    "description": "Orders choices by length, from longest to shortest text"
                },
            "placeCorrectChoiceFirst":
                {
                    "method": "correct_first",
                    "description": "Places the correct answer as the first choice"
                },
            "placeCorrectChoiceFourth":
                {
                    "method": "correct_last",
                    "description": "Places the correct answer as the last choice"
                },
            "alphabeticalSort":
                {
                    "method": "alphabetical",
                    "description": "Orders choices alphabetically (A to Z)"
                },
            "alphabeticalSortReverse":
                {
                    "method": "reverse_alphabetical",
                    "description": "Orders choices in reverse alphabetical order (Z to A)"
                }
        }
        choices_order = choices_order_map[shuffle]

        class SeparatorMap:
            SEPARATOR_MAP = {
                "space": "\\s",
                " ": "\\s",
                "newline": "\n",
                "comma": ", ",
                "semicolon": "; ",
                "pipe": " | ",
                "OR": " OR ",
                "or": " or ",
                "orLower": " or ",
                "OrCapital": " OR "
            }

            @classmethod
            def get_separator(cls, separator: str) -> str:
                return cls.SEPARATOR_MAP.get(separator, separator)

        separator = SeparatorMap.get_separator(choice_sep)
        return enumerator, separator, choices_order

    def _get_template(self, run_unitxt_recipe: dict, logger=None) -> str:
        template_name = run_unitxt_recipe['template'].split('templates.huji_workshop.')[-1]
        if template_name in self._template_cache:
            return self._template_cache[template_name]

        catalog_path = Constants.TemplatesGeneratorConstants.CATALOG_PATH
        template_name = template_name.replace(".", "/")
        template_path = Path(catalog_path, template_name + '.json')

        with open(template_path, 'r') as f:
            template_data = json.load(f)

        self._template_cache[template_name] = template_data['input_format']
        prompt_paraphrasing = self.get_prompt_paraphrasing(run_unitxt_recipe['card'])
        input_format = template_data['input_format']
        for prompt in prompt_paraphrasing.get_all_prompts():
            if prompt.text == input_format:
                return prompt.name
        raise ValueError(f"Template {template_name} not found in catalog")

    def _build_prompt_section(self, recipe: Dict,
                              logger=None) -> Dict:
        """Build prompt configuration section of schema."""
        # Process demonstration examples
        enumerator, separator, choices_order = self.parse_template_params(recipe)
        return {
            "shots": int(recipe['num_demos']),
            "template": self._get_template(recipe, logger),
            "separator": separator,
            "enumerator": enumerator,
            "choices_order": choices_order['method'],
        }

    def _build_output_section(self, row: pd.Series) -> Tuple[str, float]:
        """Build output section of schema."""
        return row['generated_text'], row['cumulative_logprob']

    def convert_quantization(self, value: str) -> str:
        """Convert quantization value to schema format."""
        mapping = {
            "None": QuantizationPrecision.NONE,
            "half": QuantizationPrecision.FLOAT16,
            "float16": QuantizationPrecision.FLOAT16,
            "float32": QuantizationPrecision.FLOAT32,
            "int8": QuantizationPrecision.INT8,
            "int4": QuantizationPrecision.INT4
        }
        if value not in mapping:
            raise ValueError(f"Unsupported quantization value: {value}")
        return mapping[value]

    def _build_quantization_section(self, row: pd.Series) -> str:
        """Build quantization section of schema."""
        return self.convert_quantization(row['run_quantization_bit_count'])

    def _build_evaluation_section(self, row: pd.Series) -> Dict:
        """Build evaluation section of schema."""
        ground_truth = row['references'][0]
        score = row['scores']['score']
        return ground_truth, score

    def get_prompt_paraphrasing(self, dataset):
        if "mmlu_pro" in dataset:
            return DatasetConfigFactory.get_instruct_prompts("MMLU_PRO")
        elif "mmlu" in dataset:
            return DatasetConfigFactory.get_instruct_prompts("MMLU")
        elif "ai2_arc" in dataset:
            return DatasetConfigFactory.get_instruct_prompts("AI2_ARC")
        elif "hellaswag" in dataset:
            return DatasetConfigFactory.get_instruct_prompts("HellaSwag")
        elif "openbook_qa" in dataset:
            return DatasetConfigFactory.get_instruct_prompts("OpenBookQA")
        elif "social_iqa" in dataset:
            return DatasetConfigFactory.get_instruct_prompts("Social_IQa")
        elif "race" in dataset:
            return DatasetConfigFactory.get_instruct_prompts("Race")
        elif "quality" in dataset:
            return DatasetConfigFactory.get_instruct_prompts("QuALITY")
        else:
            raise ValueError(f"Dataset {dataset} not found in DatasetConfigFactory")

    def build_instance_section(self, row: pd.Series, recipe) -> Dict:
        map_file_name = recipe['card'].split('.')[1]
        if map_file_name == "ai2_arc":
            map_file_name = recipe['card'].split('.')[1]
        if map_file_name.startswith("global_mmlu"):
            # add also the language in the second part of the string
            map_file_name = f"{map_file_name}.{recipe['card'].split('.')[2]}"
        current_dir = Path(__file__).parents[2]
        map_file_path = current_dir / "experiments/experiment_preparation/dataset_scheme/conversions/hf_map_data/" / f"{map_file_name}_samples.json"
        if map_file_name in self._index_map_cache:
            index_map = self._index_map_cache[map_file_name]
        else:
            with open(map_file_path, 'r') as file:
                index_map = json.load(file)
            self._index_map_cache[map_file_name] = index_map
        if row['question'] not in index_map:
            return -1
        return index_map[row['question']]['index']

    def get_closest_answer(self, answer, options):
        ai_answer = answer.strip().split("\n")
        ai_answer = ai_answer[0].strip()
        return get_close_matches(ai_answer, options, n=1, cutoff=0.0)[0]

    def process_row(self, row_data: pd.Series) -> str:
        choices = row_data['options']
        generated_answer = row_data['generated_text']
        return self.get_closest_answer(generated_answer, choices)

    def convert_dataframe(self, df: pd.DataFrame):
        """Convert entire DataFrame using vectorized operations."""
        # Get recipes
        combined_strings = df['run_id'].astype(str) + df['id'].astype(str)
        evaluation_id = combined_strings.apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
        evaluation_id.name = 'evaluation_id'

        recipes = df['run_unitxt_recipe'].apply(self._parse_config_string)
        index_map = df.apply(
            lambda row: self.build_instance_section(row, self._parse_config_string(row['run_unitxt_recipe'])),
            axis=1).rename('sample_index')
        df['run_model_args'] = df['run_model_args'].apply(lambda x: json.loads(x))
        df['references'] = df['references'].apply(lambda x: eval(x))
        df['scores'] = df['scores'].apply(lambda x: eval(x))
        # Build all sections in parallel
        model_sections = df['run_model_args'].apply(lambda x: x['model'])
        model_sections.name = 'model'
        quantization = df.apply(self._build_quantization_section, axis=1, result_type='expand')
        quantization.name = 'quantization'
        # self.convert_quantization(
        #     row['run_quantization_bit_count']
        prompt_sections = recipes.apply(lambda x: pd.Series(self._build_prompt_section(x)))

        output_sections = df.apply(self._build_output_section, axis=1, result_type='expand') \
            .rename(columns={0: 'generated_text', 1: 'cumulative_logprob'})

        eval_sections = df.apply(self._build_evaluation_section, axis=1, result_type='expand') \
            .rename(columns={0: 'ground_truth', 1: 'score'})

        dataset = recipes.str['card'].str.split('.').str[1:].str.join('.').rename('dataset')
        # run get_closest_answer on 2 columns : generated_text and options and crate a new column with the result
        closest_answer = df.apply(self.process_row, axis=1)
        closest_answer.name = 'closest_answer'
        # Combine all sections
        result_df = pd.concat([
            evaluation_id,
            dataset,
            index_map,
            model_sections,
            quantization,
            prompt_sections,
            output_sections,
            closest_answer,
            eval_sections
        ], axis=1)

        result_df.reset_index(drop=True, inplace=True)
        return result_df


def main(file_path: Path = Path(f"~/Downloads/data_2025-01.parquet"),
         process_input_dir: Path = Path(f"~/Downloads/data_2025-01.parquet"),
         logger=None):
    """Main function to demonstrate usage."""
    parquet_path = file_path
    logger.info(f"Processing file: {parquet_path}")
    processor = RunOutputMerger(parquet_path)
    models_metadata_path = Path(Constants.ExperimentConstants.MODELS_METADATA_PATH)
    converter = Converter(models_metadata_path)

    # Create output path
    output_path = os.path.join(process_input_dir,
                               f"processed_{parquet_path.name}")
    if os.path.exists(output_path):
        logger.info(f"Output file already exists: {output_path}")
        return
    logger.info(f"Output path: {output_path}")

    # Initialize writer as None
    writer = None
    num_of_batches = 0

    # Process all batches
    for processed_batch in processor.process_batches():
        num_of_batches += 1
        logger.info(f"Batches processed: {num_of_batches}")

        # Convert the batch
        data = converter.convert_dataframe(processed_batch)
        # Skip empty batches
        if len(data) == 0:
            continue

        def clean_text(text):
            if isinstance(text, str):
                return text.encode('utf-8', 'ignore').decode('utf-8')
            return text

        # נקה את כל העמודות מסוג טקסט
        for column in data.select_dtypes(include=['object']):
            data[column] = data[column].apply(clean_text)

        table = pa.Table.from_pandas(data)

        # Initialize writer with first non-empty batch's schema
        if writer is None:
            writer = pq.ParquetWriter(
                output_path,
                schema=table.schema,
                compression='snappy',  # או 'gzip'
                use_dictionary=True,
                write_statistics=True,
                coerce_timestamps='ms',
                allow_truncated_timestamps=False
            )
            # writer = pq.ParquetWriter(
            #     output_path,
            #     schema=table.schema,
            # )

        # Write batch
        writer.write_table(table)

    # Close the writer if it was created
    if writer:
        writer.close()
        logger.info(f"Processed {num_of_batches} batches.")
        logger.info(f"Output saved to: {output_path}")
    else:
        logger.warning("No data was written - all batches were empty")


def download_huggingface_files_parllel(input_dir: Path, process_input_dir, file_names, scheme_files_dir):
    """
    Downloads parquet files from Hugging Face to a specified directory
/Users/ehabba/PycharmProjects/LLM-Evaluation/src/experiments/experiment_preparation/dataset_scheme/analsyis/save_data_for_analysis.py
    """
    # Create directory if it doesn't exist
    os.makedirs(input_dir, exist_ok=True)

    num_processes = min(24, cpu_count() - 1)
    num_processes = 12
    logger = setup_logging()
    logger.info("Starting processing...")
    logger.info(f"Starting parallel processing with {num_processes} processes...")

    with Manager() as manager:
        process_func = partial(process_file_safe, input_dir=input_dir, process_input_dir=process_input_dir,
                               scheme_files_dir=scheme_files_dir, logger=logger)

        with Pool(processes=num_processes) as pool:
            list(tqdm(
                pool.imap_unordered(process_func, file_names),
                total=len(file_names),
                desc="Processing files"
            ))


def process_file_safe(url, input_dir: Path, process_input_dir, scheme_files_dir, logger):
    pid = os.getpid()
    start_time = time.time()
    try:
        logger.info(f"Process {pid} starting to process file {url}")
        process_file(url, input_dir=input_dir, process_input_dir=process_input_dir,
                     scheme_files_dir=scheme_files_dir,
                     logger=logger)
    except Exception as e:
        logger.error(f"Process {pid} encountered error processing file {url}: {e}")
        return None
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Process {pid} finished processing {url}. Processing time: {elapsed_time:.2f} seconds")


def process_file(url, input_dir: Path, process_input_dir: Path, scheme_files_dir, logger):
    # Extract date from URL for the output filename
    # Example: from URL get '2024-12-16' for the filename
    original_filename = url.split('/')[-1]
    output_path = input_dir / original_filename
    if not output_path.exists():
        logger.info(f"File already not exists: {output_path}")
        return
    main(output_path, process_input_dir, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_names', nargs='+', help='List of file_names to download')
    parser.add_argument('--input_dir', help='Output directory')
    parser.add_argument('--batch_size', type=int, help='Batch size for processing parquet files', default=1000)
    parser.add_argument('--repo_name', help='Repository name for the schema files',
                        default="eliyahabba/llm-evaluation-analysis")
    parser.add_argument('--scheme_files_dir', help='Directory to store the scheme files')
    args = parser.parse_args()
    args.input_dir = "/cs/snapless/gabis/eliyahabba/ibm_results_data_full"
    # args.input_dir = "/cs/labs/gabis/eliyahabba/ibm_results_data_full"

    process_input_dir = "/cs/snapless/gabis/eliyahabba/ibm_results_data_full_processed"
    if not os.path.exists(process_input_dir):
        process_input_dir = "/Users/ehabba/ibm_results_data_full_processed"
        os.makedirs(process_input_dir, exist_ok=True)
    if not os.path.exists(args.input_dir):
        args.input_dir = "/Users/ehabba/ibm_results_data_full"
        os.makedirs(args.input_dir, exist_ok=True)

    # parquet_path = Path("~/Downloads/data_sample.parquet")
    # convert_to_scheme_format(parquet_path, batch_size=100)
    # List of URLs to download

    # args.input_dir = "/Users/ehabba/Downloads/"
    # args.file_names = ["/Users/ehabba/Downloads/data_2025-01.parquet"]
    # process_input_dir = "/Users/ehabba/PycharmProjects/LLM-Evaluation/src/experiments/experiment_preparation/dataset_scheme/conversions/process_input_dir"
    # os.makedirs(process_input_dir, exist_ok=True)

    fs = HfFileSystem()
    existing_files = fs.ls(f"datasets/{args.repo_name}", detail=False)
    existing_files = [Path(file).stem.split("processed_")[1] for file in existing_files if file.endswith('.parquet')]
    args.file_names = [file for file in os.listdir(args.input_dir) if file.endswith('.parquet')]
    args.file_names = [file for file in os.listdir(args.input_dir) if (f'processed_{file}' not in os.listdir(process_input_dir))]

    random.shuffle(args.file_names)
    print(args.file_names)
    print(len(args.file_names))

    # args.input_dir = ("/Users/ehabba/Downloads/")
    # args.file_names = ["data_2025-01-15T03_00_00+00_00_2025-01-15T04_00_00+00_00.parquet"]
    # process_input_dir = "/Users/ehabba/PycharmProjects/LLM-Evaluation/src/experiments/experiment_preparation/dataset_scheme/conversions/process_input_dir"
    # os.makedirs(process_input_dir, exist_ok=True)

    download_huggingface_files_parllel(input_dir=Path(args.input_dir), process_input_dir=process_input_dir,
                                       file_names=args.file_names,
                                       scheme_files_dir=args.scheme_files_dir)
