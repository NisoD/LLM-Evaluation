import argparse
import json
import logging
import os
import shutil
import tempfile
import time
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count, Manager
from pathlib import Path
from typing import Dict, Optional, Iterator, Union, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytz
import requests
from datasets import load_dataset
from tqdm import tqdm

from src.analysis.create_plots.DataLoader import DataLoader
from src.experiments.experiment_preparation.dataset_scheme.conversions.download_ibm_data import generate_file_names
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


def main(pair: Path, scheme_files_dir: Path,
         logger=None):
    """Main function to demonstrate usage."""
    data_loader = DataLoader()
    model_name, shots = pair
    df_partial = data_loader.load_and_process_data(model_name, shots=shots, drop=False)
    # save as parquet file in scheme_files_dir as model_name_shots.parquet
    output_path = scheme_files_dir / f"{model_name.replace('/', '_')}_{shots}_shots.parquet"
    df_partial.to_parquet(output_path)
    logger.info(f"Output saved to: {output_path}")


def download_huggingface_files_parllel(process_output_dir):
    """
    Downloads parquet files from Hugging Face to a specified directory
    """

    num_processes = min(1, cpu_count() - 1)
    logger = setup_logging()
    logger.info("Starting processing...")
    logger.info(f"Starting parallel processing with {num_processes} processes...")
    shots_to_evaluate = [0, 5]
    models_to_evaluate = [
        'allenai/OLMoE-1B-7B-0924-Instruct',
        'meta-llama/Llama-3.2-1B-Instruct',
        'meta-llama/Meta-Llama-3-8B-Instruct',
        'meta-llama/Llama-3.2-3B-Instruct',
        'mistralai/Mistral-7B-Instruct-v0.3',
    ]
    pairs = [(model, shots) for model in models_to_evaluate for shots in shots_to_evaluate]
    # pairs = pairs[:1]

    with Manager() as manager:
        process_func = partial(process_file_safe,
                               scheme_files_dir=process_output_dir, logger=logger)

        with Pool(processes=num_processes) as pool:
            list(tqdm(
                pool.imap_unordered(process_func, pairs),
                total=len(pairs),
                desc="Processing files"
            ))


def process_file_safe(pair, scheme_files_dir, logger):
    pid = os.getpid()
    start_time = time.time()
    try:
        logger.info(f"Process {pid} starting to process file {pair}")
        process_file(pair,
                     scheme_files_dir=scheme_files_dir,
                     logger=logger)
    except Exception as e:
        logger.error(f"Process {pid} encountered error processing file {pair}: {e}")
        return None
    finally:
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Process {pid} finished processing {pair}. Processing time: {elapsed_time:.2f} seconds")


def process_file(pair, scheme_files_dir, logger):
    # Extract date from URL for the output filename
    # Example: from URL get '2024-12-16' for the filename
    main(pair, scheme_files_dir, logger)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--urls', nargs='+', help='List of urls to download')
    parser.add_argument('--output_dir', help='Output directory')
    parser.add_argument('--batch_size', type=int, help='Batch size for processing parquet files', default=1000)
    parser.add_argument('--repo_name', help='Repository name for the schema files')
    parser.add_argument('--scheme_files_dir', help='Directory to store the scheme files')
    args = parser.parse_args()
    args.output_dir = "/cs/snapless/gabis/eliyahabba/ibm_results_data_full"
    process_output_dir = "/cs/snapless/gabis/eliyahabba/ibm_results_data_full_processed_split"
    if not os.path.exists(process_output_dir):
        process_output_dir = "/Users/ehabba/ibm_results_data_full_processed_split"
        os.makedirs(process_output_dir, exist_ok=True)
    process_output_dir  = Path(process_output_dir)
    # parquet_path = Path("~/Downloads/data_sample.parquet")
    # convert_to_scheme_format(parquet_path, batch_size=100)
    download_huggingface_files_parllel(process_output_dir=process_output_dir)
