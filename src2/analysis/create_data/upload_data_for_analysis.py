import argparse
import logging
import os
import time
from datetime import datetime
from multiprocessing import Pool, cpu_count, Manager
from pathlib import Path
from functools import partial

from huggingface_hub import (
    HfApi,
    create_repo,
    CommitOperationAdd,
    create_commit
)
from tqdm import tqdm

from config.get_config import Config

def setup_logging():
    """
    הגדרת לוגים בסיסית עם תאריך ושעה, כתיבה לקובץ + הדפסה למסך
    """
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


def chunkify(lst, chunk_size):
    """
    מחלקת רשימה (lst) לצ'אנקים בגודל chunk_size (גנרטור).
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i+chunk_size]


def upload_chunk_files(args):
    """
    פונקציית העלאה המתבצעת עבור צ'אנק אחד.
    פרמטרים (args) מגיעים כטאפ"ל:
    (chunk_index, chunk_files, repo_name, token, logger)
    """
    chunk_index, chunk_files, repo_name, token, logger = args
    logger.info(f"Uploading chunk #{chunk_index} with {len(chunk_files)} files...")

    try:
        operations = []
        for local_file in chunk_files:
            operations.append(
                CommitOperationAdd(
                    path_in_repo=os.path.basename(local_file),  # שם הקובץ ב-HF
                    path_or_fileobj=local_file
                )
            )

        # יוצרים commit יחיד עם כל הקבצים שבצ'אנק
        create_commit(
            repo_id=repo_name,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Upload chunk {chunk_index}",
            token=token
        )

        logger.info(f"Successfully uploaded chunk #{chunk_index}")
    except Exception as e:
        logger.error(f"Error uploading chunk #{chunk_index}: {e}")


def upload_files_in_chunks(process_output_dir, repo_name, token, chunk_size=5):
    """
    - אוספת את רשימת הקבצים בתיקייה process_output_dir.
    - מחלקת לצ'אנקים (קבוצות של chunk_size).
    - מריצה העלאה מקבילית לכל צ'אנק.
    """
    logger = setup_logging()
    logger.info("Starting chunked upload...")

    # 1. יוצרים Repo אם לא קיים
    api = HfApi()
    try:
        create_repo(repo_id=repo_name, token=token, private=True, repo_type="dataset")
        logger.info(f"Repository '{repo_name}' created successfully.")
    except Exception as e:
        logger.info(f"Repository '{repo_name}' might already exist or error occurred: {e}")

    # 2. אוספים את כל הקבצים בתיקייה
    all_files = []
    for fname in os.listdir(process_output_dir):
        fpath = os.path.join(process_output_dir, fname)
        if os.path.isfile(fpath):
            all_files.append(fpath)

    logger.info(f"Total files found: {len(all_files)}")

    # 3. מחלקים לצ'אנקים
    chunked = list(chunkify(all_files, chunk_size))
    logger.info(f"Splitting files into {len(chunked)} chunks (chunk_size={chunk_size})")

    # 4. מקביליות - הגדרת כמות התהליכים
    num_processes = min(2, cpu_count() - 1)
    logger.info(f"Starting parallel upload with {num_processes} processes...")

    # יוצרים רשימת משימות
    tasks = []
    for idx, chunk_files in enumerate(chunked):
        tasks.append((idx, chunk_files, repo_name, token, logger))

    # 5. העלאה במקביל באמצעות Pool
    with Pool(processes=num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(upload_chunk_files, tasks),
                      total=len(tasks),
                      desc="Uploading chunks"):
            pass

    logger.info("All chunks finished uploading.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=5, help="Number of files per chunk")
    args = parser.parse_args()

    config = Config()
    TOKEN = config.config_values.get("hf_access_token", "")
    repo_name = "eliyahabba/llm-evaluation-analysis"

    process_output_dir = "/cs/snapless/gabis/eliyahabba/ibm_results_data_full_processed_all_cols_deduped"
    os.makedirs(process_output_dir, exist_ok=True)

    upload_files_in_chunks(
        process_output_dir=process_output_dir,
        repo_name=repo_name,
        token=TOKEN,
        chunk_size=args.chunk_size
    )


if __name__ == "__main__":
    main()