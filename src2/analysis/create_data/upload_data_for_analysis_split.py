import argparse
import logging
import os
from datetime import datetime
from multiprocessing import Pool, cpu_count

from huggingface_hub import (
    HfApi,
    create_repo,
    CommitOperationAdd,
    create_commit
)
from tqdm import tqdm

from config.get_config import Config


def chunkify(lst, chunk_size):
    """
    פונקציית עזר שמחזירה גנרטור של "חתיכות" מרשימה גדולה.
    לדוגמה, chunk_size=100 => כל איטרציה תחזיר עד 100 פריטים.
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def setup_logging():
    """
    הגדרת לוגים בסיסית עם תאריך ושעה, כתיבה לקובץ + הדפסה למסך.
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


def upload_chunk_files(args):
    """
    פונקציה שמקבלת טאפַל עם הנתונים הבאים:
    (chunk_index, chunk_files, repo_name, token, logger)

    מבצעת create_commit אחד המכיל את כל הקבצים באותו Chunk.
    """
    chunk_index, chunk_files, repo_name, token, logger = args
    logger.info(f"Uploading chunk #{chunk_index} with {len(chunk_files)} files...")

    # הכנת רשימת פעולות (הוספת כל קובץ שיש ב- chunk_files)
    operations = []
    for local_file in chunk_files:
        operations.append(
            CommitOperationAdd(
                path_in_repo=os.path.basename(local_file),  # שם הקובץ כפי שיופיע ברepo
                path_or_fileobj=local_file  # הנתיב המקומי
            )
        )

    try:
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


def upload_in_chunks_parallel(process_output_dir, repo_name, token, chunk_size=10):
    """
    פונקציה שלוקחת את כל הקבצים בתיקייה process_output_dir,
    מחלקת אותם ל-Chunks בגודל chunk_size, ואז מריצה העלאה של כל Chunk במקביל.
    """
    logger = setup_logging()
    api = HfApi()

    # 1. אוספים את כל הקבצים מהתיקייה
    files = [os.path.join(process_output_dir, f) for f in os.listdir(process_output_dir)]
    logger.info(f"Total files found: {len(files)}")

    # 2. יוצרים Repo (אם לא קיים) או מתעלמים משגיאה (אם כבר קיים)
    try:
        create_repo(repo_id=repo_name, token=token, private=True, repo_type="dataset")
        logger.info(f"Repository '{repo_name}' created.")
    except Exception as e:
        logger.info(f"Repository '{repo_name}' might already exist or error occurred: {e}")

    # 3. מחלקים את הקבצים למנות (Chunks)
    chunked_files = list(chunkify(files, chunk_size))
    logger.info(f"Splitting files into {len(chunked_files)} chunks (chunk_size={chunk_size})")

    # 4. הגדרת מספר תהליכים מקביליים (Pool)
    num_processes = min(24, cpu_count() - 1)
    logger.info(f"Starting parallel upload with {num_processes} processes...")

    # 5. העלאת כל Chunk במקביל
    tasks = []
    for idx, cfiles in enumerate(chunked_files):
        tasks.append((idx, cfiles, repo_name, token, logger))

    with Pool(processes=num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(upload_chunk_files, tasks), total=len(tasks), desc="Uploading Chunks"):
            pass

    logger.info("All chunks finished uploading.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=100, help="Number of files per chunk")
    args = parser.parse_args()

    # טוענים את הטוקן מהקונפיג
    config = Config()
    TOKEN = config.config_values.get("hf_access_token", "")
    repo_name = "eliyahabba/llm-evaluation-analysis-split"

    process_output_dir = "/cs/snapless/gabis/eliyahabba/ibm_results_data_full_processed_split_all_cols_deduped"
    os.makedirs(process_output_dir, exist_ok=True)

    # העלאה בפעימות של chunks, במקביל
    upload_in_chunks_parallel(
        process_output_dir=process_output_dir,
        repo_name=repo_name,
        token=TOKEN,
        chunk_size=args.chunk_size
    )


if __name__ == "__main__":
    main()
