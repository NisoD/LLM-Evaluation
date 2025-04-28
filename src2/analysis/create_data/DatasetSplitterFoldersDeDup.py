import os
from pathlib import Path
import polars as pl
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Configure basic logger
logger = logging.getLogger("dedup_logger")
logger.setLevel(logging.INFO)

# Add StreamHandler (to display logs on the console) if not already added
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def process_parquet_file(parquet_file: Path, input_path: Path, dedup_eval_id_dir: Path, all_cols_dedup_dir: Path):
    """
    Processes a single Parquet file:
    1. Reads the file.
    2. Creates two deduplicated versions:
       a. Deduplication based on 'evaluation_id'.
       b. Deduplication based on all columns (excluding 'index' if present).
    3. Saves both versions in corresponding output directories.
    4. Returns processing metadata (row count, duplicate count) for logging.
    """
    try:
        logger.info(f"[START] - Processing file: {parquet_file}")

        # Read the Parquet file
        df = pl.read_parquet(str(parquet_file))
        logger.info(f"  Read {df.shape[0]} rows and {df.shape[1]} columns from {parquet_file}")

        # 1) Deduplicate based on 'evaluation_id'
        df_eval_id_dedup = df.unique(subset=["evaluation_id"])
        logger.info(f"  Deduplication by 'evaluation_id': {df_eval_id_dedup.shape[0]} rows remaining (out of {df.shape[0]})")

        # 2) Deduplicate based on all columns (excluding 'index' if present)
        exclude_cols = ["index"] if "index" in df.columns else []
        all_cols_except_index = [c for c in df.columns if c not in exclude_cols]

        total_rows = df.shape[0]
        distinct_by_all = df.unique(subset=all_cols_except_index).shape[0]
        duplicates_count_all_cols = total_rows - distinct_by_all
        df_all_cols_dedup = df.unique(subset=all_cols_except_index)

        logger.info(f"  Deduplication by all columns (excluding {exclude_cols}): "
                    f"Found {duplicates_count_all_cols} duplicates, "
                    f"{df_all_cols_dedup.shape[0]} rows remaining")

        # 3) Save results in a mirrored directory structure
        relative_path = parquet_file.relative_to(input_path)

        # a) Save deduplicated file based on 'evaluation_id'
        eval_id_file_path = dedup_eval_id_dir / relative_path
        eval_id_file_path.parent.mkdir(parents=True, exist_ok=True)
        df_eval_id_dedup.write_parquet(str(eval_id_file_path))
        logger.info(f"  Saved deduplicated file (by 'evaluation_id') at: {eval_id_file_path}")

        # b) Save deduplicated file based on all columns
        all_cols_file_path = all_cols_dedup_dir / relative_path
        all_cols_file_path.parent.mkdir(parents=True, exist_ok=True)
        df_all_cols_dedup.write_parquet(str(all_cols_file_path))
        logger.info(f"  Saved deduplicated file (by all columns) at: {all_cols_file_path}")

        logger.info(f"[DONE] - Finished processing: {parquet_file}\n")
        return parquet_file, total_rows, duplicates_count_all_cols

    except Exception as e:
        logger.error(f"[ERROR] Error processing file {parquet_file}: {e}", exc_info=True)
        return parquet_file, None, None, str(e)


def process_parquet_files_parallel(input_dir: str, max_workers: int = 4):
    """
    1. Creates two output directories (mirroring the input structure):
       - <input_dir>_eval_id_deduped
       - <input_dir>_all_cols_deduped
    2. Recursively searches for Parquet files.
    3. Processes them in parallel using ThreadPoolExecutor.
    """
    input_path = Path(input_dir).resolve()

    # Define output directories
    dedup_eval_id_dir = input_path.parent / (input_path.name + "_eval_id_deduped")
    all_cols_dedup_dir = input_path.parent / (input_path.name + "_all_cols_deduped")
    dedup_eval_id_dir.mkdir(parents=True, exist_ok=True)
    all_cols_dedup_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = list(input_path.rglob("*.parquet"))
    if not parquet_files:
        logger.warning(f"[WARNING] No .parquet files found under {input_dir}")
        return

    logger.info(f"Found {len(parquet_files)} Parquet files in directory {input_dir}")
    logger.info(f"Starting parallel processing with max_workers={max_workers}\n")

    # Parallel processing of the files
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_parquet_file, pf, input_path, dedup_eval_id_dir, all_cols_dedup_dir)
            for pf in parquet_files
        ]

        for future in as_completed(futures):
            results.append(future.result())

    # Print processing summary
    logger.info("=== Processing Summary ===")
    for item in results:
        if len(item) == 4 and item[3] is not None:
            # Error occurred
            logger.error(f"[FAILED] File: {item[0]}, Error: {item[3]}")
        else:
            parquet_file, total_rows, duplicates_count_all_cols = item
            logger.info(
                f"[OK] - {parquet_file}\n"
                f"       Total rows: {total_rows}, Duplicates by all columns: {duplicates_count_all_cols}"
            )

    logger.info("\nDone processing all Parquet files in parallel!\n")


if __name__ == "__main__":
    input_dir = "/cs/snapless/gabis/eliyahabba/ibm_results_data_full_processed_split"
    process_parquet_files_parallel(input_dir, max_workers=8)