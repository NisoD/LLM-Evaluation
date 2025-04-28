import logging
import os
from datetime import datetime
from pathlib import Path
from huggingface_hub import HfApi, create_repo


class HFDirectoryUploader:
    def __init__(self, local_dir: str, repo_name: str, token: str):
        self.local_dir = Path(local_dir)
        self.repo_name = repo_name
        self.token = token
        self.api = HfApi(token=token)
        self.setup_logger()
        self.logger.info(f"Initialized HFDirectoryUploader with local_dir: {local_dir}, repo_name: {repo_name} with token: {token}")

    def setup_logger(self):
        """Setup logging configuration"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = logging.getLogger('HFUploader')
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(f'hf_upload_{timestamp}.log')
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def upload_inner_content(self):
        """Upload inner directory content to HuggingFace"""
        self.logger.info(f"Starting upload of inner content from {self.local_dir} to {self.repo_name}")

        # Ensure repository exists
        try:
            create_repo(
                repo_id=self.repo_name,
                token=self.token,
                private=True,
                repo_type="dataset"
            )
        except Exception as e:
            self.logger.info(f"Repository already exists or error occurred: {e}")

        # Get immediate subdirectories
        subdirs = [d for d in self.local_dir.iterdir() if d.is_dir()]

        # Upload each subdirectory
        for subdir in subdirs:
            try:
                self.logger.info(f"Uploading directory: {subdir.name}")
                self.api.upload_folder(
                    folder_path=str(subdir),
                    repo_id=self.repo_name,
                    repo_type="dataset",
                    path_in_repo=subdir.name  # Upload to root of repo with subdir name
                )
                self.logger.info(f"Successfully uploaded {subdir.name}")

            except Exception as e:
                self.logger.error(f"Error uploading directory {subdir.name}: {e}")
                continue

        # Upload any files in the root directory
        root_files = [f for f in self.local_dir.iterdir() if f.is_file()]
        if root_files:
            try:
                self.logger.info("Uploading files from root directory")
                for file in root_files:
                    self.api.upload_file(
                        path_or_fileobj=str(file),
                        path_in_repo=file.name,
                        repo_id=self.repo_name,
                        repo_type="dataset"
                    )
                self.logger.info("Successfully uploaded root files")
            except Exception as e:
                self.logger.error(f"Error uploading root files: {e}")

        self.logger.info("Upload completed")


if __name__ == "__main__":
    # Configuration
    LOCAL_DIR = "/cs/snapless/gabis/eliyahabba/ibm_results_data_full_processed_split_to_folders_all_cols_deduped"
    REPO_NAME = "Evaluating-LLM-Evaluation/evaluating_llm_evaluation_lite"
    from config.get_config import Config


    config = Config()
    TOKEN = config.config_values.get("hf_access_token", "")
    # Initialize and run uploader
    uploader = HFDirectoryUploader(
        local_dir=LOCAL_DIR,
        repo_name=REPO_NAME,
        token=TOKEN
    )

    uploader.upload_inner_content()
