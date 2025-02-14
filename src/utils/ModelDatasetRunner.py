from pathlib import Path
from typing import Callable

from src.utils.Constants import Constants

TemplatesGeneratorConstants = Constants.TemplatesGeneratorConstants
ExperimentConstants = Constants.ExperimentConstants
LLMProcessorConstants = Constants.LLMProcessorConstants


class ModelDatasetRunner:
    def __init__(self, structured_input_folder_path: str, evaluate_on: list):
        self.structured_input_folder_path = structured_input_folder_path
        self.evaluate_on = evaluate_on

    def run_function_on_all_models_and_datasets(self, processing_function: Callable,
                                                **kwargs) -> None:
        results_folder = Path(self.structured_input_folder_path)
        eval_on = self.evaluate_on
        models_names = [model.split('/')[1] for model in LLMProcessorConstants.MODEL_NAMES.values()]
        chosen_models_names = ["GEMMA_7B", "OLMo-1.7-7B-hf", "Mistral-7B-Instruct-v0.3", "Mistral-7B-Instruct-v0.2",
                               "Llama-2-13b-hf"]
        chosen_models_names = ["Llama-2-13b-hf"]
        models_names = [model for model in models_names if model in chosen_models_names]
        # models_names = [model for model in models_names if  "Llama-2-13b-chat-hf" in model]
        models_folders = [Path(results_folder / model_name) for model_name in models_names]

        for model_name in models_folders:
            print(f"Model {model_name.name}")
            datasets = sorted([file for file in model_name.glob("*") if file.is_dir()])
            # datasets = [datasets for datasets in datasets if datasets.name == "mmlu.high_school_psychology"]
            for dataset_folder in datasets:
                shots = [file for file in dataset_folder.glob("*") if file.is_dir()]
                shots = [shots for shots in shots if shots.name == "three_shot"]
                for shot in shots:
                    formats = [file for file in shot.glob("*") if file.is_dir()]
                    for format_folder in formats:
                        for eval_value in eval_on:
                            try:
                                processing_function(format_folder, eval_value, **kwargs)
                            except Exception as e:
                                print(f"Error in {model_name.name}/{dataset_folder.name}/{shot.name}/"
                                      f"{format_folder.name} for {eval_value}: {e}")

            print("\n")


# Example usage:
def check_comparison_matrix(folder, eval_value):
    print(f"Checking comparison matrix in {folder} for {eval_value}")


if __name__ == "__main__":
    results_folder = ExperimentConstants.MAIN_RESULTS_PATH / Path(
        TemplatesGeneratorConstants.MULTIPLE_CHOICE_STRUCTURED_FOLDER_NAME)
    eval_on = ExperimentConstants.EVALUATE_ON_INFERENCE
    runner = ModelDatasetRunner(results_folder, eval_on)
    runner.run_function_on_all_models_and_datasets(check_comparison_matrix)
    # runner.run_function_on_all_models_and_datasets(check_results_files)
