from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Tuple
from tqdm import tqdm
import logging
from pathlib import Path
from src.analysis.create_plots.DataLoader import DataLoader

from helpers import (
    PipelineConfig,
    ModelAnalyzer,
    save_results,
    logger
)


def process_configuration(
        config: Tuple[str, int, str],
        data_loader: 'DataLoader',
        analyzer: ModelAnalyzer,
        pipeline_config: PipelineConfig
) -> Dict:
    """Process a single model/dataset configuration and prepare visualization data."""
    model_name, shots, dataset = config

    try:
        logger.info(f"Processing {model_name} with {shots} shots for {dataset}")

        # Load and filter data
        df = data_loader.load_and_process_data(
            model_name=model_name,
            shots=shots,
            datasets=[dataset]
        )

        if df.empty:
            return {'status': 'empty', 'config': config}

        # Filter specific configurations
        df = df[
            ((df['shots'] != 5) |
             (~df['choices_order'].isin(["correct_first", "correct_last"])))
        ]

        # Analyze performance
        results = analyzer.analyze_low_performance(df)

        if results is not None:
            # Enhance results with visualization data
            results['dataset_question'] = f"{dataset}_q{results['sample_index']}"

            # Find most chosen option for each question
            for idx in results['sample_index'].unique():
                choices = [f'choice_{i}_pct' for i in range(1, 5)]
                most_chosen = max(choices, key=lambda x: results.loc[results['sample_index'] == idx, x].iloc[0])
                results.loc[results['sample_index'] == idx, 'most_chosen_option'] = int(most_chosen.split('_')[1])

            save_results(
                results,
                pipeline_config,
                model_name,
                dataset,
                shots
            )

            return {
                'status': 'success',
                'config': config,
                'visualization_data': results[['dataset_question', 'most_chosen_option', 'accuracy']].to_dict('records')
            }

        return {'status': 'success', 'config': config}

    except Exception as e:
        logger.error(f"Error processing {config}: {str(e)}")
        return {'status': 'error', 'config': config, 'error': str(e)}


def main():
    """Main entry point for the analysis pipeline."""
    # Initialize configuration and components
    config = PipelineConfig()
    analyzer = ModelAnalyzer(threshold=config.performance_threshold)
    data_loader = DataLoader()  # You'll need to implement this class

    # Generate configurations to process
    configs = [
        (model, shots, dataset)
        for model in config.models
        for shots in config.shots
        for dataset in config.all_datasets
    ]

    # Process configurations in parallel
    with ProcessPoolExecutor(max_workers=config.num_processes) as executor:
        futures = [
            executor.submit(
                process_configuration,
                cfg,
                data_loader,
                analyzer,
                config
            )
            for cfg in configs
        ]

        results = list(tqdm(
            futures,
            total=len(configs),
            desc="Processing configurations"
        ))

    # Summarize results
    success = sum(1 for r in results if r.result()['status'] == 'success')
    errors = sum(1 for r in results if r.result()['status'] == 'error')
    empty = sum(1 for r in results if r.result()['status'] == 'empty')

    logger.info(f"""
    Pipeline completed:
    - Successful: {success}
    - Errors: {errors}
    - Empty: {empty}
    """)


if __name__ == "__main__":
    main()