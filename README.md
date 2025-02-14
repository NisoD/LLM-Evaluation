# Efficient and Robust Evaluation of LLMs

## Overview

Overview
This repository contains scripts for conducting exhaustive evaluations of large language models (LLMs) across various
datasets. The focus is on different prompt patterns, models, and tasks, aiming to identify explainable patterns in LLM
behavior that optimize model performance for specific tasks like question answering on scientific documents.This project
aims to evaluate the robustness of Language Model Models (LLMs)

## Prerequisites

- Access to a SLURM cluster with GPU capabilities.
- Clone the Repository:
    
    ```
    git clone git@github.com:eliyahabba/LLM-Evaluation.git
    ```
  
- Install required Python packages:

```
cd LLM-Evaluation
pip install -r requirements.txt
```

- Hugging Face Access Token: token: Ensure you have an **access_token** file in the utils folder to authenticate API
  calls:
  Obtain a Hugging Face access token by creating an account at Hugging Face and navigating to your account settings to
  generate a new token.
  This token is required to authenticate API calls, especially for accessing models that require permissions.
  
Open the **local_config.yaml** file located at **config/local_config.yaml**.
Add or update the access_token key with your Hugging Face access token as follows:

```
  access_token: "YOUR_ACCESS_TOKEN_HERE"
```

Replace 'YOUR_ACCESS_TOKEN' with your actual access token.

-Virtual Environment Path
Open the local_config.yaml file located at config/local_config.yaml.
Add or update the venv_path key with the path to your Python virtual environment as follows:
    
```
venv_path: "/path/to/your/virtualenv"
```
Replace "/path/to/your/virtualenv" with the actual path to your virtual environment. This path is essential for the Bash scripts that may need to activate the virtual environment.

## Usage Instructions

### General

Both scripts request memory and GPU resources as estimates. You may need to adjust these values depending on the dataset
size and model complexity. Modify the --mem, --gres, and other related SLURM directives in the scripts according to your
needs.

### Running on All Datasets

To launch evaluations across all datasets:

Choose a model, for example, *Llama* and navigate to scheduler environment.

```
cd src/sh_files/{scheduler environment - (slrum/bsub)}
```

Run the script:

```
sbatch llama/run_70b_on_all_the_datasets.sh
```

This script processes each dataset according to predefined configurations.

### Running on Specific Datasets

Choose and navigate to the appropriate model directory as described above. If the directory does not exist, create it:

```
mkdir -p src/sh_files/llama
cd src/sh_files
```

Run the script:

```
sbatch llama/run_dataset_on_model_70b.sh --card {example: cards.mmlu.clinical_knowledge}   --template_range {min: 0, max: 10}
```

[//]: # (The project is organized as follows:)

[//]: # ()

[//]: # (- `data/`: Contains original and modified datasets.)

[//]: # (- `models/`: Contains pretrained LLM models.)

[//]: # (- `Modifiers/`: Contains scripts for modifying datasets.)

[//]: # (- `experiments/`: Contains scripts for running experiments and analyzing results.)

[//]: # (- `README.md`: Documentation explaining the project and its structure.)

[//]: # ()

[//]: # (## Dataset Modification)

[//]: # ()

[//]: # (Different types of dataset modifications are implemented based on specific tasks:)

[//]: # ()

[//]: # (- `MultipleChoiceModifier`: Shuffles answer choices for multiple-choice questions.)

[//]: # (- `QAModifier`: Implements modifications for question-answering datasets.)

[//]: # (- `NLIModifier`: Implements modifications for natural language inference datasets.)

[//]: # ()

[//]: # (## Experimentation)

[//]: # ()

[//]: # (The project conducts experiments to evaluate LLM performance:)

[//]: # ()

[//]: # (1. Identifying tasks and models for evaluation.)

[//]: # (2. Modifying datasets using appropriate modifiers.)

[//]: # (3. Running models on both original and modified datasets.)

[//]: # (4. Analyzing results to assess model robustness.)

[//]: # ()

[//]: # (## Usage)

[//]: # ()

[//]: # (1. Clone the repository: `git clone https://github.com/your-username/robust-llm-evaluation.git`)

[//]: # (2. Install dependencies: `pip install -r requirements.txt`)

[//]: # (3. Run dataset modifications: `python dataset_modification/modify_dataset.py`)

[//]: # (4. Run experiments: `python experiments/run_experiment.py`)

[//]: # (5. Analyze results: `python experiments/analyze_results.py`)

## Contributors

## License

This project is licensed under the [MIT License](LICENSE).
