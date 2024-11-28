# DVC Setup & Directory Structure

This README provides a guide to understand the structure of the `src` directory. The directory is designed to integrate with Data Version Control (DVC) to ensure reproducibility and efficient management of the project's machine learning pipeline.

## Repository Overview
This directory contains modular scripts and functions organized to support a DVC workflow. Each script performs a specific task within the pipeline, such as data preparation, model training, or evaluation. The pipeline stages are connected through DVC.

### Key Features
- Modular and reusable scripts.
- Clear separation of pipeline stages.
- Integrated with DVC for tracking dependencies, outputs, and metadata.
- Compatible with YAML-based pipeline configurations.

---

## Directory Structure
```
/src
├── stage_load_data.py          # Scripts for ingesting raw data
├── stage_prep_features.py      # Scripts for data cleaning, transformation, and feature engineering
├── stage_train_model.py        # Script to train machine learning models
├── stage_select_and_tune.py    # Script to perform hyperparameter tuning and feature selection
├── stage_evaluate_model.py     # Script to evaluate model performance
└── transfer_learning.py        # Script for running inference
```

---

## Setting Up the Environment
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/wri/plantation_classifier.git
   cd plantation_classifier
   ```

2. **Install Dependencies:**
   Create a virtual environment and install dependencies from `requirements.txt`:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Install DVC:**
   Ensure that DVC is installed for managing the pipeline:
   ```bash
   pip install dvc
   ```

4. **Initialize DVC:**
   If DVC is not already initialized in the repository:
   ```bash
   dvc init
   ```

---

## Using DVC Pipelines
The pipeline is defined in the `dvc.yaml` file, with dependencies, parameters, and outputs explicitly stated for each stage.

### Common Commands
1. **Check the Pipeline:**
   Verify the pipeline stages and dependencies:
   ```bash
   dvc dag
   ```

2. **Run the Pipeline:**
   Execute the entire pipeline or specific stages:
   ```bash
   dvc repro
   ```

3. **Track Data Files:**
   Add data files to DVC for versioning:
   ```bash
   dvc add data/raw_data.csv
   ```

4. **Push Data to Remote Storage:**
   Ensure remote storage is configured in `.dvc/config`:
   ```bash
   dvc push
   ```

5. **Pull Data from Remote Storage:**
   Retrieve data files for the pipeline:
   ```bash
   dvc pull
   ```

---

## Parameters Management
Pipeline parameters are defined in `params.yaml`. This file centralizes hyperparameters and configuration options for each stage of the pipeline. Update parameters as needed, and rerun the pipeline using `dvc repro` to propagate changes.

---

## Additional Resources
- [DVC Documentation](https://dvc.org/doc)
- [Main Repository](https://github.com/wri/plantation_classifier)
