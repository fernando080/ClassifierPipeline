# ModelTrainingPipeline

A machine learning pipeline project for training, tuning, and evaluating different models, including Logistic Regression, Neural Networks, and XGBoost, using a structured approach with modular code.

IMPORTANT: This repository is probably outdated and not maintained. Please use it as a reference only.

## Project Structure

The project directory is organized as follows:
├── dataset 
│ └── data_test.xlsx # Dataset file used for model training and evaluation 
├── models 
│ ├── nn 
│ │ ├── logs # TensorBoard logs for neural network training 
├── .gitignore # Git ignore file 
├── logReg_model.py # Script for training and tuning Logistic Regression model 
├── nn_model.py # Script for training and tuning Neural Network model 
├── poetry.lock # Poetry lock file for dependencies
├── pyproject.toml # Poetry project file with dependencies and configuration 
├── train_model_pipeline.py # Main pipeline script to train different models 
└── xgboost_model.py # Script for training and tuning XGBoost model


## Requirements

This project uses [Poetry](https://python-poetry.org/) for dependency management. Make sure you have Poetry installed:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

## Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/ModelTrainingPipeline.git
cd ModelTrainingPipeline
```
2. Install dependencies using Poetry:

```bash
poetry install
```

3. Activate the Poetry environment:

```bash
poetry shell
```

## Usage
This project provides different scripts to train specific models as well as a main pipeline script (train_model_pipeline.py) to handle the entire workflow.

### Running the Pipeline Script
To run the main pipeline script, use:

```bash
poetry run python train_model_pipeline.py
```

You can specify the model type (e.g., logistic, nn, xgboost) in the main() function of train_model_pipeline.py to switch between models.

### Running Individual Model Scripts
You can also run specific model scripts independently. For example:

1. Logistic Regression:

```bash
poetry run python logReg_model.py
```

2. Neural Network:

```bash
poetry run python nn_model.py
```

3. XGBoost:

```bash
poetry run python xgboost_model.py
```

## Logging and Model Outputs
1. Neural Network Logs: TensorBoard logs for neural network training are stored in models/nn/logs.
2. Model Checkpoints: The trained model for the neural network is saved in models/nn/risk_nn_model.h5.

## Customizing Hyperparameters
Hyperparameter grids for each model are defined within their respective scripts. Modify the parameter grids in logReg_model.py, nn_model.py, and xgboost_model.py to experiment with different configurations.

## Directory Overview
1. dataset: Contains the dataset file, data_test.xlsx, which is used for training and validation.
2. models: Stores model outputs and TensorBoard logs.
3. train_model_pipeline.py: The main pipeline script that orchestrates data loading, preprocessing, model training, and evaluation.
4. .gitignore: Specifies files and directories to be ignored by Git, such as environment files and model logs.
5. pyproject.toml and poetry.lock: Poetry configuration files that manage project dependencies.
