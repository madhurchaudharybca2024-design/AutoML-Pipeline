# AutoML-Pipeline

Automated Machine Learning Workflow for Rapid Prototyping  
**Production-ready, beginner-friendly Python package**

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Updated Project Structure](#updated-project-structure)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Extending the Pipeline](#extending-the-pipeline)
- [Docker Support](#docker-support)
- [API Reference](#api-reference)
- [Testing](#testing)
- [FAQ](#faq)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

**AutoML-Pipeline** automates the end-to-end machine learning workflow for tabular data:  
- Loads your data
- Preprocesses features
- Selects and trains models
- Evaluates performance
- Logs results

All with **easy configuration** and **one command**.

---

## Features

- Automated data loading and preprocessing (with NaN handling and category encoding)
- Model selection: RandomForest, LogisticRegression, XGBoost, LightGBM
- Training, evaluation, and metric reporting
- YAML config for reproducibility
- Logging and error handling
- Unit tests for reliability
- Docker containerization (easy deploy anywhere)
- Beginner-friendly and extensible

---

## Updated Project Structure

```
AutoML-Pipeline/
│
├── automl_pipeline/
│   ├── __init__.py
│   ├── config.py           # YAML config management & validation
│   ├── data_loader.py      # Robust CSV loading
│   ├── preprocess.py       # Numeric & categorical preprocessing, NaN handling
│   ├── model_selector.py   # Safe model selection (with error messages)
│   ├── trainer.py          # Train/test split, model training
│   ├── evaluator.py        # Classification metrics & safe evaluation
│   ├── logger.py           # Logging setup (no duplicate handlers)
│   └── pipeline.py         # Main orchestrator with error handling
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocess.py
│   ├── test_model_selector.py
│   ├── test_trainer.py
│   ├── test_evaluator.py
│   └── test_pipeline.py
│
├── examples/
│   ├── sample_config.yaml
│   └── sample_data.csv
│
├── Dockerfile              # For containerized deployment
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
└── LICENSE
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/madhurchaudharybca2024-design/AutoML-Pipeline.git
cd AutoML-Pipeline
```

### 2. Install Dependencies

> **Recommended:** Use a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Quickstart

### 1. Prepare Your Data

- Place your CSV file in `examples/sample_data.csv`
- Example format:
    ```
    feature1,feature2,feature3,label
    1.0,A,5.4,0
    2.1,B,3.2,1
    3.2,C,6.1,0
    ```

### 2. Configure the Pipeline

- Edit `examples/sample_config.yaml`:
    ```yaml
    data_path: examples/sample_data.csv
    target: label
    models:
      - name: random_forest
        params:
          n_estimators: 100
          max_depth: 5
      - name: logistic_regression
        params:
          solver: liblinear
    test_size: 0.2
    random_state: 42
    metrics: [accuracy, f1]
    ```

### 3. Run the Pipeline

```bash
python -m automl_pipeline.pipeline --config examples/sample_config.yaml
```

You'll see logs for each step, and results like:
```
Results for random_forest: {'accuracy': 0.80, 'f1': 0.78}
Results for logistic_regression: {'accuracy': 0.76, 'f1': 0.75}
```

---

## Configuration

All major options are controlled via a YAML config (`examples/sample_config.yaml`):

| Field         | Description                                  | Example           |
|---------------|----------------------------------------------|-------------------|
| data_path     | CSV file path                                | `examples/data.csv`|
| target        | Target column name                           | `label`           |
| models        | List of models + parameters                  | see above         |
| test_size     | Train/test split ratio                       | `0.2`             |
| random_state  | Seed for reproducibility                     | `42`              |
| metrics       | Metrics to evaluate (`accuracy`, `f1`, etc.) | `[accuracy, f1]`  |

---

## Extending the Pipeline

### Adding a New Model

1. Edit `automl_pipeline/model_selector.py`
2. Add a new `elif` for your model, e.g.:
    ```python
    elif name == 'svc':
        from sklearn.svm import SVC
        return SVC(**params)
    ```
3. Update your config file with the new model name and parameters.

### Adding a New Preprocessing Step

1. Edit `automl_pipeline/preprocess.py`
2. Add your custom logic in `fit_transform` or `transform`.

---

## Docker Support

### 1. Build the Docker Image

```bash
docker build -t automl-pipeline .
```

### 2. Run the Pipeline in Docker

```bash
docker run --rm -v $(pwd)/examples:/app/examples automl-pipeline --config /app/examples/sample_config.yaml
```

- This mounts your local `examples` folder into the container for data/config access.

### 3. Dockerfile Example

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "-m", "automl_pipeline.pipeline", "--config", "examples/sample_config.yaml"]
```

---

## API Reference

**Modules:**

- `automl_pipeline/config.py` - Loads YAML config
- `automl_pipeline/data_loader.py` - Reads CSVs
- `automl_pipeline/preprocess.py` - Scales/encodes features, handles NaNs
- `automl_pipeline/model_selector.py` - Chooses ML models (with error handling)
- `automl_pipeline/trainer.py` - Splits, trains, predicts
- `automl_pipeline/evaluator.py` - Computes metrics
- `automl_pipeline/logger.py` - Logs steps/results
- `automl_pipeline/pipeline.py` - Orchestrates everything

---

## Testing

Run all unit tests:

```bash
pytest tests/
```

---

## FAQ

**Q: Can I use Excel files?**  
A: Only CSV is supported out-of-the-box. For Excel, modify `data_loader.py` to use `pd.read_excel`.

**Q: Can I use regression models?**  
A: Yes! Add regression models in `model_selector.py`, update `metrics` in config.

**Q: How do I deploy this?**  
A: Use Docker for container deployment, or package via `setup.py`.

---

## Troubleshooting

- **ImportError for xgboost/lightgbm:**  
  Remove those models from your config if you don't need them, or install via pip (`pip install xgboost lightgbm`).
- **FileNotFoundError:**  
  Double-check the paths in your config file.
- **ValueError in model selection:**  
  Ensure the model name in config matches `model_selector.py`.

---

## License

MIT

---

**Beginner Tip:**  
If you get stuck, start with the provided sample data and config.  
Once it works, swap in your own CSV and adjust the `target` field!

---