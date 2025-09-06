import argparse
from automl_pipeline.config import Config
from automl_pipeline.data_loader import DataLoader
from automl_pipeline.preprocess import Preprocessor
from automl_pipeline.model_selector import get_model
from automl_pipeline.trainer import Trainer
from automl_pipeline.evaluator import Evaluator
from automl_pipeline.logger import get_logger

def run_pipeline(config_path):
    logger = get_logger()
    try:
        config = Config(config_path)
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return

    logger.info("Loading data...")
    try:
        data = DataLoader(config.data_path).load()
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    if config.target not in data.columns:
        logger.error(f"Target column '{config.target}' not found in data.")
        return

    X = data.drop(config.target, axis=1)
    y = data[config.target]
    logger.info("Preprocessing data...")
    try:
        preprocessor = Preprocessor()
        X = preprocessor.fit_transform(X)
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return

    for model_cfg in config.models:
        model_name = model_cfg.get('name')
        params = model_cfg.get('params', {})
        logger.info(f"Selecting model: {model_name}")
        try:
            model = get_model(model_name, params)
        except Exception as e:
            logger.error(f"Error selecting model: {e}")
            continue

        trainer = Trainer(model, config.test_size, config.random_state)
        try:
            X_train, X_test, y_train, y_test = trainer.split(X, y)
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            continue

        logger.info(f"Training model: {model_name}")
        try:
            trainer.train(X_train, y_train)
        except Exception as e:
            logger.error(f"Error during training: {e}")
            continue

        try:
            y_pred = trainer.predict(X_test)
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            continue

        evaluator = Evaluator(config.metrics)
        results = evaluator.evaluate(y_test, y_pred)
        logger.info(f"Results for {model_name}: {results}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    run_pipeline(args.config)