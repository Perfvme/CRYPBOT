from bot.core.ml_models import MLModel
from bot.core.data_processing import DataProcessor
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def retrain_models(model_type='random_forest'):
    """Retrain the ML model and save it."""
    try:
        processor = DataProcessor()
        model = MLModel(model_type=model_type)

        # Fetch historical data
        data = processor.fetch_data("BTCUSDT", "1h", limit=1000)

        # Preprocess:  Get BOTH X, y, *and* the full preprocessed DataFrame
        X_train, X_test, y_train, y_test = processor.preprocess_for_training(data)
        df_train = processor.preprocess_data(data)  # Get the full DataFrame
        df_train = processor._engineer_features(df_train) # Feature engineering
        df_train = df_train.iloc[:-1]  # Drop last row (same as in preprocess_for_training)


        # Feature Selection
        selected_features = model.select_features(X_train, y_train)
        X_train = X_train[selected_features]
        X_test = X_test[selected_features]

        # Hyperparameter Tuning
        model.tune_hyperparameters(X_train, y_train)

        # Train
        model.train(X_train, y_train)

        # Evaluate
        accuracy = model.evaluate(X_test, y_test)
        logger.info(f"Model accuracy on test set: {accuracy:.2f}")

        # Backtesting (pass the *preprocessed* DataFrame)
        backtest_results = model.backtest(df_train, "BTCUSDT") # Pass the full df
        logger.info(f"Backtesting Results: {backtest_results}")

        # Save
        model_path = f"models/{model_type}_model.joblib"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save_model(model_path)
        logger.info(f"Model retrained and saved successfully at {model_path}.")

        # Update last retraining time
        with open("last_retraining_time.txt", "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    except Exception as e:
        logger.exception(f"Error during model retraining: {e}") # Use logger.exception

if __name__ == "__main__":
    retrain_models(model_type='random_forest')
    # retrain_models(model_type='gradient_boosting')
    # retrain_models(model_type='logistic_regression')
