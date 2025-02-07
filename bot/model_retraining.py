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

def retrain_models(model_type='random_forest'): # Add model_type argument
    """Retrain the ML model and save it to disk."""
    try:
        processor = DataProcessor()
        model = MLModel(model_type=model_type) # Pass model_type to MLModel

        # Fetch historical data
        data = processor.fetch_data("BTCUSDT", "1h", limit=1000) # Fetch more data

        # Preprocess data into features and labels, and split into train/test
        X_train, X_test, y_train, y_test = processor.preprocess_for_training(data)

        # --- Feature Selection ---
        selected_features = model.select_features(X_train, y_train, n_features_to_select=30) # Select top 30
        X_train = X_train[selected_features]  # Use selected features
        X_test = X_test[selected_features]

        # Tune hyperparameters (optional, but highly recommended)
        model.tune_hyperparameters(X_train, y_train)

        # Train the model
        model.train(X_train, y_train)

        # Evaluate the model on the TEST set
        accuracy = model.evaluate(X_test, y_test)
        logger.info(f"Model accuracy on test set: {accuracy:.2f}")

        # --- Backtesting ---
        backtest_results = model.backtest(data, "BTCUSDT")
        logger.info(f"Backtesting Results: {backtest_results}")
        # You can add more detailed logging of backtest results here

        # Save the trained model
        model_path = f"models/{model_type}_model.joblib"  # Use model_type in filename
        os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure directory exists
        model.save_model(model_path)
        logger.info(f"Model retrained and saved successfully at {model_path}.")

        # Update last retraining time
        with open("last_retraining_time.txt", "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as e:
        logger.error(f"Error during model retraining: {e}")

if __name__ == "__main__":
    # Example usage:  Retrain different model types
    retrain_models(model_type='random_forest')
    # retrain_models(model_type='gradient_boosting')  # Uncomment to retrain other models
    # retrain_models(model_type='logistic_regression')
