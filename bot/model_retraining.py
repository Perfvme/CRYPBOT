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

def retrain_models():
    """Retrain the ML model and save it to disk."""
    try:
        processor = DataProcessor()
        model = MLModel()

        # Fetch historical data
        data = processor.fetch_data("BTCUSDT", "1h")

        # Preprocess data into features and labels, and split into train/test
        X_train, X_test, y_train, y_test = processor.preprocess_for_training(data)

        # Tune hyperparameters (optional, but highly recommended)
        model.tune_hyperparameters(X_train, y_train)

        # Train the model
        model.train(X_train, y_train)

        # Evaluate the model on the TEST set
        accuracy = model.evaluate(X_test, y_test)
        logger.info(f"Model accuracy on test set: {accuracy:.2f}")


        # Save the trained model
        model_path = "models/m1h_model.joblib"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure directory exists
        model.save_model(model_path)
        logger.info(f"Model retrained and saved successfully at {model_path}.")

        # Update last retraining time
        with open("last_retraining_time.txt", "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as e:
        logger.error(f"Error during model retraining: {e}")

if __name__ == "__main__":
    retrain_models()
