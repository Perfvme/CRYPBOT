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

        # Preprocess data into features and labels
        X, y = processor.preprocess_for_training(data)

        # Train the model
        model.train(X, y)

        # Save the trained model
        model_path = "models/m1h_model.joblib"
        model.save_model(model_path)
        logger.info(f"Model retrained and saved successfully at {model_path}.")

        # Update last retraining time
        with open("last_retraining_time.txt", "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as e:
        logger.error(f"Error during model retraining: {e}")