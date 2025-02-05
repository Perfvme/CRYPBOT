import os
from datetime import datetime
from bot.core.ml_models import MLModel
from bot.core.data_processing import DataProcessor
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def retrain_models():
    """Retrain the ML model and save it to disk."""
    try:
        logger.info("Starting model retraining process...")
        
        # Initialize components
        processor = DataProcessor()
        model = MLModel()

        # Fetch historical data
        logger.info("Fetching historical data from Binance...")
        data = processor.fetch_data("BTCUSDT", "1h")
        if not data:
            logger.error("No data fetched from Binance. Exiting.")
            return
        
        logger.debug(f"Fetched {len(data)} rows of data.")

        # Preprocess data into features and labels
        logger.info("Preprocessing data for training...")
        X, y = processor.preprocess_for_training(data)
        logger.debug(f"Features shape: {X.shape}, Labels shape: {y.shape}")
        if X.empty or y.empty:
            logger.error("Preprocessing failed. No valid features or labels found.")
            return

        # Train the model
        logger.info("Training the Random Forest model...")
        model.train(X, y)

        # Save the trained model
        model_path = "models/m1h_model.joblib"
        logger.info(f"Saving the trained model to {model_path}...")
        model.save_model(model_path)

        # Update last retraining time
        with open("last_retraining_time.txt", "w") as f:
            f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        logger.info("Model retraining completed successfully.")
    except Exception as e:
        logger.error(f"Error during model retraining: {e}")

if __name__ == "__main__":
    retrain_models()
