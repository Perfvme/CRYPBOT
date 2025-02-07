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
        X_test = X
