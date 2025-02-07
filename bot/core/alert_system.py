from bot.core.strategy_engine import StrategyEngine
from bot.core.data_processing import DataProcessor
from bot.api.gemini_client import GeminiClient
from bot.core.ml_models import MLModel
import numpy as np
import logging
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class AlertSystem:
    def __init__(self):
        self.engine = StrategyEngine()
        self.processor = DataProcessor()
        self.gemini_client = GeminiClient()
        # Don't load the model here. Load it in telegram_bot.py and pass it.
        # self.ml_model = MLModel(model_path="models/m1h_model.joblib")

    def calculate_ml_confidence(self, df, model):
        """Calculate ML-based confidence using the trained model."""
        try:
            # Check if the model has selected_features
            if model.selected_features is None:
                logger.warning("ML model has no selected features.  Returning default confidence.")
                return 50.0

            # Filter the DataFrame to include *only* the selected features
            features_df = df[model.selected_features]

            # Check if features_df is empty
            if features_df.empty:
                logger.warning("Feature DataFrame is empty after filtering. Returning default confidence.")
                return 50.0

            # Use the LAST row of the processed features
            features = features_df.tail(1)

            # Predict probability of price increase
            probabilities = model.predict_proba(features)[0]
            confidence = probabilities[1] * 100  # Probability of "1" (price increase)

            return confidence
        except Exception as e:
            logger.exception(f"Error calculating ML confidence: {e}") # Use logger.exception
            return 50.0  # Default confidence if prediction fails

    def calculate_ds_confidence(self, df):
        """Calculate Gemini-based confidence."""
        try:
            # Use the last 50 rows for sentiment analysis
            df_str = df.tail(50).to_string()
            sentiment_score = self.gemini_client.analyze_sentiment(df_str)
            confidence = (sentiment_score + 1) * 50  # Scale to 0-100%
            return confidence
        except Exception as e:
            print(f"Error calculating Gemini confidence: {e}")
            return 50  # Default confidence if analysis fails

    def calculate_combined_confidence(self, df, model):
        """Calculate combined confidence for a single timeframe."""
        ml_confidence = self.calculate_ml_confidence(df, model) # Pass the model
        ds_confidence = self.calculate_ds_confidence(df)
        return (ml_confidence + ds_confidence) / 2

    def generate_automatic_signal(self, df, model):
        """Generate automatic signal based on ML model confidence."""
        try:
            # Check if the model has selected_features
            if model.selected_features is None:
                logger.warning("ML model has no selected features. Returning HOLD signal.")
                return "HOLD", 50.0

            # Filter the DataFrame!
            features_df = df[model.selected_features]

            if features_df.empty:
                logger.warning("Feature DataFrame is empty after filtering. Returning HOLD signal.")
                return "HOLD", 50.0

            features = features_df.iloc[[-1]]  # Use iloc[[-1]] for DataFrame
            probabilities = model.predict_proba(features)[0]
            confidence = probabilities[1] * 100

            if confidence >= 80:
                prediction = model.predict(features)[0]
                signal = "BUY" if prediction == 1 else "SELL"
                return signal, confidence
            else:
                return "HOLD", confidence

        except Exception as e:
            logger.exception(f"Error generating automatic signal: {e}")
            return "HOLD", 50
