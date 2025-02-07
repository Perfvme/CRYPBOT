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
        self.ml_model = MLModel(model_path="models/random_forest_model.joblib")  # Load the trained model

    def calculate_ml_confidence(self, df):
        """Calculate ML-based confidence using the trained model."""
        try:
            # Get enough data for feature engineering
            temp_df = df.tail(50).copy()  # Use at least 50 rows
            features_df = self.processor.get_prediction_features(temp_df)

            # Check if features_df is empty or if selected features are missing
            if features_df.empty or not all(feature in features_df.columns for feature in self.ml_model.selected_features):
                logger.warning("Feature DataFrame is empty or missing selected features. Returning default confidence.")
                return 50

            # Select only the features used during training
            features = features_df[self.ml_model.selected_features].tail(1)

            # Predict probability of price increase
            probabilities = self.ml_model.model.predict_proba(features)[0]
            confidence = probabilities[1] * 100  # Probability of "1" (price increase)

            return confidence
        except Exception as e:
            logger.exception(f"Error calculating ML confidence: {e}") # Use logger.exception
            return 50  # Default confidence if prediction fails

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

    def calculate_combined_confidence(self, df):
        """Calculate combined confidence for a single timeframe."""
        ml_confidence = self.calculate_ml_confidence(df)
        ds_confidence = self.calculate_ds_confidence(df)
        return (ml_confidence + ds_confidence) / 2

    def generate_automatic_signal(self, df):
        """Generate automatic signal based on ML model confidence."""
        try:
            temp_df = df.tail(50).copy()
            features_df = self.processor.get_prediction_features(temp_df)

            # Check for empty DataFrame or missing features
            if features_df.empty or (self.ml_model.selected_features is not None and not all(feature in features_df.columns for feature in self.ml_model.selected_features)):
                logger.warning("Feature DataFrame is empty or missing selected features. Returning HOLD signal.")
                return "HOLD", 50  # Return HOLD and default confidence

            # Use .iloc[[-1]] to get a DataFrame, even with a single row.  This is important!
            features = features_df[self.ml_model.selected_features].iloc[[-1]]
            probabilities = self.ml_model.model.predict_proba(features)[0]
            confidence = probabilities[1] * 100 # Probability of price increase

            if confidence >= 80:
                # Determine signal based on predicted class (0 or 1)
                prediction = self.ml_model.model.predict(features)[0]
                signal = "BUY" if prediction == 1 else "SELL"
                return signal, confidence
            else:
                return "HOLD", confidence

        except Exception as e:
            logger.exception(f"Error generating automatic signal: {e}")
            return "HOLD", 50 # Default to HOLD on error
