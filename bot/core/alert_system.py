from bot.core.strategy_engine import StrategyEngine
from bot.core.data_processing import DataProcessor
from bot.api.gemini_client import GeminiClient
from bot.core.ml_models import MLModel
import numpy as np
import logging

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
        self.ml_model = MLModel(model_path="models/m1h_model.joblib")  # Load the trained model

    def calculate_ml_confidence(self, df):
        """Calculate ML-based confidence using the trained model."""
        try:
            # Prepare features for prediction
            features = df[['rsi', 'macd', 'bb_upper', 'bb_lower', 'ema', 'atr', 'vwap']].tail(1)

            # Predict probability of price increase
            probabilities = self.ml_model.model.predict_proba(features)[0]
            confidence = probabilities[1] * 100  # Probability of "1" (price increase)

            return confidence
        except Exception as e:
            logger.error(f"Error calculating ML confidence: {e}")
            return 50  # Default confidence if prediction fails

    def calculate_ds_confidence(self, df):
        """Calculate Gemini-based confidence."""
        try:
            # Analyze market sentiment using Gemini API
            last_row = df.iloc[-1]
            support, resistance = self.engine.calculate_support_resistance(df)
            fib_levels = self.engine.calculate_fibonacci_levels(df)
        
            text = (
                f"Market analysis for BTCUSDT:\n"
                f"Close price: {last_row['close']:.2f}\n"
                f"RSI: {last_row['rsi']:.2f} (Overbought if > 70, Oversold if < 30)\n"
                f"MACD: {last_row['macd']:.2f}, Signal Line: {last_row['macdsignal']:.2f}\n"
                f"Support Level: {support:.2f}, Resistance Level: {resistance:.2f}\n"
                f"Bollinger Bands: Upper={last_row['bb_upper']:.2f}, Middle={last_row['bb_middle']:.2f}, Lower={last_row['bb_lower']:.2f}\n"
                f"EMA (50): {last_row['ema']:.2f}\n"
                f"ATR (14): {last_row['atr']:.2f}\n"
                f"Volume-Weighted Average Price (VWAP): {last_row['vwap']:.2f}\n"
                f"Fibonacci Levels:\n"
                f"   - 61.8% Retracement: {fib_levels['61.8%']:.2f}\n"
                f"   - 38.2% Retracement: {fib_levels['38.2%']:.2f}\n"
                f"   - 0% (High): {fib_levels['0%']:.2f}, 100% (Low): {fib_levels['100%']:.2f}\n"
                f"Volume: {last_row['volume']:.2f}\n"
            )
        
            sentiment_score = self.gemini_client.analyze(text)
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