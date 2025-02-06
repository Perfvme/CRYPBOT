import os
import google.generativeai as genai
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class GeminiClient:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

        # Initialize the Gemini client
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-001') # Initialize the model here

    def analyze_sentiment(self, text):
        """Analyze text for sentiment and return a score between -1 and 1."""
        try:
            prompt = (
                "You are a financial sentiment analysis model. Analyze the provided market data and return ONLY a sentiment score between -1 (bearish) and +1 (bullish). "
                f"Market Data: {text}"
            )

            response = self.model.generate_content(prompt)
            logger.debug(f"Gemini API raw response (sentiment): {response.text}")

            try:
                sentiment_score = float(response.text.strip())
                return sentiment_score
            except ValueError:
                logger.warning("Failed to parse sentiment score. Returning 0.")
                logger.debug(f"Raw response content: {response.text}")
                return 0  # Default fallback
        except Exception as e:
            logger.error(f"Error calling Gemini API (sentiment): {e}")
            return 0

    def analyze_strategy_confidence(self, symbol, strategy_name, raw_data, indicator_data):
        """Analyze market data for strategy confidence and return a score (0-100)."""
        try:
            prompt = (
                f"Analyze the following market data for {symbol} ({strategy_name}):\n"
                f"Raw Data: {raw_data}\n"
                f"Indicator Data: {indicator_data}\n"
                f"Provide a confidence level (0-100%) for the overall sentiment.\n"
                f"Format your response as:\nConfidence: [value]"
            )

            response = self.model.generate_content(prompt)
            logger.debug(f"Gemini API raw response (strategy confidence): {response.text}")

            try:
                ai_confidence = float(response.text.split("Confidence: ")[1].split("\n")[0])
                return ai_confidence
            except (IndexError, ValueError): # Catch more specific errors
                logger.warning(f"Failed to parse AI confidence for {strategy_name}. Returning 50.")
                return 50.0

        except Exception as e:
            logger.error(f"Error calling Gemini API (strategy confidence): {e}")
            return 50.0

    def analyze_global_recommendation(self, symbol, raw_data, indicator_data):
        """Analyze market data for global recommendation and return structured data."""
        try:
            prompt = (
                f"Analyze the following market data for {symbol} (Global Recommendation):\n"
                f"Raw Data: {raw_data}\n"
                f"Indicator Data: {indicator_data}\n"
                f"Provide an ideal entry point, stop-loss, take-profit, and confidence level (0-100%).\n"
                f"Format your response as:\n"
                f"Entry Point: [value]\nStop Loss: [value]\nTake Profit: [value]\nConfidence: [value]"
            )

            response = self.model.generate_content(prompt)
            logger.debug(f"Gemini API raw response (global recommendation): {response.text}")

            try:
                gemini_entry = float(response.text.split("Entry Point: ")[1].split("\n")[0])
                gemini_stop_loss = float(response.text.split("Stop Loss: ")[1].split("\n")[0])
                gemini_take_profit = float(response.text.split("Take Profit: ")[1].split("\n")[0])
                gemini_confidence = float(response.text.split("Confidence: ")[1].split("\n")[0])
                return {
                    "entry_point": gemini_entry,
                    "stop_loss": gemini_stop_loss,
                    "take_profit": gemini_take_profit,
                    "confidence": gemini_confidence,
                }
            except (IndexError, ValueError):
                logger.warning("Failed to parse global recommendation from Gemini. Returning defaults.")
                return {
                    "entry_point": 0.0,
                    "stop_loss": 0.0,
                    "take_profit": 0.0,
                    "confidence": 50.0,
                }

        except Exception as e:
            logger.error(f"Error calling Gemini API (global recommendation): {e}")
            return {
                "entry_point": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "confidence": 50.0,
            }
