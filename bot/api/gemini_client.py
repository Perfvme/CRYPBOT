import os
import google.generativeai as genai
import logging
import re  # Import the regular expression library

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
                # Use regular expression to find a number (integer or decimal)
                match = re.search(r"[-+]?\d*\.?\d+", response.text)
                if match:
                    sentiment_score = float(match.group(0))
                    return sentiment_score
                else:
                    raise ValueError("No numeric sentiment score found.")
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
                # More flexible regex: Find "Confidence:" anywhere, then capture the number
                match = re.search(r"Confidence:.*?(\d+)", response.text, re.IGNORECASE | re.DOTALL)
                if match:
                    ai_confidence = float(match.group(1))
                    return ai_confidence
                else:
                    logger.warning(f"No 'Confidence:' value found in: {response.text}") # More specific logging
                    raise ValueError("No 'Confidence:' value found.")
            except (ValueError, AttributeError):
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
                # More flexible regex: Find each label anywhere, then capture the number
                entry_match = re.search(r"Entry Point:.*?([\d.]+)", response.text, re.IGNORECASE | re.DOTALL)
                stop_loss_match = re.search(r"Stop Loss:.*?([\d.]+)", response.text, re.IGNORECASE | re.DOTALL)
                take_profit_match = re.search(r"Take Profit:.*?([\d.]+)", response.text, re.IGNORECASE | re.DOTALL)
                confidence_match = re.search(r"Confidence:.*?(\d+)", response.text, re.IGNORECASE | re.DOTALL)

                if entry_match and stop_loss_match and take_profit_match and confidence_match:
                    gemini_entry = float(entry_match.group(1))
                    gemini_stop_loss = float(stop_loss_match.group(1))
                    gemini_take_profit = float(take_profit_match.group(1))
                    gemini_confidence = float(confidence_match.group(1))
                    return {
                        "entry_point": gemini_entry,
                        "stop_loss": gemini_stop_loss,
                        "take_profit": gemini_take_profit,
                        "confidence": gemini_confidence,
                    }
                else:
                    logger.warning(f"Could not find all values in: {response.text}") # More specific logging
                    raise ValueError("Could not find all required values in Gemini response.")

            except (ValueError, AttributeError):
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
