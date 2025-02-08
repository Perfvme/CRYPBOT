import os
import google.generativeai as genai
import logging
import re
from typing import Dict, Union

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
        self.model = genai.GenerativeModel('gemini-2.0-flash-001')

    def analyze_sentiment(self, text: str) -> float:
        """Analyze text for sentiment and return a score between -1 and 1."""
        try:
            prompt = (
                "You are a financial sentiment analysis model. Analyze the provided market data and return ONLY a sentiment score between -1 (bearish) and +1 (bullish). "
                f"Market Data: {text}"
            )

            response = self.model.generate_content(prompt)
            logger.debug(f"Gemini API raw response (sentiment): {response.text}")

            try:
                match = re.search(r"[-+]?\d*\.?\d+", response.text)
                if match:
                    sentiment_score = float(match.group(0))
                    return sentiment_score
                else:
                    raise ValueError("No numeric sentiment score found.")
            except ValueError:
                logger.warning("Failed to parse sentiment score. Returning 0.")
                logger.debug(f"Raw response content: {response.text}")
                return 0.0
        except Exception as e:
            logger.error(f"Error calling Gemini API (sentiment): {e}")
            return 0.0

    def analyze_strategy_confidence(self, symbol: str, strategy_name: str, ohlc_data: str, indicator_data: str) -> float:
        """Analyze market data, return confidence (0-100). Robust parsing."""
        try:
            examples = """..."""  # Your examples
            prompt = (
                f"You are a financial analysis model. Analyze the following market data for {symbol} ({strategy_name}) and provide a confidence level (0-100) for the overall trading signal.\n"
                f"{examples}\n"
                f"Input:\nSymbol: {symbol}, Strategy: {strategy_name}\n"
                f"OHLC Data:\n{ohlc_data}\n"
                f"Indicator Data:\n{indicator_data}\n"
                "Output:\nConfidence:"
            )

            response = self.model.generate_content(prompt)
            response_text = response.text
            logger.debug(f"Gemini API raw response (strategy confidence): {response_text}")

            # --- Robust Regex Parsing ---
            match = re.search(r".*?Confidence.*?:.*?(\d+)", response_text, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    ai_confidence = float(match.group(1))
                    return ai_confidence
                except (ValueError, TypeError):
                    logger.warning("Failed to parse confidence. Returning 50.")
                    return 50.0
            else:
                logger.warning("No 'Confidence:' value found. Returning 50.")
                return 50.0

        except Exception as e:
            logger.exception(f"Error calling Gemini API (strategy confidence): {e}")
            return 50.0

    def analyze_global_recommendation(self, symbol: str, ohlc_data: str, indicator_data: str) -> Dict[str, float]:
        """Analyze market data, return recommendation dict. Robust parsing."""
        try:
            examples = """..."""  # Your examples
            prompt = (
                f"You are a financial analysis model. Analyze the following market data for {symbol} and provide a global trading recommendation, including reasoning.\n"
                "Follow these steps:\n"
                "1. Identify the overall trend (Bullish, Bearish, or Neutral).  Be concise.\n"
                "2. Identify key support and resistance levels. Be concise.\n"
                "3. Based on the trend and levels, provide an ideal entry point, stop-loss, and take-profit. Be concise.\n"
                "4. Provide a confidence level (0-100) for the recommendation. Be concise.\n\n"
                f"{examples}\n"
                f"Input:\nSymbol: {symbol}\n"
                f"OHLC Data:\n{ohlc_data}\n"
                f"Indicator Data:\n{indicator_data}\n"
                "Output:\nReasoning:"  # Guide output
            )

            response = self.model.generate_content(prompt)
            response_text = response.text
            logger.debug(f"Gemini API raw response (global recommendation): {response_text}")

            # --- Robust Regex Parsing (Handles variations in Gemini's output) ---
            try:
                # Extract all relevant values in one go
                match = re.search(
                    r".*Entry Point:.*?([\d.]+).*Stop Loss:.*?([\d.]+).*Take Profit:.*?([\d.]+).*Confidence:.*?(\d+)",
                    response_text,
                    re.IGNORECASE | re.DOTALL,
                )

                if match:
                    entry_point = float(match.group(1))
                    stop_loss = float(match.group(2))
                    take_profit = float(match.group(3))
                    confidence = float(match.group(4))

                    return {
                        "entry_point": entry_point,
                        "stop_loss": stop_loss,
                        "take_profit": take_profit,
                        "confidence": confidence,
                    }
                else:
                    logger.warning("Failed to parse global recommendation using regex. Returning defaults.")
                    return {
                        "entry_point": 0.0,
                        "stop_loss": 0.0,
                        "take_profit": 0.0,
                        "confidence": 50.0,
                    }

            except (AttributeError, ValueError, TypeError) as e:
                logger.warning(f"Failed to parse global recommendation. Returning defaults. Error: {e}")
                return {
                    "entry_point": 0.0,
                    "stop_loss": 0.0,
                    "take_profit": 0.0,
                    "confidence": 50.0,
                }

        except Exception as e:
            logger.exception(f"Error calling Gemini API (global recommendation): {e}")
            return {
                "entry_point": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "confidence": 50.0,
            }
