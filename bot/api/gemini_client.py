import os
import google.generativeai as genai
import logging
import re
from typing import Dict, Union, List

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
        self.model = genai.GenerativeModel('gemini-2.0-flash') # Using gemini-pro

    def _parse_gemini_response(self, response_text: str, keys: List[str]) -> Dict[str, float]:
        """Parses a Gemini response, extracting values for specified keys.

        Args:
            response_text: The raw text response from Gemini.
            keys: A list of keys to extract (e.g., ["Confidence", "Entry Point"]).

        Returns:
            A dictionary where keys are the input keys (lowercase, spaces replaced
            with underscores) and values are the extracted float values (or default
            values if not found).
        """
        results = {}
        for key in keys:
            # More robust regex: case-insensitive, handles any chars before/after,
            # captures digits and .
            match = re.search(rf".*?{key}.*?:.*?([\d.]+)", response_text, re.IGNORECASE | re.DOTALL)
            try:
                if match:
                    results[key.lower().replace(" ", "_")] = float(match.group(1))  # Convert to float
                else:
                    logger.warning(f"Could not find key '{key}' in Gemini response.")
                    # Use a default value appropriate for the key
                    if key == "Confidence":
                        results[key.lower().replace(" ", "_")] = 50.0  # Default confidence
                    else:
                        results[key.lower().replace(" ", "_")] = 0.0  # Default for prices
            except (ValueError, TypeError) as e:
                logger.warning(f"Error parsing value for {key}: {e}. Using default.")
                if key == "Confidence":
                    results[key.lower().replace(" ", "_")] = 50.0
                else:
                    results[key.lower().replace(" ", "_")] = 0.0
        return results


    def analyze_sentiment(self, text: str) -> float:
        """Analyze text for sentiment and return a score between -1 and 1."""
        try:
            prompt = (
                "You are a financial sentiment analysis model. Analyze the provided market data and return ONLY a sentiment score between -1 (bearish) and +1 (bullish). "
                f"Market Data: {text}"
            )

            response = self.model.generate_content(prompt)
            logger.debug(f"Gemini API raw response (sentiment): {response.text}")
            return self._parse_gemini_response(response.text, ["sentiment_score"]).get("sentiment_score", 0.0) # Use helper and default

        except Exception as e:
            logger.error(f"Error calling Gemini API (sentiment): {e}")
            return 0.0

    def analyze_strategy_confidence(self, symbol: str, strategy_name: str, ohlc_data: str, indicator_data: str) -> float:
        """Analyze market data for strategy confidence (0-100).  Returns a float."""
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

            # Use the helper function for consistent parsing
            result = self._parse_gemini_response(response_text, ["Confidence"])
            return result.get("confidence", 50.0)  # Return the float value

        except Exception as e:
            logger.exception(f"Error calling Gemini API (strategy confidence): {e}")
            return 50.0

    def analyze_global_recommendation(self, symbol: str, ohlc_data: str, indicator_data: str) -> Dict[str, float]:
        """Analyze market data for a global recommendation. Returns a dict."""
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

            # Use the helper function for consistent parsing
            return self._parse_gemini_response(response_text, ["Entry Point", "Stop-Loss", "Take-Profit", "Confidence"])

        except Exception as e:
            logger.exception(f"Error calling Gemini API (global recommendation): {e}")
            return {
                "entry_point": 0.0,
                "stop_loss": 0.0,
                "take_profit": 0.0,
                "confidence": 50.0,
            }
