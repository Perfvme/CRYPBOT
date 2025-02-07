import os
from google.generativeai import GenerativeModel, configure # Import specifics
import logging
import re
import json
import requests # Add import

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
        configure(api_key=self.api_key)  # Use configure directly
        self.model = GenerativeModel('gemini-2.0-flash-001')  # Use GenerativeModel

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
                match = re.search(r"[-+]?\d*\.?\d+", response.text)
                if match:
                    sentiment_score = float(match.group(0))
                    return sentiment_score
                else:
                    raise ValueError("No numeric sentiment score found.")
            except ValueError:
                logger.warning("Failed to parse sentiment score. Returning 0.")
                logger.debug(f"Raw response content: {response.text}")
                return 0
        except Exception as e:
            logger.error(f"Error calling Gemini API (sentiment): {e}")
            return 0

    def analyze_strategy_confidence(self, symbol, strategy_name, ohlc_data, indicator_data):
        """Analyze market data for strategy confidence (0-100).  Returns a float."""
        try:
            examples = """
Example 1:
Input:
Symbol: BTCUSDT, Strategy: Scalping
OHLC Data:
             open      high       low     close
2024-01-28  42000.0  42100.0  41900.0  42050.0
Indicator Data:
    volume        rsi       macd  macdsignal
  1500.0  55.2  12.5  8.2
Output:
Confidence: 68

Example 2:
Input:
Symbol: ETHUSDT, Strategy: Swing Trading
OHLC Data:
             open      high       low     close
2024-01-28  2200.0  2220.0  2180.0  2190.0
Indicator Data:
   volume        rsi      macd  macdsignal
  2500.0  42.8  -5.3  -2.1
Output:
Confidence: 35
"""
            prompt = (
                f"You are a financial analysis model. Analyze the following market data for {symbol} ({strategy_name}) and provide a confidence level (0-100) for the overall trading signal.\n"
                f"{examples}\n"
                f"Input:\nSymbol: {symbol}, Strategy: {strategy_name}\n"
                f"OHLC Data:\n{ohlc_data}\n"
                f"Indicator Data:\n{indicator_data}\n"
                "Output:\nConfidence:"
            )

            response = self.model.generate_content(prompt)
            logger.debug(f"Gemini API raw response (strategy confidence): {response.text}")

            # --- Parse as JSON, handle errors ---
            try:
                response_json = json.loads(response.text)
                ai_confidence = float(response_json.get("Confidence", 50.0)) # Get value, default to 50
                return ai_confidence
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.warning(f"Failed to parse AI confidence for {strategy_name}. Returning 50. Error: {e}")
                return 50.0

        except Exception as e:
            logger.error(f"Error calling Gemini API (strategy confidence): {e}")
            return 50.0

    def analyze_global_recommendation(self, symbol, ohlc_data, indicator_data):
        """Analyze market data for a global recommendation. Returns a dict."""
        try:
            examples = """
Example 1:
Input:
Symbol: BTCUSDT
OHLC Data:
             open      high       low     close
2024-01-28  42000.0  42100.0  41900.0  42050.0
Indicator Data:
    volume        rsi       macd  macdsignal
  1500.0  55.2  12.5  8.2
Output:
Reasoning: The price is above the EMA, and the RSI is in a neutral range. The MACD is positive, suggesting bullish momentum.
Trend: Slightly Bullish
Support: 41900
Resistance: 42100
Entry Point: 42060
Stop Loss: 41850
Take Profit: 42200
Confidence: 70

Example 2:
Input:
Symbol: ETHUSDT
OHLC Data:
             open      high       low     close
2024-01-28  2200.0  2220.0  2180.0  2190.0
Indicator Data:
   volume        rsi      macd  macdsignal
  2500.0  42.8  -5.3  -2.1
Output:
Reasoning: The price is below the EMA, and the RSI is approaching oversold. The MACD is negative.
Trend: Bearish
Support: 2180
Resistance: 2220
Entry Point: 2185
Stop Loss: 2225
Take Profit: 2150
Confidence: 60
"""
            prompt = (
                f"You are a financial analysis model. Analyze the following market data for {symbol} and provide a global trading recommendation.\n"
                "Follow these steps:\n"
                "1. Identify the overall trend (Bullish, Bearish, or Neutral) based on the OHLC data and indicators.\n"
                "2. Identify key support and resistance levels.\n"
                "3. Based on the trend and levels, provide an ideal entry point, stop-loss, and take-profit.\n"
                "4. Provide a confidence level (0-100) for the recommendation.\n\n"
                f"{examples}\n"
                f"Input:\nSymbol: {symbol}\n"
                f"OHLC Data:\n{ohlc_data}\n"
                f"Indicator Data:\n{indicator_data}\n"
                "Output:\nReasoning:"
            )

            response = self.model.generate_content(prompt)
            logger.debug(f"Gemini API raw response (global recommendation): {response.text}")

            # --- Parse as JSON, handle errors ---
            try:
                response_json = json.loads(response.text)
                # Use .get() with defaults for all keys
                return {
                    "entry_point": float(response_json.get("Entry Point", 0.0)),
                    "stop_loss": float(response_json.get("Stop Loss", 0.0)),
                    "take_profit": float(response_json.get("Take Profit", 0.0)),
                    "confidence": float(response_json.get("Confidence", 50.0)),
                }
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
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
