import os
import google.generativeai as genai
import logging
import re
import json
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
            response_text = response.text
            logger.debug(f"Gemini API raw response (strategy confidence): {response_text}")

            # --- Prioritize Regex, then JSON as fallback ---
            match = re.search(r".*Confidence:.*?(\d+)", response_text, re.IGNORECASE | re.DOTALL)
            if match:
                try:
                    ai_confidence = float(match.group(1))
                    return ai_confidence
                except (ValueError, TypeError):
                    logger.warning("Failed to parse confidence from regex match. Trying JSON...")
            else:
                logger.warning("No 'Confidence:' value found using regex. Trying JSON...")

            try:
                response_json = json.loads(response_text)
                ai_confidence = float(response_json.get("Confidence", 50.0))
                return ai_confidence
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.warning(f"Failed to parse AI confidence. Returning 50. Error: {e}")
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
                "1. Identify the overall trend (Bullish, Bearish, or Neutral).\n"
                "2. Identify key support and resistance levels.\n"
                "3. Based on the trend and levels, provide an ideal entry point, stop-loss, and take-profit.\n"
                "4. Provide a confidence level (0-100) for the recommendation.\n\n"
                f"{examples}\n"
                f"Input:\nSymbol: {symbol}\n"
                f"OHLC Data:\n{ohlc_data}\n"
                f"Indicator Data:\n{indicator_data}\n"
                "Output:\nReasoning:"  # Guide output
            )

            response = self.model.generate_content(prompt)
            response_text = response.text
            logger.debug(f"Gemini API raw response (global recommendation): {response_text}")

            # --- Prioritize Regex, then JSON as fallback ---
            try:
                entry_match = re.search(r"Entry Point:\s*([\d.]+)", response_text, re.IGNORECASE | re.DOTALL)
                stop_loss_match = re.search(r"Stop Loss:\s*([\d.]+)", response_text, re.IGNORECASE | re.DOTALL)
                take_profit_match = re.search(r"Take Profit:\s*([\d.]+)", response_text, re.IGNORECASE | re.DOTALL)
                confidence_match = re.search(r"Confidence:\s*(\d+)", response_text, re.IGNORECASE | re.DOTALL)

                entry_point = float(entry_match.group(1)) if entry_match else 0.0
                stop_loss = float(stop_loss_match.group(1)) if stop_loss_match else 0.0
                take_profit = float(take_profit_match.group(1)) if take_profit_match else 0.0
                confidence = float(confidence_match.group(1)) if confidence_match else 50.0

                return {
                    "entry_point": entry_point,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "confidence": confidence,
                }
            except (AttributeError, ValueError):
                logger.warning("Failed to parse global recommendation using regex. Trying JSON...")

            try:
                response_json = json.loads(response_text)
                return {
                    "entry_point": float(response_json.get("Entry Point", 0.0)),
                    "stop_loss": float(response_json.get("Stop Loss", 0.0)),
                    "take_profit": float(response_json.get("Take Profit", 0.0)),
                    "confidence": float(response_json.get("Confidence", 50.0)),
                }
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
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
