import os
import google.generativeai as genai
import logging
import re
import json
import requests

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
            examples = """..."""  # Your examples (same as before)
            prompt = (
                f"You are a financial analysis model. Analyze the following market data for {symbol} ({strategy_name}) and provide a confidence level (0-100) for the overall trading signal.\n"
                f"{examples}\n"
                f"Input:\nSymbol: {symbol}, Strategy: {strategy_name}\n"
                f"OHLC Data:\n{ohlc_data}\n"
                f"Indicator Data:\n{indicator_data}\n"
                "Output:\nConfidence:"
            )

            response = self.model.generate_content(prompt)
            response_text = response.text  # Get text *before* any parsing
            logger.debug(f"Gemini API raw response (strategy confidence): {response_text}")

            try:
                # Try parsing as JSON first
                response_json = json.loads(response_text)
                ai_confidence = float(response_json.get("Confidence", 50.0))
                return ai_confidence
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                # If JSON parsing fails, try regex, and be more permissive
                logger.warning("Failed to parse as JSON. Trying regex...")
                match = re.search(r"Confidence:\s*(\d+)", response_text, re.IGNORECASE)  # More robust regex
                if match:
                    try:
                        ai_confidence = float(match.group(1))
                        return ai_confidence
                    except (ValueError, TypeError):
                        logger.warning("Failed to parse confidence from regex match. Returning 50.")
                        return 50.0
                else:
                    logger.warning("No 'Confidence:' value found using regex. Returning 50.")
                    return 50.0

        except Exception as e:
            logger.error(f"Error calling Gemini API (strategy confidence): {e}")
            return 50.0

    def analyze_global_recommendation(self, symbol, ohlc_data, indicator_data):
        """Analyze market data for a global recommendation. Returns a dict."""
        try:
            examples = """..."""  # Your examples
            prompt = (
                f"You are a financial analysis model. Analyze the following market data for {symbol} and provide a global trading recommendation.\n"
                "Follow these steps:\n"
                "1. Identify the overall trend (Bullish, Bearish, or Neutral)...\n"  # Rest of your prompt
                f"Input:\nSymbol: {symbol}\n"
                f"OHLC Data:\n{ohlc_data}\n"
                f"Indicator Data:\n{indicator_data}\n"
                "Output:\nReasoning:"
            )

            response = self.model.generate_content(prompt)
            response_text = response.text
            logger.debug(f"Gemini API raw response (global recommendation): {response_text}")

            try:
                # Try parsing as JSON first
                response_json = json.loads(response_text)
                return {
                    "entry_point": float(response_json.get("Entry Point", 0.0)),
                    "stop_loss": float(response_json.get("Stop Loss", 0.0)),
                    "take_profit": float(response_json.get("Take Profit", 0.0)),
                    "confidence": float(response_json.get("Confidence", 50.0)),
                }
            except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                # If JSON parsing fails, try regex, and be more permissive
                logger.warning("Failed to parse as JSON. Trying regex...")
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
                    logger.warning("Failed to parse global recommendation using regex. Returning defaults.")
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
