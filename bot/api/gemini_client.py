import os
import google.generativeai as genai
import logging
import re

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
        """Analyze market data for strategy confidence (0-100)."""
        try:
            # Few-shot examples (replace with your own, relevant examples)
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
                "Output:\nConfidence:"  # Guide the output format
            )


            response = self.model.generate_content(prompt)
            logger.debug(f"Gemini API raw response (strategy confidence): {response.text}")

            try:
                # More flexible regex: Handles "Confidence Level", optional "/", and extra text
                match = re.search(r"Confidence(?: Level)?:\s*([\d.]+)(?:/\d+)?.*", response.text, re.IGNORECASE | re.DOTALL)
                if match:
                    ai_confidence = float(match.group(1))
                    return ai_confidence
                else:
                    logger.warning(f"No 'Confidence:' value found in: {response.text}")
                    raise ValueError("No 'Confidence:' value found.")
            except (ValueError, AttributeError):
                logger.warning(f"Failed to parse AI confidence for {strategy_name}. Returning 50.")
                return 50.0

        except Exception as e:
            logger.error(f"Error calling Gemini API (strategy confidence): {e}")
            return 50.0

    def analyze_global_recommendation(self, symbol, ohlc_data, indicator_data):
        """Analyze market data for a global recommendation."""
        try:
            # Few-shot examples, chain-of-thought, and specific output format
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
                "Output:\nReasoning:" # Guide to start with reasoning
            )

            response = self.model.generate_content(prompt)
            logger.debug(f"Gemini API raw response (global recommendation): {response.text}")

            try:
                # More robust parsing, expecting a structured response
                entry_match = re.search(r"Entry Point:\s*([\d.]+)", response.text, re.IGNORECASE | re.DOTALL)
                stop_loss_match = re.search(r"Stop Loss:\s*([\d.]+)", response.text, re.IGNORECASE | re.DOTALL)
                take_profit_match = re.search(r"Take Profit:\s*([\d.]+)", response.text, re.IGNORECASE | re.DOTALL)
                confidence_match = re.search(r"Confidence:\s*(\d+)", response.text, re.IGNORECASE | re.DOTALL)

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
                    logger.warning(f"Could not find all values in: {response.text}")
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
