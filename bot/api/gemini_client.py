import os
import google.generativeai as genai
import logging
import re
import pandas as pd

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
                # More flexible regex: Handles "Confidence Level", optional "/", and extra text *non-greedy*
                match = re.search(r"Confidence(?: Level)?:\s*([\d.]+)(?:/\d+)?.*?", response.text, re.IGNORECASE | re.DOTALL)
                if match:
                    ai_confidence = float(match.group(1))
                    print(f"DEBUG: Matched confidence string: {match.group(0)}")  # Add this line
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
        """Analyze market data for a global recommendation with multiple take-profit levels."""
        try:
            # Enhanced few-shot examples with detailed explanations and multiple TPs
            examples = """
Example 1:
Input:
Symbol: BTCUSDT
OHLC Data:
             open      high       low     close
2024-01-28  42000.0  42100.0  41900.0  42050.0
2024-01-29  42050.0  42200.0  42000.0  42150.0
2024-01-30  42150.0  42300.0  42100.0  42250.0
Indicator Data:
    volume        rsi       macd  macdsignal  bb_upper  bb_lower
  1500.0  55.2  12.5  8.2  42350.0  41850.0
  1600.0  58.1  13.8  9.1  42400.0  41900.0
  1700.0  60.5  15.2  10.3  42450.0  41950.0
Output:
Reasoning: The price is showing bullish momentum, breaking above the previous resistance.  The RSI is rising but not yet overbought.  The MACD is positive and increasing.  There's a potential order block around 42000.
Trend: Bullish
Support: 41900 (Strong Support), 42000 (Order Block)
Resistance: 42300 (Recent High), 42500 (Next Resistance)
Ideal Entry: 42000 (Near Order Block and Support)
Stop Loss: 41850 (Below Support)
Take Profit 1: 42300 (First Resistance)
Take Profit 2: 42500 (Second Resistance)
Take Profit 3: 42700 (Fibonacci Extension 1.618)
Confidence: 75

Example 2:
Input:
Symbol: ETHUSDT
OHLC Data:
             open      high       low     close
2024-01-28  2200.0  2220.0  2180.0  2190.0
2024-01-29  2190.0  2195.0  2170.0  2175.0
2024-01-30  2175.0  2180.0  2150.0  2160.0
Indicator Data:
   volume        rsi      macd  macdsignal  bb_upper  bb_lower
  2500.0  42.8  -5.3  -2.1  2230.0  2170.0
  2400.0  38.5  -6.8  -3.5  2220.0  2160.0
  2300.0  35.2  -8.1  -4.8  2210.0  2150.0
Output:
Reasoning: The price is in a downtrend, breaking below support. The RSI is low and decreasing. The MACD is negative and widening.  A fair value gap exists between 2180 and 2190.
Trend: Bearish
Support: 2150 (Recent Low), 2130 (Next Support)
Resistance: 2180 (Fair Value Gap), 2200 (Previous Support)
Ideal Entry: 2180 (Near Fair Value Gap and Resistance)
Stop Loss: 2205 (Above Resistance)
Take Profit 1: 2150 (First Support)
Take Profit 2: 2130 (Second Support)
Take Profit 3: 2100 (Fibonacci Extension 1.618)
Confidence: 65
"""
            prompt = (
                f"You are a financial analysis model. Analyze the following market data for {symbol} and provide a global trading recommendation.\n"
                "Follow these steps:\n"
                "1. Identify the overall trend (Bullish, Bearish, or Neutral) based on the OHLC data and indicators.\n"
                "2. Identify key support and resistance levels, including strong support/resistance, order blocks, and fair value gaps.\n"
                "   - **Strong Support/Resistance:**  Areas where the price has repeatedly bounced or been rejected.\n"
                "   - **Order Blocks:**  Significant price ranges where large institutional orders are likely placed.\n"
                "   - **Fair Value Gaps (FVG):**  Imbalances between buying and selling pressure, often seen as single large candles with little overlap from adjacent candles.\n"
                "3. Determine the **Ideal Entry Point** based on the identified support/resistance, order blocks, and fair value gaps.  Prioritize strong support/resistance for entry.\n"
                "4. Set a **Stop Loss** just below a strong support level (for buys) or above a strong resistance level (for sells).\n"
                "5. Define **multiple Take Profit levels** based on:\n"
                "   - **Take Profit 1:**  The nearest significant resistance (for buys) or support (for sells).\n"
                "   - **Take Profit 2:**  The next significant resistance/support level beyond TP1.\n"
                "   - **Take Profit 3:**  A Fibonacci extension level (e.g., 1.618) or a further significant resistance/support.\n"
                "6. Provide a confidence level (0-100) for the recommendation.\n\n"
                f"{examples}\n"
                f"Input:\nSymbol: {symbol}\n"
                f"OHLC Data:\n{ohlc_data}\n"
                f"Indicator Data:\n{indicator_data}\n"
                "Output:\nReasoning:"  # Guide to start with reasoning
            )

            response = self.model.generate_content(prompt)
            logger.debug(f"Gemini API raw response (global recommendation): {response.text}")

            try:
                # More robust parsing, expecting a structured response
                entry_match = re.search(r"Ideal Entry:\s*([\d.]+)", response.text, re.IGNORECASE | re.DOTALL)
                stop_loss_match = re.search(r"Stop Loss:\s*([\d.]+)", response.text, re.IGNORECASE | re.DOTALL)
                tp1_match = re.search(r"Take Profit 1:\s*([\d.]+)", response.text, re.IGNORECASE | re.DOTALL)
                tp2_match = re.search(r"Take Profit 2:\s*([\d.]+)", response.text, re.IGNORECASE | re.DOTALL)
                tp3_match = re.search(r"Take Profit 3:\s*([\d.]+)", response.text, re.IGNORECASE | re.DOTALL)
                confidence_match = re.search(r"Confidence:\s*(\d+)", response.text, re.IGNORECASE | re.DOTALL)

                if (entry_match and stop_loss_match and tp1_match and
                        tp2_match and tp3_match and confidence_match):
                    gemini_entry = float(entry_match.group(1))
                    gemini_stop_loss = float(stop_loss_match.group(1))
                    gemini_tp1 = float(tp1_match.group(1))
                    gemini_tp2 = float(tp2_match.group(1))
                    gemini_tp3 = float(tp3_match.group(1))
                    gemini_confidence = float(confidence_match.group(1))
                    return {
                        "entry_point": gemini_entry,
                        "stop_loss": gemini_stop_loss,
                        "take_profit_1": gemini_tp1,
                        "take_profit_2": gemini_tp2,
                        "take_profit_3": gemini_tp3,
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
                    "take_profit_1": 0.0,
                    "take_profit_2": 0.0,
                    "take_profit_3": 0.0,
                    "confidence": 50.0,
                }

        except Exception as e:
            logger.error(f"Error calling Gemini API (global recommendation): {e}")
            return {
                "entry_point": 0.0,
                "stop_loss": 0.0,
                "take_profit_1": 0.0,
                "take_profit_2": 0.0,
                "take_profit_3": 0.0,
                "confidence": 50.0,
            }
