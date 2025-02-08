import os
from dotenv import load_dotenv
import telebot
import logging
import pandas as pd
import numpy as np
import time
import threading
import schedule
from bot.core.alert_system import AlertSystem
from bot.core.data_processing import DataProcessor
from bot.api.binance_client import BinanceClient
import psutil
from bot.api.gemini_client import GeminiClient
from bot.core.ml_models import MLModel
from bot.model_retraining import retrain_models
import argparse  # Import argparse
from typing import Dict, List, Tuple, Optional, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
logger.info(f"Loading .env file from: {dotenv_path}")
load_dotenv(dotenv_path)

# Initialize the Telegram bot
telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
if not telegram_token:
    logger.error("TELEGRAM_BOT_TOKEN is not set in the environment variables.")
    raise ValueError("TELEGRAM_BOT_TOKEN is not set in the environment variables.")
bot = telebot.TeleBot(telegram_token)
alert_system = AlertSystem()

# --- Correct model initialization with argparse ---
parser = argparse.ArgumentParser(description="Run the Telegram bot with a specific ML model.")
parser.add_argument('--model_path', type=str, default='models/logistic_regression_model.joblib',
                    help='Path to the trained ML model file.')
args = parser.parse_args()
ml_model = MLModel(model_path=args.model_path)  # Use args.model_path
# -------------------------------------------------


# Function to fetch data (placeholder implementation)
def fetch_data(symbol: str, interval: str) -> pd.DataFrame:
    """Fetch and preprocess data for a given symbol and interval."""
    processor = DataProcessor()
    raw_data = processor.fetch_data(symbol, interval)
    return raw_data

# Track bot start time for uptime calculation
start_time = time.time()

# Function to calculate bot uptime
def get_uptime() -> str:
    """Calculate bot uptime."""
    uptime_seconds = time.time() - start_time
    days = int(uptime_seconds // (24 * 3600))
    hours = int((uptime_seconds % (24 * 3600)) // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    return f"{days}d {hours}h {minutes}m"

# Function to check API health
def check_api_health() -> dict:
    """Check the health of external APIs."""
    api_health = {}
    try:
        # Check Binance API
        client = BinanceClient()
        client.client.ping()
        api_health["Binance"] = "✅ (Ping Successful)"
    except Exception as e:
        api_health["Binance"] = f"❌ (Error: {str(e)})"

    try:
        # Check Gemini API
        gemini_client = GeminiClient()
        gemini_client.analyze_sentiment("Test text")  # Dummy call
        api_health["Gemini"] = "✅ (Test Successful)"
    except Exception as e:
        api_health["Gemini"] = f"❌ (Error: {str(e)})"

    return api_health

# Function to get resource usage
def get_resource_usage() -> dict:
    """Get CPU, RAM, and storage usage."""
    cpu_usage = f"{psutil.cpu_percent()}%"
    ram = psutil.virtual_memory()
    ram_usage = f"{ram.used / (1024 ** 3):.1f}/{ram.total / (1024 ** 3):.1f}GB"
    disk = psutil.disk_usage('/')
    disk_usage = f"{disk.used / (1024 ** 3):.1f}/{disk.total / (1024 ** 3):.1f}GB"
    return {
        "CPU": cpu_usage,
        "RAM": ram_usage,
        "Storage": disk_usage,
    }

# Function to get last retraining time
def get_last_retraining_time() -> str:
    """Get the last retraining time from the file."""
    try:
        with open("last_retraining_time.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "Never"

# Function to evaluate model performance
def evaluate_model_performance(model: MLModel) -> str:
    """Evaluate the ML model's accuracy."""
    try:
        processor = DataProcessor()
        data = processor.fetch_data("BTCUSDT", "1h")
        X_train, X_test, y_train, y_test = processor.preprocess_for_training(data)
        # Ensure feature selection is applied consistently
        if model.selected_features is not None:
            X_test = X_test[model.selected_features]
        accuracy = model.evaluate(X_test, y_test) * 100
        return f"{accuracy:.1f}%"
    except Exception as e:
        return f"Error: {str(e)}"

# Function to get backtesting results (simplified for display)
def get_backtesting_results(model: MLModel) -> dict:
    """Gets backtesting results using the provided model."""
    try:
        processor = DataProcessor()
        data = processor.fetch_data("BTCUSDT", "1h", limit=1000)
        df = processor.preprocess_for_strategy(data)  # Use preprocess_for_strategy
        results = model.backtest(df, "BTCUSDT")
        return results
    except Exception as e:
        logger.error(f"Error during backtesting: {e}")
        return {}

# Function to check signals and send alerts
def check_and_send_alerts(model: MLModel):
    logger.info("Checking signals for automatic alerts...")
    try:
        # List of assets to monitor
        assets = ["BTCUSDT", "SOLUSDT", "ETHUSDT", "LTCUSDT"]
        timeframe_data = {"5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h"}

        for asset in assets:
            signals = []
            ds_confidences = []
            support_levels = []
            resistance_levels = []
            atr_values = []

            # Fetch and analyze data for each timeframe
            for tf, interval in timeframe_data.items():
                data = fetch_data(asset, interval)
                # Use preprocess_for_strategy here
                df = alert_system.processor.preprocess_for_strategy(data)

                # Generate signal and calculate DS confidence
                signal, _ = alert_system.engine.generate_signal(df)
                ds_confidence = alert_system.calculate_ds_confidence(df)

                # Collect support/resistance and ATR values
                support, resistance = alert_system.engine.calculate_support_resistance(df)
                atr = df['atr'].iloc[-1]

                signals.append(signal)
                ds_confidences.append(ds_confidence)
                support_levels.append(support)
                resistance_levels.append(resistance)
                atr_values.append(atr)

            # Calculate averages
            avg_support = np.mean(support_levels)
            avg_resistance = np.mean(resistance_levels)
            avg_atr = np.mean(atr_values)
            avg_ds_confidence = sum(ds_confidences) / len(ds_confidences)

            # Check if 3/4 timeframes agree and DS confidence is above 75%
            buy_count = signals.count("BUY")
            sell_count = signals.count("SELL")
            hold_count = signals.count("HOLD")

            if (buy_count >= 3 or sell_count >= 3 or hold_count >= 3) and avg_ds_confidence > 75:
                # Determine the majority signal
                majority_signal = "BUY" if buy_count >= 3 else "SELL" if sell_count >= 3 else "HOLD"

                # Calculate risk parameters based on the majority signal
                if majority_signal == "BUY":
                    entry_point = avg_support
                    stop_loss = avg_support - avg_atr
                    take_profit = avg_resistance + avg_atr
                elif majority_signal == "SELL":
                    entry_point = avg_resistance
                    stop_loss = avg_resistance + avg_atr
                    take_profit = avg_support - avg_atr
                else:  # Handle HOLD case explicitly
                    entry_point = (avg_support + avg_resistance) / 2
                    stop_loss = entry_point - avg_atr  # Example, adjust as needed
                    take_profit = entry_point + avg_atr # Example, adjust as needed

                # Prepare the alert message
                alert_message = (
                    f"🚨 ALERT: {asset}\n"
                    f"Majority Signal: {majority_signal}\n"
                    f"Timeframe Signals: {', '.join(signals)}\n"
                    f"Average DS Confidence: {avg_ds_confidence:.1f}%\n"
                    f"Entry Point: ${entry_point:.2f}\n"
                    f"Stop Loss: ${stop_loss:.2f}\n"
                    f"Take Profit: ${take_profit:.2f}\n"
                )

                # Send the alert to the specified Telegram chat ID
                chat_id = os.getenv("TELEGRAM_CHAT_ID")
                if chat_id:
                    bot.send_message(chat_id, alert_message)
                    logger.info(f"Alert sent for {asset}: {alert_message}")
                else:
                    logger.error("TELEGRAM_CHAT_ID is not set in environment variables.")

            # --- Automatic Signal Check (ML-based) ---
            data_1h = fetch_data(asset, "1h")
            df_1h = alert_system.processor.preprocess_for_strategy(data_1h) # Use preprocess_for_strategy
            auto_signal, ml_confidence = alert_system.generate_automatic_signal(df_1h, model) # Pass the model

            if auto_signal != "HOLD":
                alert_message = (
                    f"🤖 AUTOMATIC SIGNAL: {asset}\n"
                    f"Signal: {auto_signal}\n"
                    f"ML Confidence: {ml_confidence:.1f}%\n"
                )
                chat_id = os.getenv("TELEGRAM_CHAT_ID")
                if chat_id:
                    bot.send_message(chat_id, alert_message)
                    logger.info(f"Automatic signal sent for {asset}: {alert_message}")

    except Exception as e:
        logger.error(f"Error during automatic alert check: {e}")

# Function to run scheduled tasks in the background
def run_scheduler(model: MLModel):  # Pass model to scheduler functions
    while True:
        schedule.run_pending()
        time.sleep(1)

# Schedule the retraining task (optional, if you have model retraining logic)
schedule.every(24).hours.do(retrain_models)

# Schedule the alert task to run every 10 minutes
schedule.every(10).minutes.do(check_and_send_alerts, ml_model)  # Pass ml_model


# Command: /start
@bot.message_handler(commands=['start'])
def send_welcome(message):
    logger.info(f"Received message: /start from user {message.from_user.id}")
    bot.reply_to(message, "👋 Welcome to Crypto Signal Bot!\nUse /help for available commands.")

# Command: /help
@bot.message_handler(commands=['help'])
def send_help(message):
    logger.info(f"Received message: /help from user {message.from_user.id}")
    help_text = (
        "📚 Available Commands:\n"
        "- /start: Start the bot and get a welcome message.\n"
        "- /signal <coin>: Generate a trading signal for the specified coin (e.g., /signal BTCUSDT).\n"
        "- /ml_status: View the bot's system status and model performance.\n"
        "- /help: Display this help message."
    )
    bot.reply_to(message, help_text)

# Command: /signal <coin>
@bot.message_handler(commands=['signal'])
def send_signal(message):
    logger.info(f"Received message: /signal from user {message.from_user.id}")
    try:
        args = message.text.split()
        if len(args) < 2:
            bot.reply_to(message, "Usage: /signal <coin> (e.g., /signal BTCUSDT)")
            return
        symbol = args[1].upper()

        scalping_timeframes = {"5m": fetch_data(symbol, "5m"), "15m": fetch_data(symbol, "15m"), "1h": fetch_data(symbol, "1h")}
        swing_timeframes = {"1h": fetch_data(symbol, "1h"), "4h": fetch_data(symbol, "4h")}
        all_timeframes = {**scalping_timeframes, **swing_timeframes}

        def calculate_trade_parameters(timeframes: Dict[str, pd.DataFrame], strategy_name: str) -> Dict[str, Any]:
            signals = []
            support_levels = []
            resistance_levels = []
            atr_values = []
            close_prices = []
            dataframes = []

            for tf, data in timeframes.items():
                # Use preprocess_for_strategy here
                df = alert_system.processor.preprocess_for_strategy(data)
                dataframes.append(df)
                signal, _ = alert_system.engine.generate_signal(df)
                support, resistance = alert_system.engine.calculate_support_resistance(df)
                fib_levels = alert_system.engine.calculate_fibonacci_levels(df)
                atr = df['atr'].iloc[-1]
                close_prices.append(df['close'].iloc[-1])

                signals.append(signal)
                support_levels.append(support)
                resistance_levels.append(resistance)
                atr_values.append(atr)

            buy_count = signals.count("BUY")
            sell_count = signals.count("SELL")
            hold_count = signals.count("HOLD")
            majority_signal = "BUY" if buy_count > sell_count else "SELL" if sell_count > buy_count else "HOLD"

            avg_support = np.mean(support_levels)
            avg_resistance = np.mean(resistance_levels)
            avg_atr = np.mean(atr_values)
            avg_close = np.mean(close_prices)

            if majority_signal == "BUY":
                entry_point = avg_support
                stop_loss = avg_support - avg_atr
                take_profit = avg_resistance + avg_atr
            elif majority_signal == "SELL":
                entry_point = avg_resistance
                stop_loss = avg_resistance + avg_atr
                take_profit = avg_support - avg_atr
            else:
                entry_point = (avg_support + avg_resistance) / 2
                stop_loss = entry_point -  avg_atr  # Use avg_atr
                take_profit = entry_point +  avg_atr # Use avg_atr
            risk_reward_ratio = abs(take_profit - entry_point) / abs(entry_point - stop_loss) if abs(entry_point - stop_loss) >0 else 0

            # --- ML Confidence (using selected features) ---
            combined_df = dataframes[-1]  # Get the last DataFrame
            # Use get_prediction_features to ensure correct feature set, and pass the model
            ml_confidence = alert_system.calculate_ml_confidence(combined_df, ml_model)


            # --- GEMINI ANALYSIS (Consistent Data) ---
            ohlc_data = combined_df[['open', 'high', 'low', 'close']].tail(50).to_string()
            indicator_data = combined_df.drop(columns=['open', 'high', 'low', 'close', 'volume']).tail(50).to_string()

            # --- Corrected Gemini Confidence Parsing ---
            ai_confidence = alert_system.gemini_client.analyze_strategy_confidence(
                symbol, strategy_name, ohlc_data, indicator_data
            )

            return {
                "strategy": strategy_name,
                "signal": majority_signal,
                "entry_point": entry_point,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "risk_reward_ratio": risk_reward_ratio,
                "ml_confidence": ml_confidence,
                "ai_confidence": ai_confidence,  # Correctly using parsed value
            }

        def get_global_recommendation(all_timeframes: Dict[str, pd.DataFrame]) -> Dict[str, float]:
            dataframes = []
            for tf, data in all_timeframes.items():
                # Use preprocess_for_strategy here
                df = alert_system.processor.preprocess_for_strategy(data)
                dataframes.append(df)

            combined_df = pd.concat(dataframes).tail(50)
            ohlc_data = combined_df[['open', 'high', 'low', 'close']].to_string()
            indicator_data = combined_df.drop(columns=['open', 'high', 'low', 'close', 'volume']).to_string()

            recommendation = alert_system.gemini_client.analyze_global_recommendation(
                symbol, ohlc_data, indicator_data
            )
            return recommendation


        scalping_params = calculate_trade_parameters(scalping_timeframes, "Scalping")
        swing_params = calculate_trade_parameters(swing_timeframes, "Swing Trading")
        global_recommendation = get_global_recommendation(all_timeframes)

        response = (
            f"🔍 {symbol} Analysis [{pd.Timestamp.now().strftime('%H:%M UTC')}]\n"
            f"📊 Scalping Strategy:\n"
            f"│ Signal: {scalping_params['signal']}\n"
            f"│ Entry Point: ${scalping_params['entry_point']:.2f}\n"
            f"│ Stop Loss: ${scalping_params['stop_loss']:.2f}\n"
            f"│ Take Profit: ${scalping_params['take_profit']:.2f}\n"
            f"│ Risk-Reward Ratio: {scalping_params['risk_reward_ratio']:.2f}\n"
            f"│ ML Confidence: {scalping_params['ml_confidence']:.1f}%\n"
            f"│ AI Confidence: {scalping_params['ai_confidence']:.1f}%\n\n"  # Correctly using ai_confidence
            f"📈 Swing Trading Strategy:\n"
            f"│ Signal: {swing_params['signal']}\n"
            f"│ Entry Point: ${swing_params['entry_point']:.2f}\n"
            f"│ Stop Loss: ${swing_params['stop_loss']:.2f}\n"
            f"│ Take Profit: ${swing_params['take_profit']:.2f}\n"
            f"│ Risk-Reward Ratio: {swing_params['risk_reward_ratio']:.2f}\n"
            f"│ ML Confidence: {swing_params['ml_confidence']:.1f}%\n"
            f"│ AI Confidence: {swing_params['ai_confidence']:.1f}%\n\n"  # Correctly using ai_confidence
            f"🌍 Global Recommendation:\n"
            f"│ Entry Point: ${global_recommendation['entry_point']:.2f}\n"
            f"│ Stop Loss: ${global_recommendation['stop_loss']:.2f}\n"
            f"│ Take Profit: ${global_recommendation['take_profit']:.2f}\n"
            f"│ Confidence: {global_recommendation['confidence']:.1f}%\n"
        )
        bot.reply_to(message, response)
    except Exception as e:
        logger.error(f"Error processing /signal command: {e}")
        bot.reply_to(message, "An error occurred while generating the signal. Please try again later.")

# Command: /ml_status
@bot.message_handler(commands=['ml_status'])
def send_ml_status(message):
    logger.info(f"Received message: /ml_status from user {message.from_user.id}")
    try:
        # Fetch real-time metrics
        uptime = get_uptime()
        api_health = check_api_health()
        resource_usage = get_resource_usage()
        last_retraining_time = get_last_retraining_time()
        model_accuracy = evaluate_model_performance(ml_model) #pass the model
        # Get backtesting results
        backtest_results = get_backtesting_results(ml_model)

        # Format the response
        response = (
            "🖥️ System Status Report\n"
            f"▫️ Uptime: {uptime}\n"
            "▫️ API Health:\n"
        )
        for api, status in api_health.items():
            response += f"   - {api}: {status}\n"
        response += "▫️ Resource Usage:\n"
        for resource, usage in resource_usage.items():
            response += f"   - {resource}: {usage}\n"
        response += f"▫️ Last Model Retraining: {last_retraining_time}\n"
        response += f"▫️ Model Accuracy: {model_accuracy}\n"

        # Add backtesting results to the response
        if backtest_results:
            response += (
                "📊 Backtesting Results (BTCUSDT, 1h):\n"
                f"  - Final Capital: ${backtest_results['final_capital']:.2f}\n"
                f"  - Win Rate: {backtest_results['win_rate']:.2%}\n"
                f"  - Profit Factor: {backtest_results['profit_factor']:.2f}\n"
                f"  - Max Drawdown: {backtest_results['max_drawdown']:.2%}\n"
                f"  - Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}\n"
            )
        else:
            response += "▫️ Backtesting Results:  Not available\n"

        bot.reply_to(message, response)
    except Exception as e:
        logger.error(f"Error processing /ml_status command: {e}")
        bot.reply_to(message, "An error occurred while fetching system status.")

# Generic message handler (for debugging purposes)
@bot.message_handler(func=lambda message: True)
def echo_all(message):
    logger.info(f"Received message: {message.text} from user {message.from_user.id}")
    bot.reply_to(message, f"You said: {message.text}")

# Start polling
if __name__ == "__main__":
    logger.info("Starting bot...")
    parser = argparse.ArgumentParser(description="Run the Telegram bot with a specific ML model.")
    parser.add_argument('--model_path', type=str, default='models/logistic_regression_model.joblib',
                        help='Path to the trained ML model file.')
    args = parser.parse_args()
    ml_model = MLModel(model_path=args.model_path)  # Use args.model_path for loading the model
    # Pass ml_model to functions that need it
    scheduler_thread = threading.Thread(target=run_scheduler, args=(ml_model,), daemon=True)
    schedule.every(10).minutes.do(check_and_send_alerts, ml_model)  # Pass ml_model to check_and_send_alerts
    try:
        bot.infinity_polling()
    except Exception as e:
        logger.error(f"Critical error while running the bot: {e}")
        # Consider adding a retry mechanism here, or sending an alert to yourself
