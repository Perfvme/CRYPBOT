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

# Function to fetch data (placeholder implementation)
def fetch_data(symbol, interval):
    """Fetch and preprocess data for a given symbol and interval."""
    processor = DataProcessor()
    raw_data = processor.fetch_data(symbol, interval)
    return raw_data

# Track bot start time for uptime calculation
start_time = time.time()

# Function to calculate bot uptime
def get_uptime():
    """Calculate bot uptime."""
    uptime_seconds = time.time() - start_time
    days = int(uptime_seconds // (24 * 3600))
    hours = int((uptime_seconds % (24 * 3600)) // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    return f"{days}d {hours}h {minutes}m"

# Function to check API health
def check_api_health():
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
        gemini_client.analyze("Test text")  # Dummy call to check connectivity
        api_health["Gemini"] = "✅ (Ping Successful)"
    except Exception as e:
        api_health["Gemini"] = f"❌ (Error: {str(e)})"

    return api_health

# Function to get resource usage
def get_resource_usage():
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
def get_last_retraining_time():
    """Get the last retraining time from the file."""
    try:
        with open("last_retraining_time.txt", "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "Never"

# Function to evaluate model performance
def evaluate_model_performance(model):
    """Evaluate the ML model's accuracy."""
    try:
        processor = DataProcessor()
        data = processor.fetch_data("BTCUSDT", "1h")
        X, y = processor.preprocess_for_training(data)
        accuracy = model.evaluate(X, y) * 100
        return f"{accuracy:.1f}%"
    except Exception as e:
        return f"Error: {str(e)}"

# Function to check signals and send alerts
def check_and_send_alerts():
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
                df = alert_system.processor.preprocess_data(data)
                
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
                
                # Calculate risk parameters
                stop_loss = avg_support - avg_atr
                take_profit = avg_resistance + avg_atr
                
                # Prepare the alert message
                alert_message = (
                    f"🚨 ALERT: {asset}\n"
                    f"Majority Signal: {majority_signal}\n"
                    f"Timeframe Signals: {', '.join(signals)}\n"
                    f"Average DS Confidence: {avg_ds_confidence:.1f}%\n"
                    f"Entry Point: ${avg_support:.2f}\n"
                    f"Stop Loss: ${stop_loss:.2f}\n"
                    f"Take Profit: ${take_profit:.2f}\n"
                )
                
                # Send the alert to the specified Telegram chat ID
                chat_id = os.getenv("TELEGRAM_CHAT_ID")  # Replace with your chat ID in .env
                if chat_id:
                    bot.send_message(chat_id, alert_message)
                    logger.info(f"Alert sent for {asset}: {alert_message}")
                else:
                    logger.error("TELEGRAM_CHAT_ID is not set in environment variables.")
    except Exception as e:
        logger.error(f"Error during automatic alert check: {e}")

# Function to run scheduled tasks in the background
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)

# Schedule the retraining task (optional, if you have model retraining logic)
# schedule.every(24).hours.do(retrain_models)

# Schedule the alert task to run every 10 minutes
schedule.every(10).minutes.do(check_and_send_alerts)

# Start the scheduler in a separate thread
scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()

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
        # Extract the coin symbol from the command
        args = message.text.split()
        if len(args) < 2:
            bot.reply_to(message, "Usage: /signal <coin> (e.g., /signal BTCUSDT)")
            return
        symbol = args[1].upper()  # Ensure the symbol is uppercase
        timeframe_data = {
            "5m": fetch_data(symbol, "5m"),
            "15m": fetch_data(symbol, "15m"),
            "1h": fetch_data(symbol, "1h"),
            "4h": fetch_data(symbol, "4h"),
        }
        # Generate signals for all timeframes
        responses = []
        signals = []
        support_levels = []
        resistance_levels = []
        atr_values = []
        for tf, data in timeframe_data.items():
            df = alert_system.processor.preprocess_data(data)
            signal, reason = alert_system.engine.generate_signal(df)
            ml_confidence = alert_system.calculate_ml_confidence(df)
            ds_confidence = alert_system.calculate_ds_confidence(df)
            combined_confidence = (ml_confidence + ds_confidence) / 2
            # Collect support/resistance and ATR values
            support, resistance = alert_system.engine.calculate_support_resistance(df)
            atr = df['atr'].iloc[-1]
            support_levels.append(support)
            resistance_levels.append(resistance)
            atr_values.append(atr)
            responses.append(
                f"│ {tf.upper()} │ {signal} │ ML: {ml_confidence}% | DS: {ds_confidence}% │"
            )
            signals.append(signal)
        # Determine the recommended action
        if "BUY" in signals:
            recommended_action = "BUY"
        elif "SELL" in signals:
            recommended_action = "SELL"
        else:
            recommended_action = "HOLD"
        # Calculate risk parameters
        avg_support = np.mean(support_levels)
        avg_resistance = np.mean(resistance_levels)
        avg_atr = np.mean(atr_values)
        stop_loss = avg_support - avg_atr
        take_profit = avg_resistance + avg_atr
        risk_reward_ratio = (take_profit - avg_support) / (avg_support - stop_loss)
        position_size = min(0.02 * 100000 / (avg_support - stop_loss), 100000)  # Example portfolio size: $100,000
        # Combine all responses into a single message
        response = (
            f"🔍 {symbol} Analysis [{pd.Timestamp.now().strftime('%H:%M UTC')}]\n"
            f"Timeframe Signals:\n" +
            "\n".join(responses) +
            f"\n📊 Combined Confidence: {np.mean([alert_system.calculate_combined_confidence(alert_system.processor.preprocess_data(data)) for data in timeframe_data.values()]):.1f}%\n"
            f"🎯 Recommended Action: {recommended_action}\n"
            f"💡 Entry Point: ${avg_support:.2f} (wait for retest)\n"
            f"⚠️ Risk Parameters:\n"
            f"   - Stop Loss: ${stop_loss:.2f}\n"
            f"   - Take Profit: ${take_profit:.2f}\n"
            f"   - R/R Ratio: {risk_reward_ratio:.2f}\n"
            f"   - Position Size: ${position_size:.2f}"
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
        model_accuracy = evaluate_model_performance(alert_system.ml_model)

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
    try:
        bot.infinity_polling()
    except Exception as e:
        logger.error(f"Critical error while running the bot: {e}")