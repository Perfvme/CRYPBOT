import os
import pandas as pd
import talib
from bot.api.binance_client import BinanceClient
import logging
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.client = BinanceClient()

    def fetch_data(self, symbol, interval, limit=500):
        """Fetch historical klines data."""
        try:
            logger.info(f"Fetching data for {symbol} with interval {interval}...")
            return self.client.get_klines(symbol, interval, limit)
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return []

    def preprocess_data(self, data):
        """Preprocess raw data into a DataFrame with technical indicators."""
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)

        # Add technical indicators
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['macd'], df['macdsignal'], _ = talib.MACD(df['close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
        df['ema'] = talib.EMA(df['close'], timeperiod=50)
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

        return df # REMOVE df.dropna() from here

    def preprocess_for_training(self, data):
        """Preprocess for training, including feature engineering and train/test split."""
        df = self.preprocess_data(data)
        df = self._engineer_features(df) # Call feature engineering

        # Labels: Future price movement (1 for increase, 0 for decrease)
        df['future_close'] = df['close'].shift(-1)  # Next time period's close price
        df['label'] = (df['future_close'] > df['close']).astype(int)  # 1 if price increases, else 0

        # Drop the last row (no future price available)
        df = df.iloc[:-1]

        # Extract features and labels
        X = df[self.get_feature_columns()] # Use helper function
        y = df['label']

        # --- Train/Test Split ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False  # Important: shuffle=False
        )

        logger.debug(f"Features shape: {X_train.shape}, Labels shape: {y_train.shape}")
        return X_train, X_test, y_train, y_test

    def _engineer_features(self, df):
        """Helper function to engineer features.  Keeps training and prediction consistent."""
        # Lagged Features (example: lag RSI by 1 period)
        df['rsi_lag1'] = df['rsi'].shift(1)
        df['macd_lag1'] = df['macd'].shift(1)

        # Rolling Statistics (example: 5-period rolling mean of RSI)
        df['rsi_rollmean5'] = df['rsi'].rolling(window=5).mean()
        df['rsi_rollstd5'] = df['rsi'].rolling(window=5).std()

        # Price-Based Features
        df['pct_change'] = df['close'].pct_change()  # Percentage change
        df['price_diff'] = df['close'].diff()       # Price difference
        df['high_low_range'] = df['high'] - df['low'] #high-low range

        df.dropna(inplace=True) # Drop NaNs here, AFTER all calculations
        return df

    def get_feature_columns(self):
        """Returns a list of feature column names."""
        return ['rsi', 'macd', 'bb_upper', 'bb_lower', 'ema', 'atr', 'vwap',
                'rsi_lag1', 'macd_lag1', 'rsi_rollmean5', 'rsi_rollstd5',
                'pct_change', 'price_diff', 'high_low_range']

    def get_prediction_features(self, df):
        """Preprocesses data and returns features for prediction (no labels, no split)."""
        df = self.preprocess_data(df)
        df = self._engineer_features(df)  # Apply feature engineering
        return df[self.get_feature_columns()]
