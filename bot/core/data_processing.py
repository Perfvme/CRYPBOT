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
        df = self._engineer_features(df)  # Call feature engineering

        # Labels: Future price movement (1 for increase, 0 for decrease)
        df['future_close'] = df['close'].shift(-1)  # Next time period's close price
        df['label'] = (df['future_close'] > df['close']).astype(int)  # 1 if price increases, else 0

        # Drop the last row (no future price available)
        df = df.iloc[:-1]

        # Extract features and labels
        X = df[self.get_feature_columns()]  # Use helper function
        y = df['label']

        # --- Train/Test Split ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False  # Important: shuffle=False
        )

        logger.debug(f"Features shape: {X_train.shape}, Labels shape: {y_train.shape}")
        return X_train, X_test, y_train, y_test

    def _engineer_features(self, df):
        """Helper function to engineer features. Keeps training and prediction consistent."""

        # Lagged Features
        for lag in [1, 2, 3, 5, 10]:
            df[f'rsi_lag{lag}'] = df['rsi'].shift(lag)
            df[f'macd_lag{lag}'] = df['macd'].shift(lag)
            df[f'close_lag{lag}'] = df['close'].shift(lag)

        # Rolling Statistics
        for window in [5, 10, 20, 50]:
            df[f'rsi_rollmean{window}'] = df['rsi'].rolling(window=window).mean()
            df[f'rsi_rollstd{window}'] = df['rsi'].rolling(window=window).std()
            df[f'close_rollmean{window}'] = df['close'].rolling(window=window).mean()
            df[f'close_rollstd{window}'] = df['close'].rolling(window=window).std()
            df[f'volume_rollmean{window}'] = df['volume'].rolling(window=window).mean()

        # Price-Based Features
        for period in [1, 2, 3, 5]:
            df[f'pct_change_{period}'] = df['close'].pct_change(periods=period)
            df[f'price_diff_{period}'] = df['close'].diff(periods=period)
        df['high_low_range'] = df['high'] - df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        df['high_open_ratio'] = df['high'] / df['open']
        df['low_open_ratio'] = df['low'] / df['open']

        # Indicator-Based Features
        df['rsi_ema_diff'] = df['rsi'] - df['ema']
        df['macd_hist_change'] = df['macd'].diff()
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_pct'] = (df['close'] - df['bb_lower']) / df['bb_width']  # Normalized distance from BB

        # Time-Based Features
        df['hour_of_day'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day

        # Interaction Features
        df['rsi_macd'] = df['rsi'] * df['macd']
        df['rsi_bb_pct'] = df['rsi'] * df['bb_pct']

        df.dropna(inplace=True)  # Drop NaNs here, AFTER all calculations
        return df

    def get_feature_columns(self):
        """Returns a list of *all* engineered feature column names."""
        # This is important:  It should reflect *all* columns created in _engineer_features
        #  *except* 'label' and 'future_close'.
        return [col for col in self._engineer_features(self.preprocess_data(self.fetch_data('BTCUSDT', '1h'))).columns
                if col not in ['label', 'future_close']]

    def get_prediction_features(self, df):
        """Preprocesses data and returns features for prediction (no labels, no split)."""
        df = self.preprocess_data(df)
        df = self._engineer_features(df)  # Apply feature engineering
        return df[self.get_feature_columns()]
