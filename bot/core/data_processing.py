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
        """Preprocess raw data into a DataFrame."""
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)
        return df

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
        """Corrected feature engineering: Lagged RETURNS."""
        for lag in [1, 2, 3, 5, 10, 20]:
            df[f'return_{lag}'] = df['close'].pct_change(periods=lag) # Calculate percentage change
        df.dropna(inplace=True) # Drop NaN values AFTER calculations
        return df

    def get_feature_columns(self):
        """Returns a list of feature column names."""
        # Dynamically get feature columns, excluding 'label' and 'future_close'
        return [col for col in self._engineer_features(self.preprocess_data(self.fetch_data('BTCUSDT', '1h'))).columns
                if col not in ['label', 'future_close']]

    def get_prediction_features(self, df):
        """Preprocesses data and returns features for prediction."""
        df = self.preprocess_data(df)
        df = self._engineer_features(df)
        return df[self.get_feature_columns()]
