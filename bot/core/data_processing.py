import pandas as pd
from bot.api.binance_client import BinanceClient
import talib

class DataProcessor:
    def __init__(self):
        self.client = BinanceClient()

    def fetch_data(self, symbol, interval, limit=500):
        """Fetch historical klines data."""
        return self.client.get_klines(symbol, interval, limit)

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
        return df

    def preprocess_for_training(self, data):
        """
        Preprocess raw data into features (X) and labels (y) for ML training.
        Labels are binary: 1 if the price increases in the next time period, 0 otherwise.
        """
        df = self.preprocess_data(data)

        # Features: Technical indicators
        X = df[['rsi', 'macd', 'bb_upper', 'bb_lower', 'ema', 'atr', 'vwap']].dropna()

        # Labels: Future price movement (1 for increase, 0 for decrease)
        df['future_close'] = df['close'].shift(-1)  # Next time period's close price
        df['label'] = (df['future_close'] > df['close']).astype(int)  # 1 if price increases, else 0
        y = df['label'].iloc[:-1]  # Exclude the last row (no future price available)

        return X, y