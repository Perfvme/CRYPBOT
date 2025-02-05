import os
from binance.client import Client

class BinanceClient:
    def __init__(self):
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        self.client = Client(api_key, api_secret)

    def get_klines(self, symbol, interval, limit=500):
        """Fetch klines data."""
        return self.client.get_klines(symbol=symbol, interval=interval, limit=limit)