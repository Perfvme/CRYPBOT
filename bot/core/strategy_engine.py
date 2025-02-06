import numpy as np
import pandas as pd

class StrategyEngine:
    def calculate_support_resistance(self, df):
        """Calculate support and resistance zones."""
        # Use a shorter window for more dynamic support/resistance
        window = 20
        support = df['low'].rolling(window=window).min().iloc[-1]
        resistance = df['high'].rolling(window=window).max().iloc[-1]

        # Check if support/resistance are NaN (can happen with short windows)
        if pd.isna(support):
            support = df['low'].iloc[-window:].min()  # Fallback to min of last 'window' lows
        if pd.isna(resistance):
            resistance = df['high'].iloc[-window:].max() # Fallback to max of last 'window' highs

        return support, resistance

    def calculate_fibonacci_levels(self, df):
        """Calculate Fibonacci retracement levels."""
        # Use the same window as support/resistance for consistency
        window = 20
        high = df['high'].iloc[-window:].max()  # Use .iloc for index-based slicing
        low = df['low'].iloc[-window:].min()

        # Handle cases where high == low (avoid division by zero)
        if high == low:
            return {} # Return empty dict if no range

        diff = high - low
        levels = {
            "0%": high,
            "23.6%": high - 0.236 * diff,
            "38.2%": high - 0.382 * diff,
            "50%": high - 0.5 * diff,
            "61.8%": high - 0.618 * diff,
            "100%": low
        }
        return levels

    def generate_signal(self, df):
        """Generate buy/sell signals based on indicators and conditions."""
        last_row = df.iloc[-1]
        support, resistance = self.calculate_support_resistance(df)
        fib_levels = self.calculate_fibonacci_levels(df)

        # Entry conditions
        if last_row['close'] < support and last_row['rsi'] < 30:
            return "BUY", "Support Zone"
        elif last_row['close'] > resistance and last_row['rsi'] > 70:
            return "SELL", "Resistance Zone"
        elif last_row['close'] < fib_levels.get("61.8%", last_row['close']): # Use .get()
            return "BUY", "Fibonacci Level"
        elif last_row['close'] > fib_levels.get("38.2%", last_row['close']): # Use .get()
            return "SELL", "Fibonacci Level"
        return "HOLD", None
