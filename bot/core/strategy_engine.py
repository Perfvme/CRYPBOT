import numpy as np

class StrategyEngine:
    def calculate_support_resistance(self, df):
        """Calculate support and resistance zones."""
        support = df['low'].rolling(window=20).min().iloc[-1]
        resistance = df['high'].rolling(window=20).max().iloc[-1]
        return support, resistance

    def calculate_fibonacci_levels(self, df):
        """Calculate Fibonacci retracement levels."""
        high = df['high'].max()
        low = df['low'].min()
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
        elif last_row['close'] < fib_levels["61.8%"]:
            return "BUY", "Fibonacci Level"
        elif last_row['close'] > fib_levels["38.2%"]:
            return "SELL", "Fibonacci Level"
        return "HOLD", None