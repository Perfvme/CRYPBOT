import pandas as pd
from config.indicators import TechnicalIndicators

# Step 1: Simulate OHLCV data
def create_sample_data():
    """
    Creates a sample DataFrame with OHLCV data for testing.
    """
    data = {
        'timestamp': pd.date_range(start="2023-01-01", periods=100, freq='T'),  # 1-minute intervals
        'open': [100 + i * 0.1 for i in range(100)],  # Simulated open prices
        'high': [101 + i * 0.1 for i in range(100)],  # Simulated high prices
        'low': [99 + i * 0.1 for i in range(100)],    # Simulated low prices
        'close': [100.5 + i * 0.1 for i in range(100)],  # Simulated close prices
        'volume': [1000 + i * 10 for i in range(100)]   # Simulated volume
    }
    return pd.DataFrame(data)

# Step 2: Test indicator calculation
def test_indicator_calculation():
    """
    Tests the TechnicalIndicators.calculate_all method with sample data.
    """
    # Create sample data
    df = create_sample_data()
    print("Initial DataFrame:")
    print(df.head())

    # Initialize TechnicalIndicators and calculate all indicators
    indicators = TechnicalIndicators()
    df_with_indicators = indicators.calculate_all(df)

    # Print the final DataFrame
    print("\nFinal DataFrame Columns:", df_with_indicators.columns.tolist())
    print("\nFirst few rows of RSI:")
    print(df_with_indicators['rsi'].head())

    # Check if 'rsi' column exists
    if 'rsi' in df_with_indicators.columns:
        print("\n✅ RSI column successfully calculated!")
    else:
        print("\n❌ RSI column is missing!")

# Step 3: Run the test
if __name__ == "__main__":
    test_indicator_calculation()