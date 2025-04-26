import sys
import os

# Add the current directory to the path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.forecasting import get_stock_forecast

def test_forecast():
    print("Testing stock forecast with AAPL...")
    try:
        forecast, historical_data, metrics = get_stock_forecast("AAPL", prediction_days=10)
        print("✅ Forecast completed successfully!")
        print(f"Forecast length: {len(forecast)}")
        print(f"Historical data length: {len(historical_data)}")
        print(f"Metrics: {list(metrics.keys())}")
        return True
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_forecast()
    sys.exit(0 if success else 1) 