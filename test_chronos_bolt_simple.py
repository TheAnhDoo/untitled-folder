import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from chronos import ChronosBoltPipeline
import yfinance as yf

def main():
    """Test the Chronos-Bolt model with a simple example"""
    print("=== Testing Chronos-Bolt Forecasting ===")
    
    # Download stock data for Apple
    symbol = "AAPL"
    stock_data = yf.download(symbol, period="2y", auto_adjust=True, progress=False)
    
    if len(stock_data) == 0:
        print(f"No data available for {symbol}")
        return
        
    print(f"Downloaded {len(stock_data)} days of data for {symbol}")
    
    # Format the data
    stock_data = stock_data.reset_index().rename(columns={"Date": "ds", "Close": "y"})
    stock_data = stock_data[["ds", "y"]].copy()
    
    # Set up device
    device = "cpu"  # Use CPU for now
    precision = torch.float32
    
    print(f"Using device: {device} with precision: {precision}")
    
    # Prediction parameters
    prediction_days = 30
    
    # Load the Chronos-Bolt model
    try:
        print("Loading Chronos-Bolt model...")
        bolt_model = ChronosBoltPipeline.from_pretrained(
            "amazon/chronos-bolt-small",
            device_map=device,
            torch_dtype=precision,
        )
        print("Model loaded successfully")
        
        # Generate forecast
        print(f"Generating forecast for {prediction_days} days...")
        context = torch.tensor(stock_data['y'].values)
        quantiles, _ = bolt_model.predict_quantiles(
            context=context,
            prediction_length=prediction_days,
            quantile_levels=[0.1, 0.5, 0.9]
        )
        
        # Create a dataframe with forecast results
        forecast = pd.DataFrame({
            'forecast_median': quantiles[0, :, 1].numpy(),
            'forecast_lower': quantiles[0, :, 0].numpy(),
            'forecast_high': quantiles[0, :, 2].numpy()
        })
        
        print("Forecast generated successfully")
        
        # Visualization
        plt.figure(figsize=(12, 6))
        
        # Plot historical data
        plt.plot(stock_data['ds'].values, stock_data['y'].values, label='Historical Data', color='blue')
        
        # Create forecast index dates
        last_date = stock_data['ds'].iloc[-1]
        forecast_dates = pd.date_range(start=last_date, periods=prediction_days + 1)[1:]
        
        # Plot forecast
        plt.plot(forecast_dates, forecast['forecast_median'].values, label='Median Forecast', color='red')
        plt.fill_between(
            forecast_dates,
            forecast['forecast_lower'].values,
            forecast['forecast_high'].values,
            color='red',
            alpha=0.2,
            label='80% Confidence Interval'
        )
        
        # Calculate percent change
        last_price = float(stock_data['y'].iloc[-1])
        forecast_price = float(forecast['forecast_median'].iloc[-1])
        percent_change = ((forecast_price - last_price) / last_price) * 100
        
        plt.title(f"{symbol} Stock Price Forecast\nPredicted change: {percent_change:.2f}% over {prediction_days} days", fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Price", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(f"{symbol}_forecast.png")
        print(f"Forecast chart saved as {symbol}_forecast.png")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 