import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from chronos import ChronosPipeline, ChronosBoltPipeline
import yfinance as yf
from modules.core import ensemble_forecasts

def get_stock_data(symbol, period="1y"):
    """Download stock data and format it for forecasting"""
    stock_data = yf.download(symbol, period=period, auto_adjust=True, progress=False)
    
    if len(stock_data) == 0:
        raise ValueError(f"No data available for {symbol}")
        
    # Format the data
    stock_data = stock_data.reset_index().rename(columns={"Date": "ds", "Close": "y"})
    stock_data = stock_data[["ds", "y"]].copy()
    
    print(f"Downloaded {len(stock_data)} days of data for {symbol}")
    return stock_data

def preprocess_time_series(data, scaling_method='robust'):
    """Preprocess time series data for better forecasting performance"""
    # Make a copy to avoid modifying the original
    processed_data = data.copy()
    
    # Handle outliers using IQR method
    q1 = processed_data['y'].quantile(0.25)
    q3 = processed_data['y'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # Store original values for diagnostics
    processed_data['original_y'] = processed_data['y'].copy()
    
    # Cap outliers
    outliers = (processed_data['y'] < lower_bound) | (processed_data['y'] > upper_bound)
    if outliers.any():
        # Cap outliers instead of removing them
        processed_data.loc[processed_data['y'] < lower_bound, 'y'] = lower_bound
        processed_data.loc[processed_data['y'] > upper_bound, 'y'] = upper_bound
        print(f"Capped {outliers.sum()} outliers in the data")
    
    # Apply scaling if requested
    if scaling_method:
        if scaling_method == 'robust':
            # Robust scaling using median and IQR
            median = processed_data['y'].median()
            iqr = np.percentile(processed_data['y'], 75) - np.percentile(processed_data['y'], 25)
            if iqr == 0:  # Handle constant series
                iqr = 1.0
            processed_data['y_scaled'] = (processed_data['y'] - median) / iqr
            processed_data['scaling_params'] = pd.Series([median, iqr])
    
    return processed_data

def test_chronos_model(stock_symbol="AAPL", prediction_days=30, use_ensemble=True):
    """Test the enhanced Chronos model with a sample stock"""
    print(f"\n=== Testing Chronos-Bolt forecasting on {stock_symbol} for {prediction_days} days ===\n")
    
    # Get stock data
    stock_data = get_stock_data(stock_symbol, period="2y")
    
    # Preprocess data
    processed_data = preprocess_time_series(stock_data)
    
    # Use scaled data if available
    if 'y_scaled' in processed_data.columns:
        forecast_input = processed_data['y_scaled'].values
        scaling_params = processed_data['scaling_params']
    else:
        forecast_input = processed_data['y'].values
        scaling_params = None
    
    # Set up device
    device = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
    precision = torch.float16 if device == "cuda" else torch.float32
    
    print(f"Using device: {device} with precision: {precision}")
    
    # Create models for ensemble
    if use_ensemble:
        # Model 1: Chronos-Bolt
        start_time = time.time()
        try:
            bolt_model = ChronosBoltPipeline.from_pretrained(
                "amazon/chronos-bolt-small",
                device_map=device,
                torch_dtype=precision,
            )
            print("Loaded Chronos-Bolt model")
            
            # Generate forecast
            bolt_start = time.time()
            bolt_quantiles, _ = bolt_model.predict_quantiles(
                context=torch.tensor(forecast_input),
                prediction_length=prediction_days,
                quantile_levels=[0.1, 0.5, 0.9]
            )
            bolt_time = time.time() - bolt_start
            
            # Create a dataframe with forecast results
            bolt_forecast = pd.DataFrame({
                'forecast_median': bolt_quantiles[0, :, 1].cpu().numpy(),
                'forecast_lower': bolt_quantiles[0, :, 0].cpu().numpy(),
                'forecast_high': bolt_quantiles[0, :, 2].cpu().numpy()
            })
            
            print(f"Chronos-Bolt inference took {bolt_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error loading Chronos-Bolt model: {str(e)}")
            use_ensemble = False
            bolt_forecast = None
            bolt_time = 0
        
        # Model 2: Standard Chronos
        try:
            standard_model = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-small",
                device_map=device,
                torch_dtype=precision,
            )
            print("Loaded standard Chronos model")
            
            # Generate forecast
            std_start = time.time()
            std_forecast_tensor = standard_model.predict(
                torch.tensor(forecast_input),
                prediction_length=prediction_days
            )
            std_time = time.time() - std_start
            
            # Create a dataframe with forecast results
            std_forecast = pd.DataFrame({
                'forecast_median': np.median(std_forecast_tensor[0].cpu().numpy(), axis=0),
                'forecast_lower': np.quantile(std_forecast_tensor[0].cpu().numpy(), 0.1, axis=0),
                'forecast_high': np.quantile(std_forecast_tensor[0].cpu().numpy(), 0.9, axis=0)
            })
            
            print(f"Standard Chronos inference took {std_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error loading standard Chronos model: {str(e)}")
            use_ensemble = False if bolt_forecast is None else True
            std_forecast = None
        
        # Ensemble forecasts if both models available
        if use_ensemble and bolt_forecast is not None and std_forecast is not None:
            # Use 70/30 weighting favoring Chronos-Bolt
            weights = [0.7, 0.3]
            ensemble_start = time.time()
            final_forecast = ensemble_forecasts([bolt_forecast, std_forecast], weights)
            ensemble_time = time.time() - ensemble_start
            print(f"Ensemble generation took {ensemble_time:.2f} seconds")
            total_time = time.time() - start_time
            print(f"Total processing time: {total_time:.2f} seconds")
        elif bolt_forecast is not None:
            final_forecast = bolt_forecast
            print("Using only Chronos-Bolt forecast (no ensemble)")
        elif std_forecast is not None:
            final_forecast = std_forecast
            print("Using only standard Chronos forecast (no ensemble)")
        else:
            raise ValueError("No forecasts were successfully generated")
    
    else:
        # Single model approach: use Chronos-Bolt only
        start_time = time.time()
        bolt_model = ChronosBoltPipeline.from_pretrained(
            "amazon/chronos-bolt-small",
            device_map=device,
            torch_dtype=precision,
        )
        
        # Generate forecast
        bolt_quantiles, _ = bolt_model.predict_quantiles(
            context=torch.tensor(forecast_input),
            prediction_length=prediction_days,
            quantile_levels=[0.1, 0.5, 0.9]
        )
        
        # Create a dataframe with forecast results
        final_forecast = pd.DataFrame({
            'forecast_median': bolt_quantiles[0, :, 1].cpu().numpy(),
            'forecast_lower': bolt_quantiles[0, :, 0].cpu().numpy(),
            'forecast_high': bolt_quantiles[0, :, 2].cpu().numpy()
        })
        
        total_time = time.time() - start_time
        print(f"Single model processing time: {total_time:.2f} seconds")
    
    # Rescale forecast if needed
    if scaling_params is not None:
        try:
            # Safely access scaling parameters
            if isinstance(scaling_params, pd.Series):
                if len(scaling_params) >= 2:
                    median = scaling_params.iloc[0]
                    iqr = scaling_params.iloc[1]
                else:
                    print("Warning: Incomplete scaling parameters")
                    median, iqr = None, None
            else:
                try:
                    median = scaling_params[0]
                    iqr = scaling_params[1]
                except (IndexError, TypeError):
                    print("Warning: Invalid scaling parameters")
                    median, iqr = None, None
            
            # Only rescale if we have valid parameters
            if median is not None and iqr is not None:
                # Reverse robust scaling
                final_forecast['forecast_median'] = final_forecast['forecast_median'] * iqr + median
                final_forecast['forecast_lower'] = final_forecast['forecast_lower'] * iqr + median
                final_forecast['forecast_high'] = final_forecast['forecast_high'] * iqr + median
        except Exception as e:
            print(f"Error applying scaling: {str(e)}")
    
    # Visualization
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    plt.plot(stock_data['ds'], stock_data['y'], label='Historical Data', color='blue')
    
    # Create forecast index dates
    last_date = stock_data['ds'].iloc[-1]
    forecast_dates = pd.date_range(start=last_date, periods=prediction_days + 1)[1:]
    
    # Plot forecast
    plt.plot(forecast_dates, final_forecast['forecast_median'], label='Median Forecast', color='red')
    plt.fill_between(
        forecast_dates,
        final_forecast['forecast_lower'],
        final_forecast['forecast_high'],
        color='red',
        alpha=0.2,
        label='80% Confidence Interval'
    )
    
    # Calculate percent change
    last_price = stock_data['y'].iloc[-1]
    forecast_price = final_forecast['forecast_median'].iloc[-1]
    percent_change = ((forecast_price - last_price) / last_price) * 100
    
    plt.title(f"{stock_symbol} Stock Price Forecast\nPredicted change: {percent_change:.2f}% over {prediction_days} days", fontsize=14)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f"{stock_symbol}_forecast.png")
    print(f"Forecast chart saved as {stock_symbol}_forecast.png")
    
    # Calculate metrics - assuming the last 30 days as "validation"
    validation_length = min(30, len(stock_data) - 30)
    if validation_length > 0:
        val_data = stock_data[-validation_length:].copy()
        # Generate a forecast for the validation period
        if 'y_scaled' in processed_data.columns:
            val_input = processed_data.iloc[:-validation_length]['y_scaled'].values
        else:
            val_input = processed_data.iloc[:-validation_length]['y'].values
        
        # Use Chronos-Bolt for validation forecast
        try:
            val_quantiles, _ = bolt_model.predict_quantiles(
                context=torch.tensor(val_input),
                prediction_length=validation_length,
                quantile_levels=[0.1, 0.5, 0.9]
            )
            
            val_forecast = pd.DataFrame({
                'forecast_median': val_quantiles[0, :, 1].cpu().numpy(),
                'forecast_lower': val_quantiles[0, :, 0].cpu().numpy(),
                'forecast_high': val_quantiles[0, :, 2].cpu().numpy()
            })
            
            # Rescale if needed
            if scaling_params is not None:
                try:
                    # Safely access scaling parameters
                    if isinstance(scaling_params, pd.Series):
                        if len(scaling_params) >= 2:
                            median = scaling_params.iloc[0]
                            iqr = scaling_params.iloc[1]
                        else:
                            print("Warning: Incomplete scaling parameters")
                            median, iqr = None, None
                    else:
                        try:
                            median = scaling_params[0]
                            iqr = scaling_params[1]
                        except (IndexError, TypeError):
                            print("Warning: Invalid scaling parameters")
                            median, iqr = None, None
                    
                    # Only rescale if we have valid parameters
                    if median is not None and iqr is not None:
                        val_forecast['forecast_median'] = val_forecast['forecast_median'] * iqr + median
                        val_forecast['forecast_lower'] = val_forecast['forecast_lower'] * iqr + median
                        val_forecast['forecast_high'] = val_forecast['forecast_high'] * iqr + median
                except Exception as e:
                    print(f"Error applying scaling: {str(e)}")
            
            # Calculate error metrics
            actual = val_data['y'].values
            predicted = val_forecast['forecast_median'].values
            
            mae = np.mean(np.abs(actual - predicted))
            rmse = np.sqrt(np.mean((actual - predicted)**2))
            mape = np.mean(np.abs((actual - predicted) / np.where(actual == 0, 1e-10, actual))) * 100
            
            # Direction accuracy
            actual_diff = np.diff(actual)
            predicted_diff = np.diff(predicted)
            direction_matches = np.sign(actual_diff) == np.sign(predicted_diff)
            direction_accuracy = np.mean(direction_matches)
            
            # R-squared
            ss_total = np.sum((actual - np.mean(actual)) ** 2)
            ss_residual = np.sum((actual - predicted) ** 2)
            r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
            
            print("\n=== Validation Metrics ===")
            print(f"MAE: {mae:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAPE: {mape:.2f}%")
            print(f"Direction Accuracy: {direction_accuracy*100:.2f}%")
            print(f"R²: {r2:.2f}")
            
        except Exception as e:
            print(f"Unable to calculate validation metrics: {str(e)}")
    
    return final_forecast, stock_data

if __name__ == "__main__":
    # Test with Apple stock
    test_chronos_model("AAPL", prediction_days=30, use_ensemble=True)
    
    # Test with Tesla stock
    test_chronos_model("TSLA", prediction_days=30, use_ensemble=True) 