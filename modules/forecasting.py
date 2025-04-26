import os
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import concurrent.futures
from typing import Dict, List, Tuple, Union, Optional

from modules.core import determine_device, get_chronos_model, calculate_forecast_metrics, ensemble_forecasts
from modules.data_fetcher import get_stock_data, try_alternative_vn_symbols
from modules.technical_indicators import calculate_technical_indicators

# Models to use in the ensemble
MODEL_CONFIGS = [
    {"use_bolt": True, "model_size": "small", "weight": 0.7},  # Chronos-Bolt Small (primary)
    {"use_bolt": False, "model_size": "small", "weight": 0.3}  # Standard Chronos (secondary)
]

def preprocess_time_series(data: pd.DataFrame, 
                           scaling_method: str = 'robust',
                           handle_outliers: bool = True,
                           interpolate_missing: bool = True) -> pd.DataFrame:
    """
    Preprocess time series data for better forecasting performance
    
    Args:
        data: DataFrame with time series data
        scaling_method: Method to scale the data ('robust', 'standard', or None)
        handle_outliers: Whether to detect and handle outliers
        interpolate_missing: Whether to interpolate missing values
    
    Returns:
        Preprocessed DataFrame
    """
    if data is None or len(data) < 2:
        return data
    
    # Make a copy to avoid modifying the original
    processed_data = data.copy()
    
    # Handle missing values
    if interpolate_missing and processed_data['y'].isnull().any():
        processed_data['y'] = processed_data['y'].interpolate(method='time')
        # Fill any remaining NAs (at the edges)
        processed_data['y'] = processed_data['y'].fillna(method='bfill').fillna(method='ffill')
    
    # Handle outliers using IQR method
    if handle_outliers:
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
        # Store original values if not already done
        if 'original_y' not in processed_data.columns:
            processed_data['original_y'] = processed_data['y'].copy()
            
        if scaling_method == 'robust':
            # Robust scaling using median and IQR
            median = processed_data['y'].median()
            iqr = np.percentile(processed_data['y'], 75) - np.percentile(processed_data['y'], 25)
            if iqr == 0:  # Handle constant series
                iqr = 1.0
            processed_data['y_scaled'] = (processed_data['y'] - median) / iqr
            processed_data['scaling_params'] = pd.Series([median, iqr])
            
        elif scaling_method == 'standard':
            # Standard scaling using mean and std
            mean = processed_data['y'].mean()
            std = processed_data['y'].std()
            if std == 0:  # Handle constant series
                std = 1.0
            processed_data['y_scaled'] = (processed_data['y'] - mean) / std
            processed_data['scaling_params'] = pd.Series([mean, std])
    
    return processed_data

def postprocess_forecast(forecast: pd.DataFrame, 
                         processed_data: pd.DataFrame,
                         scaling_method: str = 'robust') -> pd.DataFrame:
    """
    Postprocess the forecast to reverse preprocessing transformations
    
    Args:
        forecast: DataFrame with forecast data
        processed_data: DataFrame with processed data containing scaling parameters
        scaling_method: Method used for scaling
    
    Returns:
        Postprocessed forecast DataFrame
    """
    if scaling_method is None or 'scaling_params' not in processed_data:
        return forecast
    
    # Make a copy to avoid modifying the original
    result = forecast.copy()
    
    # Get scaling parameters
    if scaling_method == 'robust':
        scaling_params = processed_data['scaling_params']
        # Safely access the values without unpacking
        if isinstance(scaling_params, pd.Series):
            if len(scaling_params) >= 2:
                median = scaling_params.iloc[0]
                iqr = scaling_params.iloc[1]
            else:
                # Handle case with insufficient values
                print("Warning: Incomplete scaling parameters for robust scaling")
                return forecast
        else:
            # Handle case where scaling_params is not a Series
            try:
                median = scaling_params[0]
                iqr = scaling_params[1]
            except (IndexError, TypeError):
                print("Warning: Invalid scaling parameters for robust scaling")
                return forecast
        
        # Reverse robust scaling
        result['forecast_median'] = result['forecast_median'] * iqr + median
        result['forecast_lower'] = result['forecast_lower'] * iqr + median
        result['forecast_high'] = result['forecast_high'] * iqr + median
        
    elif scaling_method == 'standard':
        scaling_params = processed_data['scaling_params']
        # Safely access the values without unpacking
        if isinstance(scaling_params, pd.Series):
            if len(scaling_params) >= 2:
                mean = scaling_params.iloc[0]
                std = scaling_params.iloc[1]
            else:
                # Handle case with insufficient values
                print("Warning: Incomplete scaling parameters for standard scaling")
                return forecast
        else:
            # Handle case where scaling_params is not a Series
            try:
                mean = scaling_params[0]
                std = scaling_params[1]
            except (IndexError, TypeError):
                print("Warning: Invalid scaling parameters for standard scaling")
                return forecast
        
        # Reverse standard scaling
        result['forecast_median'] = result['forecast_median'] * std + mean
        result['forecast_lower'] = result['forecast_lower'] * std + mean
        result['forecast_high'] = result['forecast_high'] * std + mean
    
    return result

def get_stock_forecast(stock_symbol: str, prediction_days: int = 10, 
                       use_ensemble: bool = True, 
                       scaling_method: str = 'robust',
                       handle_outliers: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Get stock forecast for a specific symbol with enhanced accuracy
    
    Args:
        stock_symbol: Stock symbol to forecast
        prediction_days: Number of days to forecast
        use_ensemble: Whether to use model ensembling for better accuracy
        scaling_method: Method to scale the data ('robust', 'standard', or None)
        handle_outliers: Whether to detect and handle outliers
    
    Returns:
        Tuple of (forecast DataFrame, historical DataFrame, metrics dictionary)
    """
    print(f"Getting forecast for {stock_symbol} for {prediction_days} days")
    
    # Set an environment variable that other functions can check
    os.environ["CURRENT_FORECAST_SYMBOL"] = stock_symbol
    
    # Check if we should use all available historical data
    use_all_data = os.environ.get("USE_ALL_HISTORICAL_DATA", "false").lower() == "true"
    
    try:
        # Get historical data based on user preference
        if use_all_data:
            # Fetch maximum available historical data (5 years)
            historical_data = get_stock_data(stock_symbol, period="5y")
            print(f"Using all available historical data for {stock_symbol}")
        else:
            # Default behavior: fetch 1 year of data
            historical_data = get_stock_data(stock_symbol)
            print(f"Using 1 year of historical data for {stock_symbol}")
        
        # If data is too small, try to fetch more regardless of user preference
        if len(historical_data) < 60 and not use_all_data:
            try:
                print(f"Initial data set too small, fetching more data for {stock_symbol}")
                historical_data = get_stock_data(stock_symbol, period="5y")
            except:
                pass
        
        if historical_data is None or historical_data.empty or len(historical_data) < 20:
            raise ValueError(f"Insufficient data for {stock_symbol}")
            
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        
        # Try alternative VN symbols as a last resort
        vn_result = try_alternative_vn_symbols(stock_symbol)
        
        if vn_result is not None:
            historical_data = vn_result
        else:
            # All attempts failed
            raise ValueError(f"Could not retrieve data for {stock_symbol}")
    
    # Error if still no data
    if historical_data is None or historical_data.empty or len(historical_data) < 10:
        raise ValueError(f"Insufficient data points for {stock_symbol}. Need at least 10 data points for forecasting.")
    
    # Preprocess the data
    processed_data = preprocess_time_series(
        historical_data, 
        scaling_method=scaling_method,
        handle_outliers=handle_outliers
    )
    
    # Use scaled data if available, otherwise use original
    if 'y_scaled' in processed_data.columns:
        forecast_input_data = processed_data.copy()
        forecast_input_data['y'] = forecast_input_data['y_scaled']
    else:
        forecast_input_data = processed_data
    
    # Make a copy to avoid modifying the original DataFrame
    historical_copy = forecast_input_data.copy()
    
    # Set validation days based on data availability
    total_days = len(historical_copy)
    
    if total_days <= 30:
        validation_days = 0
    elif total_days <= 60:
        validation_days = 5
    elif total_days <= 100:
        validation_days = 10
    elif total_days <= 200:
        validation_days = 15
    else:
        validation_days = 30
    
    print(f"Using {validation_days} days for validation out of {total_days} total data points")
    
    # Determine the appropriate device for model inference
    device_map, precision = determine_device()
    
    # Run forecasting with single model or ensemble
    if use_ensemble and len(MODEL_CONFIGS) > 1:
        forecasts = []
        weights = []
        
        # Generate forecasts from each model configuration
        for config in MODEL_CONFIGS:
            model = get_chronos_model(
                device_map=device_map, 
                model_precision=precision, 
                use_bolt=config["use_bolt"], 
                model_size=config["model_size"]
            )
            
            # Create a validation set if we have enough data
            if validation_days > 0:
                # Split into training and validation
                train_data = historical_copy.iloc[:-validation_days].copy()
                validation_data = historical_copy.iloc[-validation_days:].copy()
                
                # Generate forecast for validation period using the chosen method
                if hasattr(model, 'predict_quantiles'):
                    # For Chronos-Bolt models
                    quantiles, _ = model.predict_quantiles(
                        context=torch.tensor(train_data['y'].values),
                        prediction_length=validation_days,
                        quantile_levels=[0.1, 0.5, 0.9]
                    )
                    
                    val_forecast = pd.DataFrame({
                        'forecast_median': quantiles[0, :, 1].numpy(),
                        'forecast_lower': quantiles[0, :, 0].numpy(),
                        'forecast_high': quantiles[0, :, 2].numpy()
                    })
                else:
                    # For standard Chronos models
                    context = torch.tensor(train_data['y'].values)
                    val_forecast_tensor = model.predict(context, prediction_length=validation_days)
                    
                    # Take quantiles from samples
                    val_forecast = pd.DataFrame({
                        'forecast_median': np.median(val_forecast_tensor[0].numpy(), axis=0),
                        'forecast_lower': np.quantile(val_forecast_tensor[0].numpy(), 0.1, axis=0),
                        'forecast_high': np.quantile(val_forecast_tensor[0].numpy(), 0.9, axis=0)
                    })
                
                # Calculate validation metrics
                validation_metrics = calculate_forecast_metrics(
                    validation_data['y'].values, 
                    val_forecast['forecast_median'].values
                )
                
                # Adjust weight based on validation performance - prefer models with better R²
                adjusted_weight = config["weight"] * (1 + validation_metrics.get("r2", 0))
            else:
                # Not enough data for validation, use default weight
                validation_metrics = {}
                adjusted_weight = config["weight"]
            
            # Generate actual forecast
            if hasattr(model, 'predict_quantiles'):
                # For Chronos-Bolt models
                quantiles, _ = model.predict_quantiles(
                    context=torch.tensor(historical_copy['y'].values),
                    prediction_length=prediction_days,
                    quantile_levels=[0.1, 0.5, 0.9]
                )
                
                model_forecast = pd.DataFrame({
                    'forecast_median': quantiles[0, :, 1].numpy(),
                    'forecast_lower': quantiles[0, :, 0].numpy(),
                    'forecast_high': quantiles[0, :, 2].numpy()
                })
            else:
                # For standard Chronos models
                context = torch.tensor(historical_copy['y'].values)
                forecast_tensor = model.predict(context, prediction_length=prediction_days)
                
                # Create a dataframe with forecast results
                model_forecast = pd.DataFrame({
                    'forecast_median': np.median(forecast_tensor[0].numpy(), axis=0),
                    'forecast_lower': np.quantile(forecast_tensor[0].numpy(), 0.1, axis=0),
                    'forecast_high': np.quantile(forecast_tensor[0].numpy(), 0.9, axis=0)
                })
            
            # Add to ensemble
            forecasts.append(model_forecast)
            weights.append(adjusted_weight)
        
        # Combine forecasts using weighted ensemble
        forecast = ensemble_forecasts(forecasts, weights)
        validation_status = "completed" if validation_days > 0 else "skipped"
        
    else:
        # Single model approach
        # Get the Chronos model - default to Bolt if available
        model = get_chronos_model(device_map, precision)
        
        # Create a validation set if we have enough data
        if validation_days > 0:
            # Split into training and validation
            train_data = historical_copy.iloc[:-validation_days].copy()
            validation_data = historical_copy.iloc[-validation_days:].copy()
            
            # Generate forecast for validation period
            if hasattr(model, 'predict_quantiles'):
                # For Chronos-Bolt models
                quantiles, _ = model.predict_quantiles(
                    context=torch.tensor(train_data['y'].values),
                    prediction_length=validation_days,
                    quantile_levels=[0.1, 0.5, 0.9]
                )
                
                val_forecast = pd.DataFrame({
                    'forecast_median': quantiles[0, :, 1].numpy(),
                    'forecast_lower': quantiles[0, :, 0].numpy(),
                    'forecast_high': quantiles[0, :, 2].numpy()
                })
            else:
                # For standard Chronos models
                context = torch.tensor(train_data['y'].values)
                val_forecast_tensor = model.predict(context, prediction_length=validation_days)
                
                # Take quantiles from samples
                val_forecast = pd.DataFrame({
                    'forecast_median': np.median(val_forecast_tensor[0].numpy(), axis=0),
                    'forecast_lower': np.quantile(val_forecast_tensor[0].numpy(), 0.1, axis=0),
                    'forecast_high': np.quantile(val_forecast_tensor[0].numpy(), 0.9, axis=0)
                })
            
            # Calculate validation metrics
            validation_metrics = calculate_forecast_metrics(
                validation_data['y'].values, 
                val_forecast['forecast_median'].values
            )
            validation_status = "completed"
        else:
            # Not enough data for validation
            validation_metrics = {}
            validation_status = "skipped"
        
        # Generate actual forecast
        if hasattr(model, 'predict_quantiles'):
            # For Chronos-Bolt models
            quantiles, _ = model.predict_quantiles(
                context=torch.tensor(historical_copy['y'].values),
                prediction_length=prediction_days,
                quantile_levels=[0.1, 0.5, 0.9]
            )
            
            forecast = pd.DataFrame({
                'forecast_median': quantiles[0, :, 1].numpy(),
                'forecast_lower': quantiles[0, :, 0].numpy(),
                'forecast_high': quantiles[0, :, 2].numpy()
            })
        else:
            # For standard Chronos models
            context = torch.tensor(historical_copy['y'].values)
            forecast_tensor = model.predict(context, prediction_length=prediction_days)
            
            # Create a dataframe with forecast results
            forecast = pd.DataFrame({
                'forecast_median': np.median(forecast_tensor[0].numpy(), axis=0),
                'forecast_lower': np.quantile(forecast_tensor[0].numpy(), 0.1, axis=0),
                'forecast_high': np.quantile(forecast_tensor[0].numpy(), 0.9, axis=0)
            })
    
    # Reverse any scaling transformations
    forecast = postprocess_forecast(forecast, processed_data, scaling_method)
    
    # Calculate volatility in forecast
    forecast_volatility = np.std(np.diff(forecast['forecast_median'].values) / forecast['forecast_median'].values[:-1]) * 100 if len(forecast) > 1 else 0
    
    # Calculate technical indicators
    technical_indicators = calculate_technical_indicators(historical_data)
    
    # Return everything needed for display and analysis
    metrics = {
        'requested_symbol': stock_symbol,
        'actual_symbol': os.environ.get("ACTUAL_SYMBOL_USED", stock_symbol),
        'data_source': os.environ.get("DATA_SOURCE_USED", "unknown"),
        'prediction_days': prediction_days,
        'historical_days': len(historical_data),
        'forecast_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'validation_days': validation_days,
        'validation_status': validation_status,
        'forecast_volatility': forecast_volatility,
        'technical_indicators': technical_indicators,
        'model_info': "Chronos-Bolt Ensemble" if use_ensemble else "Chronos-Bolt",
        'preprocessing': {
            'scaling_method': scaling_method,
            'outliers_handled': handle_outliers
        }
    }
    
    # Add validation metrics if available
    if validation_metrics:
        metrics.update(validation_metrics)
    
    return forecast, historical_data, metrics

def generate_forecast_summary(forecast_data, historical_data, metrics, language="en"):
    """
    Generate a comprehensive natural language summary of forecast insights and trends.
    Supports both English and Vietnamese output based on language settings.
    """
    try:
        # Extract key data points
        last_historical_price = historical_data['y'].iloc[-1]
        median_forecast = forecast_data['forecast_median'].iloc[-1]
        lower_bound = forecast_data['forecast_lower'].iloc[-1]
        upper_bound = forecast_data['forecast_high'].iloc[-1]
        
        # Calculate percent change for forecast
        percent_change = ((median_forecast - last_historical_price) / last_historical_price) * 100
        
        # Determine forecast trend
        if percent_change > 3:
            forecast_trend = "upward"
        elif percent_change < -3:
            forecast_trend = "downward"
        else:
            forecast_trend = "sideways"
        
        # Assess confidence level based on multiple factors
        forecast_volatility = metrics.get('forecast_volatility', 0)
        validation_r2 = metrics.get('r2', 0)
        
        # Better confidence assessment using both volatility and R² score
        if validation_r2 > 0.7 and forecast_volatility < 1.5:
            confidence_level = "high"
        elif validation_r2 > 0.5 and forecast_volatility < 2.5:
            confidence_level = "moderate"
        else:
            confidence_level = "low"
        
        # Translate confidence level for Vietnamese
        if language == "vi":
            if confidence_level == "high":
                confidence_level = "cao"
            elif confidence_level == "moderate":
                confidence_level = "trung bình"
            else:
                confidence_level = "thấp"
        
        # Generate a basic summary
        if language == "en":
            summary = f"The forecast indicates a {forecast_trend} trend with a projected change of {percent_change:.1f}% "
            summary += f"over the next {metrics['prediction_days']} days. "
            
            # Add confidence information
            summary += f"Based on the model's {confidence_level} confidence level "
            summary += f"({metrics.get('direction_accuracy', 0)*100:.1f}% direction accuracy), "
            
            # Add price targets
            summary += f"the price is projected to be between {lower_bound:.2f} and {upper_bound:.2f}, "
            summary += f"with a median forecast of {median_forecast:.2f}."
            
            # Add model information if validation was done
            if metrics.get('validation_status') == 'completed':
                summary += f" The model achieved an R² of {metrics.get('r2', 0):.2f} "
                summary += f"and RMSE of {metrics.get('rmse', 0):.2f} during validation."
                
            # Add a note about the model used
            summary += f" This forecast was generated using the {metrics.get('model_info', 'Chronos')} model "
            summary += f"with {metrics.get('historical_days', 0)} days of historical data."
        else:  # Vietnamese
            trend_translation = "đi lên" if forecast_trend == "upward" else "đi xuống" if forecast_trend == "downward" else "đi ngang"
            summary = f"Dự báo cho thấy xu hướng {trend_translation} "
            summary += f"với mức thay đổi dự kiến là {percent_change:.1f}% "
            summary += f"trong {metrics['prediction_days']} ngày tới. "
            
            # Add confidence information
            summary += f"Dựa trên độ tin cậy {confidence_level} của mô hình "
            summary += f"(độ chính xác hướng đi {metrics.get('direction_accuracy', 0)*100:.1f}%), "
            
            # Add price targets
            summary += f"giá được dự báo sẽ nằm trong khoảng từ {lower_bound:.2f} đến {upper_bound:.2f}, "
            summary += f"với mức dự báo trung bình là {median_forecast:.2f}."
            
            # Add model information if validation was done
            if metrics.get('validation_status') == 'completed':
                summary += f" Mô hình đạt được R² là {metrics.get('r2', 0):.2f} "
                summary += f"và RMSE là {metrics.get('rmse', 0):.2f} trong quá trình xác thực."
                
            # Add a note about the model used
            summary += f" Dự báo này được tạo ra bằng mô hình {metrics.get('model_info', 'Chronos')} "
            summary += f"với {metrics.get('historical_days', 0)} ngày dữ liệu lịch sử."
            
        return summary
    except Exception as e:
        # Fallback simple summary
        if language == "en":
            return f"Forecast predicts a {'rise' if percent_change > 0 else 'fall'} of approximately {abs(percent_change):.1f}% over the next {metrics['prediction_days']} days."
        else:
            return f"Dự báo dự đoán {'tăng' if percent_change > 0 else 'giảm'} khoảng {abs(percent_change):.1f}% trong {metrics['prediction_days']} ngày tới."

def process_multiple_stocks(stock_symbols, prediction_days=10, use_ensemble=True):
    """Process multiple stocks in parallel for comparison"""
    results = {}
    
    # Check for empty input
    if not stock_symbols:
        return results
    
    # Define a worker function for parallel processing
    def process_stock(symbol):
        try:
            forecast, historical, metrics = get_stock_forecast(
                symbol, 
                prediction_days, 
                use_ensemble=use_ensemble
            )
            return symbol, (forecast, historical, metrics)
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            return symbol, None
    
    # Process stocks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(stock_symbols))) as executor:
        future_to_symbol = {executor.submit(process_stock, symbol): symbol for symbol in stock_symbols}
        
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                symbol, result = future.result()
                if result:
                    results[symbol] = result
            except Exception as e:
                print(f"Error with {symbol}: {str(e)}")
    
    return results 