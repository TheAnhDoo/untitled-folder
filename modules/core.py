import torch
import os
import sys
import subprocess
from chronos import ChronosPipeline, ChronosBoltPipeline
import numpy as np

def check_ollama_installation():
    """Check if Ollama is installed and running"""
    try:
        # Try to run the ollama command to check version
        result = subprocess.run(
            ["ollama", "version"], 
            capture_output=True, 
            text=True, 
            timeout=2
        )
        
        if result.returncode == 0:
            version = result.stdout.strip()
            return f"Ollama detected: {version}"
        else:
            return "Ollama installed but not responding"
    except Exception as e:
        return "Ollama not detected"

def check_device_compatibility():
    """Check device compatibility and available hardware acceleration"""
    device_info = "Using CPU for inference"
    
    # Check for CUDA (NVIDIA GPU)
    if torch.cuda.is_available():
        device_info = f"Using GPU: {torch.cuda.get_device_name(0)}"
        
    # Check for MPS (Apple Metal)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_info = "Using Apple Metal Performance Shaders (MPS)"
        
    return device_info

def get_chronos_model(device_map="auto", model_precision=torch.float32, use_bolt=True, model_size="small"):
    """Get cached Chronos model or load a new one
    
    Args:
        device_map (str): Device to use for model inference ('auto', 'cuda', 'mps', 'cpu')
        model_precision: Precision to use for model inference
        use_bolt (bool): Whether to use the faster and more accurate Chronos-Bolt model
        model_size (str): Size of the model ('tiny', 'mini', 'small', 'base')
    
    Returns:
        A Chronos model pipeline
    """
    # Create a cache for the model
    if not hasattr(get_chronos_model, "_cache"):
        get_chronos_model._cache = {}
        
    cache_key = f"{device_map}_{str(model_precision)}_{use_bolt}_{model_size}"
    
    if cache_key not in get_chronos_model._cache:
        if use_bolt:
            print(f"Loading Chronos-Bolt model (size={model_size}) with device_map={device_map}, precision={model_precision}")
            try:
                # Try loading Chronos-Bolt model
                model_path = f"amazon/chronos-bolt-{model_size}"
                get_chronos_model._cache[cache_key] = ChronosBoltPipeline.from_pretrained(
                    model_path,
                    device_map=device_map,
                    torch_dtype=model_precision,
                )
            except Exception as e:
                print(f"Failed to load Chronos-Bolt model: {str(e)}. Falling back to standard Chronos model.")
                model_path = f"amazon/chronos-t5-{model_size}"
                get_chronos_model._cache[cache_key] = ChronosPipeline.from_pretrained(
                    model_path,
                    device_map=device_map,
                    torch_dtype=model_precision,
                )
        else:
            print(f"Loading standard Chronos model (size={model_size}) with device_map={device_map}, precision={model_precision}")
            model_path = f"amazon/chronos-t5-{model_size}"
            get_chronos_model._cache[cache_key] = ChronosPipeline.from_pretrained(
                model_path,
                device_map=device_map,
                torch_dtype=model_precision,
            )
    
    return get_chronos_model._cache[cache_key]

def determine_device():
    """Determine the appropriate device for model inference"""
    if torch.cuda.is_available():
        return "cuda", torch.float16  # Use half precision for CUDA
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps", torch.float32  # MPS requires float32
    else:
        return "cpu", torch.float32  # CPU works well with float32

def calculate_forecast_metrics(actual, forecast):
    """Calculate common forecast performance metrics"""
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    if len(actual) != len(forecast):
        raise ValueError("Actual and forecast arrays must have same length")
    
    # Mean Absolute Error
    mae = mean_absolute_error(actual, forecast)
    
    # Root Mean Square Error
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((actual - forecast) / np.where(actual == 0, 1e-10, actual))) * 100
    
    # Direction Accuracy
    actual_diff = np.diff(actual)
    forecast_diff = np.diff(forecast)
    direction_matches = np.sign(actual_diff) == np.sign(forecast_diff)
    direction_accuracy = np.mean(direction_matches)
    
    # Calculate R-squared
    ss_total = np.sum((actual - np.mean(actual)) ** 2)
    ss_residual = np.sum((actual - forecast) ** 2)
    r2 = 1 - (ss_residual / ss_total) if ss_total != 0 else 0
    
    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "direction_accuracy": direction_accuracy,
        "r2": r2,
    }

def ensemble_forecasts(forecasts_list, weights=None):
    """Ensemble multiple forecast outputs for better accuracy
    
    Args:
        forecasts_list: List of forecast DataFrames
        weights: Optional list of weights for each forecast
    
    Returns:
        DataFrame with ensembled forecasts
    """
    import pandas as pd
    
    if not forecasts_list:
        raise ValueError("No forecasts provided for ensembling")
    
    if len(forecasts_list) == 1:
        return forecasts_list[0]
    
    # If weights not provided, use equal weights
    if weights is None:
        weights = [1/len(forecasts_list)] * len(forecasts_list)
    else:
        # Normalize weights
        weights = [w / sum(weights) for w in weights]
    
    # Initialize with first forecast's structure
    ensemble_forecast = forecasts_list[0].copy()
    
    # Reset forecast columns
    ensemble_forecast.loc[:, 'forecast_median'] = 0
    ensemble_forecast.loc[:, 'forecast_lower'] = 0
    ensemble_forecast.loc[:, 'forecast_high'] = 0
    
    # Weighted average of forecasts
    for i, forecast in enumerate(forecasts_list):
        ensemble_forecast['forecast_median'] += weights[i] * forecast['forecast_median']
        ensemble_forecast['forecast_lower'] += weights[i] * forecast['forecast_lower']
        ensemble_forecast['forecast_high'] += weights[i] * forecast['forecast_high']
    
    return ensemble_forecast 