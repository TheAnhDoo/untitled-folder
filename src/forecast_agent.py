import requests
import json
import os
import yfinance as yf
import torch
from chronos import ChronosPipeline
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional
import re
import sys
import time
from functools import lru_cache
from sklearn.metrics import mean_absolute_error, mean_squared_error
import concurrent.futures
import unicodedata
import pandas_datareader as pdr

# Alpha Vantage API key - hardcoded for reliability
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "HJ0QFRRH06A759E0")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Ollama not available. Will use fallback parsing method.")

# Global cache for the Chronos model to avoid reloading
_CHRONOS_MODEL_CACHE = {}

def normalize_vietnamese(text):
    """
    Normalize Vietnamese text by removing diacritics and converting to lowercase
    This helps with pattern matching regardless of accents
    """
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return text.lower()

def get_chronos_model(device_map="auto", model_precision=torch.float32):
    """Get cached Chronos model or load a new one"""
    cache_key = f"{device_map}_{str(model_precision)}"
    
    if cache_key not in _CHRONOS_MODEL_CACHE:
        print(f"Loading Chronos model with device_map={device_map}, precision={model_precision}")
        _CHRONOS_MODEL_CACHE[cache_key] = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-small",
            device_map=device_map,
            torch_dtype=model_precision,
        )
    return _CHRONOS_MODEL_CACHE[cache_key]

def get_alphavantage_data(symbol: str) -> pd.DataFrame:
    """Fetch stock data from Alpha Vantage API with enhanced reliability."""
    try:
        # Remove any exchange suffixes for Alpha Vantage
        base_symbol = symbol.split('.')[0].split(':')[0]
        
        # Try multiple symbol formats for Alpha Vantage
        alpha_symbols_to_try = []
        
        # For Vietnamese stocks, try multiple formats
        if '.VN' in symbol or ':VN' in symbol or any(ext in symbol for ext in ['.HOSE', '.HNX', '.UPCOM', '.HO', '.HA']):
            alpha_symbols_to_try = [
                f"VNM:{base_symbol}",  # Vietnamese market prefix
                base_symbol,           # Plain symbol
                f"{base_symbol}.HO",   # Ho Chi Minh exchange
                f"{base_symbol}.HA"    # Hanoi exchange
            ]
        else:
            # For non-Vietnamese stocks
            alpha_symbols_to_try = [base_symbol]
        
        # Try each symbol format until one works
        for alpha_symbol in alpha_symbols_to_try:
            print(f"Trying Alpha Vantage with symbol: {alpha_symbol}")
            
            # Construct the Alpha Vantage API URL for daily adjusted data with full output size
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={alpha_symbol}&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
            
            response = requests.get(url)
            data = response.json()
            
            # Check for API limit message
            if "Note" in data and "API call frequency" in data["Note"]:
                print(f"Alpha Vantage API limit reached: {data['Note']}")
                # Try a second endpoint format as a backup
                url_backup = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={alpha_symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
                response_backup = requests.get(url_backup)
                data_backup = response_backup.json()
                
                if "Global Quote" in data_backup and data_backup["Global Quote"]:
                    print(f"Successfully got quote data for {alpha_symbol} from Alpha Vantage")
                    # Create a minimal dataset with today's data
                    today = datetime.now().strftime("%Y-%m-%d")
                    records = [{
                        "ds": today,
                        "y": float(data_backup["Global Quote"]["05. price"])
                    }]
                    df = pd.DataFrame(records)
                    df['ds'] = pd.to_datetime(df['ds'])
                    df['unique_id'] = alpha_symbol
                    
                    # Set data source for metrics
                    os.environ["DATA_SOURCE_USED"] = "alphavantage_quote"
                    os.environ["ACTUAL_SYMBOL_USED"] = alpha_symbol
                    
                    print(f"WARNING: Limited to current price only due to API limits")
                    return df
            
            # Check if we got valid time series data
            if "Time Series (Daily)" in data:
                # Convert Alpha Vantage data to DataFrame
                time_series = data["Time Series (Daily)"]
                
                # Create a list of dictionaries for each date
                records = []
                for date, values in time_series.items():
                    records.append({
                        "ds": date,
                        "y": float(values["4. close"])
                    })
                    
                # Create DataFrame and sort by date
                df = pd.DataFrame(records)
                df['ds'] = pd.to_datetime(df['ds'])
                df = df.sort_values('ds')
                
                # Add unique_id column
                df['unique_id'] = alpha_symbol
                
                print(f"Successfully fetched {len(df)} days of data from Alpha Vantage for {alpha_symbol}")
                
                # Set data source for metrics
                os.environ["DATA_SOURCE_USED"] = "alphavantage"
                os.environ["ACTUAL_SYMBOL_USED"] = alpha_symbol
                
                return df
            else:
                error_message = data.get("Error Message", "Unknown error")
                print(f"Alpha Vantage error for {alpha_symbol}: {error_message}")
        
        # If we've tried all symbols and none worked, return None
        return None
        
    except Exception as e:
        print(f"Error fetching data from Alpha Vantage: {str(e)}")
        return None

@lru_cache(maxsize=32)
def get_stock_data(stock_symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Cached function to get stock data to avoid redundant API calls"""
    error_messages = []
    
    try:
        # Check if we should use the maximum available historical data
        if os.environ.get("USE_MAX_HISTORY", "false").lower() == "true":
            period = "max"
        else:
            # Ensure we have enough data for proper validation
            period = max(period, "2y")  # Minimum 2 years of data for better validation
        
        # For Vietnamese stocks, try different formats right away
        alternate_symbols = [stock_symbol]
        
        # If it looks like a Vietnamese stock, prepare alternatives
        if '.VN' in stock_symbol or (len(stock_symbol) <= 3 and stock_symbol.isupper()):
            if '.VN' in stock_symbol:
                base = stock_symbol.replace('.VN', '')
                alternate_symbols.extend([
                    base,                    # Without .VN
                    f"{base}:VN",           # With :VN separator
                    f"{base}.HNX",          # Hanoi Stock Exchange
                    f"{base}.HOSE",         # Ho Chi Minh Stock Exchange
                    f"{base}.UPCOM",        # Unlisted Public Company Market
                    f"{base}.HO",           # Alternative Ho Chi Minh format
                    f"{base}.HA",           # Alternative Hanoi format
                    f"{base}.HM"            # Alternative format
                ])
            else:
                # If no .VN suffix, add it and other variations
                alternate_symbols.extend([
                    f"{stock_symbol}.VN",    # With .VN
                    f"{stock_symbol}:VN",    # With :VN
                    f"{stock_symbol}.HNX",   # Hanoi
                    f"{stock_symbol}.HOSE",  # Ho Chi Minh
                    f"{stock_symbol}.UPCOM", # UPCOM
                    f"{stock_symbol}.HO",    # HO
                    f"{stock_symbol}.HA"     # HA
                ])
        
        # Try each symbol until one works
        for symbol in alternate_symbols:
            try:
                print(f"Trying to fetch data for symbol: {symbol}")
                # Try to get minimal data first to verify the ticker exists
                info = yf.Ticker(symbol).info
                
                if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                    # Download the actual data
                    stock_data = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
                    
                    if len(stock_data) > 0:
                        print(f"Successfully fetched {len(stock_data)} days of data for {symbol}")
                        
                        # Format the data
                        stock_data = stock_data.reset_index().rename(columns={"Date": "ds", "Close": "y"})
                        stock_data = stock_data[["ds", "y"]].copy()
                        stock_data.columns = ["ds", "y"]
                        stock_data['unique_id'] = symbol
                        
                        # Store the successful symbol for metrics
                        os.environ["ACTUAL_SYMBOL_USED"] = symbol
                        os.environ["DATA_SOURCE_USED"] = "yahoo"
                        
                        return stock_data
                    else:
                        error_messages.append(f"No data returned for {symbol}")
                else:
                    error_messages.append(f"Invalid market data for {symbol}")
            except Exception as ticker_error:
                error_messages.append(f"Error with {symbol}: {str(ticker_error)}")
                continue
        
        print("Yahoo Finance attempts failed. Trying alternative data sources...")
        
        # Try Alpha Vantage first as it often has better coverage
        alpha_data = get_alphavantage_data(stock_symbol)
        if alpha_data is not None and len(alpha_data) > 0:
            return alpha_data
        else:
            error_messages.append("Alpha Vantage data retrieval failed")
        
        # If Alpha Vantage fails, try Stooq as a second fallback
        try:
            base_symbol = stock_symbol.replace('.VN', '')
            stooq_symbol = f"{base_symbol}.VN" if '.VN' in stock_symbol or len(stock_symbol) <= 3 else stock_symbol
            
            print(f"Trying alternative data source (stooq) for {stooq_symbol}")
            start_date = datetime.now() - timedelta(days=365*5)  # 5 years of data
            end_date = datetime.now()
            
            stock_data = pdr.data.DataReader(
                stooq_symbol, 
                'stooq', 
                start=start_date, 
                end=end_date
            )
            
            if len(stock_data) > 0:
                print(f"Successfully fetched {len(stock_data)} days of data from stooq for {stooq_symbol}")
                
                # Format the data
                stock_data = stock_data.reset_index().rename(columns={"Date": "ds", "Close": "y"})
                stock_data = stock_data[["ds", "y"]].copy()
                stock_data.columns = ["ds", "y"]
                stock_data['unique_id'] = stooq_symbol
                
                # Store the successful symbol for metrics
                os.environ["ACTUAL_SYMBOL_USED"] = stooq_symbol
                os.environ["DATA_SOURCE_USED"] = "stooq"
                
                return stock_data
            else:
                error_messages.append(f"No data from Stooq for {stooq_symbol}")
        except Exception as stooq_error:
            error_messages.append(f"Stooq data source error: {str(stooq_error)}")
        
        # If all attempts failed, raise a comprehensive error
        raise ValueError(f"Could not fetch data for {stock_symbol} or any of its alternatives. Errors: {'; '.join(error_messages)}")
    except Exception as e:
        # Collect all error messages for better diagnostics
        full_error = f"Error processing stock data for '{stock_symbol}': {str(e)}"
        if error_messages:
            full_error += f" Attempted alternatives with errors: {'; '.join(error_messages)}"
        raise ValueError(full_error)

def calculate_technical_indicators(historical_data: pd.DataFrame) -> dict:
    """
    Calculate technical indicators for a given stock based on historical data
    Returns a dictionary of indicator values
    """
    # Make sure we have enough data
    if len(historical_data) < 30:
        return {}
        
    # Extract close prices
    prices = historical_data['y'].values
    
    # Create a DataFrame for TA calculations
    df = pd.DataFrame({
        'close': prices,
        'high': prices,  # If we don't have high/low data, use close as approximation
        'low': prices,
        'open': prices,
    })
    
    # Simple Moving Averages
    sma20 = np.mean(prices[-20:]) if len(prices) >= 20 else None
    sma50 = np.mean(prices[-50:]) if len(prices) >= 50 else None
    sma200 = np.mean(prices[-200:]) if len(prices) >= 200 else None
    
    # Exponential Moving Averages
    def ema(data, span):
        if len(data) >= span:
            alpha = 2 / (span + 1)
            weights = (1 - alpha) ** np.arange(span)
            weights = weights / weights.sum()
            return np.sum(data[-span:] * weights[::-1])
        return None
    
    ema12 = ema(prices, 12)
    ema26 = ema(prices, 26)
    
    # MACD
    macd_signal = None
    if ema12 is not None and ema26 is not None:
        macd_line = ema12 - ema26
        macd_signal = "bullish" if macd_line > 0 else "bearish"
    
    # RSI (Relative Strength Index)
    def calculate_rsi(prices, window=14):
        if len(prices) <= window:
            return None
        
        # Calculate price changes
        deltas = np.diff(prices)
        deltas = deltas[-window:]  # Take the last 'window' deltas
        
        # Calculate gains and losses
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        
        if avg_loss == 0:
            return 100  # No losses, so RSI is 100
            
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(prices)
    
    # Bollinger Bands
    def calculate_bollinger_bands(prices, window=20, num_std=2):
        if len(prices) < window:
            return None, None, None
            
        rolling_mean = np.mean(prices[-window:])
        rolling_std = np.std(prices[-window:])
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        # Determine position relative to bands
        current_price = prices[-1]
        if current_price > upper_band:
            position = "upper"
        elif current_price < lower_band:
            position = "lower"
        else:
            position = "middle"
            
        return position
    
    bollinger_position = calculate_bollinger_bands(prices)
    
    # Stochastic Oscillator
    def calculate_stochastic(prices, window=14):
        if len(prices) < window:
            return None
            
        # Get the window of prices
        window_prices = prices[-window:]
        
        # Find highest high and lowest low
        highest_high = np.max(window_prices)
        lowest_low = np.min(window_prices)
        
        # Calculate %K
        if highest_high - lowest_low == 0:
            return 50  # To avoid division by zero
            
        k_percent = ((prices[-1] - lowest_low) / (highest_high - lowest_low)) * 100
        return k_percent
    
    stochastic = calculate_stochastic(prices)
    
    # Money Flow Index (MFI)
    def calculate_mfi(high, low, close, volume, period=14):
        # For this simple implementation, we'll use close prices as volume estimate
        if len(close) < period + 1:
            return None
            
        # Calculate typical price
        typical_price = close
        
        # Calculate raw money flow
        raw_money_flow = typical_price * close
        
        # Get positive and negative money flow
        money_flow_pos = np.zeros(len(close))
        money_flow_neg = np.zeros(len(close))
        
        for i in range(1, len(close)):
            if typical_price[i] > typical_price[i-1]:
                money_flow_pos[i] = raw_money_flow[i]
            else:
                money_flow_neg[i] = raw_money_flow[i]
        
        # Sum positive and negative money flow over period
        pos_flow = np.sum(money_flow_pos[-period-1:-1])
        neg_flow = np.sum(money_flow_neg[-period-1:-1])
        
        # Calculate money flow ratio and MFI
        if neg_flow == 0:
            return 100
            
        money_ratio = pos_flow / neg_flow
        mfi = 100 - (100 / (1 + money_ratio))
        
        return mfi
    
    # Since we don't have volume, we'll estimate MFI using close prices
    # This is a very rough approximation
    mfi = calculate_mfi(prices, prices, prices, prices)
    
    # Average True Range (ATR) - simplified
    def calculate_atr(high, low, close, period=14):
        if len(close) < period:
            return None
            
        # Calculate true ranges (we're using the same value for high, low, close)
        tr = np.zeros(len(close))
        for i in range(1, len(close)):
            tr[i] = abs(high[i] - low[i])
        
        # Average true range
        atr = np.mean(tr[-period:])
        return atr
    
    atr = calculate_atr(prices, prices, prices)
    
    # Standard deviation of returns
    def calculate_std_dev(prices, period=20):
        if len(prices) < period:
            return None
            
        returns = np.diff(prices) / prices[:-1]
        std_dev = np.std(returns[-period:]) * 100  # Convert to percentage
        return std_dev
    
    std_dev = calculate_std_dev(prices)
    
    # SMA trend indicators
    sma_trend = None
    if sma20 is not None and sma50 is not None:
        if prices[-1] > sma20:
            sma_trend = "Price is above 20-day SMA" 
        else:
            sma_trend = "Price is below 20-day SMA"
            
    # EMA trend
    ema_trend = None
    if ema12 is not None and ema26 is not None:
        if prices[-1] > ema12:
            ema_trend = "Price is above 12-day EMA"
        else:
            ema_trend = "Price is below 12-day EMA"
    
    # Compile all indicators into a dictionary
    indicators = {}
    
    if sma20 is not None: indicators["SMA"] = f"{sma20:.2f}"
    if ema12 is not None: indicators["EMA"] = f"{ema12:.2f}"
    if macd_signal is not None: indicators["MACD"] = macd_signal
    if rsi is not None: indicators["RSI"] = f"{rsi:.2f}"
    if stochastic is not None: indicators["Stochastic"] = f"{stochastic:.2f}"
    if mfi is not None: indicators["MFI"] = f"{mfi:.2f}"
    if bollinger_position is not None: indicators["Bollinger Bands"] = bollinger_position
    if atr is not None: indicators["ATR"] = f"{atr:.2f}"
    if std_dev is not None: indicators["Standard Deviation"] = f"{std_dev:.2f}"
    if sma_trend is not None: indicators["SMA Trend"] = sma_trend
    if ema_trend is not None: indicators["EMA Trend"] = ema_trend
    
    return indicators

def generate_forecast_summary(forecast_data, historical_data, metrics, language):
    """
    Generate a comprehensive natural language summary of forecast insights and trends.
    Supports both English and Vietnamese output based on language settings.
    
    Args:
        forecast_data (pd.DataFrame): Forecast data with quantiles
        historical_data (pd.DataFrame): Historical price data
        metrics (dict): Forecast metrics including MAE, RMSE, etc.
        
    Returns:
        dict: Summary text and confidence level
    """
    try:
        # Extract key data points
        last_historical_price = historical_data['y'].iloc[-1]
        median_forecast = forecast_data['forecast_median'].iloc[-1]
        lower_bound = forecast_data['forecast_lower'].iloc[-1]
        upper_bound = forecast_data['forecast_high'].iloc[-1]
        
        # Get data for technical analysis
        historical_prices = historical_data['y'].values
        forecast_prices = forecast_data['forecast_median'].values
        all_prices = np.concatenate([historical_prices[-60:], forecast_prices])
        
        # Forecast direction and magnitude
        percent_change = ((median_forecast - last_historical_price) / last_historical_price) * 100
        
        # Volatility assessment
        forecast_volatility = metrics.get('forecast_volatility', 0)
        historical_volatility = historical_data['y'].pct_change().std() * 100
        relative_volatility = forecast_volatility / (historical_volatility if historical_volatility > 0 else 1)
        
        # Price pattern in forecast
        forecast_trend = "upward" if percent_change > 3 else "downward" if percent_change < -3 else "sideways"
        
        # Calculate if the forecast shows consistent trend or volatility
        q25_trend = (forecast_data['q25'].iloc[-1] - forecast_data['q25'].iloc[0]) > 0
        q75_trend = (forecast_data['q75'].iloc[-1] - forecast_data['q75'].iloc[0]) > 0
        consistent_direction = q25_trend == q75_trend
        
        # Recent price action (last 30 days)
        recent_window = min(30, len(historical_data))
        recent_change = ((historical_data['y'].iloc[-1] - historical_data['y'].iloc[-recent_window]) / 
                         historical_data['y'].iloc[-recent_window]) * 100
        
        # Identify price patterns in historical data
        # Simple moving averages
        short_window = min(20, len(historical_prices) - 1)
        long_window = min(50, len(historical_prices) - 1)
        
        if len(historical_prices) > long_window:
            sma_short = np.mean(historical_prices[-short_window:])
            sma_long = np.mean(historical_prices[-long_window:])
            sma_trend = "bullish" if sma_short > sma_long else "bearish"
            
            # Calculate momentum
            momentum = historical_prices[-1] - historical_prices[-10] if len(historical_prices) >= 10 else 0
            momentum_signal = "positive" if momentum > 0 else "negative"
            
            # Check for reversal patterns
            last_5_changes = np.diff(historical_prices[-6:])
            consecutive_direction = np.all(last_5_changes > 0) or np.all(last_5_changes < 0)
            potential_reversal = consecutive_direction and (forecast_trend != "upward" if np.all(last_5_changes > 0) else forecast_trend != "downward")
        else:
            sma_trend = "undetermined"
            momentum_signal = "undetermined"
            potential_reversal = False
        
        # Support and resistance analysis
        # Simplified approach - look at recent highs and lows
        if len(historical_prices) > 30:
            recent_high = np.max(historical_prices[-30:])
            recent_low = np.min(historical_prices[-30:])
            
            # Check if current price is near support or resistance
            near_support = abs(last_historical_price - recent_low) / recent_low < 0.03  # Within 3% of recent low
            near_resistance = abs(last_historical_price - recent_high) / recent_high < 0.03  # Within 3% of recent high
            
            # Predict breakout/breakdown based on forecast
            potential_breakout = near_resistance and forecast_trend == "upward"
            potential_breakdown = near_support and forecast_trend == "downward"
        else:
            near_support = False
            near_resistance = False
            potential_breakout = False
            potential_breakdown = False
        
        # Confidence assessment with more factors
        confidence_factors = []
        if relative_volatility > 1.5:
            confidence_factors.append("high forecast volatility" if language == "en" else "độ biến động dự báo cao")
        if not consistent_direction:
            confidence_factors.append("inconsistent direction across quantiles" if language == "en" else "hướng không nhất quán giữa các phân vị")
        if 'direction_accuracy' in metrics and metrics['direction_accuracy'] < 0.6:
            confidence_factors.append("below average direction accuracy in validation" if language == "en" else "độ chính xác hướng thấp hơn trung bình trong xác thực")
        if 'validation_status' in metrics and metrics['validation_status'] != 'completed':
            confidence_factors.append("incomplete validation" if language == "en" else "xác thực không hoàn chỉnh")
            
        # Determine overall confidence
        if len(confidence_factors) >= 2 or ('validation_status' in metrics and metrics['validation_status'] != 'completed'):
            confidence_level = "low" if language == "en" else "thấp"
        elif len(confidence_factors) == 1 or relative_volatility > 1.2:
            confidence_level = "moderate" if language == "en" else "trung bình"
        else:
            confidence_level = "high" if language == "en" else "cao"
        
        # Range estimation (how likely price will stay within forecast range)
        range_ratio = (upper_bound - lower_bound) / last_historical_price
        range_certainty = "wide" if range_ratio > 0.15 else "moderate" if range_ratio > 0.07 else "narrow"
        
        # Vietnamese translation for range_certainty
        if language == "vi":
            range_certainty = "rộng" if range_ratio > 0.15 else "trung bình" if range_ratio > 0.07 else "hẹp"
        
        # Market behavior description - English and Vietnamese versions
        if language == "en":
            if percent_change > 10:
                strength_desc = "surge significantly"
            elif percent_change > 5:
                strength_desc = "rise considerably"
            elif percent_change > 2:
                strength_desc = "increase moderately"
            elif percent_change > 0:
                strength_desc = "show slight gains"
            elif percent_change > -2:
                strength_desc = "experience minor decline"
            elif percent_change > -5:
                strength_desc = "decrease moderately"
            elif percent_change > -10:
                strength_desc = "fall considerably"
            else:
                strength_desc = "drop significantly"
        else:  # Vietnamese
            if percent_change > 10:
                strength_desc = "tăng mạnh đáng kể"
            elif percent_change > 5:
                strength_desc = "tăng đáng kể"
            elif percent_change > 2:
                strength_desc = "tăng vừa phải"
            elif percent_change > 0:
                strength_desc = "tăng nhẹ"
            elif percent_change > -2:
                strength_desc = "giảm nhẹ"
            elif percent_change > -5:
                strength_desc = "giảm vừa phải"
            elif percent_change > -10:
                strength_desc = "giảm đáng kể"
            else:
                strength_desc = "giảm mạnh đáng kể"
        
        # Technical terms translations
        if language == "en":
            tr_bullish = "bullish"
            tr_bearish = "bearish"
            tr_positive = "positive"
            tr_negative = "negative"
            tr_upward = "upward"
            tr_downward = "downward"
            tr_sideways = "sideways"
            tr_support = "support"
            tr_resistance = "resistance"
            tr_breakout = "breakout"
            tr_breakdown = "breakdown"
            tr_consolidation = "consolidation"
            tr_volatility = "volatility"
        else:
            tr_bullish = "tăng giá"
            tr_bearish = "giảm giá"
            tr_positive = "tích cực"
            tr_negative = "tiêu cực"
            tr_upward = "đi lên"
            tr_downward = "đi xuống"
            tr_sideways = "đi ngang"
            tr_support = "hỗ trợ"
            tr_resistance = "kháng cự"
            tr_breakout = "bứt phá"
            tr_breakdown = "phá vỡ"
            tr_consolidation = "tích lũy"
            tr_volatility = "biến động"
            
        # Trend direction in the right language
        if forecast_trend == "upward":
            tr_forecast_trend = tr_upward
        elif forecast_trend == "downward":
            tr_forecast_trend = tr_downward
        else:
            tr_forecast_trend = tr_sideways
            
        # SMA trend in the right language
        if sma_trend == "bullish":
            tr_sma_trend = tr_bullish
        else:
            tr_sma_trend = tr_bearish
            
        # Momentum in the right language
        if momentum_signal == "positive":
            tr_momentum = tr_positive
        else:
            tr_momentum = tr_negative
        
        # Generate the comprehensive summary
        stock_name = metrics.get('actual_symbol', metrics.get('requested_symbol', 'The stock'))
        
        if language == "en":
            # English summary
            # Introduction with forecast direction and magnitude
            summary = f"{stock_name} is forecasted to {strength_desc} over the coming period, "
            summary += f"with a projected change of {percent_change:.1f}% from the current price of {last_historical_price:.2f}. "
            
            # Recent price context and technical analysis
            if abs(recent_change) > 10:
                recent_action = "strong " + ("upward" if recent_change > 0 else "downward")
                summary += f"This forecast follows a {recent_action} movement of {abs(recent_change):.1f}% over the past {recent_window} trading days. "
            else:
                summary += f"Over the past {recent_window} trading days, the stock has moved {recent_change:.1f}%. "
            
            # Add technical analysis insights if we have enough data
            if len(historical_prices) > long_window:
                summary += f"Technical indicators show a {tr_sma_trend} trend based on moving averages, with {tr_momentum} momentum. "
                
                if near_support:
                    summary += f"The price is currently testing a support level around {recent_low:.2f}. "
                    if potential_breakdown:
                        summary += "The forecast suggests a possible breakdown below this support, which could accelerate selling pressure. "
                elif near_resistance:
                    summary += f"The price is currently testing a resistance level around {recent_high:.2f}. "
                    if potential_breakout:
                        summary += "The forecast suggests a possible breakout above this resistance, which could trigger further buying. "
                
                if potential_reversal:
                    summary += "Recent price action shows consistent movement in one direction, but the forecast indicates a potential reversal pattern forming. "
            
            # Volatility and risk assessment
            summary += f"The forecast shows {range_certainty} range estimation with a {confidence_level} confidence level. "
            
            if relative_volatility > 1.2:
                summary += f"Volatility is projected to be {relative_volatility:.1f}x higher than historical levels, suggesting significant price fluctuations may occur. "
            elif relative_volatility < 0.8:
                summary += f"Volatility is projected to be lower than historical patterns ({relative_volatility:.1f}x), suggesting more stable price movement. "
            
            # Add validation metrics if available
            if 'mae' in metrics and 'direction_accuracy' in metrics:
                summary += f"Based on backtesting, the model demonstrates a direction accuracy of {metrics['direction_accuracy']*100:.1f}% "
                summary += f"with a Mean Absolute Error of {metrics['mae']:.2f}. "
            
            # Add specific pattern insights
            if forecast_trend == "sideways":
                summary += "The price structure appears to be forming a consolidation pattern, with price likely to trade within a defined range. "
                summary += f"Investors might consider range-bound strategies between approximately {lower_bound:.2f} and {upper_bound:.2f}. "
            elif forecast_trend == "upward":
                summary += f"The model indicates a primarily bullish trend over the forecast period with potential resistance around {upper_bound:.2f}. "
                if 'forecast_volatility' in metrics and metrics['forecast_volatility'] < 0.5:
                    summary += "The steady upward trajectory suggests a sustainable trend rather than a speculative surge. "
            else:  # downward
                summary += f"The model indicates a primarily bearish trend over the forecast period with potential support around {lower_bound:.2f}. "
                if potential_reversal and potential_breakdown:
                    summary += "Traders should be cautious of accelerated selling if key support levels are breached. "
            
            # Conclusion with risk factors
            if confidence_factors:
                summary += f"Key risk factors to consider include: {', '.join(confidence_factors)}. "
            
            summary += f"Given these factors, the overall confidence in this forecast is {confidence_level}."
            
        else:
            # Vietnamese summary
            # Introduction with forecast direction and magnitude
            summary = f"Dự báo cho thấy {stock_name} sẽ {strength_desc} trong thời gian tới, "
            summary += f"với mức thay đổi dự kiến {percent_change:.1f}% so với giá hiện tại là {last_historical_price:.2f}. "
            
            # Recent price context and technical analysis
            if abs(recent_change) > 10:
                recent_action = "mạnh " + ("đi lên" if recent_change > 0 else "đi xuống")
                summary += f"Dự báo này đi sau biến động {recent_action} {abs(recent_change):.1f}% trong {recent_window} ngày giao dịch gần đây. "
            else:
                summary += f"Trong {recent_window} ngày giao dịch vừa qua, cổ phiếu đã biến động {recent_change:.1f}%. "
            
            # Add technical analysis insights if we have enough data
            if len(historical_prices) > long_window:
                summary += f"Các chỉ báo kỹ thuật cho thấy xu hướng {tr_sma_trend} dựa trên đường trung bình động, với đà {tr_momentum}. "
                
                if near_support:
                    summary += f"Giá hiện đang kiểm định vùng hỗ trợ quanh mức {recent_low:.2f}. "
                    if potential_breakdown:
                        summary += f"Dự báo cho thấy khả năng phá vỡ vùng hỗ trợ này, có thể dẫn đến áp lực bán gia tăng. "
                elif near_resistance:
                    summary += f"Giá hiện đang kiểm định vùng kháng cự quanh mức {recent_high:.2f}. "
                    if potential_breakout:
                        summary += f"Dự báo cho thấy khả năng bứt phá vùng kháng cự này, có thể kích hoạt lực mua gia tăng. "
                
                if potential_reversal:
                    summary += f"Diễn biến giá gần đây cho thấy chuyển động nhất quán theo một hướng, nhưng dự báo cho thấy khả năng hình thành mô hình đảo chiều. "
            
            # Volatility and risk assessment
            summary += f"Dự báo cho thấy biên độ dao động {range_certainty} với mức độ tin cậy {confidence_level}. "
            
            if relative_volatility > 1.2:
                summary += f"Độ biến động dự kiến cao hơn {relative_volatility:.1f} lần so với mức lịch sử, cho thấy khả năng xảy ra biến động giá đáng kể. "
            elif relative_volatility < 0.8:
                summary += f"Độ biến động dự kiến thấp hơn so với mẫu lịch sử ({relative_volatility:.1f} lần), cho thấy chuyển động giá ổn định hơn. "
            
            # Add validation metrics if available
            if 'mae' in metrics and 'direction_accuracy' in metrics:
                summary += f"Dựa trên kiểm tra ngược, mô hình thể hiện độ chính xác về hướng là {metrics['direction_accuracy']*100:.1f}% "
                summary += f"với Sai số Tuyệt đối Trung bình (MAE) là {metrics['mae']:.2f}. "
            
            # Add specific pattern insights
            if forecast_trend == "sideways":
                summary += f"Cấu trúc giá có vẻ đang hình thành mô hình tích lũy, với giá có khả năng giao dịch trong biên độ xác định. "
                summary += f"Nhà đầu tư có thể xem xét chiến lược giao dịch trong khoảng giá từ {lower_bound:.2f} đến {upper_bound:.2f}. "
            elif forecast_trend == "upward":
                summary += f"Mô hình cho thấy xu hướng chủ yếu là tăng giá trong thời gian dự báo với vùng kháng cự tiềm năng quanh mức {upper_bound:.2f}. "
                if 'forecast_volatility' in metrics and metrics['forecast_volatility'] < 0.5:
                    summary += f"Quỹ đạo đi lên ổn định cho thấy xu hướng bền vững hơn là một đợt tăng mang tính đầu cơ. "
            else:  # downward
                summary += f"Mô hình cho thấy xu hướng chủ yếu là giảm giá trong thời gian dự báo với vùng hỗ trợ tiềm năng quanh mức {lower_bound:.2f}. "
                if potential_reversal and potential_breakdown:
                    summary += f"Nhà giao dịch nên thận trọng với khả năng bán tháo nếu các mức hỗ trợ quan trọng bị phá vỡ. "
            
            # Conclusion with risk factors
            if confidence_factors:
                summary += f"Các yếu tố rủi ro chính cần xem xét bao gồm: {', '.join(confidence_factors)}. "
            
            summary += f"Với những yếu tố này, mức độ tin cậy tổng thể cho dự báo này là {confidence_level}."
        
        return {
            "summary": summary,
            "forecast_trend": forecast_trend,
            "percent_change": percent_change,
            "confidence_level": confidence_level
        }
        
    except Exception as e:
        print(f"Error generating forecast summary: {str(e)}")
        return {
            "summary": "Unable to generate comprehensive forecast summary due to insufficient data or processing error." if language == "en" else "Không thể tạo bản tóm tắt dự báo toàn diện do dữ liệu không đủ hoặc lỗi xử lý.",
            "forecast_trend": "unknown",
            "percent_change": 0,
            "confidence_level": "unknown" if language == "en" else "không xác định"
        }

def get_stock_forecast(stock_symbol: str, prediction_days: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Get stock forecast for a specific symbol"""
    print(f"Getting forecast for {stock_symbol} for {prediction_days} days")
    
    # Set an environment variable that other functions can check
    os.environ["CURRENT_FORECAST_SYMBOL"] = stock_symbol
    
    # First attempt with Yahoo Finance
    try:
        historical_data = get_stock_data(stock_symbol)
        
        # If data is too small, try to fetch more
        if len(historical_data) < 60:
            try:
                historical_data = get_stock_data(stock_symbol, period="5y")
            except:
                pass
        
        if historical_data is None or historical_data.empty or len(historical_data) < 20:
            raise ValueError(f"Insufficient data for {stock_symbol}")
        
        os.environ["DATA_SOURCE_USED"] = "yfinance"
        os.environ["ACTUAL_SYMBOL_USED"] = stock_symbol
            
    except Exception as e:
        print(f"Yahoo Finance error: {str(e)}")
        
        # Try Stooq as a backup
        try:
            # Prepare symbol for Stooq (remove exchange suffixes)
            stooq_symbol = stock_symbol.split('.')[0].split(':')[0]
            print(f"Trying Stooq with symbol: {stooq_symbol}")
            
            # Fetch data from Stooq
            stooq_data = pdr.get_data_stooq(stooq_symbol)
            
            if stooq_data is not None and not stooq_data.empty:
                # Convert to expected format
                historical_data = pd.DataFrame({
                    'ds': stooq_data.index,
                    'y': stooq_data['Close'].values
                })
                
                # Add unique_id column
                historical_data['unique_id'] = stooq_symbol
                
                print(f"Successfully fetched {len(historical_data)} days of data from Stooq")
                
                # Set data source for metrics
                os.environ["DATA_SOURCE_USED"] = "stooq"
                os.environ["ACTUAL_SYMBOL_USED"] = stooq_symbol
            else:
                raise ValueError(f"No data from Stooq for {stooq_symbol}")
                
        except Exception as stooq_error:
            print(f"Stooq error: {str(stooq_error)}")
            
            # Try Alpha Vantage as a last resort
            try:
                alpha_data = get_alphavantage_data(stock_symbol)
                
                if alpha_data is not None and not alpha_data.empty:
                    historical_data = alpha_data
                    print(f"Using Alpha Vantage data with {len(historical_data)} rows")
                else:
                    raise ValueError(f"No data from Alpha Vantage for {stock_symbol}")
            except Exception as alpha_error:
                print(f"Alpha Vantage error: {str(alpha_error)}")
                # Finally, try some alternative symbol formats for Vietnamese stocks
                vn_result = try_alternative_vn_symbols(stock_symbol)
                
                if vn_result is not None:
                    historical_data = vn_result
                else:
                    # All attempts failed, raise the original error
                    raise ValueError(f"Could not retrieve data for {stock_symbol} from any source. Original error: {str(e)}")
    
    # Error if still no data
    if historical_data is None or historical_data.empty or len(historical_data) < 10:
        raise ValueError(f"Insufficient data points for {stock_symbol}. Need at least 10 data points for forecasting.")
    
    # Set validation days based on data availability
    # For longer histories, use more days for validation
    total_days = len(historical_data)
    
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
    
    # Make a copy to avoid modifying the original DataFrame
    historical_copy = historical_data.copy()
    
    # Determine the appropriate device for model inference
    if torch.cuda.is_available():
        device_map = "cuda"
        precision = torch.float16  # Use half precision for CUDA
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device_map = "mps"
        precision = torch.float32  # MPS requires float32
    else:
        device_map = "cpu"
        precision = torch.float32  # CPU works well with float32
    
    # Get the Chronos model (using cache if available)
    model = get_chronos_model(device_map, precision)
    
    # Create a validation set if we have enough data
    if validation_days > 0:
        # Split into training and validation
        train_data = historical_copy.iloc[:-validation_days].copy()
        validation_data = historical_copy.iloc[-validation_days:].copy()
        
        # Generate forecast for validation period
        val_forecast = model.forecast(
            train_data,
            prediction_length=validation_days
        )
        
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
    forecast = model.forecast(
        historical_copy,
        prediction_length=prediction_days
    )
    
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
        'technical_indicators': technical_indicators
    }
    
    # Add validation metrics if available
    if validation_metrics:
        metrics.update(validation_metrics)
    
    return forecast, historical_data, metrics

def process_multiple_stocks(stock_symbols: List[str], prediction_days: int = 10) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, Dict]]:
    """Process multiple stocks in parallel and return results"""
    language = os.environ.get("LANGUAGE", "en")
    use_advanced = os.environ.get("USE_ADVANCED_ANALYSIS", "true").lower() == "true"
    results = {}
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(stock_symbols), 4)) as executor:
        # Submit tasks
        future_to_symbol = {
            executor.submit(get_stock_forecast, symbol, prediction_days): symbol 
            for symbol in stock_symbols
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                forecast, historical_data, metrics = future.result()
                
                # Calculate technical indicators if advanced analysis is enabled
                if use_advanced and 'technical_indicators' not in metrics and historical_data is not None:
                    metrics['technical_indicators'] = calculate_technical_indicators(historical_data)
                
                # Generate summary with the correct language
                summary = generate_forecast_summary(forecast, historical_data, metrics, language)
                metrics['forecast_summary'] = summary
                
                results[symbol] = (forecast, historical_data, metrics)
            except Exception as e:
                print(f"Error processing {symbol}: {str(e)}")
                # Add empty result to indicate error
                results[symbol] = (None, None, {"error": str(e), "requested_symbol": symbol})
    
    return results

def extract_stock_info_with_regex(prompt: str) -> Dict:
    """Extract stock symbols and prediction days using regex patterns
    Returns a dictionary with 'stock_symbols' and 'prediction_days'"""
    
    # Normalize the prompt to lowercase
    prompt_lower = prompt.lower()
    
    # First, try to extract the forecast period (in days)
    # Look for patterns like "next X days", "X days", "X-day forecast"
    day_patterns = [
        r'(\d+)\s*(?:day|ngày)',  # 10 day, 10 days, 10 ngày
        r'next\s*(\d+)',  # next 10
        r'(\d+)[- ]day',  # 10-day, 10 day
        r'(\d+)[- ]ngày',  # Vietnamese: 10-ngày, 10 ngày
    ]
    
    prediction_days = 10  # Default if not specified
    
    for pattern in day_patterns:
        day_match = re.search(pattern, prompt_lower)
        if day_match:
            prediction_days = int(day_match.group(1))
            break
    
    # Also handle weeks and months
    week_match = re.search(r'(\d+)\s*(?:week|tuần)', prompt_lower)
    month_match = re.search(r'(\d+)\s*(?:month|tháng)', prompt_lower)
    
    if week_match:
        prediction_days = int(week_match.group(1)) * 7
    elif month_match:
        prediction_days = int(month_match.group(1)) * 30
    
    # Cap prediction days to a reasonable value
    prediction_days = min(max(prediction_days, 1), 365)
    
    # Extract stock symbols
    # Common stock symbols are 1-5 characters, all caps
    stock_pattern = r'\b([A-Z]{1,5})[\.:]?([A-Z]{2})?\b'
    
    # Also look for common company names
    company_names = {
        'apple': 'AAPL',
        'microsoft': 'MSFT',
        'amazon': 'AMZN',
        'google': 'GOOGL',
        'alphabet': 'GOOGL',
        'facebook': 'META',
        'meta': 'META',
        'tesla': 'TSLA',
        'netflix': 'NFLX',
        'alibaba': 'BABA',
        'nvidia': 'NVDA',
        'amd': 'AMD',
        'intel': 'INTC',
        'ibm': 'IBM',
        'oracle': 'ORCL',
        'cisco': 'CSCO',
        'adidas': 'ADDYY',
        'nike': 'NKE',
        'walmart': 'WMT',
        'target': 'TGT',
        'costco': 'COST',
        'coca cola': 'KO',
        'coca-cola': 'KO',
        'pepsi': 'PEP',
        'boeing': 'BA',
        'airbus': 'EADSY',
        'mastercard': 'MA',
        'visa': 'V',
        'paypal': 'PYPL',
        'jp morgan': 'JPM',
        'jpmorgan': 'JPM',
        'goldman sachs': 'GS',
        'bank of america': 'BAC',
        'disney': 'DIS',
        'verizon': 'VZ',
        'att': 'T',
        'at&t': 'T',
        'ford': 'F',
        'general motors': 'GM',
        'gm': 'GM',
        'starbucks': 'SBUX',
        'mcdonalds': 'MCD',
        'vietcombank': 'VCB.VN',
        'vcb': 'VCB.VN',
        'vingroup': 'VIC.VN',
        'vic': 'VIC.VN',
        'vinhomes': 'VHM.VN',
        'vhm': 'VHM.VN',
        'viettel': 'VTG.VN',
        'vpbank': 'VPB.VN',
        'vpb': 'VPB.VN',
        'masan': 'MSN.VN',
        'sabeco': 'SAB.VN',
        'fpt': 'FPT.VN',
        'vnm': 'VNM.VN',
        'vinamilk': 'VNM.VN',
        'hòa phát': 'HPG.VN',
        'hoa phat': 'HPG.VN',
        'hpg': 'HPG.VN',
        'techcombank': 'TCB.VN',
        'tcb': 'TCB.VN',
        'vietjet': 'VJC.VN',
        'vjc': 'VJC.VN',
        'vn-index': '^VNINDEX',
        'vnindex': '^VNINDEX',
        'vn index': '^VNINDEX',
        'dow jones': '^DJI',
        'dow': '^DJI',
        's&p 500': '^GSPC',
        's&p': '^GSPC',
        'sp500': '^GSPC',
        'nasdaq': '^IXIC',
    }
    
    # Extract stock symbols from the prompt
    stock_symbols = []
    
    # First try to extract by symbol
    symbol_matches = re.findall(stock_pattern, prompt)
    for match in symbol_matches:
        symbol = match[0]
        suffix = match[1] if match[1] else ""
        
        # Skip common words that match the pattern
        if symbol.lower() in ['a', 'i', 'in', 'to', 'for', 'on', 'at', 'by', 'of']:
            continue
            
        # Build the full symbol
        full_symbol = symbol + suffix
        
        # Check if it's a Vietnamese stock
        if any(vn_term in prompt_lower for vn_term in ['vietnam', 'việt nam', 'vn-index', 'vnindex', 'hose', 'hnx']):
            if not suffix and symbol not in ['^VNINDEX', '^DJI', '^GSPC', '^IXIC']:
                full_symbol = symbol + '.VN'
                
        stock_symbols.append(full_symbol)
    
    # Then look for company names
    for company, symbol in company_names.items():
        if company in prompt_lower:
            if symbol not in stock_symbols:
                stock_symbols.append(symbol)
    
    # If no symbols found, check for some common cases
    if not stock_symbols:
        if 's&p' in prompt_lower or 'sp500' in prompt_lower or 'standard and poor' in prompt_lower:
            stock_symbols.append('^GSPC')
        elif 'dow' in prompt_lower or 'dow jones' in prompt_lower:
            stock_symbols.append('^DJI')
        elif 'nasdaq' in prompt_lower:
            stock_symbols.append('^IXIC')
            
    # Remove duplicates while preserving order
    seen = set()
    stock_symbols = [x for x in stock_symbols if not (x in seen or seen.add(x))]
    
    return {
        'stock_symbols': stock_symbols,
        'prediction_days': prediction_days
    }

@lru_cache(maxsize=32)
def get_ollama_response(prompt: str, model: str = "llama3") -> Dict:
    """Cached function to get Ollama responses for faster repeated queries"""
    response = ollama.chat(
        model=model, 
        messages=[{"role": "user", "content": prompt}]
    )
    return response

def stock_forecast_agent(prompt: str) -> Union[Tuple[pd.DataFrame, pd.DataFrame, Dict], Dict[str, Tuple[pd.DataFrame, pd.DataFrame, Dict]]]:
    """AI agent for forecasting stocks based on natural language prompts"""
    print(f"Processing forecast query: {prompt}")
    
    # Start time for performance tracking
    start_time = time.time()
    
    # Get language from environment for localization (defaults to English)
    language = os.environ.get("LANGUAGE", "en")
    
    # Flag for advanced analysis (technical indicators)
    use_advanced = os.environ.get("USE_ADVANCED_ANALYSIS", "true").lower() == "true"
    
    # Try to get the response from ollama
    try:
        if OLLAMA_AVAILABLE:
            ollama_result = get_ollama_response(prompt)
            
            if ollama_result and 'stock_symbols' in ollama_result:
                stock_symbols = ollama_result.get('stock_symbols', [])
                prediction_days = ollama_result.get('prediction_days', 10)
                print(f"Ollama extracted: {stock_symbols}, {prediction_days} days")
                
                # Set batch processing for multiple stocks
                batch_processing = len(stock_symbols) > 1 or os.environ.get("BATCH_PROCESSING", "false").lower() == "true"
                
                # For multiple stocks, process in parallel with batch processing
                if len(stock_symbols) > 1 and batch_processing:
                    return process_multiple_stocks(stock_symbols, prediction_days)
                
                # For single stock or without batch processing
                if stock_symbols:
                    forecast, historical_data, metrics = get_stock_forecast(stock_symbols[0], prediction_days)
                    
                    # Calculate technical indicators if advanced analysis is enabled
                    if use_advanced and 'technical_indicators' not in metrics and historical_data is not None:
                        metrics['technical_indicators'] = calculate_technical_indicators(historical_data)
                    
                    # Generate summary with the correct language
                    summary = generate_forecast_summary(forecast, historical_data, metrics, language)
                    metrics['forecast_summary'] = summary
                    
                    # Return forecast results
                    return forecast, historical_data, metrics
    except Exception as e:
        print(f"Ollama error: {str(e)}. Using fallback method.")
    
    # Fallback method using regex if Ollama fails
    print("Using fallback extraction with regex")
    result = extract_stock_info_with_regex(prompt)
    
    # Set batch processing for multiple stocks
    batch_processing = len(result['stock_symbols']) > 1 or os.environ.get("BATCH_PROCESSING", "false").lower() == "true"
    
    # For multiple stocks, process in parallel with batch processing
    if len(result['stock_symbols']) > 1 and batch_processing:
        return process_multiple_stocks(result['stock_symbols'], result['prediction_days'])
    
    # For single stock or without batch processing
    if result['stock_symbols']:
        forecast, historical_data, metrics = get_stock_forecast(result['stock_symbols'][0], result['prediction_days'])
        
        # Calculate technical indicators if advanced analysis is enabled
        if use_advanced and 'technical_indicators' not in metrics and historical_data is not None:
            metrics['technical_indicators'] = calculate_technical_indicators(historical_data)
        
        # Generate summary with the correct language
        summary = generate_forecast_summary(forecast, historical_data, metrics, language)
        metrics['forecast_summary'] = summary
        
        # Return forecast results
            return forecast, historical_data, metrics
    else:
        raise ValueError("No valid stock symbols found in the prompt")