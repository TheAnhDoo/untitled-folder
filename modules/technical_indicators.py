import numpy as np
import pandas as pd

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
    
    # Calculate various indicators
    indicators = {}
    
    # Simple Moving Averages
    sma_values = calculate_sma(prices)
    indicators.update(sma_values)
    
    # Exponential Moving Averages and MACD
    ema_values = calculate_ema_macd(prices)
    indicators.update(ema_values)
    
    # RSI
    rsi = calculate_rsi(prices)
    if rsi is not None:
        indicators["RSI"] = f"{rsi:.2f}"
    
    # Bollinger Bands
    bb_position = calculate_bollinger_bands(prices)
    if bb_position is not None:
        indicators["Bollinger Bands"] = bb_position
    
    # Stochastic
    stoch = calculate_stochastic(prices)
    if stoch is not None:
        indicators["Stochastic"] = f"{stoch:.2f}"
    
    # Money Flow Index
    mfi = calculate_mfi(prices, prices, prices, prices)  # Using price as volume estimate
    if mfi is not None:
        indicators["MFI"] = f"{mfi:.2f}"
    
    # Average True Range
    atr = calculate_atr(prices, prices, prices)
    if atr is not None:
        indicators["ATR"] = f"{atr:.2f}"
    
    # Standard deviation of returns
    std_dev = calculate_std_dev(prices)
    if std_dev is not None:
        indicators["Standard Deviation"] = f"{std_dev:.2f}"
    
    return indicators

def calculate_sma(prices, short_window=20, long_window=50):
    """Calculate Simple Moving Averages and related signals"""
    result = {}
    
    # SMA values
    sma20 = np.mean(prices[-short_window:]) if len(prices) >= short_window else None
    sma50 = np.mean(prices[-long_window:]) if len(prices) >= long_window else None
    sma200 = np.mean(prices[-200:]) if len(prices) >= 200 else None
    
    if sma20 is not None:
        result["SMA"] = f"{sma20:.2f}"
    
    # SMA trend
    if sma20 is not None and sma50 is not None:
        if prices[-1] > sma20:
            result["SMA Trend"] = "Price is above 20-day SMA"
        else:
            result["SMA Trend"] = "Price is below 20-day SMA"
    
    return result

def calculate_ema(data, span):
    """Calculate Exponential Moving Average"""
    if len(data) >= span:
        alpha = 2 / (span + 1)
        weights = (1 - alpha) ** np.arange(span)
        weights = weights / weights.sum()
        return np.sum(data[-span:] * weights[::-1])
    return None

def calculate_ema_macd(prices):
    """Calculate EMAs and MACD"""
    result = {}
    
    # EMA calculations
    ema12 = calculate_ema(prices, 12)
    ema26 = calculate_ema(prices, 26)
    
    if ema12 is not None:
        result["EMA"] = f"{ema12:.2f}"
    
    # MACD
    if ema12 is not None and ema26 is not None:
        macd_line = ema12 - ema26
        macd_signal = "bullish" if macd_line > 0 else "bearish"
        result["MACD"] = macd_signal
        
        # EMA trend
        if prices[-1] > ema12:
            result["EMA Trend"] = "Price is above 12-day EMA"
        else:
            result["EMA Trend"] = "Price is below 12-day EMA"
    
    return result

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
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

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """Calculate Bollinger Bands and determine position"""
    if len(prices) < window:
        return None
        
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

def calculate_stochastic(prices, window=14):
    """Calculate Stochastic Oscillator"""
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

def calculate_mfi(high, low, close, volume, period=14):
    """Calculate Money Flow Index"""
    # For this implementation, we'll use close prices as volume estimate
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

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    if len(close) < period:
        return None
        
    # Calculate true ranges (simplified since we're using same value for high, low, close)
    tr = np.zeros(len(close))
    for i in range(1, len(close)):
        tr[i] = abs(high[i] - low[i])
    
    # Average true range
    atr = np.mean(tr[-period:])
    return atr

def calculate_std_dev(prices, period=20):
    """Calculate Standard Deviation of returns"""
    if len(prices) < period:
        return None
        
    returns = np.diff(prices) / prices[:-1]
    std_dev = np.std(returns[-period:]) * 100  # Convert to percentage
    return std_dev

def _handle_rsi_signal(value):
    """Determine signal for RSI indicator"""
    if value >= 70:
        return {"icon": "↗️", "text": "Overbought", "color": "#F44336"}
    elif value <= 30:
        return {"icon": "↘️", "text": "Oversold", "color": "#4CAF50"}
    else:
        return {"icon": "↔️", "text": "Neutral", "color": "#FF9800"}

def _handle_macd_signal(value):
    """Determine signal for MACD indicator"""
    if value == "bullish":
        return {"icon": "↗️", "text": "Bullish", "color": "#4CAF50"}
    else:
        return {"icon": "↘️", "text": "Bearish", "color": "#F44336"}

def _handle_moving_average_signal(value):
    """Determine signal for SMA and EMA indicators"""
    if "above" in value:
        return {"icon": "↗️", "text": "Bullish", "color": "#4CAF50"}
    else:
        return {"icon": "↘️", "text": "Bearish", "color": "#F44336"}

def _handle_bollinger_bands_signal(value):
    """Determine signal for Bollinger Bands indicator"""
    if value == "upper":
        return {"icon": "↗️", "text": "Overbought", "color": "#F44336"}
    elif value == "lower":
        return {"icon": "↘️", "text": "Oversold", "color": "#4CAF50"}
    else:
        return {"icon": "↔️", "text": "Neutral", "color": "#FF9800"}

def _handle_oscillator_signal(value):
    """Determine signal for oscillator-type indicators (Stochastic, MFI)"""
    if value >= 80:
        return {"icon": "↗️", "text": "Overbought", "color": "#F44336"}
    elif value <= 20:
        return {"icon": "↘️", "text": "Oversold", "color": "#4CAF50"}
    else:
        return {"icon": "↔️", "text": "Neutral", "color": "#FF9800"}

def _handle_volatility_signal():
    """Determine signal for volatility indicators (ATR, Standard Deviation)"""
    return {"icon": "ℹ️", "text": "Info", "color": "#1E88E5"}

def get_indicator_signal(indicator, value):
    """Determine signal based on indicator value"""
    # Default signal (neutral/info)
    result = {"icon": "ℹ️", "text": "Neutral", "color": "#FF9800"}
    
    try:
        # Convert value to float if it's a string with a number
        if isinstance(value, str):
            # Handle cases like "45.32 other text"
            try:
                value = float(value.split()[0])
            except ValueError:
                # If conversion fails, keep the original value
                pass
        
        # Route to the appropriate handler based on indicator type
        if indicator == "RSI":
            result = _handle_rsi_signal(value)
        
        elif indicator == "MACD":
            result = _handle_macd_signal(value)
        
        elif indicator in ["SMA", "EMA"]:
            # Just the value, no signal
            result = {"icon": "ℹ️", "text": "Info", "color": "#1E88E5"}
        
        elif indicator in ["SMA Trend", "EMA Trend"]:
            result = _handle_moving_average_signal(value)
        
        elif indicator == "Bollinger Bands":
            result = _handle_bollinger_bands_signal(value)
        
        elif indicator in ["Stochastic", "MFI"]:
            result = _handle_oscillator_signal(value)
        
        elif indicator in ["ATR", "Standard Deviation"]:
            result = _handle_volatility_signal()
        
        # Add more indicators as needed
    except Exception as e:
        # If we can't interpret the value, return generic info signal
        print(f"Error interpreting indicator {indicator}: {str(e)}")
        result = {"icon": "ℹ️", "text": "Info", "color": "#1E88E5"}
    
    return result 