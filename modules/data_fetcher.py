import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
from functools import lru_cache
import pandas_datareader as pdr
import unicodedata

# Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "HJ0QFRRH06A759E0")

def normalize_vietnamese(text):
    """Normalize Vietnamese text by removing diacritics"""
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return text.lower()

def get_alphavantage_data(symbol: str) -> pd.DataFrame:
    """Fetch stock data from Alpha Vantage API"""
    try:
        # Remove any exchange suffixes for Alpha Vantage
        base_symbol = symbol.split('.')[0].split(':')[0]
        
        # Try multiple symbol formats for Alpha Vantage
        alpha_symbols_to_try = []
        
        # For Vietnamese stocks, try multiple formats
        if '.VN' in symbol or ':VN' in symbol or any(ext in symbol for ext in ['.HOSE', '.HNX', '.UPCOM', '.HO', '.HA']):
            alpha_symbols_to_try = [
                f"VNM:{base_symbol}",
                base_symbol,
                f"{base_symbol}.HO",
                f"{base_symbol}.HA"
            ]
        else:
            # For non-Vietnamese stocks
            alpha_symbols_to_try = [base_symbol]
        
        # Try each symbol format until one works
        for alpha_symbol in alpha_symbols_to_try:
            print(f"Trying Alpha Vantage with symbol: {alpha_symbol}")
            
            # Construct the Alpha Vantage API URL
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
                    base,                    
                    f"{base}:VN",          
                    f"{base}.HNX",         
                    f"{base}.HOSE",        
                    f"{base}.UPCOM",       
                    f"{base}.HO",          
                    f"{base}.HA",          
                    f"{base}.HM"           
                ])
            else:
                # If no .VN suffix, add it and other variations
                alternate_symbols.extend([
                    f"{stock_symbol}.VN",    
                    f"{stock_symbol}:VN",    
                    f"{stock_symbol}.HNX",   
                    f"{stock_symbol}.HOSE",  
                    f"{stock_symbol}.UPCOM", 
                    f"{stock_symbol}.HO",    
                    f"{stock_symbol}.HA"     
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

def try_alternative_vn_symbols(symbol: str):
    """Try alternative formats for Vietnamese stocks when standard lookups fail"""
    try:
        # List of possible Vietnamese stock exchange extensions
        vn_extensions = ['.VN', '.HOSE', '.HNX', '.UPCOM', '.HO', '.HA', ':VN']
        
        # Strip any existing extensions
        base_symbol = symbol.split('.')[0].split(':')[0]
        
        # Try each extension
        for ext in vn_extensions:
            try_symbol = f"{base_symbol}{ext}"
            print(f"Trying alternative Vietnamese symbol format: {try_symbol}")
            
            try:
                # Try with YFinance
                data = get_stock_data(try_symbol, period="1y")
                if data is not None and len(data) > 0:
                    return data
            except:
                continue
            
            # If no success with YFinance, try Alpha Vantage
            try:
                data = get_alphavantage_data(try_symbol)
                if data is not None and len(data) > 0:
                    return data
            except:
                continue
        
        # Special case for VN market indices
        if symbol.lower() in ['vnindex', 'vn-index', 'vn index']:
            try:
                return get_stock_data('^VNINDEX', period="1y")
            except:
                pass
            
        # If we get here, no alternatives worked
        return None
        
    except Exception as e:
        print(f"Error trying alternative VN symbols: {str(e)}")
        return None 