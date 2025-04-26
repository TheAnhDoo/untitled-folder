import re
import os
from functools import lru_cache
from typing import Dict, List, Union, Tuple
import pandas as pd

# Try to import Ollama for LLM processing
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Ollama not available. Will use fallback parsing method.")

from modules.forecasting import get_stock_forecast, process_multiple_stocks, generate_forecast_summary

@lru_cache(maxsize=32)
def get_ollama_response(prompt: str, model: str = "llama3") -> Dict:
    """Cached function to get Ollama responses for faster repeated queries"""
    try:
        response = ollama.chat(
            model=model, 
            messages=[{"role": "user", "content": prompt}]
        )
        return response
    except Exception as e:
        print(f"Error getting Ollama response: {str(e)}")
        return {}

def extract_stock_info_with_regex(prompt: str) -> Dict:
    """Extract stock symbols and prediction days using regex patterns"""
    
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

def stock_forecast_agent(prompt: str) -> Union[Tuple[pd.DataFrame, pd.DataFrame, Dict], Dict[str, Tuple[pd.DataFrame, pd.DataFrame, Dict]]]:
    """AI agent for forecasting stocks based on natural language prompts"""
    print(f"Processing forecast query: {prompt}")
    
    # Get language from environment for localization (defaults to English)
    language = os.environ.get("LANGUAGE", "en")
    
    # Flag for advanced analysis (technical indicators)
    use_advanced = os.environ.get("USE_ADVANCED_ANALYSIS", "true").lower() == "true"
    
    # Try to get the response from ollama
    try:
        if OLLAMA_AVAILABLE:
            # Prompt the LLM to extract stock symbols and prediction days
            ollama_prompt = f"""
            Extract stock symbols and prediction days from this query: "{prompt}"
            
            Format your response as a JSON object with the keys 'stock_symbols' (an array of strings) 
            and 'prediction_days' (an integer). For example:
            {{
                "stock_symbols": ["AAPL", "MSFT"],
                "prediction_days": 10
            }}
            
            If the query is about Vietnamese stocks, add .VN to the symbols unless they already have a suffix.
            If no prediction days are specified, use 10 days as the default.
            If no stock symbols are found, try to infer from context. For example, 'Apple' → 'AAPL'.
            """
            
            ollama_result = get_ollama_response(ollama_prompt)
            
            if ollama_result and hasattr(ollama_result, 'get') and 'content' in ollama_result:
                try:
                    # Try to extract JSON from the response
                    import json
                    content = ollama_result['content']
                    
                    # Find JSON block if present
                    json_match = re.search(r'({.*})', content, re.DOTALL)
                    if json_match:
                        extracted_json = json_match.group(1)
                        result = json.loads(extracted_json)
                        
                        stock_symbols = result.get('stock_symbols', [])
                        prediction_days = result.get('prediction_days', 10)
                        
                        print(f"Ollama extracted: {stock_symbols}, {prediction_days} days")
                        
                        # Set batch processing for multiple stocks
                        batch_processing = len(stock_symbols) > 1
                        
                        # For multiple stocks, process in parallel with batch processing
                        if len(stock_symbols) > 1 and batch_processing:
                            return process_multiple_stocks(stock_symbols, prediction_days)
                        
                        # For single stock or without batch processing
                        if stock_symbols:
                            forecast, historical_data, metrics = get_stock_forecast(stock_symbols[0], prediction_days)
                            
                            # Generate summary with the correct language
                            summary = generate_forecast_summary(forecast, historical_data, metrics, language)
                            metrics['forecast_summary'] = summary
                            
                            # Return forecast results
                            return forecast, historical_data, metrics
                except Exception as json_error:
                    print(f"Error processing Ollama JSON response: {str(json_error)}")
    except Exception as e:
        print(f"Ollama error: {str(e)}. Using fallback method.")
    
    # Fallback method using regex if Ollama fails
    print("Using fallback extraction with regex")
    result = extract_stock_info_with_regex(prompt)
    
    # Set batch processing for multiple stocks
    batch_processing = len(result['stock_symbols']) > 1
    
    # For multiple stocks, process in parallel with batch processing
    if len(result['stock_symbols']) > 1 and batch_processing:
        return process_multiple_stocks(result['stock_symbols'], result['prediction_days'])
    
    # For single stock or without batch processing
    if result['stock_symbols']:
        forecast, historical_data, metrics = get_stock_forecast(result['stock_symbols'][0], result['prediction_days'])
        
        # Generate summary with the correct language
        summary = generate_forecast_summary(forecast, historical_data, metrics, language)
        metrics['forecast_summary'] = summary
        
        # Return forecast results
        return forecast, historical_data, metrics
    else:
        raise ValueError("No valid stock symbols found in the prompt") 