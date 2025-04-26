#!/usr/bin/env python3

# Script to create a fixed version of app.py with correct indentation
import re

def fix_app_py():
    input_file = '/Users/theanh/Downloads/test/stock_forecast_ai_agent/app.py'
    output_file = '/Users/theanh/Downloads/test/stock_forecast_ai_agent/app_fixed.py'
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Fix the get_indicator_signal function
    for i in range(len(lines)):
        if "def get_indicator_signal(indicator, value):" in lines[i]:
            # Found the function, now insert the fixed version
            start_index = i
            # Find the end of the function (next def)
            end_index = None
            for j in range(i + 1, len(lines)):
                if "def multi_stock_analysis_page():" in lines[j]:
                    end_index = j
                    break
            
            if end_index:
                # Replace the function with a corrected version
                fixed_function = """def get_indicator_signal(indicator, value):
    \"\"\"Get the signal status for a technical indicator\"\"\"
    result = {"icon": "", "text": "", "color": "#757575"}
    
    try:
        # Handle different indicators
        if indicator == "RSI":
            value = float(value)
            if value >= 70:
                result = {"icon": "↗️", "text": "Overbought", "color": "#F44336"}
            elif value <= 30:
                result = {"icon": "↘️", "text": "Oversold", "color": "#4CAF50"}
            else:
                result = {"icon": "↔️", "text": "Neutral", "color": "#FF9800"}
        
        elif indicator == "MACD":
            if "bullish" in str(value).lower():
                result = {"icon": "↗️", "text": "Bullish", "color": "#4CAF50"}
            elif "bearish" in str(value).lower():
                result = {"icon": "↘️", "text": "Bearish", "color": "#F44336"}
            else:
                result = {"icon": "↔️", "text": "Neutral", "color": "#FF9800"}
        
        elif "SMA" in indicator or "EMA" in indicator:
            if "above" in str(value).lower():
                result = {"icon": "↗️", "text": "Bullish", "color": "#4CAF50"}
            elif "below" in str(value).lower():
                result = {"icon": "↘️", "text": "Bearish", "color": "#F44336"}
            else:
                result = {"icon": "↔️", "text": "Neutral", "color": "#FF9800"}
        
        elif indicator == "Bollinger Bands":
            if "upper" in str(value).lower():
                result = {"icon": "↗️", "text": "Overbought", "color": "#F44336"}
            elif "lower" in str(value).lower():
                result = {"icon": "↘️", "text": "Oversold", "color": "#4CAF50"}
            else:
                result = {"icon": "↔️", "text": "Middle", "color": "#FF9800"}
        
        elif indicator == "Stochastic" or indicator == "MFI":
            try:
                if isinstance(value, str) and len(value.split()) > 0:
                    value = float(value.split()[0])
                else:
                    value = float(value)
                    
                if value >= 80:
                    result = {"icon": "↗️", "text": "Overbought", "color": "#F44336"}
                elif value <= 20:
                    result = {"icon": "↘️", "text": "Oversold", "color": "#4CAF50"}
                else:
                    result = {"icon": "↔️", "text": "Neutral", "color": "#FF9800"}
            except:
                result = {"icon": "ℹ️", "text": "Info", "color": "#1E88E5"}
        
        elif indicator == "ATR" or indicator == "Standard Deviation":
            # These are volatility indicators without direct signal interpretation
            result = {"icon": "ℹ️", "text": "Info", "color": "#1E88E5"}
            
        elif "Trend" in indicator:
            # For trend indicators that contain text descriptions
            if "above" in str(value).lower():
                result = {"icon": "↗️", "text": "Bullish", "color": "#4CAF50"}
            elif "below" in str(value).lower():
                result = {"icon": "↘️", "text": "Bearish", "color": "#F44336"}
            else:
                result = {"icon": "↔️", "text": "Neutral", "color": "#FF9800"}
        
        else:
            # Generic handling for other indicators
            result = {"icon": "ℹ️", "text": "Info", "color": "#1E88E5"}
            
    except Exception as e:
        # If cannot parse, return default
        print(f"Error processing indicator {indicator}: {e}")
        result = {"icon": "ℹ️", "text": "Info", "color": "#1E88E5"}
        
    return result
"""
                # Replace the function
                lines[start_index:end_index] = [fixed_function]
                break

    # Fix multi_stock_analysis_page function try-except blocks
    for i in range(len(lines)):
        if "def multi_stock_analysis_page():" in lines[i]:
            start_index = i
            # Find all try/except blocks in this function and fix them
            # This is a simplified approach - in a real scenario, you'd need more robust parsing
            
            # Let's just fix the known problematic sections
            for j in range(start_index, len(lines)):
                # Fix specific line where there's a try without except
                if "    if prompt:" in lines[j]:
                    # The try block without except - add a proper except
                    lines[j+1] = "        try:\n"
                    
                # Fix the else indentation at line 1426
                if "            else:" in lines[j] and "# Multiple stocks case" in lines[j+1]:
                    lines[j] = "                else:\n"
                    
                # Fix exception handler indentation
                if "        except Exception as e:" in lines[j]:
                    lines[j] = "        except Exception as e:\n"
                    if j+1 < len(lines) and "st.error" in lines[j+1]:
                        lines[j+1] = "            st.error(f\"{t.get('multi_stock_error', 'Error processing multi-stock forecast:')} {str(e)}\")\n"
                
                # Stop when we reach the next function
                if j > start_index and "def " in lines[j] and "():" in lines[j]:
                    break
    
    # Fix the benchmark page indentation issues
    for i in range(len(lines)):
        if "def benchmark_page():" in lines[i]:
            start_index = i
            
            for j in range(start_index, len(lines)):
                # Fix specific indentation issues in the benchmark page
                if "                    else:" in lines[j] and "# Display instructions" in lines[j+1]:
                    lines[j] = "        else:\n"
                
                # Stop when we reach the next function or end of file
                if j > start_index and j+1 < len(lines) and "def " in lines[j+1] and "():" in lines[j+1]:
                    break
    
    # Fix the sidebar section indentation
    for i in range(len(lines)):
        if "    # Toggle language and update environment variable" in lines[i]:
            lines[i+1] = "                st.session_state['language'] = 'vi' if st.session_state['language'] == 'en' else 'en'\n"
            lines[i+2] = "                os.environ[\"LANGUAGE\"] = st.session_state['language']\n"
            lines[i+3] = "                st.rerun()\n"
    
    # Write the fixed content to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"Fixed file saved as {output_file}")

if __name__ == "__main__":
    fix_app_py() 