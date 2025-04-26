#!/usr/bin/env python3

# This script fixes the indentation issues in app.py

with open('app.py', 'r') as f:
    lines = f.readlines()

# Fixed fragment for the RSI section
rsi_lines = """        # Handle different indicators
        if indicator == "RSI":
            value = float(value)
            if value >= 70:
                result = {"icon": "↗️", "text": "Overbought", "color": "#F44336"}
            elif value <= 30:
                result = {"icon": "↘️", "text": "Oversold", "color": "#4CAF50"}
            else:
                result = {"icon": "↔️", "text": "Neutral", "color": "#FF9800"}
        
        elif indicator == "MACD":
"""

# Fixed fragment for the Stochastic/MFI section
stoch_lines = """        elif indicator == "Stochastic" or indicator == "MFI":
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
"""

# Replace the lines in the file
rsi_start = 1206  # Line number where RSI section starts
rsi_end = 1221    # Line number where RSI section ends
stoch_start = 1239  # Line number where Stochastic section starts
stoch_end = 1259    # Line number where Stochastic section ends

# Split the replacement blocks into lines
rsi_replace = rsi_lines.splitlines(True)
stoch_replace = stoch_lines.splitlines(True)

# Replace the problematic sections
lines[rsi_start:rsi_end] = rsi_replace
lines[stoch_start:stoch_end] = stoch_replace

# Write the file back
with open('app.py', 'w') as f:
    f.writelines(lines)

print("Fixed indentation issues in app.py")
