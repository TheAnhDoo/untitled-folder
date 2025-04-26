import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import torch
import traceback
from datetime import datetime, timedelta

# Initialize Streamlit app - MUST be the first st.* call
st.set_page_config(
    page_title="Stock Forecast AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'language' not in st.session_state:
    st.session_state['language'] = 'en'

# Custom CSS for styling
st.markdown("""
<style>
.ai-agent-header {
    display: flex;
    align-items: center;
    margin-bottom: 1.5rem;
}
.ai-agent-logo {
    background: linear-gradient(135deg, #1E88E5, #7B1FA2);
    width: 60px;
    height: 60px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2rem;
    color: white;
    margin-right: 1rem;
}
.ai-agent-title h1 {
    margin: 0;
    padding: 0;
    font-size: 1.8rem;
    font-weight: 600;
}
.ai-agent-title p {
    margin: 0;
    padding: 0;
    opacity: 0.8;
}
.stCard {
    border-radius: 10px;
    border: 1px solid #e0e0e0;
    background-color: white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
    overflow: hidden;
}
.pulse-element {
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% {
        box-shadow: 0 0 0 0 rgba(30, 136, 229, 0.4);
    }
    70% {
        box-shadow: 0 0 0 10px rgba(30, 136, 229, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(30, 136, 229, 0);
    }
}
.language-toggle {
    background-color: #f5f5f5;
    border-radius: 20px;
    padding: 8px 15px;
    display: flex;
    align-items: center;
    cursor: pointer;
    margin-bottom: 15px;
    transition: all 0.2s ease;
}
.language-toggle:hover {
    background-color: #e0e0e0;
}
.language-icon {
    margin-right: 8px;
}
.metrics-card {
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 16px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    margin-bottom: 20px;
    border-left: 4px solid #4CAF50;
}
.forecast-accuracy {
    font-weight: bold;
    color: #4CAF50;
    font-size: 1.2rem;
}
</style>
""", unsafe_allow_html=True)

# Import our custom modules
from modules.core import check_ollama_installation, check_device_compatibility
from modules.agent import stock_forecast_agent
from modules.ui import (
    display_metrics_dashboard, display_technical_indicators, 
    plot_forecast_chart, show_supported_tickers, 
    display_stock_error, sidebar_data_source_info,
    get_translations
)

def init_session_state():
    # Initialize session state variables if they don't exist
    if 'language' not in st.session_state:
        st.session_state.language = 'en'
    if 'page' not in st.session_state:
        st.session_state.page = 'forecast'

# Initialize session state
init_session_state()

# Custom CSS
st.markdown("""
<style>
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.stTextArea label, .stTextInput label {
    font-size: 1.1rem;
    color: #555;
    font-weight: 600;
}
.app-header {
    text-align: center;
    margin-bottom: 2rem;
}
.stTextArea textarea {
    border-radius: 10px;
    border: 1px solid #ddd;
    padding: 10px;
    font-family: 'Roboto', sans-serif;
    font-size: 1rem;
    box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.05);
}
.forecast-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    margin-right: 5px;
    font-size: 0.8rem;
    font-weight: 500;
}
.forecast-tag-bullish {
    background-color: rgba(0, 200, 0, 0.2);
    color: rgb(0, 100, 0);
}
.forecast-tag-bearish {
    background-color: rgba(200, 0, 0, 0.2);
    color: rgb(100, 0, 0);
}
.forecast-tag-neutral {
    background-color: rgba(100, 100, 100, 0.2);
    color: rgb(50, 50, 50);
}
.metrics-card {
    background-color: #f7f9fc;
    border-radius: 6px;
    padding: 16px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    margin-bottom: 16px;
    border-left: 4px solid #4CAF50;
}
.forecast-accuracy {
    font-weight: bold;
    color: #4CAF50;
    font-size: 1.2rem;
}
.technical-summary {
    background-color: #f8f9fa;
    border: 1px solid #eaecef;
    border-radius: 6px;
    padding: 15px;
    margin-bottom: 20px;
}
.technical-summary h3 {
    margin-top: 0;
    margin-bottom: 10px;
    color: #333;
}
.technical-summary p {
    margin-bottom: 8px;
    line-height: 1.4;
}
</style>
""", unsafe_allow_html=True)

def main_forecast_page():
    """Main forecast page for stock analysis (both single and multiple stocks)"""
    # Get the current language translations
    lang = st.session_state['language']
    t = get_translations(lang)
    
    # Create header
    st.markdown(f"""
    <div class="ai-agent-header">
        <div class="ai-agent-logo pulse-element">📊</div>
        <div class="ai-agent-title">
            <h1>{t.get("forecast_title", "AI Stock Forecaster")}</h1>
            <p>{t.get("forecast_subtitle", "Predict single or multiple stocks with natural language")}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Custom CSS for better metrics
    st.markdown("""
    <style>
    .metrics-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 4px solid #4CAF50;
    }
    .forecast-accuracy {
        font-weight: bold;
        color: #4CAF50;
        font-size: 1.2rem;
    }
    .forecast-range-note {
        background-color: #f0f7ff;
        border-left: 4px solid #1E88E5;
        padding: 10px 15px;
        border-radius: 5px;
        margin: 10px 0;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Natural language input - direct without wrapper
    prompt_label = t.get("forecast_prompt", "Ask for a stock forecast in natural language")
    
    # Show instructions above the input field instead of using placeholder
    st.markdown(f"**{prompt_label}**")
    
    # Show instruction text
    st.markdown(f"*{t.get('prompt_instruction', 'Enter a question like: "Forecast Apple stock for the next 7 days" or "Compare MSFT, AMZN and GOOGL"')}*")
    
    # Input field without placeholder
    prompt = st.text_area(
        label="",
        label_visibility="collapsed",
        height=100
    )
    
    # Advanced options in a collapsible area
    with st.expander(t.get("advanced_options", "Advanced Options")):
        col1, col2 = st.columns(2)
        
        with col1:
            confidence_interval = st.slider(
                t.get("confidence_interval", "Confidence Interval"),
                min_value=50,
                max_value=95,
                value=80,
                step=5,
                help=t.get("confidence_help", "The range that contains the true value with selected probability")
            )
            
            # Store in session state to persist
            st.session_state['confidence_interval'] = confidence_interval
            
        with col2:
            use_advanced = st.checkbox(
                t.get("advanced_analysis", "Advanced Analysis"),
                value=True,
                help=t.get("advanced_help", "Include technical indicators and detailed analysis")
            )
            
            # Add use all historical data option
            use_all_data = st.checkbox(
                t.get("use_all_data", "Use All Historical Data"),
                value=False,
                help=t.get("use_all_data_help", "Use all available historical data instead of just the last 6 months for potentially better predictions")
            )
            
            # Store in session state to persist
            st.session_state['use_all_data'] = use_all_data
            
            # Show supported tickers button
            if st.button(t.get("show_tickers", "Show Supported Tickers")):
                show_supported_tickers()
    
    # Forecast button
    forecast_button = st.button(
        t.get("generate_forecast", "Generate Forecast"),
        type="primary",
        use_container_width=True
    )
    
    # Instructions when no prompt is entered
    if not prompt and not forecast_button:
        st.info(t.get("forecast_instructions", "Enter a natural language query above to forecast one or multiple stocks. You can ask about a single stock or compare multiple stocks in the same query."))
    
    if prompt:
        try:
            # Custom loader
            loader_placeholder = st.empty()
            loader_placeholder.markdown(f"""
            <div style="display: flex; justify-content: center; margin: 2rem 0;">
                <div style="display: flex; flex-direction: column; align-items: center;">
                    <div class="ai-agent-logo pulse-element" style="width: 80px; height: 80px; font-size: 2.5rem;">🧠</div>
                    <p style="margin-top: 1rem;">{t.get("processing", "Processing your request...")}</p>
                    <p style="font-size: 0.8rem; opacity: 0.7;">{t.get("analyzing_market", "Analyzing market data and generating forecast...")}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Set environment variable for advanced analysis
            os.environ["USE_ADVANCED_ANALYSIS"] = "true" if use_advanced else "false"
            
            # Set environment variable for using all data
            os.environ["USE_ALL_HISTORICAL_DATA"] = "true" if st.session_state.get('use_all_data', False) else "false"
            
            # Call the AI agent to process the prompt
            results = stock_forecast_agent(prompt)
            
            # Remove loader once processing is complete
            loader_placeholder.empty()
            
            # Check if we got a dictionary of results (multiple stocks) or a single result
            if isinstance(results, dict):
                # Multiple stocks case
                # We got a dictionary of results: {symbol: (forecast, historical, metrics)}
                symbols = list(results.keys())
                num_stocks = len(symbols)
                
                # Success message with animation
                st.markdown(f"""
                <div style="text-align: center; margin: 2rem 0;">
                    <div style="display: inline-block; background-color: #E8F5E9; padding: 0.5rem 2rem; border-radius: 50px;">
                        <span style="color: #4CAF50; font-weight: bold;">✅ {t.get("multi_stock_success", "Successfully analyzed")} {num_stocks} {t.get("stocks", "stocks")}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Add a note explaining the forecast ranges
                st.markdown(f"""
                <div class="forecast-range-note">
                    <strong>📈 {t.get("forecast_range_note_title", "Understanding the Forecast")}</strong>
                    <p>{t.get("forecast_range_note", "The forecast shows three lines: the middle line is the most likely price trajectory, while the shaded area represents the range where prices are expected to fall with " + str(confidence_interval) + "% confidence.")}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create tabs for each stock
                st.subheader(t.get("comparison_results", "Comparison Results"))
                
                # Create tabs for each stock
                stock_tabs = st.tabs(symbols)
                
                # For each stock, display the forecast
                for i, (symbol, tab) in enumerate(zip(symbols, stock_tabs)):
                    # Check if the result is an error message (string) or valid data (tuple)
                    if isinstance(results[symbol], str):
                        with tab:
                            error_message = results[symbol]
                            display_stock_error(symbol, error_message)
                        continue
                    
                    # If it's a valid result, unpack it
                    forecast, historical_data, metrics = results[symbol]
                    
                    # Skip stocks with missing data
                    if forecast is None or historical_data is None:
                        with tab:
                            error_message = "Unable to generate forecast for this stock"
                            display_stock_error(symbol, error_message)
                        continue
                    
                    # Format dates for forecast
                    last_date = historical_data['ds'].iloc[-1]
                    forecast_dates = [last_date + timedelta(days=i+1) for i in range(len(forecast))]
                    
                    with tab:
                        # Create subtabs for each stock
                        tab_labels = [
                            t.get("chart_tab", "Chart"),
                            t.get("data_tab", "Data"),
                            t.get("metrics_tab", "Metrics"),
                            t.get("technical_indicators", "Technical Indicators")
                        ]
                        
                        s_chart_tab, s_data_tab, s_metrics_tab, s_technical_tab = st.tabs(tab_labels)
                        
                        with s_chart_tab:
                            # Plot individual forecast chart
                            plot_forecast_chart(
                                historical_data, 
                                forecast, 
                                forecast_dates, 
                                metrics.get('actual_symbol', symbol),
                                confidence_interval
                            )
                        
                        with s_data_tab:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader(t.get("historical_data_title", "Historical Data (Last 10 days)"))
                                historical_display = historical_data.tail(10).reset_index(drop=True)
                                historical_display = historical_display.rename(columns={
                                    'ds': t.get('date', 'Date'),
                                    'y': t.get('price', 'Price')
                                })
                                st.dataframe(historical_display, use_container_width=True)
                            
                            with col2:
                                st.subheader(t.get("forecast_data", "Forecast Data"))
                                forecast_df = pd.DataFrame({
                                    t.get("date", "Date"): forecast_dates,
                                    t.get("lower_bound", "Lower Bound"): forecast["forecast_lower"].round(2),
                                    t.get("median_forecast", "Median Forecast"): forecast["forecast_median"].round(2),
                                    t.get("upper_bound", "Upper Bound"): forecast["forecast_high"].round(2)
                                })
                                st.dataframe(forecast_df, use_container_width=True)
                        
                        with s_metrics_tab:
                            if metrics:
                                display_metrics_dashboard(metrics)
                        
                        with s_technical_tab:
                            if 'technical_indicators' in metrics:
                                display_technical_indicators(metrics)
                            else:
                                # Create technical indicators on-the-fly if not in metrics
                                st.info(t.get("no_technical_data", "No technical indicator data available for this forecast."))
            
            else:
                # Single stock case
                forecast, historical_data, metrics = results
                
                # Add a note explaining the forecast ranges
                st.markdown(f"""
                <div class="forecast-range-note">
                    <strong>📈 {t.get("forecast_range_note_title", "Understanding the Forecast")}</strong>
                    <p>{t.get("forecast_range_note", "The forecast shows three lines: the middle line is the most likely price trajectory, while the shaded area represents the range where prices are expected to fall with " + str(confidence_interval) + "% confidence.")}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Format dates for forecast
                last_date = historical_data['ds'].iloc[-1]
                forecast_dates = [last_date + timedelta(days=i+1) for i in range(len(forecast))]
                
                # Create tabs for different views
                tab_labels = [
                    t.get("chart_tab", "Forecast Chart"),
                    t.get("data_tab", "Data"),
                    t.get("metrics_tab", "Metrics"),
                    t.get("technical_indicators", "Technical Indicators")
                ]
                
                chart_tab, data_tab, metrics_tab, technical_tab = st.tabs(tab_labels)
                
                with chart_tab:
                    # Plot the forecast chart
                    plot_forecast_chart(
                        historical_data, 
                        forecast, 
                        forecast_dates, 
                        metrics.get('actual_symbol', metrics.get('requested_symbol', 'Stock')),
                        confidence_interval
                    )
                    
                with data_tab:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader(t.get("historical_data_title", "Historical Data (Last 10 days)"))
                        historical_display = historical_data.tail(10).reset_index(drop=True)
                        # Rename columns for display
                        historical_display = historical_display.rename(columns={
                            'ds': t.get('date', 'Date'),
                            'y': t.get('price', 'Price')
                        })
                        st.dataframe(historical_display, use_container_width=True)
                    
                    with col2:
                        st.subheader(t.get("forecast_data", "Forecast Data"))
                        forecast_df = pd.DataFrame({
                            t.get("date", "Date"): forecast_dates,
                            t.get("lower_bound", "Lower Bound"): forecast["forecast_lower"].round(2),
                            t.get("median_forecast", "Median Forecast"): forecast["forecast_median"].round(2),
                            t.get("upper_bound", "Upper Bound"): forecast["forecast_high"].round(2)
                        })
                        st.dataframe(forecast_df, use_container_width=True)
                
                with metrics_tab:
                    if metrics:
                        display_metrics_dashboard(metrics)
                
                with technical_tab:
                    if 'technical_indicators' in metrics:
                        display_technical_indicators(metrics)
                    else:
                        # Create technical indicators on-the-fly if not in metrics
                        st.info(t.get("no_technical_data", "No technical indicator data available for this forecast."))
        
        except Exception as e:
            st.error(f"{t.get('forecast_error', 'Error generating forecast:')}: {str(e)}")
            
            # Extract stock symbol from error message if possible
            error_text = str(e)
            stock_symbol = None
            
            if "Could not fetch data for " in error_text:
                try:
                    stock_symbol = error_text.split("Could not fetch data for ")[1].split(" ")[0]
                except:
                    pass
            elif "Insufficient data for " in error_text:
                try:
                    stock_symbol = error_text.split("Insufficient data for ")[1].split(" ")[0]
                except:
                    pass
            
            # Show custom error message with helpful suggestions
            if stock_symbol:
                display_stock_error(stock_symbol, error_text)
            
            # Show detailed error information in expander
            with st.expander(t.get("error_details", "Error Details")):
                st.code(traceback.format_exc(), language="python")

def about_page():
    """About page with information about the app"""
    lang = st.session_state['language']
    t = get_translations(lang)
    
    # Create header
    st.markdown(f"""
    <div class="ai-agent-header">
        <div class="ai-agent-logo pulse-element">ℹ️</div>
        <div class="ai-agent-title">
            <h1>{t.get("about_title", "About AI Stock Forecaster")}</h1>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Overview section
    st.markdown(f"""
    <div class="stCard" style="padding: 1.5rem;">
        <h3>{t.get("overview", "Overview")}</h3>
        <p>{t.get("overview_text", "AI Stock Forecaster is an advanced tool that combines the power of large language models and time-series forecasting to predict stock prices. The application is built on a modern tech stack with a focus on performance and user experience.")}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key features section
    st.markdown(f"""
    <div class="stCard" style="padding: 1.5rem;">
        <h3>{t.get("key_features", "Key Features")}</h3>
        <ul>
            <li>{t.get("feature_1", "Natural Language Interface: Simply ask for forecasts in plain language")}</li>
            <li>{t.get("feature_2", "Advanced Forecasting Model: Uses Amazon's Chronos model for probabilistic forecasting")}</li>
            <li>{t.get("feature_3", "Multiple Stock Analysis: Compare forecasts across different companies")}</li>
            <li>{t.get("feature_4", "Technical Indicators: Evaluate stocks with industry-standard technical indicators")}</li>
            <li>{t.get("feature_5", "Interactive Visualizations: Explore forecasts with detailed interactive charts")}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Main application
if __name__ == '__main__':
    # Create sidebar for settings and information
    with st.sidebar:
        # Modern sidebar header with logo
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <div style="background: linear-gradient(135deg, #1E88E5, #7B1FA2); width: 50px; height: 50px; border-radius: 10px; display: flex; align-items: center; justify-content: center;">
                <span style="color: white; font-size: 1.8rem;">📈</span>
            </div>
            <div style="margin-left: 10px;">
                <h3 style="margin: 0; padding: 0; font-size: 1.2rem;">AI Stock Agent</h3>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Get current language
        lang = st.session_state['language']
        t = get_translations(lang)
        
        st.markdown(f"### {t.get('settings', 'Settings')}")
        
        # Modern language selector
        st.markdown(f"**{t.get('language', 'Language')}:**")
        
        # Create a modern language toggle button
        current_lang = "🇺🇸 English" if lang == 'en' else "🇻🇳 Tiếng Việt"
        switch_to = "🇻🇳 Tiếng Việt" if lang == 'en' else "🇺🇸 English"
        
        # Actual language toggle button
        if st.button("Switch Language", key="lang_switch_btn", help="Toggle between English and Vietnamese"):
            # Toggle language and update environment variable
            st.session_state['language'] = 'vi' if st.session_state['language'] == 'en' else 'en'
            os.environ["LANGUAGE"] = st.session_state['language']
            st.rerun()
        
        # Navigation with improved styling
        st.markdown(f"### {t.get('navigate', 'Navigation')}")
        
        # Format navigation options with icons
        nav_options = {
            t.get("forecast", "Stock Forecast"): "📊",
            t.get("about", "About"): "ℹ️"
        }
        
        page = st.radio(
            "",
            list(nav_options.keys()),
            format_func=lambda x: f"{nav_options[x]} {x}",
            label_visibility="collapsed"
        )
        
        # System information
        with st.expander(t.get("system_info", "System Information")):
            # Device info
            device_info = check_device_compatibility()
            
            # Show device being used
            if "gpu" in device_info.lower():
                st.markdown(f"🚀 **{device_info}**")
            else:
                st.markdown(f"💻 **{device_info}**")
                
            # Check Ollama
            ollama_status = check_ollama_installation()
            st.markdown(f"**Ollama:** {ollama_status}")
            
            # Show basic system info
            st.markdown(f"""
            **Python:** {sys.version.split()[0]}  
            **OS:** {os.name.upper()}
            """)
        
        # Add data source info
        sidebar_data_source_info()
        
        # Add a small footer
        st.markdown("""
        <div style="position: fixed; bottom: 20px; left: 20px; opacity: 0.7; font-size: 0.8rem;">
            AI Stock Forecaster v2.0<br>© 2024
        </div>
        """, unsafe_allow_html=True)
    
    # Render selected page
    if page == t.get("forecast", "Stock Forecast"):
        main_forecast_page()
    else:
        about_page()