import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from datetime import datetime, timedelta
from typing import Dict

def show_supported_tickers():
    """Show supported ticker examples"""
    # Create tabs for different types of supported tickers
    tab1, tab2, tab3 = st.tabs(["U.S. Stocks", "Vietnamese Stocks", "Indices"])
    
    with tab1:
        st.markdown("""
        **Examples**: AAPL, MSFT, AMZN, GOOGL, TSLA, META, NVDA, JPM, V, WMT, DIS, NFLX, INTC, AMD, etc.
        
        You can use nearly any U.S. stock symbol traded on major exchanges (NYSE, NASDAQ).
        """)
    
    with tab2:
        st.markdown("""
        **Examples**: VIC.VN, VNM.VN, FPT.VN, VCB.VN, HPG.VN, MSN.VN, etc.
        
        Vietnamese stocks should use the format SYMBOL.VN. 
        Alternative formats like SYMBOL:VN or SYMBOL.HO may also work.
        """)
    
    with tab3:
        st.markdown("""
        **Examples**: ^GSPC (S&P 500), ^DJI (Dow Jones), ^IXIC (NASDAQ), ^VNINDEX (VN-Index)
        
        Most major global indices are supported.
        """)

def display_stock_error(stock_symbol, error_text):
    """Display custom error message for stock retrieval problems"""
    st.error(f"Error processing {stock_symbol}: {error_text}")
    
    # Show common fixes
    with st.expander("Troubleshooting suggestions"):
        st.markdown("""
        ### Common fixes:
        
        1. **Check symbol format**
           - US stocks: Use standard ticker symbols (e.g., AAPL, MSFT)
           - Vietnamese stocks: Use .VN suffix (e.g., VIC.VN, FPT.VN)
           - Indices: Include ^ prefix (e.g., ^GSPC, ^VNINDEX)
           
        2. **Try alternative symbols**
           - Some companies have different ticker symbols on different exchanges
           - For Vietnamese stocks, try with or without the .VN suffix
           
        3. **Data availability**
           - New or small companies may have limited historical data
           - Some exchanges or markets may have delayed or restricted data
           
        4. **API status**
           - Our data providers may have temporary service interruptions
           - Try again later if this is a temporary issue
        """)
    
    # Provide suggestions for similar symbols
    if '.VN' not in stock_symbol and len(stock_symbol) <= 4:
        st.info(f"If you're looking for a Vietnamese stock, try {stock_symbol}.VN instead")
    
    if stock_symbol.lower() in ['sp500', 'sp', 's&p']:
        st.info("For the S&P 500 index, try using ^GSPC")
    
    if stock_symbol.lower() in ['dow', 'dowjones']:
        st.info("For the Dow Jones Industrial Average, try using ^DJI")
    
    if stock_symbol.lower() in ['nasdaq']:
        st.info("For the NASDAQ Composite Index, try using ^IXIC")
    
    if stock_symbol.lower() in ['vnindex', 'vn-index', 'vn']:
        st.info("For the Vietnam Index, try using ^VNINDEX")

def plot_forecast_chart(historical_data, forecast_data, forecast_dates, stock_symbol, confidence_interval=80):
    """Plot interactive forecast chart using Plotly"""
    
    # Get language setting
    lang = st.session_state.get('language', 'en')
    t = get_translations(lang)
    
    # Prepare data
    historical_dates = historical_data['ds']
    historical_prices = historical_data['y']
    
    forecast_values = forecast_data['forecast_median'].values
    forecast_lower = forecast_data['forecast_lower'].values
    forecast_upper = forecast_data['forecast_high'].values
    
    # Create plot
    fig = go.Figure()
    
    # Add historical line
    fig.add_trace(go.Scatter(
        x=historical_dates,
        y=historical_prices,
        mode='lines',
        name=t.get('historical', 'Historical'),
        line=dict(color='#1E88E5', width=2)
    ))
    
    # Add forecast line
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode='lines',
        name=t.get('forecast', 'Forecast'),
        line=dict(color='#7CB342', width=2, dash='dash')
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_dates + forecast_dates[::-1],
        y=forecast_upper.tolist() + forecast_lower.tolist()[::-1],
        fill='toself',
        fillcolor='rgba(124, 179, 66, 0.2)',
        line=dict(color='rgba(124, 179, 66, 0)'),
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Configure layout
    last_price = historical_prices.iloc[-1]
    forecast_last = forecast_values[-1]
    
    # Calculate the overall range to set reasonable y-axis limits
    all_values = np.concatenate([historical_prices.values, forecast_values, forecast_lower, forecast_upper])
    y_min = np.min(all_values) * 0.95
    y_max = np.max(all_values) * 1.05
    
    # Set up chart colors based on forecast direction
    forecast_change = (forecast_last - last_price) / last_price
    theme_color = '#4CAF50' if forecast_change >= 0 else '#F44336'
    
    # Customize the layout
    fig.update_layout(
        title=f"{stock_symbol} - {t.get('price_forecast', 'Price Forecast')}",
        xaxis_title=t.get('date', 'Date'),
        yaxis_title=t.get('price', 'Price'),
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(range=[y_min, y_max]),
        margin=dict(l=10, r=10, t=50, b=10),
        height=450,
        template='plotly_white'
    )
    
    # Add annotations for confidence interval
    fig.add_annotation(
        text=f"{confidence_interval}% {t.get('confidence_interval', 'Confidence Interval')}",
        xref="paper", yref="paper",
        x=0.01, y=0.01,
        showarrow=False,
        font=dict(size=12, color="gray"),
        align="left"
    )
    
    # Calculate and display forecast metrics
    percent_change = (forecast_last - last_price) / last_price * 100
    change_text = f"{percent_change:.2f}% {t.get('projected_change', 'Projected Change')}"
    
    fig.add_annotation(
        text=change_text,
        xref="paper", yref="paper",
        x=0.99, y=0.01,
        showarrow=False,
        font=dict(size=12, color=theme_color),
        align="right"
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Add a small disclaimer
    st.markdown(
        f"<div style='text-align: center; color: gray; font-size: 0.8rem;'>{t.get('forecast_disclaimer', 'Forecast disclaimer: This is a probabilistic forecast and actual results may vary.')}</div>",
        unsafe_allow_html=True
    )

def display_metrics_dashboard(metrics: Dict):
    """Display forecast metrics in a visually appealing dashboard"""
    
    # Get language setting
    lang = st.session_state.get('language', 'en')
    t = get_translations(lang)
    
    # Check if metrics exist
    if not metrics:
        st.warning(t.get("no_metrics", "No metrics available for this forecast."))
        return
    
    # Create layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        accuracy = metrics.get('direction_accuracy', 0) * 100
        st.markdown(f"""
        <div class="metrics-card">
            <div style="font-size: 0.85rem; color: #666;">{t.get("forecast_accuracy", "Forecast Accuracy")}</div>
            <div class="forecast-accuracy">{accuracy:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        mae = metrics.get('mae', 0)
        st.markdown(f"""
        <div class="metrics-card" style="border-left-color: #1E88E5;">
            <div style="font-size: 0.85rem; color: #666;">{t.get("mean_error", "Mean Error (MAE)")}</div>
            <div style="font-weight: bold; color: #1E88E5; font-size: 1.2rem;">{mae:.3f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        volatility = metrics.get('forecast_volatility', 0)
        st.markdown(f"""
        <div class="metrics-card" style="border-left-color: #F44336;">
            <div style="font-size: 0.85rem; color: #666;">{t.get("forecast_volatility", "Forecast Volatility")}</div>
            <div style="font-weight: bold; color: #F44336; font-size: 1.2rem;">{volatility:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Additional metrics
    st.markdown(f"### {t.get('additional_metrics', 'Additional Metrics')}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        metrics_table = {
            t.get("requested_symbol", "Requested Symbol"): metrics.get('requested_symbol', 'N/A'),
            t.get("actual_symbol", "Actual Symbol Used"): metrics.get('actual_symbol', 'N/A'),
            t.get("data_source", "Data Source"): metrics.get('data_source', 'N/A'),
            t.get("prediction_days", "Prediction Days"): metrics.get('prediction_days', 'N/A'),
            t.get("rmse", "RMSE"): f"{metrics.get('rmse', 0):.4f}" if 'rmse' in metrics else "N/A",
        }
        
        # Create a DataFrame and display as a table
        df = pd.DataFrame(list(metrics_table.items()), columns=[t.get("metric", "Metric"), t.get("value", "Value")])
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    with col2:
        metrics_table2 = {
            t.get("historical_days", "Historical Days"): metrics.get('historical_days', 'N/A'),
            t.get("validation_days", "Validation Days"): metrics.get('validation_days', 'N/A'),
            t.get("validation_status", "Validation Status"): metrics.get('validation_status', 'N/A'),
            t.get("forecast_timestamp", "Forecast Timestamp"): metrics.get('forecast_timestamp', 'N/A'),
            t.get("mape", "MAPE"): f"{metrics.get('mape', 0):.2f}%" if 'mape' in metrics else "N/A",
        }
        
        # Create a DataFrame and display as a table
        df2 = pd.DataFrame(list(metrics_table2.items()), columns=[t.get("metric", "Metric"), t.get("value", "Value")])
        st.dataframe(df2, use_container_width=True, hide_index=True)
    
    # Add forecast summary from metrics if available
    if 'forecast_summary' in metrics and 'summary' in metrics['forecast_summary']:
        with st.expander(t.get("forecast_insights", "Forecast Insights"), expanded=True):
            st.markdown(metrics['forecast_summary']['summary'])

def _get_localized_signal(signal_text, translations):
    """Translate signal text based on current language"""
    signal_translations = {
        'Overbought': translations.get('overbought', 'Overbought'),
        'Oversold': translations.get('oversold', 'Oversold'),
        'Neutral': translations.get('neutral', 'Neutral'),
        'Bullish': translations.get('bullish', 'Bullish'),
        'Bearish': translations.get('bearish', 'Bearish'),
        'Info': translations.get('info', 'Info'),
        'upper': translations.get('upper_band', 'Upper Band'),
        'lower': translations.get('lower_band', 'Lower Band'),
        'middle': translations.get('middle_band', 'Middle Band')
    }
    return signal_translations.get(signal_text, signal_text)

def _format_indicator(name, value, translations):
    """Create HTML for individual indicator"""
    from modules.technical_indicators import get_indicator_signal
    
    signal = get_indicator_signal(name, value)
    localized_name = translations.get(name.lower().replace(' ', '_'), name)
    localized_signal = _get_localized_signal(signal['text'], translations)
    
    return f"""
    <div class="indicator-card">
        <div class="indicator-name">{localized_name}</div>
        <div class="indicator-value" style="color: {signal['color']};">
            {signal['icon']} {value} ({localized_signal})
        </div>
    </div>
    """

def _add_technical_indicator_css():
    """Add custom CSS for indicator styling"""
    st.markdown("""
    <style>
    .indicator-section {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 20px;
    }
    .indicator-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 10px;
        min-width: 150px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .indicator-name {
        font-weight: bold;
        font-size: 0.9rem;
        margin-bottom: 5px;
    }
    .indicator-value {
        font-size: 1.0rem;
    }
    .technical-summary {
        background-color: #f1f8e9;
        border-left: 4px solid #7CB342;
        padding: 16px;
        border-radius: 4px;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

def _render_indicator_section(section_title, indicators_list, available_indicators, translations):
    """Render a section of indicators"""
    st.markdown(f"### {translations.get(section_title.lower() + '_indicators', section_title + ' Indicators')}")
    
    # Create a container to hold the indicators
    indicator_container = st.container()
    
    # Create columns for each indicator that exists in this section
    indicators_in_section = [ind for ind in indicators_list if ind in available_indicators]
    
    if indicators_in_section:
        # Create a dynamic grid with columns
        cols = st.columns(min(len(indicators_in_section), 3))  # Max 3 columns per row
        
        # Place each indicator in a column
        for i, ind in enumerate(indicators_in_section):
            col_idx = i % len(cols)
            with cols[col_idx]:
                from modules.technical_indicators import get_indicator_signal
                
                value = available_indicators[ind]
                signal = get_indicator_signal(ind, value)
                localized_name = translations.get(ind.lower().replace(' ', '_'), ind)
                localized_signal = _get_localized_signal(signal['text'], translations)
                
                st.markdown(f"""
                <div style="background-color: #f8f9fa; border-radius: 5px; padding: 10px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <div style="font-weight: bold; font-size: 0.9rem; margin-bottom: 5px;">{localized_name}</div>
                    <div style="font-size: 1.0rem; color: {signal['color']};">
                        {signal['icon']} {value} ({localized_signal})
                    </div>
                </div>
                """, unsafe_allow_html=True)

def display_technical_indicators(metrics):
    """Display technical indicators with visual representations"""
    
    # Get language setting
    lang = st.session_state.get('language', 'en')
    t = get_translations(lang)
    
    # Check if technical indicators exist
    if not metrics or 'technical_indicators' not in metrics:
        st.warning(t.get("no_indicators", "No technical indicators available for this forecast."))
        return
    
    indicators = metrics['technical_indicators']
    
    # Define which indicators to display in each section
    momentum_indicators = ['RSI', 'MACD', 'Stochastic', 'MFI']
    trend_indicators = ['SMA', 'EMA', 'SMA Trend', 'EMA Trend']
    volatility_indicators = ['Bollinger Bands', 'ATR', 'Standard Deviation']
    
    # Add custom CSS for styling
    _add_technical_indicator_css()
    
    # Display technical analysis summary at the top
    technical_summary = get_technical_summary(indicators, lang)
    st.markdown(technical_summary, unsafe_allow_html=True)
    
    # Display indicators by category
    _render_indicator_section('Momentum', momentum_indicators, indicators, t)
    _render_indicator_section('Trend', trend_indicators, indicators, t)
    _render_indicator_section('Volatility', volatility_indicators, indicators, t)

def get_technical_summary(indicators, language='en'):
    """Generate a summary of technical indicators"""
    # This is a simplified version - full implementation would include more conditions
    if not indicators:
        return "Insufficient data for technical analysis."
    
    is_english = language == 'en'
    
    # Check RSI conditions
    rsi_signal = ""
    if 'RSI' in indicators:
        try:
            rsi_value = float(indicators['RSI'].split()[0])
            if rsi_value > 70:
                rsi_signal = "overbought (potential sell signal)" if is_english else "quá mua (tín hiệu bán tiềm năng)"
            elif rsi_value < 30:
                rsi_signal = "oversold (potential buy signal)" if is_english else "quá bán (tín hiệu mua tiềm năng)"
            else:
                rsi_signal = "neutral" if is_english else "trung tính"
        except:
            pass
    
    # Check MACD
    macd_signal = ""
    if 'MACD' in indicators:
        if indicators['MACD'] == 'bullish':
            macd_signal = "bullish (uptrend)" if is_english else "tăng giá (xu hướng đi lên)"
        else:
            macd_signal = "bearish (downtrend)" if is_english else "giảm giá (xu hướng đi xuống)"
    
    # Generate summary content
    if is_english:
        summary_content = "<h3>Technical Analysis</h3>"
        
        if rsi_signal and macd_signal:
            summary_content += f"<p>Momentum indicators show the stock is currently <strong>{rsi_signal}</strong>, "
            summary_content += f"while trend indicators suggest a <strong>{macd_signal}</strong> pattern.</p>"
        elif rsi_signal:
            summary_content += f"<p>RSI indicates the stock is currently <strong>{rsi_signal}</strong>.</p>"
        elif macd_signal:
            summary_content += f"<p>MACD indicates a <strong>{macd_signal}</strong> pattern.</p>"
            
        if 'Bollinger Bands' in indicators:
            bb_position = indicators['Bollinger Bands']
            if bb_position == 'upper':
                summary_content += "<p>Price is testing the upper Bollinger Band, suggesting potential resistance.</p>"
            elif bb_position == 'lower':
                summary_content += "<p>Price is testing the lower Bollinger Band, suggesting potential support.</p>"
            else:
                summary_content += "<p>Price is within the Bollinger Bands, suggesting normal volatility.</p>"
    else:
        summary_content = "<h3>Phân Tích Kỹ Thuật</h3>"
        
        if rsi_signal and macd_signal:
            summary_content += f"<p>Các chỉ báo đà cho thấy cổ phiếu hiện đang ở trạng thái <strong>{rsi_signal}</strong>, "
            summary_content += f"trong khi các chỉ báo xu hướng cho thấy mô hình <strong>{macd_signal}</strong>.</p>"
        elif rsi_signal:
            summary_content += f"<p>RSI cho thấy cổ phiếu hiện đang <strong>{rsi_signal}</strong>.</p>"
        elif macd_signal:
            summary_content += f"<p>MACD cho thấy mô hình <strong>{macd_signal}</strong>.</p>"
            
        if 'Bollinger Bands' in indicators:
            bb_position = indicators['Bollinger Bands']
            if bb_position == 'upper':
                summary_content += "<p>Giá đang kiểm định dải Bollinger trên, cho thấy vùng kháng cự tiềm năng.</p>"
            elif bb_position == 'lower':
                summary_content += "<p>Giá đang kiểm định dải Bollinger dưới, cho thấy vùng hỗ trợ tiềm năng.</p>"
            else:
                summary_content += "<p>Giá đang nằm trong dải Bollinger, cho thấy biến động bình thường.</p>"
    
    # Wrap the content in the styled div
    return f'<div style="background-color: #f1f8e9; border-left: 4px solid #7CB342; padding: 16px; border-radius: 4px; margin-bottom: 20px;">{summary_content}</div>'

def sidebar_data_source_info():
    """Display information about data sources in sidebar"""
    # Get language setting
    lang = st.session_state.get('language', 'en')
    t = get_translations(lang)
    
    with st.expander(t.get("data_sources", "Data Sources")):
        st.markdown(f"""
        - **Yahoo Finance**: {t.get("default_source", "Default source for most stocks")}
        - **Alpha Vantage**: {t.get("secondary_source", "Secondary source for stocks not available on Yahoo Finance")}
        - **Stooq**: {t.get("fallback_source", "Fallback source for additional coverage")}
        """)
        
        st.markdown(f"<small>{t.get('data_disclaimer', 'Data is provided for informational purposes only and may be delayed.')}</small>", unsafe_allow_html=True)

def get_translations(language='en'):
    """Get translations dictionary based on language setting"""
    # Define English translations as default
    english = {
        'settings': 'Settings',
        'language': 'Language',
        'navigate': 'Navigation',
        'forecast': 'Stock Forecast',
        'multi_stock': 'Multi-Stock Analysis',
        'about': 'About',
        'historical_data_title': 'Historical Data (Last 10 days)',
        'date': 'Date',
        'price': 'Price',
        'forecast_data': 'Forecast Data',
        'lower_bound': 'Lower Bound',
        'median_forecast': 'Median Forecast',
        'upper_bound': 'Upper Bound',
        'historical': 'Historical',
        'forecast': 'Forecast',
        'forecast_disclaimer': 'This is a probabilistic forecast and actual results may vary.',
        'no_metrics': 'No metrics available for this forecast.',
        'forecast_accuracy': 'Forecast Accuracy',
        'mean_error': 'Mean Error (MAE)',
        'forecast_volatility': 'Forecast Volatility',
        'additional_metrics': 'Additional Metrics',
        'no_indicators': 'No technical indicators available for this forecast.',
        'momentum_indicators': 'Momentum Indicators',
        'trend_indicators': 'Trend Indicators',
        'volatility_indicators': 'Volatility Indicators',
        'technical_analysis_summary': 'Technical Analysis Summary',
        'data_sources': 'Data Sources',
        'default_source': 'Default source for most stocks',
        'secondary_source': 'Secondary source for stocks not available on Yahoo Finance',
        'fallback_source': 'Fallback source for additional coverage',
        'data_disclaimer': 'Data is provided for informational purposes only and may be delayed.',
        'use_all_data': 'Use All Historical Data',
        'use_all_data_help': 'Use all available historical data instead of just the last 6 months for potentially better predictions',
        'days_forecast_help': 'Number of days to forecast into the future',
        'forecast_range_note_title': 'Understanding the Forecast',
        'forecast_range_note': 'The forecast shows three lines: the middle line is the most likely price trajectory, while the shaded area represents the range where prices are expected to fall with the selected confidence level.',
        'forecast_subtitle': 'Predict single or multiple stocks with natural language',
        'prompt_instruction': 'Enter a question like: "Forecast Apple stock for the next 7 days" or "Compare MSFT, AMZN and GOOGL"',
        'forecast_prompt': 'Ask for a stock forecast in natural language',
        'confidence_interval': 'Confidence Interval',
        'confidence_help': 'The range that contains the true value with selected probability',
        'advanced_analysis': 'Advanced Analysis',
        'advanced_help': 'Include technical indicators and detailed analysis',
        'advanced_options': 'Advanced Options',
        'forecast_instructions': 'Enter a natural language query above to forecast one or multiple stocks. You can ask about a single stock or compare multiple stocks in the same query.',
    }
    
    # Vietnamese translations
    vietnamese = {
        'settings': 'Cài Đặt',
        'language': 'Ngôn Ngữ',
        'navigate': 'Điều Hướng',
        'forecast': 'Dự Báo Cổ Phiếu',
        'multi_stock': 'Phân Tích Đa Cổ Phiếu',
        'about': 'Giới Thiệu',
        'historical_data_title': 'Dữ Liệu Lịch Sử (10 ngày gần nhất)',
        'date': 'Ngày',
        'price': 'Giá',
        'forecast_data': 'Dữ Liệu Dự Báo',
        'lower_bound': 'Biên Dưới',
        'median_forecast': 'Dự Báo Trung Vị',
        'upper_bound': 'Biên Trên',
        'historical': 'Lịch Sử',
        'forecast': 'Dự Báo',
        'forecast_disclaimer': 'Đây là dự báo xác suất và kết quả thực tế có thể khác nhau.',
        'no_metrics': 'Không có số liệu cho dự báo này.',
        'forecast_accuracy': 'Độ Chính Xác',
        'mean_error': 'Sai Số Trung Bình (MAE)',
        'forecast_volatility': 'Biến Động Dự Báo',
        'additional_metrics': 'Số Liệu Bổ Sung',
        'no_indicators': 'Không có chỉ báo kỹ thuật cho dự báo này.',
        'momentum_indicators': 'Chỉ Báo Đà',
        'trend_indicators': 'Chỉ Báo Xu Hướng',
        'volatility_indicators': 'Chỉ Báo Biến Động',
        'technical_analysis_summary': 'Tóm Tắt Phân Tích Kỹ Thuật',
        'data_sources': 'Nguồn Dữ Liệu',
        'default_source': 'Nguồn mặc định cho hầu hết các cổ phiếu',
        'secondary_source': 'Nguồn thứ cấp cho các cổ phiếu không có trên Yahoo Finance',
        'fallback_source': 'Nguồn dự phòng để bổ sung phạm vi phủ',
        'data_disclaimer': 'Dữ liệu được cung cấp chỉ với mục đích thông tin và có thể bị chậm trễ.',
        'use_all_data': 'Sử Dụng Tất Cả Dữ Liệu Lịch Sử',
        'use_all_data_help': 'Sử dụng tất cả dữ liệu lịch sử thay vì chỉ 6 tháng gần đây để có dự báo chính xác hơn',
        'days_forecast_help': 'Số ngày dự báo trong tương lai',
        'forecast_range_note_title': 'Hiểu Về Dự Báo',
        'forecast_range_note': 'Biểu đồ dự báo thể hiện ba đường: đường giữa là quỹ đạo giá có khả năng xảy ra nhất, trong khi vùng tô màu thể hiện phạm vi giá dự kiến ​​sẽ rơi vào với mức độ tin cậy đã chọn.',
        'forecast_subtitle': 'Dự đoán một hoặc nhiều cổ phiếu bằng ngôn ngữ tự nhiên',
        'prompt_instruction': 'Nhập câu hỏi như: "Dự báo cổ phiếu Apple trong 7 ngày tới" hoặc "So sánh MSFT, AMZN và GOOGL"',
        'forecast_prompt': 'Hỏi dự báo cổ phiếu bằng ngôn ngữ tự nhiên',
        'confidence_interval': 'Khoảng Tin Cậy',
        'confidence_help': 'Phạm vi chứa giá trị thực với xác suất đã chọn',
        'advanced_analysis': 'Phân Tích Nâng Cao',
        'advanced_help': 'Bao gồm các chỉ báo kỹ thuật và phân tích chi tiết',
        'advanced_options': 'Tùy Chọn Nâng Cao',
        'forecast_instructions': 'Nhập truy vấn bằng ngôn ngữ tự nhiên ở trên để dự báo một hoặc nhiều cổ phiếu. Bạn có thể hỏi về một cổ phiếu hoặc so sánh nhiều cổ phiếu trong cùng một truy vấn.',
    }
    
    # Return the appropriate translation dictionary
    if language == 'vi':
        return vietnamese
    return english 