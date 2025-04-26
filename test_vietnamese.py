#!/usr/bin/env python3
"""
Test script specifically for Vietnamese language support in the AI Stock Forecaster
This script tests the system's ability to handle Vietnamese queries
"""

import time
import pandas as pd
import numpy as np
from src.forecast_agent import (
    extract_stock_info_with_regex,
    normalize_vietnamese,
    stock_forecast_agent,
    get_stock_forecast
)

def test_vietnamese_normalization():
    """Test normalization of Vietnamese text"""
    test_cases = [
        ("Dự báo giá cổ phiếu Vingroup", "du bao gia co phieu vingroup"),
        ("Giá cổ phiếu Vietcombank trong 30 ngày tới", "gia co phieu vietcombank trong 30 ngay toi"),
        ("So sánh FPT và Fecon", "so sanh fpt va fecon"),
        ("Dự đoán xu hướng Hòa Phát Group", "du doan xu huong hoa phat group")
    ]
    
    print("\n=== Testing Vietnamese Normalization ===")
    for original, expected in test_cases:
        normalized = normalize_vietnamese(original)
        match = normalized == expected
        print(f"Original: '{original}'")
        print(f"Normalized: '{normalized}'")
        print(f"Match: {match}\n")
        assert match, f"Normalization failed for '{original}'"
    
    print("All normalization tests passed!")

def test_vietnamese_company_recognition():
    """Test recognition of Vietnamese company names"""
    test_cases = [
        ("Dự báo giá cổ phiếu Vingroup trong 10 ngày tới", "VIC.VN", 10),
        ("Giá cổ phiếu Fecon trong 30 ngày", "FCN.VN", 30),
        ("Vietcombank xu hướng 2 tuần", "VCB.VN", 14),  # 2 weeks = 14 days
        ("FPT trong 1 tháng tới", "FPT.VN", 30),  # 1 month = 30 days
        ("Hòa Phát trong 45 ngày", "HPG.VN", 45),
        ("So sánh SSI trong 20 ngày", "SSI.VN", 20)
    ]
    
    print("\n=== Testing Vietnamese Company Recognition ===")
    for query, expected_symbol, expected_days in test_cases:
        try:
            result = extract_stock_info_with_regex(query)
            symbol = result.get("stock_symbol", "")
            days = result.get("prediction_days", 0)
            
            symbol_match = symbol == expected_symbol
            days_match = days == expected_days
            
            print(f"Query: '{query}'")
            print(f"Symbol: '{symbol}' (Expected: '{expected_symbol}') - Match: {symbol_match}")
            print(f"Days: {days} (Expected: {expected_days}) - Match: {days_match}\n")
            
            assert symbol_match, f"Symbol mismatch for '{query}'"
            assert days_match, f"Days mismatch for '{query}'"
        except Exception as e:
            print(f"Error processing '{query}': {str(e)}\n")
            raise

    print("All company recognition tests passed!")

def test_time_period_extraction():
    """Test extraction of time periods from Vietnamese text"""
    test_cases = [
        ("10 ngày", 10),
        ("15 ngày tới", 15),
        ("2 tuần", 14),  # 2 weeks = 14 days
        ("3 tuần tới", 21),  # 3 weeks = 21 days
        ("1 tháng", 30),  # 1 month = 30 days
        ("2 tháng tới", 60)  # 2 months = 60 days
    ]
    
    print("\n=== Testing Time Period Extraction ===")
    for period, expected_days in test_cases:
        test_query = f"Dự báo cổ phiếu VIC.VN trong {period}"
        try:
            result = extract_stock_info_with_regex(test_query)
            days = result.get("prediction_days", 0)
            match = days == expected_days
            
            print(f"Period: '{period}'")
            print(f"Extracted days: {days} (Expected: {expected_days}) - Match: {match}\n")
            
            assert match, f"Time period extraction failed for '{period}'"
        except Exception as e:
            print(f"Error processing '{period}': {str(e)}\n")
            raise
    
    print("All time period extraction tests passed!")

def test_full_vietnamese_queries():
    """Test full Vietnamese queries with the agent"""
    test_queries = [
        "Dự báo giá cổ phiếu Vingroup trong 10 ngày tới",
        "So sánh FPT và Fecon trong 15 ngày",
        "Giá cổ phiếu Vietcombank trong 1 tháng tới sẽ như thế nào",
        "Dự báo xu hướng HPG.VN trong 3 tuần tới"
    ]
    
    print("\n=== Testing Full Vietnamese Queries ===")
    
    for query in test_queries:
        print(f"\nProcessing query: '{query}'")
        try:
            start_time = time.time()
            result = extract_stock_info_with_regex(query)
            processing_time = time.time() - start_time
            
            print(f"Extracted info: {result}")
            print(f"Processing time: {processing_time:.3f} seconds")
            
            # Try to get the actual forecast for the first stock
            if "multiple_stocks" in result:
                symbol = result["multiple_stocks"][0]
            else:
                symbol = result["stock_symbol"]
                
            print(f"Getting forecast for {symbol} for {result['prediction_days']} days...")
            try:
                # Just try to get a forecast, don't analyze results
                forecast, _, _ = get_stock_forecast(symbol, result["prediction_days"])
                print(f"Forecast successful! Generated {len(forecast)} data points.")
            except Exception as forecast_error:
                print(f"Forecast error: {str(forecast_error)}")
                
        except Exception as e:
            print(f"Error: {str(e)}")
    
    print("\nFull Vietnamese query tests completed!")

def main():
    """Run all Vietnamese language tests"""
    print("=== AI Stock Forecaster Vietnamese Language Tests ===")
    
    # Run the tests
    test_vietnamese_normalization()
    test_vietnamese_company_recognition()
    test_time_period_extraction()
    test_full_vietnamese_queries()
    
    print("\n=== All Vietnamese language tests completed! ===")

if __name__ == "__main__":
    main() 