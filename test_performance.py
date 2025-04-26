#!/usr/bin/env python3
"""
Performance testing script for AI Stock Forecaster
This script runs a series of benchmarks to measure the performance improvements
"""

import time
import pandas as pd
import numpy as np
from src.forecast_agent import (
    stock_forecast_agent, 
    get_stock_forecast, 
    process_multiple_stocks
)

def test_single_stock(stock_symbol="AAPL", prediction_days=10):
    """Test performance of single stock forecast"""
    start_time = time.time()
    forecast, history, metrics = get_stock_forecast(stock_symbol, prediction_days)
    elapsed = time.time() - start_time
    
    print(f"\n--- Single Stock Test: {stock_symbol} ---")
    print(f"Processing time: {elapsed:.3f} seconds")
    if 'mae' in metrics:
        print(f"MAE: {metrics['mae']:.4f}")
    if 'rmse' in metrics:
        print(f"RMSE: {metrics['rmse']:.4f}")
    if 'direction_accuracy' in metrics:
        print(f"Direction Accuracy: {metrics['direction_accuracy']*100:.2f}%")
    
    return elapsed, metrics

def test_agent(prompt="Get stock price forecast for Apple for the next 10 days"):
    """Test performance of AI agent parsing"""
    start_time = time.time()
    result = stock_forecast_agent(prompt=prompt)
    elapsed = time.time() - start_time
    
    print(f"\n--- Agent Test ---")
    print(f"Prompt: '{prompt}'")
    print(f"Processing time: {elapsed:.3f} seconds")
    
    if isinstance(result, dict):
        # Multiple stocks
        print(f"Detected {len(result)} stocks")
        for symbol, (_, _, metrics) in result.items():
            if metrics and 'mae' in metrics:
                print(f"{symbol} - MAE: {metrics['mae']:.4f}")
    else:
        # Single stock
        _, _, metrics = result
        stock_symbol = result[0]['ticker'].iloc[0]
        print(f"Detected stock: {stock_symbol}")
        if 'mae' in metrics:
            print(f"MAE: {metrics['mae']:.4f}")
    
    return elapsed

def test_batch_processing(symbols=["AAPL", "MSFT", "GOOGL"], prediction_days=10):
    """Test performance of batch processing multiple stocks"""
    start_time = time.time()
    results = process_multiple_stocks(symbols, prediction_days)
    elapsed = time.time() - start_time
    
    print(f"\n--- Batch Processing Test ---")
    print(f"Stocks: {', '.join(symbols)}")
    print(f"Total processing time: {elapsed:.3f} seconds")
    print(f"Average time per stock: {elapsed/len(symbols):.3f} seconds")
    
    # Calculate average metrics
    metrics_avg = {}
    for _, (_, _, metrics) in results.items():
        for key, value in metrics.items():
            if key not in metrics_avg:
                metrics_avg[key] = []
            metrics_avg[key].append(value)
    
    for key, values in metrics_avg.items():
        if key in ['mae', 'rmse', 'direction_accuracy']:
            print(f"Average {key}: {np.mean(values):.4f}")
    
    return elapsed, results

def test_cache_speedup():
    """Test speedup from caching by running the same query twice"""
    stock = "AAPL"
    print(f"\n--- Cache Speedup Test ---")
    
    # First run (without cache)
    start_time = time.time()
    get_stock_forecast(stock, 10)
    first_run = time.time() - start_time
    print(f"First run: {first_run:.3f} seconds")
    
    # Second run (with cache)
    start_time = time.time()
    get_stock_forecast(stock, 10)
    second_run = time.time() - start_time
    print(f"Second run: {second_run:.3f} seconds")
    
    speedup = first_run / second_run if second_run > 0 else float('inf')
    print(f"Speedup: {speedup:.2f}x")
    
    return first_run, second_run, speedup

def main():
    """Run all performance tests"""
    print("=== AI Stock Forecaster Performance Tests ===")
    
    results = {}
    
    # Test single stocks
    for stock in ["AAPL", "MSFT", "TSLA"]:
        elapsed, _ = test_single_stock(stock)
        results[f"single_{stock}"] = elapsed
    
    # Test Vietnamese stocks
    try:
        elapsed, _ = test_single_stock("VIC.VN")
        results["single_vietnamese"] = elapsed
    except Exception as e:
        print(f"Error testing Vietnamese stock: {e}")
    
    # Test AI agent
    for prompt in [
        "Get stock price forecast for Apple for the next 10 days",
        "What will Tesla stock be worth in 15 days?",
        "Forecast Vingroup stock for the next week"
    ]:
        elapsed = test_agent(prompt)
        results[f"agent_{prompt[:20]}"] = elapsed
    
    # Test batch processing
    elapsed, _ = test_batch_processing()
    results["batch_processing"] = elapsed
    
    # Test cache speedup
    first, second, speedup = test_cache_speedup()
    results["cache_first"] = first
    results["cache_second"] = second
    results["cache_speedup"] = speedup
    
    # Summary
    print("\n=== Performance Summary ===")
    for test, elapsed in results.items():
        print(f"{test}: {elapsed:.3f} seconds")
    
    # Save results to CSV
    pd.DataFrame([results]).to_csv("performance_results.csv", index=False)
    print("\nResults saved to performance_results.csv")

if __name__ == "__main__":
    main() 