import requests
import time
import os
import threading

def monitor_log():
    """Monitor streamlit.log for errors while testing"""
    log_file = "streamlit.log"
    initial_size = os.path.getsize(log_file) if os.path.exists(log_file) else 0
    print(f"Started monitoring {log_file} (initial size: {initial_size} bytes)")
    
    while monitoring:
        time.sleep(1)
        if os.path.exists(log_file):
            current_size = os.path.getsize(log_file)
            if current_size > initial_size:
                with open(log_file, 'r') as f:
                    f.seek(initial_size)
                    new_content = f.read()
                    if "error" in new_content.lower() or "exception" in new_content.lower():
                        print("ERROR FOUND IN LOG:")
                        for line in new_content.splitlines():
                            if "error" in line.lower() or "exception" in line.lower():
                                print(f"  {line}")
                initial_size = current_size

# Set up monitoring
monitoring = True
monitor_thread = threading.Thread(target=monitor_log)
monitor_thread.daemon = True
monitor_thread.start()

try:
    print('Testing the stock forecast app...')
    # Give Streamlit some time to fully initialize
    time.sleep(2)
    
    # First check if the server is accessible
    r = requests.get('http://localhost:8501')
    print(f'Streamlit server accessible: Status code {r.status_code}')
    
    # Now try to navigate to the forecast page
    print('Navigating to stock forecast page...')
    print('Please manually use your browser to test the stock forecast at:')
    print('http://localhost:8501/')
    print('Enter a prompt like "Forecast AAPL for next 7 days" and check for errors')
    
    # Wait for manual testing
    print("\nMonitoring logs for errors... Press Ctrl+C to stop")
    while True:
        time.sleep(5)
        
except KeyboardInterrupt:
    print("\nTest monitoring stopped by user")
except Exception as e:
    print(f'Error: {str(e)}')
finally:
    monitoring = False
    if monitor_thread.is_alive():
        monitor_thread.join(timeout=1) 