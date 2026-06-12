import time
import json
import os
import signal
import sys
import random
import traceback
import socket
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import database
from database import Database

WEBSITES = [
    # websites of your choice
    "https://cse.buet.ac.bd/moodle",
    "https://google.com",
    "https://prothomalo.com",
]

# For testing, set to 10 as required by the assignment
# You can increase this when your script is working properly
TRACES_PER_SITE = 1
FINGERPRINTING_URL = "http://localhost:5000" 
OUTPUT_PATH = "./data/dataset.json"

# Initialize the database to save trace data reliably
database.db = Database(WEBSITES)

""" Signal handler to ensure data is saved before quitting. """
def signal_handler(sig, frame):
    print("\nReceived termination signal. Exiting gracefully...")
    try:
        database.db.export_to_json(OUTPUT_PATH)
    except:
        pass
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


"""
Some helper functions to make your life easier.
"""

def is_server_running(host='127.0.0.1', port=5000):
    """Check if the Flask server is running."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

def setup_webdriver():
    """Set up the Selenium WebDriver with Chrome options."""
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Add additional options for stability
    # chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-popup-blocking")
    
    # Use the local chromedriver.exe instead of downloading it
    chromedriver_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chromedriver.exe")
    print(f"Using local ChromeDriver at: {chromedriver_path}")
    
    service = Service(executable_path=chromedriver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Set page load timeout to avoid hanging on slow websites
    driver.set_page_load_timeout(30)
    
    return driver

def retrieve_traces_from_backend(driver):
    """Retrieve traces from the backend API."""
    traces = driver.execute_script("""
        return fetch('/api/get_results')
            .then(response => response.ok ? response.json() : {traces: []})
            .then(data => data.traces || [])
            .catch(() => []);
    """)
    
    count = len(traces) if traces else 0
    print(f"  - Retrieved {count} traces from backend API" if count else "  - No traces found in backend storage")
    return traces or []

def clear_trace_results(driver, wait):
    """Clear all results from the backend by pressing the button."""
    clear_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Clear all results')]")
    clear_button.click()

    wait.until(EC.text_to_be_present_in_element(
        (By.XPATH, "//div[@role='alert']"), "All results cleared successfully!"))
    
def is_collection_complete():
    """Check if target number of traces have been collected."""
    current_counts = database.db.get_traces_collected()
    remaining_counts = {website: max(0, TRACES_PER_SITE - count) 
                      for website, count in current_counts.items()}
    return sum(remaining_counts.values()) == 0

"""
Your implementation starts here.
"""



def click_trace_button(driver, wait):
    """Try clicking the trace collection button."""
    try:
        for text in ["Collect Trace", "Start Trace"]:
            try:
                button = wait.until(EC.element_to_be_clickable((By.XPATH, f"//button[contains(text(), '{text}')]")))
                button.click()
                print(f"Clicked button: {text}")
                return True
            except:
                continue
        
        # Try finding any button with relevant keywords
        buttons = driver.find_elements(By.TAG_NAME, "button")
        for btn in buttons:
            if "trace" in btn.text.lower() or "collect" in btn.text.lower():
                btn.click()
                print(f"Clicked alternative button: {btn.text}")
                return True

        # Fallback to JavaScript click
        driver.execute_script("document.querySelector('button').click();")
        print("Clicked first button via JS")
        return True
    except Exception as e:
        print(f"Button click error: {e}")
        return False

def open_target_website(driver, website_url):
    """Open target website in a new tab and return its window handle."""
    try:
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[-1])
        driver.get(website_url)
        print(f"Opened target website: {website_url}")
        return driver.current_window_handle
    except Exception as e:
        print(f"Error opening target website: {e}")
        return None

def interact_with_site(driver, scrolls=5):
    """Simulate user interactions with the site."""
    try:
        for _ in range(scrolls):
            scroll_by = random.randint(300, 1000)
            driver.execute_script(f"window.scrollBy(0, {scroll_by});")
            time.sleep(random.uniform(0.5, 2.0))

            # Simulate mouse move
            driver.execute_script(f"""
                const ev = new MouseEvent('mousemove', {{
                    bubbles: true,
                    cancelable: true,
                    clientX: {random.randint(100, 800)},
                    clientY: {random.randint(100, 600)}
                }});
                document.dispatchEvent(ev);
            """)
    except Exception as e:
        print(f"Interaction error: {e}")

def wait_for_trace_result(wait):
    """Wait for the trace result UI to appear."""
    selectors = [
        "//div[contains(@class, 'result-item')]",
        "//div[@role='alert']",
        "//div[contains(@class, 'alert')]"
    ]
    for selector in selectors:
        try:
            wait.until(EC.visibility_of_element_located((By.XPATH, selector)))
            return True
        except:
            continue
    return False

def collect_single_trace(driver, wait, website_url):
    """Main trace collection routine."""
    try:
        # Step 1: Open fingerprinting page
        if driver.current_url != FINGERPRINTING_URL:
            driver.get(FINGERPRINTING_URL)
            time.sleep(2)

        original_window = driver.current_window_handle

        # Step 2: Click trace button
        if not click_trace_button(driver, wait):
            print("Failed to start trace")
            return False

        time.sleep(1)

        # Step 3: Open new tab for target site
        target_window = open_target_website(driver, website_url)

        # Step 4: Interact with target website
        if target_window:
            interact_with_site(driver)
        else:
            print("Skipping interaction due to tab open failure")

        # Step 5: Close target tab and return to original
        if target_window:
            driver.close()
        driver.switch_to.window(original_window)
        driver.execute_script("window.focus();")

        # Step 6: Wait for result
        if wait_for_trace_result(wait):
            print("Trace result detected.")
            return True
        else:
            print("Trace result not clearly detected. Waiting a bit more.")
            time.sleep(5)
            return True  # Optimistically assume success

    except Exception as e:
        print(f"Trace collection error for {website_url}: {e}")
        traceback.print_exc()
        try:
            driver.switch_to.window(original_window)
        except:
            pass
        return False


def collect_fingerprints(driver, target_counts=None):
    from collections import defaultdict

    wait = WebDriverWait(driver, 10)

    current_counts = database.db.get_traces_collected()

    if target_counts is None:
        target_counts = {website: TRACES_PER_SITE for website in WEBSITES}

    remaining_counts = {
        website: max(0, target_counts[website] - current_counts.get(website, 0))
        for website in WEBSITES
    }

    total_remaining = sum(remaining_counts.values())
    if total_remaining == 0:
        print("All target traces have been collected!")
        return 0

    print(f"Need to collect {total_remaining} more traces:")
    for website, count in remaining_counts.items():
        print(f"  - {website}: {count} traces remaining")

    # 2. Open the fingerprinting website once
    driver.get(FINGERPRINTING_URL)
    time.sleep(2)

    total_new_traces = 0

    try:
        while sum(remaining_counts.values()) > 0:
            for website in WEBSITES:
                if remaining_counts[website] <= 0:
                    continue

                print(f"\n[Batch] Collecting trace for {website}...")

                try:
                    clear_trace_results(driver, wait)
                    time.sleep(1)
                except Exception as e:
                    print(f"Error clearing results: {e}")

                success = collect_single_trace(driver, wait, website)

                if success:
                    traces = retrieve_traces_from_backend(driver)

                    if traces:
                        site_idx = database.db.get_traces_collected().get(website, 0) + 1
                        database.db.save_trace(website, site_idx, traces[0])
                        total_new_traces += 1
                        remaining_counts[website] -= 1

                        print(f"[{website}] Collected trace ({site_idx}/{target_counts[website]})")
                    else:
                        print(f"No trace data retrieved for {website}")
                else:
                    print(f"Trace collection failed for {website}")

                if total_new_traces % 10 == 0:
                    database.db.export_to_json(OUTPUT_PATH)

    except Exception as e:
        print(f"Error in collection process: {e}")
        traceback.print_exc()

    return total_new_traces


def main():
    """ Implement the main function to start the collection process.
    1. Check if the Flask server is running
    2. Initialize the database
    3. Set up the WebDriver
    4. Start the collection process, continuing until the target number of traces is reached
    5. Handle any exceptions and ensure the WebDriver is closed at the end
    6. Export the collected data to a JSON file
    7. Retry if the collection is not complete
    """
    # 1. Check if the Flask server is running
    if not is_server_running():
        print("Flask server is not running. Starting the server...")
        try:
            # Start the Flask server as a subprocess
            import subprocess
            import sys
            from threading import Thread
            
            def run_flask_server():
                subprocess.run([sys.executable, "app.py"], check=True)
            
            server_thread = Thread(target=run_flask_server)
            server_thread.daemon = True  # This ensures the thread will exit when the main program exits
            server_thread.start()
            
            # Wait for the server to start
            for _ in range(30):  # Try for 30 seconds
                if is_server_running():
                    print("Flask server started successfully!")
                    break
                time.sleep(1)
            else:
                print("Failed to start Flask server. Please start it manually.")
                sys.exit(1)
        except Exception as e:
            print(f"Error starting Flask server: {e}")
            sys.exit(1)
    else:
        print("Flask server is already running.")
    
    # 2. Initialize the database
    database.db.init_database()
      # 3. Set up the WebDriver
    driver = None
    webdriver_errors = 0
    max_webdriver_errors = 3
    
    while webdriver_errors < max_webdriver_errors:
        try:
            print("Setting up WebDriver...")
            driver = setup_webdriver()
            break  # If we got here, WebDriver setup was successful
        except Exception as e:
            webdriver_errors += 1
            print(f"Error setting up WebDriver (attempt {webdriver_errors}/{max_webdriver_errors}): {e}")
            traceback.print_exc()
            if webdriver_errors >= max_webdriver_errors:
                print("Failed to set up WebDriver after multiple attempts. Exiting.")
                sys.exit(1)
            print("Retrying WebDriver setup in 5 seconds...")
            time.sleep(5)
    
    try:
        # 4. Start the collection process
        retry_count = 0
        max_retries = 5
        browser_restart_count = 0
        max_browser_restarts = 3
        
        while not is_collection_complete() and retry_count < max_retries:
            if retry_count > 0:
                print(f"\nRetry attempt {retry_count}/{max_retries}...")
            
            try:
                traces_collected = collect_fingerprints(driver)
                print(f"\nCollection session complete. Collected {traces_collected} new traces.")
                
                # If we didn't collect any traces in this run, increment retry counter
                if traces_collected == 0:
                    retry_count += 1
                else:
                    # Reset retry counter if we made progress
                    retry_count = 0
                
                # 6. Export the collected data to a JSON file
                database.db.export_to_json(OUTPUT_PATH)
                
                # Check if we're done
                if is_collection_complete():
                    print("\nCollection complete! All target traces have been collected.")
                    break
                
            except Exception as e:
                # If there's an error in the collection process, we might need to restart the browser
                print(f"Error during collection: {e}")
                traceback.print_exc()
                retry_count += 1
                
                # If we've had multiple failures, try restarting the browser
                browser_restart_count += 1
                if browser_restart_count < max_browser_restarts:
                    print(f"Attempting to restart browser (attempt {browser_restart_count}/{max_browser_restarts})...")
                    try:
                        if driver:
                            driver.quit()
                    except:
                        pass
                    
                    try:
                        driver = setup_webdriver()
                        print("Browser restarted successfully.")
                    except Exception as restart_e:
                        print(f"Error restarting browser: {restart_e}")
                        traceback.print_exc()
                        if browser_restart_count >= max_browser_restarts:
                            print("Too many browser restart failures. Exiting.")
                            break
                        
            # Small pause before the next collection session
            print("Pausing before next collection cycle...")
            time.sleep(5)
        
        if not is_collection_complete():
            print("\nWarning: Could not collect all required traces after maximum retry attempts.")
        
    except Exception as e:
        print(f"Error in main collection process: {e}")
        traceback.print_exc()
    finally:
        # 5. Clean up: ensure the WebDriver is closed
        if driver:
            print("Closing WebDriver...")
            try:
                driver.quit()
            except Exception as e:
                print(f"Error closing WebDriver: {e}")
                
        # Make sure data is saved regardless of how we exited
        try:
            print("Saving final data export...")
            database.db.export_to_json(OUTPUT_PATH)
        except Exception as e:
            print(f"Error during final data export: {e}")
    
    print("\nData collection process finished.")
    print(f"The collected dataset is available at: {os.path.abspath(OUTPUT_PATH)}")
    database.db.export_to_json(OUTPUT_PATH)  # Final export

if __name__ == "__main__":
    main()
