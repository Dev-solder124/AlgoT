import requests
import os
from dotenv import load_dotenv
import logging
from datetime import datetime, UTC

# Setup logging
today_str = datetime.now().strftime("%Y-%m-%d")
log_filename = f"testmarket_data_{today_str}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

load_dotenv("upstox.env")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

if not ACCESS_TOKEN:
    raise RuntimeError("ACCESS_TOKEN not found in environment")

data = {
    'quantity': 1,
    'product': 'D',
    'validity': 'DAY',
    'price': 0,
    'tag': 'string',
    'instrument_token': 'NSE_EQ|INE002A01018',
    'order_type': 'MARKET',
    'transaction_type': 'BUY',
    'disclosed_quantity': 0,
    'trigger_price': 0,
    'is_amo': False
}

if data['quantity'] >= 1500:
    data['slice'] = True

def place_order(order_payload: dict) -> requests.Response:
    url = 'https://api-hft.upstox.com/v3/order/place'
    headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Authorization': f'Bearer {ACCESS_TOKEN}',
    }

    start_ts = datetime.now(UTC)
    response = requests.post(url, json=order_payload, headers=headers, timeout=5)
    latency_ms = (datetime.now(UTC) - start_ts).total_seconds() * 1000

    response.raise_for_status()

    try:
        response_data = response.json()
    except Exception:
        logging.error("Invalid JSON in response")
        raise

    order_id = response_data.get("data", {}).get("order_id", "N/A")
    logging.info("Order placed | Status: %s | Latency: %.2f ms | Order ID: %s | Body: %s",
                 response.status_code, latency_ms, order_id, response_data)
    return response

try:
    place_order(data)
except requests.HTTPError as http_err:
    logging.error("HTTP error occurred: %s | Response: %s", str(http_err), http_err.response.text)
except Exception as e:
    logging.error("An error occurred: %s", str(e))
