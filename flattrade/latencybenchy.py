import os
import time
import json
import threading
import logging
from dotenv import load_dotenv
from flattrade_api import FlattradeAPI  # assumes you saved your class as flattrade_api.py

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

# Load credentials
load_dotenv("flattrade.env")
USER_ID = os.getenv("USER_ID")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

if not USER_ID or not ACCESS_TOKEN:
    logging.error("USER_ID or ACCESS_TOKEN missing in .env")
    exit(1)

api = FlattradeAPI(USER_ID, USER_ID, ACCESS_TOKEN)

# 🔧 Config
NUM_ORDERS = 3
SYMBOL = "TATAMOTORS"
EXCHANGE = "BSE"
QUANTITY = 1
PRICE = 0  # Use 0 for market orders

# 📦 Order function
def place_order_task(i):
    start = time.perf_counter()
    result = api.place_order(
        exchange=EXCHANGE,
        symbol=SYMBOL,
        quantity=QUANTITY,
        price=PRICE,
        transaction_type="B"
    )
    end = time.perf_counter()
    print(f"🧾 Order {i+1} → took {(end - start)*1000:.2f} ms")
    return result

# 🔁 Serial benchmark
def benchmark_serial():
    print("\n🔹 SERIAL ORDER PLACEMENT")
    t0 = time.perf_counter()
    for i in range(NUM_ORDERS):
        place_order_task(i)
    t1 = time.perf_counter()
    print(f"⏱️ Serial Total Time: {(t1 - t0)*1000:.2f} ms")

# 🔀 Threaded benchmark
def benchmark_threaded():
    print("\n🔹 THREADED ORDER PLACEMENT")
    threads = []
    t0 = time.perf_counter()

    for i in range(NUM_ORDERS):
        t = threading.Thread(target=place_order_task, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    t1 = time.perf_counter()
    print(f"⏱️ Threaded Total Time: {(t1 - t0)*1000:.2f} ms")

# 🚀 Run both tests
if __name__ == "__main__":
    benchmark_serial()
    benchmark_threaded()
