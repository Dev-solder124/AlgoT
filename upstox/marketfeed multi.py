import asyncio
import json
import ssl
import websockets
import requests
from datetime import datetime
from google.protobuf.json_format import MessageToDict
import MarketDataFeedV3_pb2 as pb
import os
import csv
from dotenv import load_dotenv
import logging


today_str = datetime.now().strftime("%Y-%m-%d")
log_filename = f"testmarket_data_{today_str}.log"

logging.basicConfig(
    level=logging.ERROR,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)


# Load environment variables
load_dotenv("upstox.env")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

# Mapping of instrument keys to human-readable names
INSTRUMENTS = {
    "NSE_INDEX|Nifty 50": "Nifty 50",
    "NSE_EQ|INE002A01018": "Reliance (NSE)",
    "BSE_EQ|INE002A01018": "Reliance (BSE)",
    "NSE_EQ|INE155A01022": "TATA MOTORS (NSE)",
    "BSE_EQ|INE155A01022": "TATA MOTORS (BSE)",
    "NSE_EQ|INE081A01020": "TATA STEEL (NSE)",
    "BSE_EQ|INE081A01020": "TATA STEEL (BSE)",
    "NSE_EQ|INE127D01025": "HDFC AMC (NSE)",
    "BSE_EQ|INE127D01025": "HDFC AMC (BSE)",
    "NSE_EQ|INE040A01034": "HDFC BANK (NSE)",
    "BSE_EQ|INE040A01034": "HDFC BANK (BSE)",
    "BSE_EQ|INE238A01034": "AXIS BANK(BSE)",
    "NSE_EQ|INE238A01034": "AXIS BANK(NSE)",
    "BSE_EQ|INE498L01015": "LTF (BSE)",
    "NSE_EQ|INE498L01015": "LTF (NSE)",
    "NSE_EQ|INE237A01028": "KOTAKBANK(NSE)",
    "BSE_EQ|INE237A01028": "KOTAKBANK(BSE)",
    "NSE_EQ|INE0V6F01027": "HYUNDAI(NSE)",
    "BSE_EQ|INE0V6F01027": "HYUNDAI(BSE)",
    "NSE_EQ|INE021A01026": "ASIANPAINT(NSE)",
    "BSE_EQ|INE021A01026": "ASIANPAINT(BSE)"
}

def get_market_data_feed_authorize_v3():
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {ACCESS_TOKEN}'
    }
    url = 'https://api.upstox.com/v3/feed/market-data-feed/authorize'
    api_response = requests.get(url=url, headers=headers)
    return api_response.json()

def decode_protobuf(buffer):
    feed_response = pb.FeedResponse()
    feed_response.ParseFromString(buffer)
    return feed_response

async def connect_and_fetch():
    last_print_times = {key: 0 for key in INSTRUMENTS}
    while True:
        try:
            await fetch_market_data(last_print_times)
        except Exception as e:
            logging.info(f" Error: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

async def fetch_market_data(last_print_times):
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    response = get_market_data_feed_authorize_v3()
    authorized_uri = response["data"]["authorized_redirect_uri"]

    async with websockets.connect(
        authorized_uri,
        ssl=ssl_context,
        ping_interval=10,
        ping_timeout=5,
    ) as websocket:
        logging.info(' Connection established')

        await asyncio.sleep(1)

        sub_data = {
            "guid": "someguid",
            "method": "sub",
            "data": {
                "mode": "full",
                "instrumentKeys": list(INSTRUMENTS.keys())
            }
        }

        await websocket.send(json.dumps(sub_data).encode('utf-8'))
        logging.info(' Subscription request sent.')

        while True:
            message = await websocket.recv()
            decoded_data = decode_protobuf(message)
            data_dict = MessageToDict(decoded_data)

            if "feeds" in data_dict:
                for instrument_key, feed in data_dict["feeds"].items():
                    full_feed = feed.get("fullFeed", {})

                    # Determine if it's an index or equity feed
                    if "indexFF" in full_feed:
                        ltpc = full_feed["indexFF"].get("ltpc", {})
                    elif "marketFF" in full_feed:
                        ltpc = full_feed["marketFF"].get("ltpc", {})
                    else:
                        continue  # No valid LTP data

                    ltp = ltpc.get("ltp", "N/A")
                    ltt = int(ltpc.get("ltt", "0")) / 1000
                    dt = datetime.fromtimestamp(ltt)
                    ltt_str = dt.strftime("%Y-%m-%d %H:%M:%S")

                    # Only logging.info and log if 0.5 seconds passed since last logging.info
                    if ltt - last_print_times[instrument_key] >= 0.5:
                        name = INSTRUMENTS[instrument_key]
                        logging.info(f" {ltt_str} - {name} LTP: {ltp}")
                        last_print_times[instrument_key] = ltt

                        # ✅ Write to CSV file
                        filename = f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.csv"
                        with open(filename, mode='a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([ltt_str, ltp])

# Run the client
asyncio.run(connect_and_fetch())
         