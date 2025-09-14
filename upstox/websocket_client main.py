import asyncio
import json
import ssl
import websockets
import requests
from datetime import datetime
from google.protobuf.json_format import MessageToDict
import MarketDataFeedV3_pb2 as pb
import os
from dotenv import load_dotenv
import logging


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
    last_printed_minute = None
    while True:
        try:
            await fetch_market_data(last_printed_minute)
        except Exception as e:
            logging.info(f" Error: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

async def fetch_market_data(last_printed_minute):
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    response = get_market_data_feed_authorize_v3()
    authorized_uri = response["data"]["authorized_redirect_uri"]

    async with websockets.connect(
        authorized_uri,
        ssl=ssl_context,
        ping_interval=10,  # Send ping every 10 seconds
        ping_timeout=5,    # Wait 5 seconds for a pong response
    ) as websocket:
        logging.info('Connection established')

        await asyncio.sleep(1)

        data = {
            "guid": "someguid",
            "method": "sub",
            "data": {
                "mode": "full",
                "instrumentKeys": ["NSE_INDEX|Nifty 50"]
            }
        }

        binary_data = json.dumps(data).encode('utf-8')
        await websocket.send(binary_data)
        logging.info('Subscription request sent.')

        last_ltp_print_time = 0

        while True:
            message = await websocket.recv()
            decoded_data = decode_protobuf(message)
            data_dict = MessageToDict(decoded_data)

            if "feeds" in data_dict:
                for instrument, feed in data_dict["feeds"].items():
                    if instrument == "NSE_INDEX|Nifty 50":
                        full_feed = feed.get("fullFeed", {})
                        index_ff = full_feed.get("indexFF", {})
                        ltpc = index_ff.get("ltpc", {})

                        ltp = ltpc.get("ltp", "N/A")
                        ltt = int(ltpc.get("ltt", "0")) / 1000
                        dt = datetime.fromtimestamp(ltt)
                        ltt_str = dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                        current_minute = dt.strftime("%Y-%m-%d %H:%M")

                        if ltt - last_ltp_print_time >= 0:
                            logging.info(f" {ltt_str} - Nifty 50 LTP: {ltp}")
                            last_ltp_print_time = ltt

                            if (dt.second == 0) or (dt.second==1):
                                ohlc = index_ff.get("marketOHLC", {}).get("ohlc", [])[1]
                                logging.info(ohlc)

# Run the client
asyncio.run(connect_and_fetch())
