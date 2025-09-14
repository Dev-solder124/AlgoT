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
import time

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

def get_market_data_feed_authorize_v3():
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {ACCESS_TOKEN}'
    }
    url = 'https://api.upstox.com/v3/feed/market-data-feed/authorize'
    return requests.get(url=url, headers=headers).json()

def decode_protobuf(buffer):
    feed_response = pb.FeedResponse()
    feed_response.ParseFromString(buffer)
    return feed_response

async def connect_and_fetch():
    last_printed_ltt = 0
    while True:
        try:
            await fetch_market_data(last_printed_ltt)
        except Exception as e:
            logging.info(f" Error: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

async def fetch_market_data(last_printed_ltt):
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
        max_queue=16,
        compression=None
    ) as websocket:
        logging.info('Connection established')

        await websocket.send(json.dumps({
            "guid": "someguid",
            "method": "sub",
            "data": {
                "mode": "full",
                "instrumentKeys": ["NSE_INDEX|Nifty 50"]
            }
        }).encode('utf-8'))
        logging.info('Subscription request sent.')

        while True:
            raw_msg = await websocket.recv()
            decoded_data = decode_protobuf(raw_msg)
            data_dict = MessageToDict(decoded_data, preserving_proto_field_name=True)

            feed = data_dict.get("feeds", {}).get("NSE_INDEX|Nifty 50", {})
            full_feed = feed.get("fullFeed", {}).get("indexFF", {})
            ltpc = full_feed.get("ltpc", {})

            ltp = ltpc.get("ltp")
            ltt = int(ltpc.get("ltt", 0))


            
            
            ltt_str = datetime.fromtimestamp(ltt / 1000).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            # ✅ Corrected latency calculation
            current_ts = time.time()
            latency_ms = (current_ts - ltt / 1000) * 1000
            logging.info(f" {ltt_str} - Nifty 50 LTP: {ltp} | Latency: {latency_ms:.1f} ms")

            if datetime.fromtimestamp(ltt / 1000).second in [0, 1]:
                ohlc = full_feed.get("marketOHLC", {}).get("ohlc", [])
                if len(ohlc) > 1:
                    logging.info(f"OHLC: {ohlc[1]}")

asyncio.run(connect_and_fetch())
