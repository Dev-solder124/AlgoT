import asyncio
import aiohttp
import json
import ssl
import websockets
import requests
from datetime import datetime
from google.protobuf.json_format import MessageToDict
import MarketDataFeedV3_pb2 as pb
from dotenv import dotenv_values
import logging

# === Logging Configuration ===
today_str = datetime.now().strftime("%Y-%m-%d")
log_filename = f"market_data_{today_str}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

#===========================FLATTRADE_ORDERPLACING===========================
flattrade = dotenv_values("flattrade.env")
FACCESS_TOKEN = flattrade.get("ACCESS_TOKEN")
FUSER_ID = flattrade.get("USER_ID")

class FlattradeAPI:
    def __init__(self, user_id, account_id, auth_token, session: aiohttp.ClientSession):
        self.base_url = "https://piconnect.flattrade.in/PiConnectTP"
        self.user_id = user_id
        self.account_id = account_id
        self.auth_token = auth_token
        self.session = session
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def _build_body(self, data: dict) -> str:
        jdata = json.dumps(data, separators=(",", ":"))
        return f"jData={jdata}&jKey={self.auth_token}"

    async def place_order(
        self,
        exchange: str,
        symbol: str,
        quantity: int,
        price: float,
        transaction_type: str,
        product: str = "I",
        order_type: str = "LMT",
        validity: str = "DAY",
        trigger_price: float = None,
        remarks: str = "",
        amo: str = None
    ) -> dict:
        endpoint = f"{self.base_url}/PlaceOrder"
        order = {
            "uid": self.user_id,
            "actid": self.account_id,
            "exch": exchange,
            "tsym": symbol.upper().replace(" ", "-"),
            "qty": str(quantity),
            "prc": str(price),
            "prd": product,
            "trantype": transaction_type.upper(),
            "prctyp": order_type,
            "ret": validity,
            "remarks": remarks,
            "ordersource": "API",
        }
        if trigger_price is not None and order_type.startswith("SL-"):
            order["trgprc"] = str(trigger_price)
        if amo:
            order["amo"] = amo

        body = self._build_body(order)

        try:
            async with self.session.post(endpoint, data=body, headers=self.headers, timeout=10) as resp:
                result = await resp.json()
                logging.info("PlaceOrder response:\n%s", json.dumps(result, indent=2))
                return result
        except Exception as e:
            logging.error(f"Error placing order: {e}")
            return {}

#===========================UPSTOX_MARKETFEED===========================
upstox = dotenv_values("upstox.env")
UACCESS_TOKEN = upstox.get("ACCESS_TOKEN")

INSTRUMENTS = {
    "NSE_EQ|INE0V6F01027": "HYUNDAI(NSE)",
    "BSE_EQ|INE0V6F01027": "HYUNDAI(BSE)"
}

ltp_buffers = {key: [] for key in INSTRUMENTS}

def get_market_data_feed_authorize_v3():
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {UACCESS_TOKEN}'
    }
    url = 'https://api.upstox.com/v3/feed/market-data-feed/authorize'
    api_response = requests.get(url=url, headers=headers)
    return api_response.json()

def decode_protobuf(buffer):
    feed_response = pb.FeedResponse()
    feed_response.ParseFromString(buffer)
    return feed_response

async def connect_and_fetch():
    while True:
        try:
            await fetch_market_data()
        except Exception as e:
            logging.warning(f" Error: {e}. Reconnecting in 5 seconds...")
            await asyncio.sleep(5)

async def fetch_market_data():
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
                    if "indexFF" in full_feed:
                        ltpc = full_feed["indexFF"].get("ltpc", {})
                    elif "marketFF" in full_feed:
                        ltpc = full_feed["marketFF"].get("ltpc", {})
                    else:
                        continue

                    ltp = ltpc.get("ltp", "N/A")
                    name = INSTRUMENTS[instrument_key]

                    if name == "HYUNDAI(NSE)":
                        await HYUNDAI.meanrev(NSEPrice=ltp)
                    if name == "HYUNDAI(BSE)":
                        await HYUNDAI.meanrev(BSEPrice=ltp)

                    ltt = int(ltpc.get("ltt", "0")) / 1000
                    ltt_str = datetime.fromtimestamp(ltt).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    # logging.info(f" {ltt_str} - {name} LTP: {ltp}")

#==========================STRATEGY==============================
class stock:
    NSE=""
    BSE=""
    NSEPrice=0
    BSEPrice=0
    Tgap=0
    Position=False
    Direction=False
    def __init__(self, NSE="", BSE="", NSEPrice=0, BSEPrice=0, Tgap=0,Sgap=0, fapi=None, tolerance=0):
        self.NSE = NSE
        self.BSE = BSE
        self.NSEPrice = NSEPrice
        self.BSEPrice = BSEPrice
        self.Tgap = Tgap # target trade opening gap
        self.Sgap = Sgap # target trade closing gap
        self.fapi = fapi
        self.tol=tolerance #price limit tolerance

    async def SellBSE(self):
        return await self.fapi.place_order("BSE", self.BSE, 1, self.BSEPrice - self.tol, "S")


    async def BuyBSE(self):
        return await self.fapi.place_order("BSE", self.BSE, 1, self.BSEPrice + self.tol, "B")
    

    async def meanrev(self,NSEPrice=0,BSEPrice=0):
        if NSEPrice!=0:
            self.NSEPrice=NSEPrice
        if BSEPrice!=0:
            self.BSEPrice=BSEPrice

        gap=self.NSEPrice-self.BSEPrice
        if self.NSEPrice==0 or self.BSEPrice==0:
            gap=0
        if self.Position==False:
            if abs(gap)>=self.Tgap:
                if gap>0:
                    await self.BuyBSE()
                    logging.info("BUY_BSE TRADE OPEN")                    
                    self.Position=True
                    self.Direction=True
                elif gap<0:
                    await self.SellBSE()
                    logging.info("SELL_BSE TRADE OPEN")
                    self.Position=True
                    self.Direction=False
        else:
            if self.Direction and gap<=-self.Sgap:
                await self.SellBSE()
                logging.info("SELL_BSE TRADE CLOSE")
                self.Position=False
            elif not self.Direction and gap>=self.Sgap:
                await self.BuyBSE()
                logging.info("BUY_BSE TRADE CLOSE")
                self.Position=False

        # logging.info(f"POSITION: {self.Position} | GAP: {gap:.3f} | NSE: {self.NSEPrice:.3f} | BSE: {self.BSEPrice:.3f}")

#===================PERIODIC ORDER TASK===================
async def periodic_order_placer(fapi, exchange, symbol, qty, price, txn_type, interval_sec):
    while True:
        await fapi.place_order(exchange, symbol, qty, price, txn_type)
        logging.info(f"Placed dummy {txn_type} order for {symbol} on {exchange} every {interval_sec} sec")
        await asyncio.sleep(interval_sec)

# ==== MAIN RUN FUNCTION ====
async def main():
    async with aiohttp.ClientSession() as session:
        fapi = FlattradeAPI(FUSER_ID, FUSER_ID, FACCESS_TOKEN, session)


        global HYUNDAI
        HYUNDAI=stock(NSE="HYUNDAI-EQ",BSE="HYUNDAI",Tgap=1.0,fapi=fapi,tolerance=0.1)

        await asyncio.gather(
            connect_and_fetch(),
            periodic_order_placer(
                fapi=fapi,
                exchange="NSE",
                symbol="TATAMO",
                qty=1,
                price=0, 
                txn_type="B",
                interval_sec=3
            )
        )

# ==== ENTRY POINT ====
if __name__ == "__main__":
    asyncio.run(main())