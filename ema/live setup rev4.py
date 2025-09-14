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
from dotenv import dotenv_values
import logging
import threading

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
# Load environment variables
flattrade = dotenv_values("flattrade.env")
FACCESS_TOKEN = flattrade.get("ACCESS_TOKEN")
FUSER_ID=flattrade.get("USER_ID")

class FlattradeAPI:
    def __init__(self, user_id, account_id, auth_token):
        self.base_url = "https://piconnect.flattrade.in/PiConnectTP"
        self.user_id = user_id
        self.account_id = account_id
        self.auth_token = auth_token
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

    def _build_body(self, data: dict) -> str:
        """
        Build the raw request body as:
          jData={…}&jKey=TOKEN
        exactly per FlatTrades curl example.
        """
        jdata = json.dumps(data, separators=(",", ":"))
        return f"jData={jdata}&jKey={self.auth_token}"

    def place_order(
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
        amo: str = None        # omit for in‑session, set "Yes" for AMO
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
        # logging.info("PlaceOrder body:\n%s", json.dumps(body, indent=2))

        resp = requests.post(endpoint, data=body, headers=self.headers, timeout=10)
        result= resp.json()
        logging.info("In-session PlaceOrder response:\n%s", json.dumps(result, indent=2))


    def get_order_book(self) -> dict:
        endpoint = f"{self.base_url}/OrderBook"
        body = self._build_body({"uid": self.user_id})
        logging.info("OrderBook body:\n%s", json.dumps(body, indent=2))

        resp = requests.post(endpoint, data=body, headers=self.headers, timeout=10)
        return resp.json()


#===========================UPSTOX_MARKETFEED===========================
# Load environment variables
upstox = dotenv_values("upstox.env")
UACCESS_TOKEN = upstox.get("ACCESS_TOKEN")

# Mapping of instrument keys to human-readable names
INSTRUMENTS = {
    "NSE_INDEX|Nifty 50": "NIFTY50"
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
    # asyncio.create_task(periodic_csv_writer())
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

                    # #----------------------
                    # if name=="TATA MOTORS (NSE)":
                    #     TATAMOTORS.meanrev(NSEPrice=ltp)
                    # if name=="TATA MOTORS (BSE)":
                    #     TATAMOTORS.meanrev(BSEPrice=ltp)
                    # #----------------------

                    
                    ltt = int(ltpc.get("ltt", "0")) /1000
                    ltt_str = datetime.fromtimestamp(ltt).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    logging.info(f" {ltt_str} - {name} LTP: {ltp}")
                    
                    ltp_buffers[instrument_key].append([ltt_str, ltp])


#==========================STRATEGY==============================
class NIFTY:
    
    target_increment=10
    stoploss_increment=1
    position="Empty"
    seek_status="Empty"

    temp_stoploss=0
    stoploss=0
    target=0
    buffer=0
    trtbuffer=0
    entry=0
    gap=0

    window=75
    alpha=2/(window+1)
    previous_ema=0

    order_bar=0
    order_bar_high=[]
    order_bar_low=[]

    alert_bar=0
    alert_bar_high=[]
    alert_bar_low=[]

    def __init__(self, previous_ema=0):
        self.previous_ema=previous_ema





    def strat(self, ltp: float, open=0,close=0,high=0,low=0,ltt=None):

        if self.position=="Long":

            if ((ltp<=self.stoploss)):
                self.position="Empty"

                #exit order


                logging.info(ltt,"Short exit at",ltp,"with",-ltp+self.entry,"points")


            elif ltp>=self.target:
                self.position="Empty"

                #exit order

                logging.info(ltt,"Long exit at",ltp,"with",ltp-self.entry,"points")
                
        if self.position=="Short":

            if ((ltp>=self.stoploss)):
                self.position="Empty"

                #exit order

                logging.info(ltt,"Short exit at",ltp,"with",-ltp+self.entry,"points")
    


            elif ltp<=self.target:
                self.position="Empty"

                #exit order

                logging.info(ltt,"Short exit at",ltp,"with",-ltp+self.entry,"points")

        if ltt.second==0:
            ema_value = self.alpha * close + (1 - self.alpha) * self.previous_ema
            self.previous_ema=ema_value

                    
            if self.order_bar==15:


                if self.seek_status=="Long seeking" and self.position=="Empty":

                    if ltp>=self.entry_target:
                        
                        #place long order

                        self.position="Long"
                        self.seek_status="Empty"
                        self.entry=ltp
                        
                        buffer=abs(self.entry_target-self.temp_stoploss)
                        trtbuffer=buffer
                        if buffer==0:
                            buffer=1      
                        # if buffer>30:
                        #     buffer=30
                        self.stoploss=entry-(self.stoploss_increment*buffer)
                        self.target=entry+self.target_increment*trtbuffer
                        
                        logging.info(ltp,self.position,"entry=",self.entry,"buffer:",self.buffer,"SL",self.stoploss,"Tgrt",self.target)


                    elif ltp<=self.temp_stoploss:
                        logging.info("Long abandoned, Short taken")

                        #place short order

                        self.position="Short"
                        self.seek_status="Empty"
                        
                        self.entry=ltp
                        buffer=abs(-self.temp_stoploss+self.entry_target)
                        trtbuffer=buffer
                        if buffer==0:
                            buffer=1
                        # if buffer>30:
                        #     buffer=30   
                        self.target=self.entry-self.target_increment*trtbuffer
                        self.stoploss=self.entry+self.stoploss_increment*buffer
                        logging.into(ltt,self.position,"entry=",self.entry,"buffer:",self.buffer,"SL",self.stoploss,"Tgrt",self.target)


                    else:
                        self.seek_status="Empty"
                    
                    


                if seek_status=="Short seeking" and position=="Empty":

                    if ltp<=self.entry_target:
                        position="Short"
                        seek_status="Empty"

                        entry=ltp

                        self.stoploss=self.temp_stoploss
                        buffer=abs(self.temp_stoploss-self.entry_target)
                        trtbuffer=buffer
                        if buffer==0:
                            buffer=1
                        # if buffer>30:
                        #     buffer=30  
                        self.target=self.entry-self.target_increment*trtbuffer
                        self.stoploss=self.entry+self.stoploss_increment*buffer

                        logging.info(ltt,self.position,"entry=",self.entry,"buffer:",self.buffer,"SL",self.stoploss,"Tgrt",self.target)


                    elif ltp>=self.temp_stoploss:
                        
                        print("Short abandoned, Long taken")
                        self.position="Long"
                        self.seek_status="Empty"

                        self.entry=ltp
                        buffer=abs(self.temp_stoploss-self.entry_target)
                        trtbuffer=buffer
                        self.stoploss=self.temp_stoploss
                        if buffer==0:
                            buffer=1 
                        # if buffer>30:
                        #     buffer=30             
                        self.target=self.entry+self.target_increment*trtbuffer 
                        self.stoploss=self.entry-self.stoploss_increment*buffer 

                        logging.info(ltt,self.position,"entry=",self.entry,"buffer:",self.buffer,"SL",self.stoploss,"Tgrt",self.target)

                    else:
                        self.seek_status="Empty"
                        # dat_element=[]


                self.order_bar=0

                self.order_bar+=1
            else:
                self.order_bar+=1



            if self.alert_bar==15:
                alert_high=max(self.alert_bar_high)
                alert_low=min(self.alert_bar_low)
                

                cdl_body=alert_high-alert_low

                if self.seek_status=="Empty":
                    if alert_low>ema_value:

                        self.seek_status="Short seeking"

                        self.entry_target=alert_low
                        gap=alert_low-ema_value
                        self.temp_stoploss=alert_high

                        logging.info(ltt,self.seek_status)
      
                    elif alert_high<ema_value:

                        self.seek_status="Long seeking"

                        self.entry_target=alert_high
                        gap=ema_value-alert_high
                        self.temp_stoploss=alert_low


                        logging.into(ltt,seek_status)

                self.alert_bar_high.clear()
                self.alert_bar_low.clear()


                self.alert_bar=0
                
                self.alert_bar_high.append(high)
                self.alert_bar_low.append(low)
                self.alert_bar+=1    
            else:
                self.alert_bar_high.append(high)
                self.alert_bar_low.append(low)

                self.alert_bar+=1



#===========================START===========================
# Start the program
if __name__ == "__main__":
    

    fapi = FlattradeAPI(FUSER_ID, FUSER_ID, FACCESS_TOKEN)

    
    # TATAMOTORS=stock(NSE="TATAMOTORS-EQ",BSE="TATAMOTORS",Tgap=0.49)


    asyncio.run(connect_and_fetch())
