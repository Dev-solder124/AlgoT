import os
import json
import logging
import requests
from dotenv import load_dotenv
import time
from pprint import pprint

# ——— Logging Setup ———
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
import threading

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
        self._local = threading.local()  # 🧵 thread-local session storage

    def _get_session(self) -> requests.Session:
        """Return a thread-local session."""
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
        return self._local.session

    def _build_body(self, data: dict) -> str:
        jdata = json.dumps(data, separators=(",", ":"))
        return f"jData={jdata}&jKey={self.auth_token}"

    def place_order(self, exchange, symbol, quantity, price, transaction_type,
                    product="I", order_type="MKT", validity="DAY",
                    trigger_price=None, remarks="", amo=None) -> dict:
        endpoint = f"{self.base_url}/PlaceOrder"
        order = {
            "uid": str(self.user_id),
            "actid": str(self.account_id),
            "exch": str(exchange),
            "tsym": str(symbol.upper().replace(" ", "-")),
            "qty": str(quantity),
            "prc": str(price),
            "prd": str(product),
            "trantype": str(transaction_type.upper()),
            "prctyp": str(order_type),
            "ret": str(validity),
            "remarks": str(remarks),
            "ordersource": "API"
        }

        if trigger_price is not None and order_type.startswith("SL-"):
            order["trgprc"] = str(trigger_price)
        if amo:
            order["amo"] = str(amo)

        body = self._build_body(order)
        session = self._get_session()  # 🧵 get thread-safe session

        resp = session.post(endpoint, data=body, headers=self.headers, timeout=10)
        result = resp.json()
        logging.info("In-session PlaceOrder response:\n%s", json.dumps(result, indent=2))
        return result

    def get_order_book(self) -> dict:
        endpoint = f"{self.base_url}/OrderBook"
        body = self._build_body({"uid": self.user_id})
        session = self._get_session()
        logging.info("OrderBook body:\n%s", body)

        resp = session.post(endpoint, data=body, headers=self.headers, timeout=10)
        return resp.json()



if __name__ == "__main__":
    # Load credentials
    load_dotenv("flattrade.env")
    USER_ID = os.getenv("USER_ID")
    ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

    if not USER_ID or not ACCESS_TOKEN:
        logging.error("USER_ID or ACCESS_TOKEN missing in flattrade.env")
        exit(1)

    api = FlattradeAPI(USER_ID, USER_ID, ACCESS_TOKEN)

    logging.info("orderLL")
    # Example in-session order
    api.place_order(
        exchange="BSE",
        symbol="TATAMOTORS",
        quantity=1,
        price=0,
        transaction_type="BUY"
    )
    time.sleep(3)
    logging.info("time")
    api.place_order(
        exchange="BSE",
        symbol="TATAMOTORS",
        quantity=1,
        price=0,
        transaction_type="BUY"
    )
    logging.info("time2")
    api.place_order(
        exchange="BSE",
        symbol="TATAMOTORS",
        quantity=1,
        price=0,
        transaction_type="BUY"
    )



    # # Fetch order book
    # ob = api.get_order_book()
    # logging.info("OrderBook response:\n%s", json.dumps(ob, indent=2))
