import os
import json
import logging
import requests
from dotenv import load_dotenv
from pprint import pprint
import time

# ——— Logging Setup ———
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')

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
        exactly per FlatTrade’s curl example.
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
        order_type: str = "MKT",
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
        transaction_type="B"
    )

    logging.info("time")
    api.place_order(
        exchange="BSE",
        symbol="TATAMOTORS",
        quantity=1,
        price=0,
        transaction_type="B"
    )
    logging.info("time2")
    api.place_order(
        exchange="BSE",
        symbol="TATAMOTORS",
        quantity=1,
        price=0,
        transaction_type="B"
    )
    

    # # 2. AMO order (only valid in AMO window)
    # amo_result = api.place_order(
    #     exchange="NSE",
    #     symbol="TATAMOTORS-EQ",
    #     quantity=1,
    #     price=704.50,
    #     transaction_type="B",
    #     amo="Yes"
    # )
    # logging.info("AMO PlaceOrder response:\n%s", json.dumps(amo_result, indent=2))

    # Fetch order book
    # ob = api.get_order_book()
    # logging.info("OrderBook response:\n%s", json.dumps(ob, indent=2))
