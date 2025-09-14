import os
import json
import logging
import requests
from requests import Request
from requests.adapters import HTTPAdapter
from dotenv import load_dotenv
import threading

class FlattradeAPI:
    def __init__(self, user_id, account_id, auth_token):
        self.base_url = "https://piconnect.flattrade.in/PiConnectTP"
        self.user_id = user_id
        self.account_id = account_id
        self.auth_token = auth_token
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Connection": "keep-alive"
        }
        self._local = threading.local()  # thread-local storage for sessions & requests

    def _get_session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            session = requests.Session()
            adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
            session.mount("https://", adapter)
            session.mount("http://", adapter)
            self._local.session = session
        return self._local.session

    def _get_prepared_request(self, body: str) -> requests.PreparedRequest:
        if not hasattr(self._local, "base_request"):
            # Create a base Request object only once per thread
            req = Request(
                method="POST",
                url=f"{self.base_url}/PlaceOrder",
                headers=self.headers,
                data=body  # will be overwritten on each call
            )
            self._local.base_request = req
        else:
            self._local.base_request.data = body

        return self._get_session().prepare_request(self._local.base_request)

    def _build_body(self, data: dict) -> str:
        jdata = json.dumps(data, separators=(",", ":"))
        return f"jData={jdata}&jKey={self.auth_token}"

    def place_order(self, exchange, symbol, quantity, price, transaction_type,
                    product="I", order_type="MKT", validity="DAY",
                    trigger_price=None, remarks="", amo=None) -> dict:
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
        prepared = self._get_prepared_request(body)
        session = self._get_session()

        resp = session.send(prepared, timeout=10)
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



# class FlattradeAPI:
#     def __init__(self, user_id, account_id, auth_token):
#         self.base_url = "https://piconnect.flattrade.in/PiConnectTP"
#         self.user_id = user_id
#         self.account_id = account_id
#         self.auth_token = auth_token
#         self.headers = {
#             "Content-Type": "application/json",
#             "Accept": "application/json"
#         }
#         self._local = threading.local()  # 🧵 thread-local session storage

#     def _get_session(self) -> requests.Session:
#         """Return a thread-local session."""
#         if not hasattr(self._local, "session"):
#             self._local.session = requests.Session()
#         return self._local.session

#     def _build_body(self, data: dict) -> str:
#         jdata = json.dumps(data, separators=(",", ":"))
#         return f"jData={jdata}&jKey={self.auth_token}"

#     def place_order(self, exchange, symbol, quantity, price, transaction_type,
#                     product="I", order_type="MKT", validity="DAY",
#                     trigger_price=None, remarks="", amo=None) -> dict:
#         endpoint = f"{self.base_url}/PlaceOrder"
#         order = {
#             "uid": str(self.user_id),
#             "actid": str(self.account_id),
#             "exch": str(exchange),
#             "tsym": str(symbol.upper().replace(" ", "-")),
#             "qty": str(quantity),
#             "prc": str(price),
#             "prd": str(product),
#             "trantype": str(transaction_type.upper()),
#             "prctyp": str(order_type),
#             "ret": str(validity),
#             "remarks": str(remarks),
#             "ordersource": "API"
#         }

#         if trigger_price is not None and order_type.startswith("SL-"):
#             order["trgprc"] = str(trigger_price)
#         if amo:
#             order["amo"] = str(amo)

#         body = self._build_body(order)
#         session = self._get_session()  # 🧵 get thread-safe session

#         resp = session.post(endpoint, data=body, headers=self.headers, timeout=10)
#         result = resp.json()
#         logging.info("In-session PlaceOrder response:\n%s", json.dumps(result, indent=2))
#         return result

#     def get_order_book(self) -> dict:
#         endpoint = f"{self.base_url}/OrderBook"
#         body = self._build_body({"uid": self.user_id})
#         session = self._get_session()
#         logging.info("OrderBook body:\n%s", body)

#         resp = session.post(endpoint, data=body, headers=self.headers, timeout=10)
#         return resp.json()


# class FlattradeAPI:
#     def __init__(self, user_id, account_id, auth_token):
#         self.base_url = "https://piconnect.flattrade.in/PiConnectTP"
#         self.user_id = user_id
#         self.account_id = account_id
#         self.auth_token = auth_token
#         self.headers = {
#             "Content-Type": "application/json",
#             "Accept": "application/json"
#         }

#     def _build_body(self, data: dict) -> str:
#         """
#         Build the raw request body as:
#           jData={…}&jKey=TOKEN
#         exactly per FlatTrade’s curl example.
#         """
#         jdata = json.dumps(data, separators=(",", ":"))
#         return f"jData={jdata}&jKey={self.auth_token}"

#     def place_order(
#         self,
#         exchange: str,
#         symbol: str,
#         quantity: int,
#         price: float,
#         transaction_type: str,
#         product: str = "I",
#         order_type: str = "MKT",
#         validity: str = "DAY",
#         trigger_price: float = None,
#         remarks: str = "",
#         amo: str = None        # omit for in‑session, set "Yes" for AMO
#     ) -> dict:
#         endpoint = f"{self.base_url}/PlaceOrder"
#         order = {
#             "uid": self.user_id,
#             "actid": self.account_id,
#             "exch": exchange,
#             "tsym": symbol.upper().replace(" ", "-"),
#             "qty": str(quantity),
#             "prc": str(price),
#             "prd": product,
#             "trantype": transaction_type.upper(),
#             "prctyp": order_type,
#             "ret": validity,
#             "remarks": remarks,
#             "ordersource": "API",
#         }
#         if trigger_price is not None and order_type.startswith("SL-"):
#             order["trgprc"] = str(trigger_price)
#         if amo:
#             order["amo"] = amo

#         body = self._build_body(order)
#         # logging.info("PlaceOrder body:\n%s", json.dumps(body, indent=2))

#         resp = requests.post(endpoint, data=body, headers=self.headers, timeout=10)
#         result= resp.json()
#         logging.info("In-session PlaceOrder response:\n%s", json.dumps(result, indent=2))

#     def get_order_book(self) -> dict:
#         endpoint = f"{self.base_url}/OrderBook"
#         body = self._build_body({"uid": self.user_id})
#         logging.info("OrderBook body:\n%s", json.dumps(body, indent=2))

#         resp = requests.post(endpoint, data=body, headers=self.headers, timeout=10)
#         return resp.json()
