import aiohttp
import asyncio
import json
import time
import logging
from dotenv import dotenv_values

flattrade = dotenv_values("flattrade.env")
FACCESS_TOKEN = flattrade.get("ACCESS_TOKEN")
FUSER_ID=flattrade.get("USER_ID")

# Optional: Configure logging to see more detailed output
logging.basicConfig(level=logging.INFO)


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
        order_type: str = "MKT",
        validity: str = "DAY",
        trigger_price: float = None,
        remarks: str = ""
    ) -> tuple[dict, float]:
        endpoint = f"{self.base_url}/PlaceOrder"
        order = {
            "uid": self.user_id,
            "actid": self.account_id,
            "exch": exchange,
            "tsym": symbol.upper().replace(" ", "-"),
            "qty": str(quantity),
            "prc": str(price),
            "prd": product,
            "trantype": transaction_type.upper(),  # 'BUY' or 'SELL'
            "prctyp": order_type,
            "ret": validity,
            "remarks": remarks,
            "ordersource": "API",
        }
        if trigger_price is not None and order_type.startswith("SL-"):
            order["trgprc"] = str(trigger_price)

        body = self._build_body(order)

        start = time.perf_counter()
        try:
            async with self.session.post(endpoint, data=body, headers=self.headers, timeout=10) as resp:
                result = await resp.json()
        except Exception as e:
            result = {"error": str(e)}
        end = time.perf_counter()
        return result, round(end - start, 3)


async def buynse_sellbse_parallel(fapi: FlattradeAPI, symbol_nse: str, symbol_bse: str, qty: int, price: float):
    start_total = time.perf_counter()

    # Prepare async order placements
    tasks = [
        fapi.place_order("NSE", symbol_nse, qty, price, "B", remarks="Buy NSE"),
        fapi.place_order("BSE", symbol_bse, qty, price, "S", remarks="Sell BSE"),
    ]

    # Run both tasks concurrently
    (nse_resp, nse_time), (bse_resp, bse_time) = await asyncio.gather(*tasks)

    end_total = time.perf_counter()
    total_time = round(end_total - start_total, 3)

    print("\nOrder Responses:")
    print(f"NSE BUY order response: {json.dumps(nse_resp, indent=2)}")
    print(f"BSE SELL order response: {json.dumps(bse_resp, indent=2)}")

    print(f"\n⏱ Time taken:")
    print(f"Buy NSE:  {nse_time} seconds")
    print(f"Sell BSE: {bse_time} seconds")
    print(f"total:    {total_time} seconds")


async def main():
    async with aiohttp.ClientSession() as session:
        fapi = FlattradeAPI(FUSER_ID, FUSER_ID, FACCESS_TOKEN, session)


        for i in range(10):
            start = time.perf_counter()
            await buynse_sellbse_parallel(
                fapi=fapi,
                symbol_nse="TATAMOTORS-EQ",  # NSE symbol
                symbol_bse="TATAMOTORS",     # BSE symbol
                qty=1,
                price=1.0  # MKT order (price is ignored)
            )
            print(f"⏱ Took {time.perf_counter() - start:.3f} sec")
            await asyncio.sleep(2)

        
        # time.sleep(3)
        # print("wait 3 sec")
        # await buynse_sellbse_parallel(
        #     fapi=fapi,
        #     symbol_nse="TATAMOTORS-EQ",  # NSE symbol
        #     symbol_bse="TATAMOTORS",     # BSE symbol
        #     qty=1,
        #     price=1.0  # MKT order (price is ignored)
        # )
        # time.sleep(3)
        # print("wait 3 sec")
        # await buynse_sellbse_parallel(
        #     fapi=fapi,
        #     symbol_nse="TATAMOTORS-EQ",  # NSE symbol
        #     symbol_bse="TATAMOTORS",     # BSE symbol
        #     qty=1,
        #     price=1.0  # MKT order (price is ignored)
        # )
        # time.sleep(5)
        # print("wait 5 sec")
        # await buynse_sellbse_parallel(
        #     fapi=fapi,
        #     symbol_nse="TATAMOTORS-EQ",  # NSE symbol
        #     symbol_bse="TATAMOTORS",     # BSE symbol
        #     qty=1,
        #     price=1.0  # MKT order (price is ignored)
        # )
        # time.sleep(10)
        # print("wait 10 sec")
        # await buynse_sellbse_parallel(
        #     fapi=fapi,
        #     symbol_nse="TATAMOTORS-EQ",  # NSE symbol
        #     symbol_bse="TATAMOTORS",     # BSE symbol
        #     qty=1,
        #     price=1.0  # MKT order (price is ignored)
        # )
        # time.sleep(30)
        # print("wait 30 sec")
        # await buynse_sellbse_parallel(
        #     fapi=fapi,
        #     symbol_nse="TATAMOTORS-EQ",  # NSE symbol
        #     symbol_bse="TATAMOTORS",     # BSE symbol
        #     qty=1,
        #     price=1.0  # MKT order (price is ignored)
        # )


if __name__ == "__main__":
    asyncio.run(main())