from kiteconnect import KiteConnect
import datetime
import os
from dotenv import load_dotenv

load_dotenv("kite.env")
access_token = os.getenv("ACCESS_TOKEN")
api_key = os.getenv("API_KEY")

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

def get_nifty_expiry():
    today = datetime.date.today()
    # NIFTY options expire on Thursday
    days_to_thursday = (3 - today.weekday()) % 7
    expiry_date = today + datetime.timedelta(days=days_to_thursday)
    return expiry_date.strftime("%d%b%y").upper()

def place_nifty_itm_order(strike, quantity, direction):
    expiry_str = get_nifty_expiry()
    tradingsymbol = f"NIFTY{expiry_str}{strike}{direction}"
    
    try:
        # Verify instrument exists
        instruments = kite.instruments("NFO")
        valid_symbols = [i['tradingsymbol'] for i in instruments 
                       if i['name'] == 'NIFTY' 
                       and i['expiry'] == expiry_str
                       and i['instrument_type'] == direction]
        
        if tradingsymbol not in valid_symbols:
            print(f"Invalid contract. Available {direction} strikes for NIFTY{expiry_str}:")
            strikes = sorted({int(s[-8:-2]) for s in valid_symbols})
            print(strikes)
            return None
            
        order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=kite.EXCHANGE_NFO,
            tradingsymbol=tradingsymbol,
            transaction_type=kite.TRANSACTION_TYPE_BUY,
            quantity=quantity,
            product=kite.PRODUCT_MIS,
            order_type=kite.ORDER_TYPE_MARKET,
            validity=kite.VALIDITY_DAY
        )
        print(f"Order placed for {tradingsymbol}. Order ID: {order_id}")
        return order_id
    except Exception as e:
        print(f"Order placement failed: {e}")
        return None

# Example usage with validation
if __name__ == "__main__":
    strike = 24500  # Change to valid strike from printed list
    quantity = 75
    direction = 'CE'  # 'CE' or 'PE'
    
    place_nifty_itm_order(strike, quantity, direction)