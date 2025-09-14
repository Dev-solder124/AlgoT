from kiteconnect import KiteConnect
import os
from dotenv import load_dotenv

load_dotenv("kite.env")
access_token = os.getenv("ACCESS_TOKEN")
api_key = os.getenv("API_KEY")

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

def place_tatamotors_intraday_order(exchange, quantity, transaction_type):
    """
    Place intraday order for TATAMOTORS in specified exchange
    
    Parameters:
    exchange (str): 'NSE' or 'BSE'
    quantity (int): Number of shares to buy/sell
    transaction_type (str): 'BUY' or 'SELL'
    """
    tradingsymbol = "TATAMOTORS"
    
    try:
        # Place order
        order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            exchange=exchange,
            tradingsymbol=tradingsymbol,
            transaction_type=transaction_type,
            quantity=quantity,
            product=kite.PRODUCT_MIS,  # MIS for intraday
            order_type=kite.ORDER_TYPE_MARKET,  # Market order
            validity=kite.VALIDITY_DAY
        )
        print(f"Order placed for {tradingsymbol} on {exchange}. Order ID: {order_id}")
        return order_id
    except Exception as e:
        print(f"Order placement failed for {exchange}: {e}")
        return None

# Example usage:
# # Place buy order for 10 shares of TATAMOTORS on NSE
# place_tatamotors_intraday_order(exchange=kite.EXCHANGE_NSE, 
#                                quantity=10, 
#                                transaction_type=kite.TRANSACTION_TYPE_BUY)

# Place sell order for 5 shares of TATAMOTORS on BSE
place_tatamotors_intraday_order(exchange=kite.EXCHANGE_BSE, 
                               quantity=5, 
                               transaction_type=kite.TRANSACTION_TYPE_SELL)