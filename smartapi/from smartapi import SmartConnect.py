from SmartApi import SmartWebSocket

# Replace with your credentials
feed_token = "eyJhbGciOiJIUzUxMiJ9.eyJ1c2VybmFtZSI6IkFBQUk0NzcwODQiLCJpYXQiOjE3NDA2NDc3NDksImV4cCI6MTc0MDczNDE0OX0.qC6kyiLDl7vXjw0YXtalWMYLQ0kXf7FVfhBdY0x8bTcjfvUqFSxtlUWUA8ryacrM0tzM-BDhKKHPtvgKt2j8Mw"
client_id = "AAAI477084"

# Symbol token for Nifty 50
symbol_token = "99926000"

# Create a WebSocket object
websocket = SmartWebSocket(feed_token, client_id)

# Define callback functions
def on_open(ws):
    print("WebSocket connection opened")
    # Subscribe to Nifty 50 real-time data
    ws.subscribe(task="mw", tokens=[symbol_token])

def on_message(ws, message):
    print("Received message:", message)

def on_close(ws):
    print("WebSocket connection closed")

# Assign callbacks
websocket.on_open = on_open
websocket.on_message = on_message
websocket.on_close = on_close

# Connect to WebSocket
websocket.connect()
