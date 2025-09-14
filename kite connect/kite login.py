import os
from dotenv import load_dotenv,dotenv_values
from kiteconnect import KiteConnect

# Your API credentials
load_dotenv("kite.env")
api_key = os.getenv("API_KEY")  # Replace with your actual API key
api_secret = os.getenv("ACCESS_TOKEN")  # Replace with your actual API secret

file_path="kite.env"
def update_env_variable(file_path, key, new_value):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        lines = []

    key_found = False
    new_lines = []

    for line in lines:
        if line.strip().startswith(f"{key}="):
            new_lines.append(f"{key}={new_value}\n")
            key_found = True
        else:
            new_lines.append(line)

    if not key_found:
        new_lines.append(f"{key}={new_value}\n")

    with open(file_path, 'w') as file:
        file.writelines(new_lines)


# Create Kite Connect instance
kite = KiteConnect(api_key=api_key)

# Step 2: Generate and display login URL
print("\nSTEP 1: Get Request Token")
print("------------------------")
login_url = kite.login_url()
print("1. Open this URL in your browser:", login_url)
print("2. Login with your Zerodha credentials")
print("3. After login, copy the request_token from the redirected URL\n")

# Step 3: Get request token from user input
request_token = input("Paste the request token here: ").strip()

try:
    # Step 4: Generate session and access token
    print("\nSTEP 2: Generating Access Token")
    print("------------------------------")
    session_data = kite.generate_session(request_token, api_secret=api_secret)
    access_token = session_data["access_token"]
    
    # Display the access token
    print("\nSUCCESS! Your access token is:")
    print("----------------------------")
    print(access_token)
    print("\nSave this token securely. It will expire after 24 hours.")
    update_env_variable(file_path,"ACCESS_TOKEN",access_token)

    
except Exception as e:
    print("\nERROR generating access token:")
    print("----------------------------")
    print(e)
    print("\nPossible reasons:")
    print("- Invalid request token")
    print("- Expired request token (they're valid for only a few minutes)")
    print("- Incorrect API secret")
    print("- System clock not synchronized")