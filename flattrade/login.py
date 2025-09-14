import os
import hashlib
import requests
from dotenv import load_dotenv

# Load environment variables from 'flattrade.env'
env_file = "flattrade.env"
load_dotenv(env_file)

API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

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

def generate_access_token():
    auth_url = f"https://auth.flattrade.in/?app_key={API_KEY}"
    print("\nSTEP 1: Authenticate with Flattrade")
    print("-----------------------------------")
    print("1. Open the following URL in your browser:")
    print(auth_url)
    print("2. Log in with your Flattrade credentials.")
    print("3. After login, you'll be redirected to your redirect URL with a 'request_code' parameter.")
    print("4. Copy the 'request_code' from the URL.\n")

    request_code = input("Enter the request_code: ").strip()

    # Generate SHA-256 hash
    hash_value = hashlib.sha256((API_KEY + request_code + API_SECRET).encode()).hexdigest()

    payload = {
        "api_key": API_KEY,
        "request_code": request_code,
        "api_secret": hash_value
    }

    print("\nSTEP 2: Generating Access Token")
    print("-------------------------------")
    response = requests.post("https://authapi.flattrade.in/trade/apitoken", json=payload)

    if response.status_code == 200:
        try:
            data = response.json()
            if data.get("stat") == "Ok":
                access_token = data["token"]
                print("\nSUCCESS! Your access token is:")
                print("-------------------------------")
                print(access_token)
                print("\nNote: This token is valid for 24 hours.")
                update_env_variable(env_file, "ACCESS_TOKEN", access_token)
                return access_token
            else:
                print(f"\nERROR: {data.get('emsg', 'Unknown error')}")
        except Exception as e:
            print(f"\nERROR parsing response JSON: {str(e)}")
    else:
        print(f"\nERROR: Failed to generate access token. HTTP Status: {response.status_code}")
        print("Response Content:", response.text)

    return None

if __name__ == "__main__":
    generate_access_token()
