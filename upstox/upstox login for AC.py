import requests
import os
from dotenv import load_dotenv,dotenv_values


# Replace with your API Key and Secret
API_KEY = 'f0188d57-094b-4892-8023-4fae9184a242'
API_SECRET = 'xxv1hhxtbl'
REDIRECT_URI = 'http://localhost'

file_path="upstox.env"
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

# Step 1: Get the authorization code
auth_url = f"https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={API_KEY}&redirect_uri={REDIRECT_URI}"
print(f"Open this URL in your browser and authorize: {auth_url}")
auth_code = input("Enter the authorization code from the URL: ")

# Step 2: Get the access token
token_url = "https://api.upstox.com/v2/login/authorization/token"
payload = {
    'code': auth_code,
    'client_id': API_KEY,
    'client_secret': API_SECRET,
    'redirect_uri': REDIRECT_URI,
    'grant_type': 'authorization_code'
}
response = requests.post(token_url, data=payload)
access_token = response.json()['access_token']
print("Access Token:", access_token)
update_env_variable(file_path,"ACCESS_TOKEN",access_token)

