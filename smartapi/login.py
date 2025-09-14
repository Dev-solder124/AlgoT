import requests

api_key = "NYCsceUP"
secret_key = "67b02371-dbe7-44e2-adc8-4715c7094255"
client_code = "AAAI477084"  # Your Angel One client ID
pin = "0518"  # Your Angel One PIN

auth_url = "https://apiconnect.angelbroking.com/rest/auth/angelbroking/user/v1/loginByPassword"

payload = {
    "clientcode": client_code,
    "password": pin,
    "totp": "4TAS2GWW75LIZ2IEXEMN7KJDWI"  # Use your TOTP if 2FA is enabled
}

headers = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "X-UserType": "USER",
    "X-SourceID": "WEB",
    "X-ClientLocalIP": "CLIENT_LOCAL_IP",
    "X-ClientPublicIP": "CLIENT_PUBLIC_IP",
    "X-MACAddress": "MAC_ADDRESS",
    "X-PrivateKey": api_key
}

response = requests.post(auth_url, json=payload, headers=headers)
auth_data = response.json()
print(response.status_code)  # Check the HTTP status code
print(response.text)        # Print the raw response
access_token = auth_data["data"]["jwtToken"]