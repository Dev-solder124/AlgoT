from datetime import datetime
print("Current date and time:", datetime.now())
while True:
    if datetime.now().second> 25 and datetime.now().second < 30:
        print(datetime.now().second)
        