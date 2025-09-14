import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from typing import Union

a=20
def func():
    
    a=40
    print(a)

def fun2(a):
    a=30
    func()
    print(a)
fun2(a)
print(a)
import os
from dotenv import load_dotenv,dotenv_values
from pprint import pprint

class stock:
    pbseprice=0
    pnseprice=0

tata= stock()
tata.pbseprice=78

print(tata.pbseprice)