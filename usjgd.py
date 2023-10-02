import pandas as pd
a = pd.read_csv("FoodBalanceSheets_E_Africa_NOFLAG (1).csv",encoding="latin-1")
print(a)
mylist=a.sum(item="wine")
print(mylist)