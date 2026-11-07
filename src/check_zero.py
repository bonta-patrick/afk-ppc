import pandas as pd

gg = pd.read_parquet("data/processed/hidden_gg_hourly.parquet")
cust = pd.read_parquet("data/processed/customers.parquet")
cap = cust["capacity_kw"].astype(float)
cap.index = cap.index.astype(str)

# câte case produc aproape zero în fiecare lună (doar 2010-2012)
m = gg.loc["2010-07-01":"2012-06-30"].div(cap, axis=1).resample("MS").sum()
rel = m.div(m.median(axis=1), axis=0)
print("Case aproape de zero, pe luni:")
print((rel < 0.1).sum(axis=1))

print()
print("Codurile poștale ale celor 5 case:")
print(cust.loc[[92, 239, 220, 234, 158]])