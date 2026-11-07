import pandas as pd

d = pd.read_csv("data/processed/decisions_test.csv", parse_dates=["date"], dtype={"home": str})

print("1) Cum sunt clasificate alarmele, pe tipul real al casei")
print(pd.crosstab(d["adevar"], d["action"]))

print("\n2) Casele cu scadere reala: prima alarma si primul CONTACTEAZA")
r = d[d["adevar"] == "scadere_reala"]
t = r.groupby("home").agg(prima_alarma=("date", "min"), alarme=("date", "count"))
t["primul_CONTACT"] = r[r["action"] == "CONTACTEAZA"].groupby("home")["date"].min()
print(t)