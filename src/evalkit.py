"""
evalkit.py - evaluare comuna pentru baseline si pentru model.

Intrare: un tabel zilnic "state" (zile x case) cu True = casa e in alarma in ziua aceea.
Aceleasi functii masoara orice metoda, ca sa fie comparatia corecta.

Metrici:
  - alarme false la 1.000 zile-casa, pe casele curate (neatinse, fara scadere reala)
  - pentru fiecare tip de defect: % prinse, zile pana la detectie, kWh pierduti pana la detectie
  - pentru fiecare tip de consum nou: % de case care au primit alarma (ar trebui sa fie mic)
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROC = Path("data/processed")
COOLDOWN_DAYS = 14      # o casa nu primeste o noua alarma in 14 zile de la precedenta (la numararea alarmelor false)


def load_period(period):
    """Incarca scenariile simulate pentru perioada ('val' sau 'test')."""
    d = PROC / "sim" / period
    if not (d / "net_faulty.parquet").exists():
        raise SystemExit(f"Lipseste {d}/net_faulty.parquet. Rulati: python src/simulate.py --period {period}")
    net = pd.read_parquet(d / "net_faulty.parquet")
    labels = pd.read_csv(d / "labels.csv", dtype={"home": str}, parse_dates=["start"])
    lost = pd.read_parquet(d / "lost_gg_hourly.parquet")
    clean = json.loads((d / "clean_homes.json").read_text(encoding="utf-8"))
    real = json.loads((d / "real_decline.json").read_text(encoding="utf-8"))
    return net, labels, lost, [str(h) for h in clean], [str(h) for h in real]


def onsets(state, cooldown=COOLDOWN_DAYS):
    """Zilele in care incepe o alarma noua (cu pauza intre alarme)."""
    s = state.fillna(False).astype(bool).to_numpy()
    prev = np.concatenate([[False], s[:-1]])
    raw = np.flatnonzero(s & ~prev)
    keep, last = [], -10 ** 9
    for i in raw:
        if i - last >= cooldown:
            keep.append(i)
            last = i
    return state.index[keep]


def false_alarm_rate(state, clean_homes, cooldown=COOLDOWN_DAYS):
    cols = [h for h in clean_homes if h in state.columns]
    n_on = sum(len(onsets(state[h], cooldown)) for h in cols)
    home_days = len(cols) * len(state)
    return n_on, home_days, 1000.0 * n_on / home_days


def fault_results(state, labels, lost):
    rows = []
    end_all = state.index[-1] + pd.Timedelta(hours=23)
    for _, r in labels[labels["kind"] == "fault"].iterrows():
        h, s = r["home"], pd.Timestamp(r["start"]).normalize()
        st = state[h].fillna(False).astype(bool)
        fresh = st & ~st.shift(1, fill_value=False)          # zi in care alarma PORNESTE
        cand = fresh[(fresh.index >= s) & fresh]
        if len(cand):
            d = cand.index[0]
            detected, delay, end = True, (d - s).days, d + pd.Timedelta(hours=23)
        else:
            detected, delay, end = False, np.nan, end_all
        rows.append({"home": h, "type": r["type"], "start": s, "detected": detected,
                     "delay_days": delay,
                     "kwh_lost_until_detection": float(lost[h].loc[s:end].sum())})
    return pd.DataFrame(rows)


def load_results(state, labels):
    rows = []
    for _, r in labels[labels["kind"] == "load"].iterrows():
        h, s = r["home"], pd.Timestamp(r["start"]).normalize()
        st = state[h].fillna(False).astype(bool)
        fresh = st & ~st.shift(1, fill_value=False)
        rows.append({"home": h, "type": r["type"], "alarmed": bool(fresh[(fresh.index >= s)].any())})
    return pd.DataFrame(rows)


def summarize(state, labels, lost, clean_homes):
    n_on, home_days, per1000 = false_alarm_rate(state, clean_homes)
    fr = fault_results(state, labels, lost)
    per_type = fr.groupby("type").agg(
        n=("home", "count"),
        prinse=("detected", "mean"),
        zile_mediane_pana_la_detectie=("delay_days", "median"),
        kWh_pierduti_pana_la_detectie=("kwh_lost_until_detection", "mean"))
    per_type["prinse"] = (per_type["prinse"] * 100).round(0)
    per_type = per_type.round(1)
    lr = load_results(state, labels)
    per_load = lr.groupby("type").agg(n=("home", "count"), cu_alarma=("alarmed", "mean"))
    per_load["cu_alarma"] = (per_load["cu_alarma"] * 100).round(0)
    clean_cols = [h for h in clean_homes if h in state.columns]
    sim = state[clean_cols].sum(axis=1)
    return {"fa_onsets": n_on, "home_days": home_days, "fa_per_1000": per1000,
            "per_type": per_type, "per_load": per_load, "faults": fr,
            "max_simultaneous_clean": int(sim.max()), "max_simultaneous_day": sim.idxmax()}


def overall_recall(summary):
    f = summary["faults"]
    return float(f["detected"].mean()) if len(f) else float("nan")


def overall_delay(summary):
    f = summary["faults"]
    d = f.loc[f["detected"], "delay_days"]
    return float(d.mean()) if len(d) else float("nan")


def print_summary(name, sm):
    print(f"\n=== {name} ===")
    print(f"Alarme false pe case curate: {sm['fa_onsets']} alarme in {sm['home_days']} zile-casa "
          f"= {sm['fa_per_1000']:.2f} la 1.000 zile-casa")
    print(f"Cele mai multe case curate in alarma simultan: {sm['max_simultaneous_clean']} "
          f"(in {sm['max_simultaneous_day']:%Y-%m-%d})")
    print("\nDefecte (ce trebuie prins):")
    print(sm["per_type"].to_string())
    print("\nConsum nou (NU trebuie sa dea alarma):")
    print(sm["per_load"].to_string())