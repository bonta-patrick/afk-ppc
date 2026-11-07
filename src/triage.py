"""
triage.py - triajul cu capacitate: cine e verificat primul cand putem verifica doar K case pe saptamana.

Simulam, zi cu zi, o coada de cazuri:
  - o alarma deschide un caz (o casa are cel mult un caz deschis; alarmele noi ii actualizeaza prioritatea)
  - in fiecare zi, echipa poate verifica cate cazuri ii permite capacitatea (K pe saptamana)
  - se verifica cazurile cu cea mai mare prioritate
  - rezultatul verificarii se afla din "adevarul" simulat:
        casa are un defect deja inceput  -> gasit si reparat: nu mai pierde productie, cazul se inchide
        casa e curata / are doar consum nou / defectul nu a inceput -> verificare irosita (30 de zile fara alte alarme)
        casa din lista celor cu scadere reala -> numarata separat (nu stim cauza)

Comparam cinci politici:
  1. B1 (regula simpla), in ordinea sosirii
  2. model, toate alarmele, in ordinea sosirii
  3. model, toate alarmele, dupa pierderea estimata
  4. model + decizie (doar CONTACTEAZA), in ordinea sosirii
  5. model + decizie (doar CONTACTEAZA), dupa scor = pierdere estimata x parte_solara     <- MeterMirror
plus doua baseline-uri "corecte": B1 si model+decizie prioritizate DUPA CAPACITATEA INSTALATA (kW),
pe care orice furnizor o stie din contract. Daca scorul nu bate capacitatea, estimarea pierderii nu aduce nimic.

Metrici: precizia verificarilor, kWh pierduti pe defect (medie si mediana, pana la reparatie), % defecte reparate.

Rulare (dupa decide.py):
    python src/triage.py --period val
    python src/triage.py --period test     # o singura data, la final
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

import evalkit as ev
import baseline as bl

PROC, CFG, REPORTS = Path("data/processed"), Path("config"), Path("reports")
COOLDOWN_AFTER_FALSE = 30
K_GRID = [1, 2, 3, 5, None]          # verificari pe saptamana; None = fara limita
LOG_K = 3                                # capacitatea pentru care salvam jurnalul zilnic (pentru aplicatie)
LOG_SCENARIO = "5. model+decizie, dupa scor"


def run_queue(events, K_week, order, fault_start, real_set, clean_set, days):
    by_day = {d: g for d, g in events.groupby("date")} if len(events) else {}
    open_cases, repaired, blocked = {}, {}, {}
    visits, open_per_day = [], []
    budget = 0.0
    for d in days:
        g = by_day.get(d)
        if g is not None:
            for r in g.itertuples():
                h = r.home
                if h in repaired or blocked.get(h, pd.Timestamp.min) > d:
                    continue
                if h in open_cases:
                    open_cases[h]["priority"] = max(open_cases[h]["priority"], r.priority)
                else:
                    open_cases[h] = {"opened": d, "priority": r.priority}
        if K_week is None:
            n = len(open_cases)
        else:
            budget = min(budget + K_week / 7.0, max(1.0, K_week / 7.0))
            n = int(np.floor(budget + 1e-9))
            n = min(n, len(open_cases))
            budget -= n
        if n > 0:
            items = list(open_cases.items())
            if order == "fifo":
                items.sort(key=lambda kv: (kv[1]["opened"], kv[0]))
            else:
                items.sort(key=lambda kv: (-kv[1]["priority"], kv[1]["opened"], kv[0]))
            for h, info in items[:n]:
                if h in fault_start and d >= fault_start[h]:
                    outcome = "tp"
                    repaired[h] = d
                elif h in real_set:
                    outcome = "real"
                    repaired[h] = d
                else:
                    outcome = ("fp_pre" if h in fault_start else ("fp_clean" if h in clean_set else "fp_load"))
                    blocked[h] = d + pd.Timedelta(days=COOLDOWN_AFTER_FALSE)
                visits.append({"date": d, "home": h, "outcome": outcome, "opened": info["opened"],
                               "wait_days": (d - info["opened"]).days, "priority": info["priority"]})
                del open_cases[h]
        open_per_day.append({"date": d, "open_cases": len(open_cases)})
    return pd.DataFrame(visits, columns=["date", "home", "outcome", "opened", "wait_days", "priority"]), pd.DataFrame(open_per_day)


def metrics(visits, fault_start, lost, end_ts, n_weeks):
    tp = visits[visits["outcome"] == "tp"]
    fp = visits[visits["outcome"].str.startswith("fp")]
    real = visits[visits["outcome"] == "real"]
    kwh, delays = [], []
    for h, s in fault_start.items():
        r = tp[tp["home"] == h]
        if len(r):
            d = r["date"].iloc[0]
            kwh.append(float(lost[h].loc[s:d + pd.Timedelta(hours=23)].sum()))
            delays.append((d - s).days)
        else:
            kwh.append(float(lost[h].loc[s:end_ts].sum()))
    n_vis = len(tp) + len(fp)
    return {"vizite_total": len(visits), "verificari_pe_sapt": round((len(visits)) / n_weeks, 2), "gasite": len(tp), "irosite": len(fp),
            "scadere_reala": len(real),
            "precizie%": round(100 * len(tp) / n_vis) if n_vis else np.nan,
            "defecte_reparate%": round(100 * len(tp) / len(fault_start)),
            "zile_mediane_pana_la_reparatie": float(np.median(delays)) if delays else np.nan,
            "kWh_pierduti_pe_defect": round(float(np.mean(kwh)), 1),
            "kWh_mediana_pe_defect": round(float(np.median(kwh)), 1)}


def load_economics():
    """config/economics.json: price_per_kwh si visit_cost (in aceeasi moneda). Le completati voi, din surse verificate.
    Cat timp sunt goale (null), coloanele de cost nu se calculeaza."""
    p = CFG / "economics.json"
    if not p.exists():
        CFG.mkdir(exist_ok=True)
        p.write_text(json.dumps({"currency": "EUR", "price_per_kwh": None, "visit_cost": None,
                                 "sources": "completati sursa si data pentru fiecare valoare"}, indent=2), encoding="utf-8")
        print(f"Am creat {p}. Completati price_per_kwh si visit_cost (cu sursa) ca sa vedeti si costul total.")
    e = json.loads(p.read_text(encoding="utf-8"))
    return e if (e.get("price_per_kwh") is not None and e.get("visit_cost") is not None) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=["val", "test"], required=True)
    args = ap.parse_args()
    net, labels, lost, clean, real = ev.load_period(args.period)
    dec = pd.read_csv(PROC / f"decisions_{args.period}.csv", parse_dates=["date"], dtype={"home": str})
    days = pd.date_range(net.index.min().normalize(), net.index.max().normalize(), freq="D")
    end_ts = days[-1] + pd.Timedelta(hours=23)
    n_weeks = len(days) / 7.0
    fault_start = {r.home: pd.Timestamp(r.start).normalize() for r in labels[labels["kind"] == "fault"].itertuples()}
    real_set, clean_set = set(real), set(clean)
    cap = pd.read_parquet(PROC / "customers.parquet")["capacity_kw"].astype(float)
    cap.index = cap.index.astype(str)

    # evenimente B1
    net_train = pd.read_parquet(PROC / "splits" / "net_train.parquet")
    thr = json.loads((CFG / "baseline_thresholds.json").read_text(encoding="utf-8"))
    runs, monitorable = bl.b1_prepare(net_train, net)
    st = bl.b1_state(runs, monitorable, thr["b1_N"])
    b1 = pd.DataFrame([(h, d) for h in st.columns for d in ev.onsets(st[h])], columns=["home", "date"])
    b1["priority"] = 0.0

    dec["loss"] = dec["est_pierdere_kwh_zi"].fillna(0.0)
    dec["score"] = dec["loss"] * dec["solar_frac"].fillna(0.0).clip(0, 1)
    contact = dec[dec["action"] == "CONTACTEAZA"]
    scenarios = {
        f"1a. B1 (N={thr['b1_N']}), ordinea sosirii": (b1, "fifo"),
        f"1b. B1 (N={thr['b1_N']}), dupa capacitate": (b1.assign(priority=b1["home"].map(cap).fillna(0.0)), "priority"),
        "2. model, toate alarmele, ordinea sosirii": (dec.assign(priority=0.0), "fifo"),
        "3. model, toate alarmele, dupa pierdere": (dec.assign(priority=dec["loss"]), "priority"),
        "4a. model+decizie, ordinea sosirii": (contact.assign(priority=0.0), "fifo"),
        "4b. model+decizie, dupa capacitate": (contact.assign(priority=contact["home"].map(cap).fillna(0.0)), "priority"),
        LOG_SCENARIO: (contact.assign(priority=contact["score"]), "priority"),
    }
    econ = load_economics()
    print(f"Perioada {args.period}: {len(fault_start)} case cu defect simulat, {len(clean_set)} case curate, "
          f"{len(real_set)} cu scadere reala; alarme: B1 {len(b1)}, model {len(dec)} (CONTACTEAZA {len(contact)})")
    rows, log = [], None
    for K in K_GRID:
        for name, (events, order) in scenarios.items():
            visits, openpd = run_queue(events, K, order, fault_start, real_set, clean_set, days)
            m = metrics(visits, fault_start, lost, end_ts, n_weeks)
            m.update({"K_pe_sapt": "fara limita" if K is None else K, "politica": name})
            if econ:
                m["cost_energie"] = round(m["kWh_pierduti_pe_defect"] * len(fault_start) * econ["price_per_kwh"], 1)
                m["cost_vizite"] = round(m["vizite_total"] * econ["visit_cost"], 1)
                m["cost_total_" + econ["currency"]] = round(m["cost_energie"] + m["cost_vizite"], 1)
            rows.append(m)
            if K == LOG_K and name == LOG_SCENARIO:
                log = visits.assign(open_cases_max=int(openpd["open_cases"].max()))
    res = pd.DataFrame(rows).set_index(["K_pe_sapt", "politica"])
    pd.set_option("display.width", 250)
    print("\n" + res.to_string())
    REPORTS.mkdir(exist_ok=True)
    res.to_csv(REPORTS / f"triage_{args.period}.csv")
    if log is not None:
        log.to_csv(PROC / f"triage_log_{args.period}.csv", index=False)
        print(f"\nJurnal de verificari pentru K={LOG_K}/sapt, politica '{LOG_SCENARIO}': "
              f"{PROC / f'triage_log_{args.period}.csv'}")


if __name__ == "__main__":
    main()