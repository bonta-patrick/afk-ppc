"""
decide.py - stratul de decizie: transforma alarmele modelului in ACTIUNI.

Pentru fiecare alarma (casa, zi) se uita la ultimele 28 de zile si se intreaba:
"golul de la orele de zi urmareste soarele?"
  - lipsa de productie solara este PROPORTIONALA cu soarele (mare in zile insorite, mica in cele innorate)
  - un consum nou (piscina, munca de acasa) este aproape CONSTANT, indiferent de soare

Pe ultimele 28 de zile se potriveste  abatere_zilnica = a + b * soare_asteptat  (soare_asteptat = radiatia x capacitatea).
    parte_solara = b * media(soare_asteptat) / media(abatere)       (1 = tot golul urmareste soarele, 0 = nu)

Actiuni:
    CONTACTEAZA     alarma + parte_solara >= prag  -> probabil lipseste productia solara
    MONITORIZEAZA   alarma, dar golul nu urmareste soarele (probabil consum nou) sau soarele a variat
                    prea putin ca sa putem spune; nu trimitem pe nimeni, reevaluam
    VERIFICA_DATE   prea multe citiri lipsa sau prea putine zile cu date in fereastra

Rulare (dupa train_eval.py --period val, cu --peer off):
    python src/decide.py --period val     # alege pragul pe validare, il salveaza in config/decision.json
    python src/decide.py --period test    # aplica pragul INGHETAT (o singura data, la final)
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

import evalkit as ev
from train_eval import cusum_alarms

PROC, CFG, MODELS, REPORTS = Path("data/processed"), Path("config"), Path("models"), Path("reports")
VARIANT = "full"
WINDOW = 28            # zile
MIN_DAYS = 14          # minim de zile cu date in fereastra
MIN_SUN_VAR = 0.15     # variatia relativa minima a soarelui (std/medie) ca sa putem deosebi cauzele
MISSING_MAX = 0.2      # peste 20% citiri lipsa -> verificam datele
MIN_EXCESS = 0.02      # kWh/ora: sub aceasta abatere medie nu putem calcula partea solara
DAY_HOURS = list(range(8, 18))
THETA_GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TARGET_CONTACT_FA = 1.0     # alarme de tip CONTACTEAZA false la 1.000 zile-casa, pe case curate
ACTION_CONTACT, ACTION_MONITOR, ACTION_DATA = "CONTACTEAZA", "MONITORIZEAZA", "VERIFICA_DATE"


# ------------------------------------------------------------------ date de intrare
def load_inputs(period, mode):
    net, labels, lost, clean, real = ev.load_period(period)
    z = pd.read_parquet(PROC / f"z_{period}_{VARIANT}_{mode}.parquet")
    resid = pd.read_parquet(PROC / f"resid_{period}_{VARIANT}.parquet")
    sc = pd.read_csv(MODELS / f"scale_{VARIANT}_{mode}.csv", index_col=0)
    sc.index = sc.index.astype(str)
    rad = pd.read_parquet(PROC / "radiation_hourly.parquet")["radiation_wm2"]
    cust = pd.read_parquet(PROC / "customers.parquet")
    cap = cust["capacity_kw"].astype(float)
    cap.index = cap.index.astype(str)
    cols = list(z.columns)
    R = resid[cols].sub(sc["bias"].reindex(cols)).to_numpy(float)          # abatere fata de normal (kWh/ora)
    rad_day = rad[rad.index.hour.isin(DAY_HOURS)].resample("D").mean().reindex(z.index)
    X = rad_day.to_numpy(float)[:, None] / 1000.0 * cap.reindex(cols).to_numpy(float)[None, :]   # "soare asteptat" (kW)
    nd = net[net.index.hour.isin(DAY_HOURS)].isna().resample("D").mean().reindex(z.index)
    M = nd[cols].to_numpy(float)                                              # fractia de citiri lipsa pe zi
    return net, labels, lost, clean, real, z, R, X, M, cols


# ------------------------------------------------------------------ trasaturi pe alarma
def event_features(R, X, M, i, j):
    lo = max(0, i - WINDOW + 1)
    y, x, m = R[lo:i + 1, j], X[lo:i + 1, j], M[lo:i + 1, j]
    ok = ~np.isnan(y) & ~np.isnan(x)
    out = {"n_days": int(ok.sum()), "missing": float(np.nanmean(m)) if np.isfinite(m).any() else np.nan,
           "solar_frac": np.nan, "excess_kwh_h": np.nan, "note": ""}
    if out["n_days"] < MIN_DAYS:
        out["note"] = "few_days"
        return out
    yy, xx = y[ok], x[ok]
    ybar, xbar = yy.mean(), xx.mean()
    out["excess_kwh_h"] = float(ybar)
    if xbar <= 0 or xx.std() / xbar < MIN_SUN_VAR:
        out["note"] = "flat_sun"
    elif ybar <= MIN_EXCESS:
        out["note"] = "small_excess"
    else:
        b = np.cov(xx, yy, bias=True)[0, 1] / xx.var()
        out["solar_frac"] = float(np.clip(b * xbar / ybar, -1.0, 2.0))
    return out


def build_events(z, R, X, M, cols, h):
    A = cusum_alarms(z, h)
    rows = []
    for j, home in enumerate(cols):
        for d in ev.onsets(A[home]):
            i = z.index.get_loc(d)
            f = event_features(R, X, M, i, j)
            f.update({"home": home, "date": d})
            rows.append(f)
    return pd.DataFrame(rows)


def assign(events, theta):
    def one(r):
        if r["missing"] > MISSING_MAX or r["note"] == "few_days":
            return ACTION_DATA
        if r["note"] in ("flat_sun", "small_excess") or np.isnan(r["solar_frac"]):
            return ACTION_MONITOR
        return ACTION_CONTACT if r["solar_frac"] >= theta else ACTION_MONITOR
    return events.apply(one, axis=1) if len(events) else pd.Series([], dtype=object)


def explain(r, action):
    loss = max(r["excess_kwh_h"], 0) * 10 if pd.notna(r["excess_kwh_h"]) else np.nan
    if action == ACTION_DATA:
        if r["note"] == "few_days":
            return (f"Inceput de perioada: doar {r['n_days']} zile de istoric, prea putin pentru a deosebi un panou "
                    f"stricat de un consum nou. Reevaluam cand avem cel putin {MIN_DAYS} zile.")
        return (f"Date nesigure ({r['missing']:.0%} din citirile de zi lipsesc): "
                f"verificam datele inainte de orice actiune.")
    if r["note"] == "flat_sun":
        return ("Abatere la orele de zi, dar soarele a variat prea putin in ultimele 28 de zile ca sa deosebim "
                "un panou stricat de un consum nou. Asteptam zile cu soare diferit.")
    if r["note"] == "small_excess" or pd.isna(r["solar_frac"]):
        return "Abatere mica si neconcludenta; continuam monitorizarea."
    if action == ACTION_CONTACT:
        strong = r["solar_frac"] >= 0.7
        return (f"De ~{r['n_days']} zile, la orele de zi consumul net e cu {loss:.1f} kWh/zi peste normal. "
                f"{min(r['solar_frac'], 1.0):.0%} din abatere creste odata cu soarele: "
                + ("indiciu puternic ca lipseste productia solara" if strong else
                   "indiciu moderat ca lipseste productia solara (merita verificat, cu incredere medie)")
                + f". Pierdere estimata: {loss:.1f} kWh/zi.")
    return (f"Consum net cu {loss:.1f} kWh/zi peste normal la orele de zi, dar doar {max(r['solar_frac'], 0):.0%} din "
            f"abatere urmareste soarele: seamana cu un consum nou, nu cu un panou stricat. Nu trimitem tehnician; reevaluam.")


# ------------------------------------------------------------------ evaluare
def evaluate_actions(events, action, labels, lost, clean, n_days, end_ts):
    contact = events[action == ACTION_CONTACT] if len(events) else events
    by_home = {h: g["date"].sort_values().reset_index(drop=True) for h, g in contact.groupby("home")} if len(contact) else {}
    clean_set = set(clean)
    n_clean_contact = int(sum(len(v) for h, v in by_home.items() if h in clean_set))
    fa = 1000.0 * n_clean_contact / (len(clean) * n_days)
    rows = []
    for _, r in labels[labels["kind"] == "fault"].iterrows():
        h, s = r["home"], pd.Timestamp(r["start"]).normalize()
        d = by_home.get(h, pd.Series([], dtype="datetime64[ns]"))
        d = d[d >= s]
        if len(d):
            rows.append({"type": r["type"], "det": True, "delay": (d.iloc[0] - s).days,
                         "kwh": float(lost[h].loc[s:d.iloc[0] + pd.Timedelta(hours=23)].sum())})
        else:
            rows.append({"type": r["type"], "det": False, "delay": np.nan, "kwh": float(lost[h].loc[s:end_ts].sum())})
    fr = pd.DataFrame(rows)
    load = {}
    for t, g in labels[labels["kind"] == "load"].groupby("type"):
        hit = [bool((by_home.get(r["home"], pd.Series([], dtype="datetime64[ns]")) >= pd.Timestamp(r["start"]).normalize()).any())
               for _, r in g.iterrows()]
        fl = []
        for _, r in g.iterrows():
            s0 = pd.Timestamp(r["start"]).normalize()
            fl.append(np.mean([bool((by_home.get(c, pd.Series([], dtype="datetime64[ns]")) >= s0).any()) for c in clean]))
        load[t] = (100 * np.mean(hit), 100 * np.mean(fl))
    return {"fa": fa, "recall": 100 * fr["det"].mean(), "delay": fr["delay"].median(),
            "kwh": fr["kwh"].mean(), "load": load, "faults": fr}


def table_row(name, res):
    row = {"regula": name, "CONTACT_FA_la_1000": round(res["fa"], 2), "defecte_prinse%": round(res["recall"]),
           "zile_mediane": res["delay"], "kWh_pierduti_medie": round(res["kwh"], 1)}
    for t, (a, f) in sorted(res["load"].items()):
        row[f"{t}_contact%(fond)"] = f"{a:.0f}({f:.0f})"
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=["val", "test"], required=True)
    ap.add_argument("--peer", choices=["on", "off"], default="off")
    args = ap.parse_args()
    mode = "peer" if args.peer == "on" else "raw"
    freeze = args.period == "test"
    net, labels, lost, clean, real, z, R, X, M, cols = load_inputs(args.period, mode)
    h = json.loads((CFG / "model_thresholds.json").read_text(encoding="utf-8"))[mode][VARIANT]
    n_days = len(z)
    end_ts = z.index[-1] + pd.Timedelta(hours=23)
    events = build_events(z, R, X, M, cols, h)
    print(f"Perioada {args.period}, model '{VARIANT}' ({mode}), prag CUSUM h={h}: {len(events)} alarme "
          f"({events['home'].isin(clean).sum()} pe case curate)")
    print("Note despre alarme:", events["note"].replace("", "ok").value_counts().to_dict())

    ref = evaluate_actions(events, pd.Series([ACTION_CONTACT] * len(events), index=events.index),
                           labels, lost, clean, n_days, end_ts)
    rows = [table_row("fara verificare (toate alarmele)", ref)]
    results = {}
    for th in THETA_GRID:
        results[th] = evaluate_actions(events, assign(events, th), labels, lost, clean, n_days, end_ts)
        rows.append(table_row(f"prag parte_solara >= {th}", results[th]))
    thr_path = CFG / "decision.json"
    if freeze:
        if not thr_path.exists():
            raise SystemExit("Rulati intai: python src/decide.py --period val")
        theta = json.loads(thr_path.read_text(encoding="utf-8"))["theta"]
        if theta not in results:
            results[theta] = evaluate_actions(events, assign(events, theta), labels, lost, clean, n_days, end_ts)
    else:
        ok = [t for t in THETA_GRID if results[t]["fa"] <= TARGET_CONTACT_FA]
        if ok:
            score = lambda t: (results[t]["recall"] - np.mean([a for a, _ in results[t]["load"].values()]), -results[t]["kwh"])
            theta = max(ok, key=score)
            note = ""
        else:
            theta = max(THETA_GRID, key=lambda t: -results[t]["fa"])
            note = f"!! Niciun prag nu atinge {TARGET_CONTACT_FA} alarme CONTACT false la 1.000; am luat cel mai strict."
        CFG.mkdir(exist_ok=True)
        thr_path.write_text(json.dumps({"theta": theta, "mode": mode, "window": WINDOW}, indent=2), encoding="utf-8")
        if note:
            print(note)
    print("\nComparatie reguli de decizie (in paranteza: % din casele curate cu CONTACT in aceeasi fereastra = fond):")
    print(pd.DataFrame(rows).set_index("regula").to_string())
    print(f"\nPrag ales{' (INGHETAT)' if freeze else ' pe validare'}: parte_solara >= {theta}")

    action = assign(events, theta)
    events = events.assign(action=action)
    events["explicatie"] = [explain(r, a) for (_, r), a in zip(events.iterrows(), action)]
    lab = labels.set_index("home")
    events["adevar"] = [("curata" if h_ in set(clean) else
                         (f"{lab.loc[h_, 'kind']}:{lab.loc[h_, 'type']}" if h_ in lab.index else "scadere_reala"))
                        for h_ in events["home"]]
    cm = pd.crosstab(events["adevar"].str.split(":").str[0], events["action"])
    print("\nActiuni pe tipul real al casei (toate alarmele):")
    print(cm.to_string())
    events["est_pierdere_kwh_zi"] = events["excess_kwh_h"].clip(lower=0) * 10
    out = PROC / f"decisions_{args.period}.csv"
    events.to_csv(out, index=False)
    print(f"\nSalvat: {out}")
    print(f"Defecte prinse cu CONTACTEAZA: {results[theta]['recall']:.0f}% | alarme CONTACT false pe case curate: "
          f"{results[theta]['fa']:.2f} la 1.000 zile-casa")
    print("Exemplu de explicatii:")
    for a in (ACTION_CONTACT, ACTION_MONITOR, ACTION_DATA):
        e = events[events["action"] == a]
        if len(e):
            r = e.iloc[0]
            print(f"  [{a}] casa {r['home']} {r['date']:%Y-%m-%d}: {r['explicatie']}")


if __name__ == "__main__":
    main()