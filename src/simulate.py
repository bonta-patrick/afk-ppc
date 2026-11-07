"""
simulate.py - simulatorul de defecte si de consum nou.

Ideea: ia datele REALE si "strica" artificial unele case, ca sa stim exact raspunsul corect.
  - DEFECTE (scade productia solara):   total, partial, intermitent, treptat
  - CONSUM NOU (cresc consumul, panourile sunt OK): aer conditionat, pompa de piscina,
    masina electrica, munca de acasa.  Sistemul NU trebuie sa dea alarma "panouri" pentru ele.

Rulare:
    python src/simulate.py --period val      # pentru reglare (tipuri: total, partial)
    python src/simulate.py --period test     # evaluarea finala (toate cele 4 tipuri)

Rezultate in data/processed/sim/<perioada>/:
    net_faulty.parquet        netul pe care il vede modelul (cu casele stricate)
    labels.csv                raspunsul corect: ce casa, ce tip, de cand, cat
    lost_gg_hourly.parquet    kWh de productie pierduti pe ora (doar casele cu defect)
    added_load_hourly.parquet kWh de consum adaugati pe ora (doar casele cu consum nou)
    real_decline.json         casele cu scadere reala (excluse din injectare)
    clean_homes.json          casele neatinse = baza pentru alarmele false
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

PROC = Path("data/processed")
CFG_DIR = Path("config")
CHK = Path("reports/checks")

DEFAULT_CFG = {
    "seed": 42,
    "n_per_type": 10,            # cate case stricam pentru fiecare tip de defect
    "n_load_per_type": 5,        # cate case primesc fiecare tip de consum nou
    "start_margin_days": 14,     # defectul nu incepe in primele 14 zile ale perioadei
    "min_days_after_start": 45,  # lasam cel putin 45 de zile dupa inceperea defectului
    "real_decline_ratio": 0.7,   # aceeasi regula ca in split_and_check.py
    "types_by_period": {
        "val":  ["total", "partial"],
        "test": ["total", "partial", "intermittent", "gradual"],
    },
    "faults": {
        "partial":      {"factor_range": [0.4, 0.7]},
        "intermittent": {"day_fraction": 0.3, "hours": [12, 17]},
        "gradual":      {"rate_per_day": 0.003, "max_loss_range": [0.3, 0.5]},
    },
    "loads": {
        "ac":   {"hours": [13, 19], "kwh_per_hour": [0.8, 1.5], "months": [11, 12, 1, 2, 3],
                 "sunny_percentile": 60, "start_window": ["11-01", "01-15"]},
        "pool": {"hours": [8, 15], "kwh_per_hour": [0.6, 1.0]},
        "ev":   {"hours": [19, 22], "kwh_per_hour": [1.5, 3.0], "day_fraction": 0.5},
        "wfh":  {"hours": [9, 16], "kwh_per_hour": [0.3, 0.6], "weekdays_only": True},
    },
}


def load_config():
    CFG_DIR.mkdir(exist_ok=True)
    p = CFG_DIR / "faults.json"
    if not p.exists():
        p.write_text(json.dumps(DEFAULT_CFG, indent=2), encoding="utf-8")
        print(f"Am creat {p} cu setarile implicite. Il puteti edita, dar NU dupa ce ati vazut rezultatele.")
    return json.loads(p.read_text(encoding="utf-8"))


def load_inputs():
    def rd(name):
        f = PROC / f"{name}.parquet"
        if not f.exists():
            raise SystemExit(f"Lipseste {f}. Rulati intai prepare.py si split_and_check.py")
        return pd.read_parquet(f)
    net, gc, gg = rd("net_hourly"), rd("hidden_gc_hourly"), rd("hidden_gg_hourly")
    rad = rd("radiation_hourly")["radiation_wm2"]
    cust = rd("customers")
    cap = cust["capacity_kw"].astype(float)
    cap.index = cap.index.astype(str)
    return net, gc, gg, rad, cap


# ---------------------------------------------------------------------------
# Eligibilitate: ce case NU au voie sa primeasca defecte simulate
# ---------------------------------------------------------------------------
def period_real_decline(gg, cap, splits, period, ratio_thr):
    """Case al caror randament in perioada data e < ratio_thr x randamentul din antrenare.
    Se foloseste doar productia, nu rezultatele vreunui model."""
    def yearly(a, b):
        days = pd.date_range(a, b, freq="h").normalize().nunique()
        return gg.loc[a:b].sum().div(cap).div(days)
    ratio = yearly(*splits[period]) / yearly(*splits["train"])
    return sorted(ratio[ratio < ratio_thr].index.tolist())


# ---------------------------------------------------------------------------
# Defecte: cum modificam productia
# ---------------------------------------------------------------------------
def fault_factor(index, ftype, start, params, rng):
    """Vector cu factorul de multiplicare al productiei (1 = normal, 0 = oprit)."""
    f = np.ones(len(index))
    after = np.asarray(index >= start)
    if ftype == "total":
        f[after] = 0.0
    elif ftype == "partial":
        f[after] = params["factor"]
    elif ftype == "intermittent":
        days = pd.date_range(start.normalize(), index[-1].normalize(), freq="D")
        bad_days = days[rng.random(len(days)) < params["day_fraction"]]
        h0, h1 = params["hours"]
        hour_ok = np.asarray((index.hour >= h0) & (index.hour <= h1))
        day_bad = np.asarray(index.normalize().isin(bad_days))
        f[after & hour_ok & day_bad] = 0.0
    elif ftype == "gradual":
        d = np.asarray((index - start).total_seconds() / 86400.0)
        fac = np.maximum(1.0 - params["rate_per_day"] * d, 1.0 - params["max_loss"])
        f[after] = fac[after]
    else:
        raise ValueError(ftype)
    return f


# ---------------------------------------------------------------------------
# Consum nou: cum adaugam consum (panourile raman sanatoase)
# ---------------------------------------------------------------------------
def load_profile(index, kind, start, params, sunny_days, rng):
    """kWh adaugati pe ora."""
    add = np.zeros(len(index))
    h0, h1 = params["hours"]
    active = np.asarray(index >= start)
    in_hours = np.asarray((index.hour >= h0) & (index.hour <= h1))
    mask = active & in_hours
    days_all = pd.date_range(start.normalize(), index[-1].normalize(), freq="D")
    if kind == "ac":
        in_season = np.asarray(index.month.isin(params["months"]))
        sunny = np.asarray(index.normalize().isin(sunny_days))
        mask = mask & in_season & sunny          # aerul conditionat merge in zilele calde, deci insorite
    elif kind == "ev":
        used = days_all[rng.random(len(days_all)) < params["day_fraction"]]
        mask = mask & np.asarray(index.normalize().isin(used))
    elif kind == "wfh" and params.get("weekdays_only"):
        mask = mask & np.asarray(index.day_of_week < 5)
    # pool: in fiecare zi, fara legatura cu soarele
    add[mask] = params["kwh"]
    return add


def pick_start(rng, p_start, p_end, margin, min_after, lo=None, hi=None):
    first = p_start.normalize() + pd.Timedelta(days=margin)
    last = p_end.normalize() - pd.Timedelta(days=min_after)
    if lo is not None:
        first = max(first, lo)
    if hi is not None:
        last = min(last, hi)
    if last < first:
        raise SystemExit("Fereastra pentru data de inceput e goala; reduceti min_days_after_start.")
    return first + pd.Timedelta(days=int(rng.integers(0, (last - first).days + 1)))


def window_dates(md_lo, md_hi, p_start):
    lo = pd.Timestamp(f"{p_start.year}-{md_lo}")
    if lo < p_start:
        lo = pd.Timestamp(f"{p_start.year + 1}-{md_lo}")
    hi = pd.Timestamp(f"{lo.year}-{md_hi}")
    if hi < lo:
        hi = pd.Timestamp(f"{lo.year + 1}-{md_hi}")
    return lo, hi


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=["val", "test"], required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    splits = json.loads((CFG_DIR / "split.json").read_text(encoding="utf-8"))
    net, gc, gg, rad, cap = load_inputs()
    a, b = splits[args.period]
    p_start, p_end = pd.Timestamp(a), pd.Timestamp(b)
    seed = (args.seed if args.seed is not None else cfg["seed"]) + {"val": 1, "test": 2}[args.period]
    rng = np.random.default_rng(seed)

    net_p, gc_p, gg_p = net.loc[a:b], gc.loc[a:b], gg.loc[a:b]
    idx = net_p.index

    # ---- 1. case eligibile
    real_val = list(json.loads((CFG_DIR / "real_decline_homes.json").read_text(encoding="utf-8")).keys())
    real_period = period_real_decline(gg, cap, splits, args.period, cfg["real_decline_ratio"])
    excluded = sorted(set(real_val) | set(real_period))
    eligible = [h for h in net.columns if h not in excluded]
    fault_types = cfg["types_by_period"][args.period]
    load_types = list(cfg["loads"].keys())
    need = cfg["n_per_type"] * len(fault_types) + cfg["n_load_per_type"] * len(load_types)
    print(f"Perioada {args.period}: {len(net.columns)} case, excluse (scadere reala): {len(excluded)}, "
          f"eligibile: {len(eligible)}, de strans: {need}")
    if need > len(eligible):
        raise SystemExit("Nu sunt destule case eligibile. Reduceti n_per_type sau n_load_per_type in config/faults.json")
    pool = list(rng.permutation(eligible))

    # zile insorite (pentru aerul conditionat): peste percentila X a zilelor de vara
    daily = rad.resample("D").mean()
    ac = cfg["loads"]["ac"]
    summer = daily[daily.index.month.isin(ac["months"])]
    sunny_days = daily[(daily >= np.percentile(summer, ac["sunny_percentile"])) &
                       daily.index.month.isin(ac["months"])].index

    net_f = net_p.copy()
    labels, lost, added = [], {}, {}
    margin, min_after = cfg["start_margin_days"], cfg["min_days_after_start"]

    # ---- 2. defecte
    for ftype in fault_types:
        for _ in range(cfg["n_per_type"]):
            h = pool.pop()
            start = pick_start(rng, p_start, p_end, margin, min_after)
            params = dict(cfg["faults"].get(ftype, {}))
            if ftype == "partial":
                params["factor"] = float(rng.uniform(*params.pop("factor_range")))
            if ftype == "gradual":
                params["max_loss"] = float(rng.uniform(*params.pop("max_loss_range")))
            fac = fault_factor(idx, ftype, start, params, rng)
            gg_new = gg_p[h].to_numpy() * fac
            loss = gg_p[h].to_numpy() - gg_new
            net_f[h] = gc_p[h].to_numpy() - gg_new
            lost[h] = pd.Series(loss, index=idx)
            after = np.asarray(idx >= start)
            n_days = max((p_end - start).days + 1, 1)
            labels.append({"home": h, "kind": "fault", "type": ftype, "start": start,
                           "params": json.dumps(params), "capacity_kw": cap[h],
                           "mean_effect_kwh_per_day": float(loss[after].sum() / n_days)})

    # ---- 3. consum nou
    for kind in load_types:
        for _ in range(cfg["n_load_per_type"]):
            h = pool.pop()
            params = dict(cfg["loads"][kind])
            params["kwh"] = float(rng.uniform(*params["kwh_per_hour"]))
            lo = hi = None
            if "start_window" in params:
                lo, hi = window_dates(*params["start_window"], p_start)
            start = pick_start(rng, p_start, p_end, margin, min_after, lo, hi)
            add = load_profile(idx, kind, start, params, sunny_days, rng)
            net_f[h] = net_p[h].to_numpy() + add
            added[h] = pd.Series(add, index=idx)
            n_days = max((p_end - start).days + 1, 1)
            labels.append({"home": h, "kind": "load", "type": kind, "start": start,
                           "params": json.dumps({k: v for k, v in params.items() if k != "kwh_per_hour"}),
                           "capacity_kw": cap[h], "mean_effect_kwh_per_day": float(add.sum() / n_days)})

    labels = pd.DataFrame(labels)

    # ---- 4. verificari de siguranta
    touched = set(labels["home"])
    untouched = [h for h in net_p.columns if h not in touched]
    assert np.allclose(net_f[untouched].to_numpy(), net_p[untouched].to_numpy(), equal_nan=True), \
        "casele neatinse au fost modificate!"
    assert net_f.isna().sum().sum() == net_p.isna().sum().sum(), "au aparut valori lipsa noi"
    for h, l in lost.items():
        assert (l.dropna() >= -1e-9).all(), "productia a crescut, nu a scazut"
    for h in labels.loc[labels.kind == "fault", "home"]:
        s = pd.Timestamp(labels.loc[labels.home == h, "start"].iloc[0])
        assert np.allclose(net_f.loc[:s - pd.Timedelta(hours=1), h], net_p.loc[:s - pd.Timedelta(hours=1), h]), \
            "netul s-a schimbat inainte de inceputul defectului"
    print("Verificari de siguranta: OK (casele neatinse identice, nimic modificat inainte de start)")

    # ---- 5. salvare
    out = PROC / "sim" / args.period
    out.mkdir(parents=True, exist_ok=True)
    net_f.columns = [str(c) for c in net_f.columns]
    net_f.to_parquet(out / "net_faulty.parquet")
    labels.to_csv(out / "labels.csv", index=False)
    pd.DataFrame(lost).to_parquet(out / "lost_gg_hourly.parquet")
    pd.DataFrame(added).to_parquet(out / "added_load_hourly.parquet")
    (out / "real_decline.json").write_text(json.dumps(excluded), encoding="utf-8")

    # ---- 6. rezumat
    print("\nRezumat scenarii:")
    summ = labels.groupby(["kind", "type"]).agg(
        n=("home", "count"),
        efect_mediu_kWh_zi=("mean_effect_kwh_per_day", "mean"),
        primul_start=("start", "min"), ultimul_start=("start", "max"))
    summ["efect_mediu_kWh_zi"] = summ["efect_mediu_kWh_zi"].round(2)
    print(summ.to_string())
    clean = [h for h in untouched if h not in excluded]
    (out / "clean_homes.json").write_text(json.dumps(clean), encoding="utf-8")
    print(f"\nCase curate, neatinse, pentru masurarea alarmelor false: {len(clean)}")
    print(f"Case cu defect sau consum nou: {len(touched)}; excluse (scadere reala): {len(excluded)}")
    print(f"Salvat in {out}/")

    # ---- 7. grafic cu exemple
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        CHK.mkdir(parents=True, exist_ok=True)
        show = [t for t in ["total", "partial", "intermittent", "gradual"] if t in fault_types]
        fig, axes = plt.subplots(len(show), 1, figsize=(10, 2.6 * len(show)), squeeze=False)
        for ax, t in zip(axes[:, 0], show):
            row = labels[(labels.kind == "fault") & (labels.type == t)].iloc[0]
            h, s = row["home"], pd.Timestamp(row["start"])
            w0, w1 = s - pd.Timedelta(days=30), min(s + pd.Timedelta(days=60, hours=23), p_end)
            orig = gg_p[h].loc[w0:w1].resample("D").sum()
            new = (gg_p[h] - lost[h]).loc[w0:w1].resample("D").sum()
            ax.plot(orig, label="productie reala", color="gray")
            ax.plot(new, label="productie dupa defect simulat")
            ax.axvline(s, color="red", linestyle="--", linewidth=1)
            ax.set_title(f"Defect '{t}', casa {h}, de la {s:%Y-%m-%d}", fontsize=10)
            ax.set_ylabel("kWh/zi")
            ax.legend(fontsize=8, loc="lower left")
        fig.tight_layout()
        fig.savefig(CHK / f"5_simulator_{args.period}.png", dpi=110)
        plt.close(fig)
        print(f"Grafic: {CHK / f'5_simulator_{args.period}.png'}")
    except Exception as e:   # graficul nu e esential
        print("(graficul nu s-a putut crea:", e, ")")


if __name__ == "__main__":
    main()