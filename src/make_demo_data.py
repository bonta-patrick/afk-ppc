"""
make_demo_data.py - ambaleaza ce are nevoie aplicatia intr-un folder mic, demo_data/<perioada>/.

Aplicatia (app/app.py) citeste DOAR din demo_data/, deci ruleaza si dupa un simplu `git clone`,
fara datele mari (data/ si models/ nu sunt in repo).

Rulati dupa ce ati terminat lantul de pe perioada ceruta (baseline, train_eval, decide):
    python src/make_demo_data.py --period test

Nu modifica nimic din modelare: doar citeste rezultatele deja calculate.
"""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import pandas as pd

import evalkit as ev
import baseline as bl

PROC, CFG, REPORTS = Path("data/processed"), Path("config"), Path("reports")
DAY_HOURS = list(range(8, 18))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=["val", "test"], default="test")
    args = ap.parse_args()
    period = args.period
    out = Path("demo_data") / period
    out.mkdir(parents=True, exist_ok=True)

    splits = json.loads((CFG / "split.json").read_text(encoding="utf-8"))
    a, b = splits[period]
    net, labels, lost, clean, real = ev.load_period(period)       # net = netul cu defecte simulate
    resid = pd.read_parquet(PROC / f"resid_{period}_full.parquet")  # abaterea zilnica a modelului 'full'
    gg = pd.read_parquet(PROC / "hidden_gg_hourly.parquet").loc[a:b]
    cust = pd.read_parquet(PROC / "customers.parquet")
    added = pd.read_parquet(PROC / "sim" / period / "added_load_hourly.parquet")

    # --- serii zilnice (mici)
    net_day = net[net.index.hour.isin(DAY_HOURS)].resample("D").mean()
    resid = resid.reindex(net_day.index)
    expected = net_day - resid[net_day.columns]                    # netul "asteptat" de model
    gg_day = gg.resample("D").sum()                                # productia reala, pe zi (kWh)
    lost_day = lost.resample("D").sum()
    added_day = added.resample("D").sum()
    for name, df in (("net_day", net_day), ("expected_day", expected), ("gg_day", gg_day),
                     ("lost_day", lost_day), ("added_day", added_day)):
        df.columns = [str(c) for c in df.columns]
        df.index.name = "date"
        df.to_parquet(out / f"{name}.parquet")

    # --- evenimente ale baseline-urilor (aceleasi functii ca in baseline.py, cu pragurile inghetate)
    net_train = pd.read_parquet(PROC / "splits" / "net_train.parquet")
    thr = json.loads((CFG / "baseline_thresholds.json").read_text(encoding="utf-8"))
    runs, monitorable = bl.b1_prepare(net_train, net)
    st1 = bl.b1_state(runs, monitorable, thr["b1_N"])
    st2 = bl.b2_state(bl.b2_prepare(net_train, net, {"val": 1, "test": 2}[period]), thr["b2_h"])
    for name, st in (("b1_events", st1), ("b2_events", st2)):
        ev_df = pd.DataFrame([(h, d) for h in st.columns for d in ev.onsets(st[h])], columns=["home", "date"])
        ev_df.to_csv(out / f"{name}.csv", index=False)

    # --- tabele mici
    labels.to_csv(out / "labels.csv", index=False)
    shutil.copy(PROC / f"decisions_{period}.csv", out / "decisions.csv")
    c = cust.copy()
    c.index = c.index.astype(str)
    c.index.name = "home"
    c.to_csv(out / "customers.csv")
    for f in (f"baseline_{period}.csv", f"model_{period}_raw.csv", f"triage_{period}.csv"):
        if (REPORTS / f).exists():
            shutil.copy(REPORTS / f, out / f)
        else:
            print(f"(lipseste {REPORTS / f}, nu l-am copiat)")
    meta = {"period": period, "start": str(net_day.index.min().date()), "end": str(net_day.index.max().date()),
            "clean": clean, "real": real, "thresholds": {"baseline": thr}}
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    size = sum(f.stat().st_size for f in out.glob("*")) / 1e6
    print(f"Gata: {out} ({size:.1f} MB). Fisiere: {sorted(f.name for f in out.glob('*'))}")


if __name__ == "__main__":
    main()