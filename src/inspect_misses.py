"""
inspect_misses.py - de ce unele defecte nu sunt prinse? (analiza erorilor)

Pentru fiecare defect simulat pe validare calculeaza raportul semnal/zgomot:
    semnal = cat net in plus apare pe ora de zi din cauza defectului (kWh/ora, in medie)
    zgomot = cat variaza in mod normal reziduul zilnic al casei (kWh/ora)
si arata care defecte sunt prinse la pragul ales si la unul foarte permisiv.

Rulare (dupa train_eval.py --period val):
    python src/inspect_misses.py            # modelul 'full', ajustare fata de vecini
    python src/inspect_misses.py --variant nb --peer off
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

import evalkit as ev
from train_eval import cusum_alarms

PROC, CFG, MODELS = Path("data/processed"), Path("config"), Path("models")
PERMISSIVE_H = 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="full")
    ap.add_argument("--peer", choices=["on", "off"], default="on")
    ap.add_argument("--period", default="val")
    args = ap.parse_args()
    mode = "peer" if args.peer == "on" else "raw"
    net, labels, lost, clean, real = ev.load_period(args.period)
    if args.variant != "full":
        raise SystemExit("Doar varianta 'full' salveaza matricea z; rulati cu --variant full")
    z = pd.read_parquet(PROC / f"z_{args.period}_{args.variant}_{mode}.parquet")
    thr = json.loads((CFG / "model_thresholds.json").read_text(encoding="utf-8"))[mode][args.variant]
    sc = pd.read_csv(MODELS / f"scale_{args.variant}_{mode}.csv", index_col=0)
    sc.index = sc.index.astype(str)

    f = labels[labels["kind"] == "fault"].copy()
    f["sd"] = f["home"].map(sc["sd"])
    f["semnal_kWh_ora"] = f["mean_effect_kwh_per_day"] / 10.0
    f["SNR"] = (f["semnal_kWh_ora"] / f["sd"]).round(2)
    end = z.index[-1]
    f["zile_ramase"] = (end - f["start"]).dt.days
    d_thr = ev.fault_results(cusum_alarms(z, thr), labels, lost).set_index(["home", "type"])["detected"]
    d_per = ev.fault_results(cusum_alarms(z, PERMISSIVE_H), labels, lost).set_index(["home", "type"])["detected"]
    f = f.set_index(["home", "type"])
    f["prins_la_h_ales"] = d_thr.reindex(f.index)
    f["prins_la_h_permisiv"] = d_per.reindex(f.index)
    f = f.reset_index().sort_values("SNR")
    cols = ["home", "type", "capacity_kw", "start", "zile_ramase", "mean_effect_kwh_per_day", "SNR",
            "prins_la_h_ales", "prins_la_h_permisiv"]
    print(f"Defecte simulate pe {args.period}, model '{args.variant}', ajustare '{mode}', h ales = {thr}, h permisiv = {PERMISSIVE_H}")
    print("Ordonate dupa SNR (semnal / zgomot zilnic): cele mai grele primele\n")
    show = f[cols].copy()
    show["start"] = show["start"].dt.strftime("%Y-%m-%d")
    print(show.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    bins = pd.cut(f["SNR"], [0, 0.5, 1, 2, np.inf], labels=["<0.5", "0.5-1", "1-2", ">2"])
    g = f.groupby(bins, observed=True).agg(n=("home", "count"),
                                           prinse_h_ales=("prins_la_h_ales", "mean"),
                                           prinse_h_permisiv=("prins_la_h_permisiv", "mean"))
    print("\nProcent prins pe benzi de SNR:")
    print((g.assign(prinse_h_ales=(g.prinse_h_ales * 100).round(0),
                    prinse_h_permisiv=(g.prinse_h_permisiv * 100).round(0))).to_string())
    miss = f[~f["prins_la_h_permisiv"].astype(bool)]
    print(f"\nNeprinse nici la pragul permisiv: {len(miss)} din {len(f)}; "
          f"SNR median al lor = {miss['SNR'].median():.2f}, al celorlalte = {f.loc[f.index.difference(miss.index), 'SNR'].median():.2f}; "
          f"zile ramase mediane la start: {miss['zile_ramase'].median():.0f}")


if __name__ == "__main__":
    main()