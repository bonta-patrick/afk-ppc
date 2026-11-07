"""
baseline.py - doua reguli simple, FARA invatare automata. Modelul vostru trebuie sa le bata.

  B1 "fara export la pranz":  alarma daca, N zile la rand, la pranz casa nu mai trimite curent in retea.
                              (se aplica doar caselor care exportau des in anul de antrenare)
  B2 "abatere fata de anul trecut": compara netul de la pranz cu acelasi interval din anul de antrenare;
                              alarma cand media abaterii pe 7 zile trece un prag.

Pragurile (N pentru B1, h pentru B2) se aleg PE VALIDARE, pe casele curate, astfel incat
alarmele false sa nu depaseasca o tinta (implicit 2 la 1.000 zile-casa). Apoi se INGHEATA.

Rulare:
    python src/baseline.py --period val     # calibreaza pragurile si le salveaza
    python src/baseline.py --period test    # aplica pragurile inghetate (rulati o singura data, la final)
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

import evalkit as ev

CFG = Path("config")
PROC = Path("data/processed")
REPORTS = Path("reports")

TARGET_FA_PER_1000 = 2.0
MIDDAY_HOURS = list(range(10, 16))      # 10:00-15:59 (ceasul din fisierul Ausgrid)
EXPORT_HOURS = list(range(11, 15))      # 11:00-14:59
MIN_EXPORT_RATE = 0.25                  # B1 se aplica doar caselor care exportau >=25% din zile in antrenare
W_B2 = 7                                # fereastra B2, in zile
N_GRID = [3, 5, 7, 10, 14, 21, 30, 45]
H_GRID = [round(x, 2) for x in np.arange(0.5, 8.01, 0.25)]


# ---------------------------------------------------------------- trasaturi zilnice
def daily_midday(net):
    return net[net.index.hour.isin(MIDDAY_HOURS)].resample("D").mean()


def daily_export(net):
    sub = net[net.index.hour.isin(EXPORT_HOURS)]
    return sub.resample("D").min() < 0           # True daca macar o ora de pranz exporta


def run_length(no_export):
    """Cate zile la rand fara export se termina in fiecare zi (pe fiecare casa)."""
    c = no_export.astype(int).cumsum()
    reset = c.where(~no_export).ffill().fillna(0)
    return (c - reset)


# ---------------------------------------------------------------- B1
def b1_prepare(net_train, net_period):
    rate = daily_export(net_train).mean()
    monitorable = rate >= MIN_EXPORT_RATE
    runs = run_length(~daily_export(net_period))
    return runs, monitorable


def b1_state(runs, monitorable, N):
    """Alarma in zilele in care seria fara export atinge N, 2N, 3N... (dupa fiecare alarma contorul se reseteaza)."""
    st = (runs > 0) & (runs % N == 0)
    st.loc[:, ~monitorable.reindex(st.columns).fillna(False).to_numpy()] = False
    return st


# ---------------------------------------------------------------- B2
def b2_prepare(net_train, net_period, years_back):
    mt = daily_midday(net_train)
    ref_s = mt.rolling(15, center=True, min_periods=7).mean()
    sd = (mt - ref_s).std().clip(lower=0.05)
    m = daily_midday(net_period)
    ref_dates = m.index - pd.DateOffset(years=years_back)
    ref = ref_s.reindex(ref_dates)
    ref.index = m.index
    z = (m - ref) / sd
    return z.rolling(W_B2, min_periods=W_B2).mean()


def b2_state(zroll, h):
    return zroll > h


# ---------------------------------------------------------------- calibrare
def calibrate(make_state, grid, labels, lost, clean, target):
    rows = []
    for g in grid:
        sm = ev.summarize(make_state(g), labels, lost, clean)
        rows.append({"param": g, "fa": sm["fa_per_1000"], "recall": ev.overall_recall(sm),
                     "delay": ev.overall_delay(sm), "kwh": ev.overall_kwh(sm)})
    t = pd.DataFrame(rows)
    best, note = ev.pick_threshold(t, target)
    return best, t, note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=["val", "test"], required=True)
    args = ap.parse_args()

    net, labels, lost, clean, real = ev.load_period(args.period)
    net_train = pd.read_parquet(PROC / "splits" / "net_train.parquet")
    years_back = {"val": 1, "test": 2}[args.period]

    runs, monitorable = b1_prepare(net_train, net)
    zroll = b2_prepare(net_train, net, years_back)
    print(f"B1: case monitorizabile (exportau >= {MIN_EXPORT_RATE:.0%} din zile in antrenare): "
          f"{int(monitorable.sum())} din {len(monitorable)}")

    thr_path = CFG / "baseline_thresholds.json"
    if args.period == "val":
        b1_best, b1_tab, n1 = calibrate(lambda N: b1_state(runs, monitorable, N), N_GRID, labels, lost, clean, TARGET_FA_PER_1000)
        b2_best, b2_tab, n2 = calibrate(lambda h: b2_state(zroll, h), H_GRID, labels, lost, clean, TARGET_FA_PER_1000)
        print("\nB1, alegerea lui N (zile fara export):"); print(b1_tab.round(3).to_string(index=False)); print(n1)
        print("\nB2, alegerea pragului h:"); print(b2_tab.round(3).to_string(index=False)); print(n2)
        thr = {"target_fa_per_1000": TARGET_FA_PER_1000,
               "b1_N": int(b1_best["param"]), "b2_h": float(b2_best["param"]), "b2_window_days": W_B2,
               "min_export_rate": MIN_EXPORT_RATE}
        CFG.mkdir(exist_ok=True)
        thr_path.write_text(json.dumps(thr, indent=2), encoding="utf-8")
        print(f"\nPraguri alese pe validare si salvate in {thr_path}: {thr}")
    else:
        if not thr_path.exists():
            raise SystemExit("Rulati intai: python src/baseline.py --period val")
        thr = json.loads(thr_path.read_text(encoding="utf-8"))
        print(f"Praguri INGHETATE din validare: {thr}")

    st1 = b1_state(runs, monitorable, thr["b1_N"])
    st2 = b2_state(zroll, thr["b2_h"])
    s1 = ev.summarize(st1, labels, lost, clean)
    s2 = ev.summarize(st2, labels, lost, clean)
    ev.print_summary(f"B1 fara export la pranz (N={thr['b1_N']} zile), perioada {args.period}", s1)
    ev.print_summary(f"B2 abatere fata de anul trecut (h={thr['b2_h']}), perioada {args.period}", s2)

    # casele cu scadere reala: cand da alarma fiecare metoda (doar informativ)
    print("\nCase cu scadere reala (nu au defecte simulate): prima alarma")
    for h in real:
        if h in st1.columns:
            o1, o2 = ev.onsets(st1[h]), ev.onsets(st2[h])
            print(f"   casa {h}: B1 {o1[0]:%Y-%m-%d}" if len(o1) else f"   casa {h}: B1 -", end="")
            print(f" | B2 {o2[0]:%Y-%m-%d}" if len(o2) else " | B2 -")

    REPORTS.mkdir(exist_ok=True)
    out = {}
    for name, sm in (("B1", s1), ("B2", s2)):
        t = sm["per_type"].copy(); t.insert(0, "metoda", name)
        out[name] = t
    res = pd.concat(out.values())
    res["fa_la_1000_zile_casa"] = np.where(res["metoda"] == "B1", s1["fa_per_1000"], s2["fa_per_1000"]).round(2)
    res.to_csv(REPORTS / f"baseline_{args.period}.csv")
    print(f"\nRezultate salvate in {REPORTS / f'baseline_{args.period}.csv'}")


if __name__ == "__main__":
    main()