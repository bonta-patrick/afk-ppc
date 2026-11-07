"""
features.py - caracteristicile (informatia) pe care o primeste modelul.

REGULA: doar ce ar vedea un furnizor real: citirile NETE ale caselor, capacitatea instalata
din contract si radiatia solara istorica. NICIODATA productia reala a panourilor.

Pentru fiecare casa si fiecare ora de zi (8-17), modelul trebuie sa ghiceasca netul "normal".
Tinta este  y = net - consumul_de_baza,  unde consumul de baza = media netului de NOAPTE
din ultimele 28 de zile (noaptea panourile nu produc, deci netul = consumul).

Trasaturi:
  hour, month, dow, is_dst    calendar (is_dst = ora de vara, ceasul Ausgrid o urmeaza)
  cap                         capacitatea panourilor (kW)
  night_mean                  consumul de baza al casei (ultimele 28 de zile, doar ZILE ANTERIOARE)
  rad, rad_cap                radiatia Open-Meteo si radiatia x capacitate
  nb_yield                    "producti-a" estimata din vecini: mediana celorlalte case (FARA casa
                              curenta) a lui (consum_de_baza - net)/capacitate. Prinde norii locali.
  home                        identitatea casei (orientare, umbra: invatate din anul de antrenare)

Nu folosim niciodata citiri recente de ZI ale casei (ar "invata" defectul ca normal).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROC = Path("data/processed")
CFG = Path("config")

DAY_HOURS = list(range(8, 18))
NIGHT_HOURS = list(range(0, 6))
NIGHT_WINDOW = 28
MIN_NEIGHBORS = 30


def loo_median(Y, min_n=MIN_NEIGHBORS):
    """Pentru fiecare rand si coloana j: mediana celorlalte coloane (fara j). Exact, vectorizat."""
    T, J = Y.shape
    nan = np.isnan(Y)
    n = (~nan).sum(axis=1)
    key = np.where(nan, np.inf, Y)
    order = np.argsort(key, axis=1, kind="stable")
    S = np.take_along_axis(key, order, axis=1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(J), (T, J)), axis=1)
    m = (n // 2)[:, None]
    odd = (n % 2 == 1)[:, None]

    def g(i):
        return np.take_along_axis(S, np.clip(i, 0, J - 1), axis=1)

    a_odd = np.where(rank < m, (g(m) + g(m + 1)) / 2,
                     np.where(rank == m, (g(m - 1) + g(m + 1)) / 2, (g(m - 1) + g(m)) / 2))
    a_even = np.where(rank < m, g(m), g(m - 1))
    out = np.where(odd, a_odd, a_even)
    out[nan] = np.nan
    out[n < min_n, :] = np.nan
    return out


def sydney_dst(ts_index):
    days = ts_index.normalize().unique()
    noon = (days + pd.Timedelta(hours=12)).tz_localize("Australia/Sydney")
    flag = pd.Series([t.dst() != pd.Timedelta(0) for t in noon], index=days)
    return flag.reindex(ts_index.normalize()).to_numpy().astype(int)


def load_base():
    net_real = pd.read_parquet(PROC / "net_hourly.parquet")
    rad = pd.read_parquet(PROC / "radiation_hourly.parquet")["radiation_wm2"]
    cust = pd.read_parquet(PROC / "customers.parquet")
    cap = cust["capacity_kw"].astype(float)
    cap.index = cap.index.astype(str)
    splits = json.loads((CFG / "split.json").read_text(encoding="utf-8"))
    return {"net_real": net_real, "rad": rad, "cap": cap, "splits": splits,
            "all_homes": [str(c) for c in net_real.columns]}


def build(period, base):
    """Returneaza (X, y, idx, cols): X lung (o linie per ora x casa), y matrice (ore x case)."""
    net_real, rad, cap, splits = base["net_real"], base["rad"], base["cap"], base["splits"]
    a, b = splits[period]
    if period == "train":
        net_p = net_real.loc[a:b]
    else:
        net_p = pd.read_parquet(PROC / "sim" / period / "net_faulty.parquet")
    cols = [str(c) for c in net_p.columns]
    idx = net_p.index[net_p.index.hour.isin(DAY_HOURS)]
    N = net_p.loc[idx].to_numpy(float)

    # consumul de baza din netul REAL de noapte (noaptea nu e afectat de niciun scenariu simulat)
    night = net_real[net_real.index.hour.isin(NIGHT_HOURS)].resample("D").mean()
    n28 = night.rolling(NIGHT_WINDOW, min_periods=7).mean().shift(1)     # doar zile anterioare
    n28.columns = [str(c) for c in n28.columns]
    NM = n28[cols].reindex(idx.normalize()).to_numpy(float)

    capv = cap.reindex(cols).to_numpy(float)
    Y = (NM - N) / capv[None, :]                    # "productie" estimata pe kW, per casa si ora
    NB = loo_median(Y)
    R = rad.reindex(idx).to_numpy(float)[:, None]
    RC = R / 1000.0 * capv[None, :]
    T, J = N.shape
    X = pd.DataFrame({
        "hour": np.repeat(idx.hour.to_numpy(), J),
        "month": np.repeat(idx.month.to_numpy(), J),
        "dow": np.repeat(idx.day_of_week.to_numpy(), J),
        "is_dst": np.repeat(sydney_dst(idx), J),
        "cap": np.tile(capv, T),
        "night_mean": NM.ravel(),
        "rad": np.broadcast_to(R, (T, J)).ravel(),
        "rad_cap": RC.ravel(),
        "nb_yield": NB.ravel(),
        "home": pd.Categorical(np.tile(np.array(cols, dtype=object), T), categories=base["all_homes"]),
    })
    y = N - NM
    return X, y, idx, cols