"""
prepare.py - curatarea datelor Ausgrid + radiatie Open-Meteo.

Ce face:
  1. citeste cele 3 fisiere Ausgrid (fiecare an are alt format de data)
  2. face citirile orare (suma a doua jumatati de ora; lipsa ramane lipsa)
  3. verifica calitatea si alege casele bune
  4. aliniaza radiatia orara cu contoarele (testeaza singur decalajul orar)
  5. salveaza rezultatele in data/processed/ + un jurnal al curatarii

Rulare din folderul proiectului:   python src/prepare.py
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

# ---- setari (le schimbati aici, nu in cod) ---------------------------------
START, END = "2010-07-01", "2013-06-30"   # perioada asteptata a datelor
MAX_MISSING = 0.05          # eliminam casele cu >5% valori lipsa (GC sau GG)
MAX_GC_KWH_PER_HOUR = 20.0  # prag pentru "consum absurd" (doar raportat)
MAX_GG_OVER_CAPACITY = 1.5  # productie orara > 1.5 x capacitate (doar raportat)
TRAIN_END = "2011-06-30 23:00"   # anul de antrenare: 2010-11
SHIFT_RANGE = range(-12, 13)     # decalaje orare testate pentru radiatie
DATE_FORMATS = ["%d-%b-%y", "%d/%m/%Y"]  # 1-Jul-10  si  1/07/2011

LOG = []
def log(msg=""):
    print(msg)
    LOG.append(str(msg))

# ---------------------------------------------------------------------------
def parse_dates(s, fname):
    """Incearca fiecare format; il accepta doar daca TOATE datele se parseaza."""
    for fmt in DATE_FORMATS:
        parsed = pd.to_datetime(s, format=fmt, errors="coerce")
        if parsed.notna().all():
            log(f"  format data: {fmt}")
            return parsed
    raise ValueError(f"{fname}: nu pot parsa datele. Exemple: {s.head(3).tolist()}")


def load_ausgrid_file(path):
    """Returneaza un tabel cu o linie per (casa, categorie, zi) si 24 coloane orare."""
    log(f"Citesc {path.name}")
    # skiprows=1: primul rand e o nota, nu antetul.
    # index_col=False: unele randuri au virgula in plus la final.
    df = pd.read_csv(path, skiprows=1, index_col=False,
                     usecols=lambda c: not str(c).startswith("Unnamed"))
    df.columns = [str(c).strip() for c in df.columns]

    meta_cols = ["Customer", "Generator Capacity", "Postcode",
                 "Consumption Category", "date"]
    missing = [c for c in meta_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: lipsesc coloanele {missing}. Am: {list(df.columns)[:8]}")

    time_cols = [c for c in df.columns if re.fullmatch(r"\d{1,2}:\d{2}", c)]
    if len(time_cols) != 48 or time_cols[0] != "0:30" or time_cols[-1] != "0:00":
        raise ValueError(f"{path.name}: coloane de timp neasteptate: {time_cols[:3]}...{time_cols[-2:]} (n={len(time_cols)})")

    out = df[meta_cols].copy()
    out["Consumption Category"] = out["Consumption Category"].astype(str).str.strip()
    out["date"] = parse_dates(out["date"].astype(str).str.strip(), path.name)

    before_na = int(df[time_cols].isna().sum().sum())
    vals = df[time_cols].apply(pd.to_numeric, errors="coerce")
    after_na = int(vals.isna().sum().sum())
    log(f"  randuri: {len(df)}, celule goale: {before_na}, celule ne-numerice transformate in lipsa: {after_na - before_na}")

    # Coloanele sunt 0:30, 1:00, ..., 23:30, 0:00 = intervale care SE TERMINA la acea ora.
    # Perechile (0:30,1:00), (1:30,2:00), ... formeaza orele 0-1, 1-2, ..., 23-24.
    # np.sum da NaN daca oricare jumatate lipseste => lipsa ramane lipsa.
    hourly = vals.to_numpy(dtype=float).reshape(len(vals), 24, 2).sum(axis=2)
    out = pd.concat([out.reset_index(drop=True),
                     pd.DataFrame(hourly, columns=list(range(24)))], axis=1)
    return out


def to_wide(rows, channel):
    """Un tabel: linii = ora de inceput, coloane = casa."""
    sub = rows[rows["Consumption Category"] == channel]
    long = sub.melt(id_vars=["Customer", "date"], value_vars=list(range(24)),
                    var_name="hour", value_name="kwh")
    long["ts"] = long["date"] + pd.to_timedelta(long["hour"].astype(int), unit="h")
    wide = long.pivot(index="ts", columns="Customer", values="kwh")
    wide.columns = wide.columns.astype(int)
    return wide.sort_index().sort_index(axis=1)


def find_radiation_shift(gg_mean, rad):
    """aligned[S] = rad[S + k ore]. Alege k cu corelatia maxima cu productia medie."""
    idx = gg_mean.index
    shifted = {k: pd.Series(rad.reindex(idx + pd.Timedelta(hours=k)).to_numpy(), index=idx)
               for k in SHIFT_RANGE}
    overall = {k: gg_mean.corr(s) for k, s in shifted.items()}
    best = max(overall, key=overall.get)
    per_month = {}
    for (y, m), grp in gg_mean.groupby([idx.year, idx.month]):
        c = {k: grp.corr(shifted[k].loc[grp.index]) for k in SHIFT_RANGE}
        per_month[(y, m)] = max(c, key=c.get)
    return best, overall, per_month, shifted


def save(df, name):
    d = df.copy()
    d.columns = [str(c) for c in d.columns]
    d.index.name = "ts" if d.index.name is None else d.index.name
    try:
        d.to_parquet(OUT / f"{name}.parquet")
        log(f"  salvat {name}.parquet  {d.shape}")
    except ImportError:
        d.to_csv(OUT / f"{name}.csv")
        log(f"  (fara pyarrow) salvat {name}.csv  {d.shape}")


# ---------------------------------------------------------------------------
def main():
    files = sorted(RAW.glob("Solar home*.csv"))
    rad_files = sorted(RAW.glob("*radiation*.csv"))
    if len(files) != 3:
        raise SystemExit(f"Astept 3 fisiere 'Solar home*.csv' in {RAW}, am gasit {len(files)}")
    if len(rad_files) != 1:
        raise SystemExit(f"Astept 1 fisier '*radiation*.csv' in {RAW}, am gasit {len(rad_files)}")

    log("== 1. CITIRE SI ORAR ==")
    rows = pd.concat([load_ausgrid_file(f) for f in files], ignore_index=True)
    log(f"Total randuri: {len(rows)}; categorii: {rows['Consumption Category'].value_counts().to_dict()}")
    log(f"Perioada gasita: {rows['date'].min().date()} .. {rows['date'].max().date()}")

    log("\n== 2. DUPLICATE ==")
    key = ["Customer", "Consumption Category", "date"]
    dup = rows.duplicated(key, keep=False)
    log(f"Randuri implicate in duplicate (casa, categorie, zi): {int(dup.sum())}")
    if dup.any():
        rows = rows.drop_duplicates(key, keep="first")
        log("  -> pastrat primul rand din fiecare grup (verificati manual daca valorile difera)")

    log("\n== 3. CASE: capacitate si cod postal ==")
    cust = rows.groupby("Customer").agg(
        capacity_kw=("Generator Capacity", "first"), postcode=("Postcode", "first"),
        n_cap=("Generator Capacity", "nunique"), n_pc=("Postcode", "nunique"))
    log(f"Case distincte: {len(cust)}; cu capacitate inconsistenta: {int((cust.n_cap > 1).sum())}; "
        f"cu cod postal inconsistent: {int((cust.n_pc > 1).sum())}")
    cust = cust.drop(columns=["n_cap", "n_pc"])

    log("\n== 4. TABELE ORARE (GC=consum, GG=productie; CL ignorat) ==")
    gc, gg = to_wide(rows, "GC"), to_wide(rows, "GG")
    full_idx = pd.date_range(f"{START} 00:00", f"{END} 23:00", freq="h")
    for name, w in (("GC", gc), ("GG", gg)):
        outside = int((~w.index.isin(full_idx)).sum())
        if outside:
            log(f"  {name}: {outside} ore in afara perioadei {START}..{END} (ignorate)")
    gc, gg = gc.reindex(full_idx), gg.reindex(full_idx)   # orele lipsa devin NaN
    homes = sorted(set(gc.columns) & set(gg.columns))
    log(f"Case cu ambele canale GC si GG: {len(homes)}")
    gc, gg = gc[homes], gg[homes]

    log("\n== 5. VERIFICARI DE BUN-SIMT ==")
    for name, w in (("GC", gc), ("GG", gg)):
        neg = int((w < 0).sum().sum())
        log(f"{name}: valori negative = {neg} -> transformate in lipsa")
    gc, gg = gc.mask(gc < 0), gg.mask(gg < 0)
    log(f"GC: ore cu consum > {MAX_GC_KWH_PER_HOUR} kWh = {int((gc > MAX_GC_KWH_PER_HOUR).sum().sum())} (doar raportat)")
    cap = cust.loc[homes, "capacity_kw"].astype(float)
    over = int((gg.div(cap, axis=1) > MAX_GG_OVER_CAPACITY).sum().sum())
    log(f"GG: ore cu productie > {MAX_GG_OVER_CAPACITY} x capacitate = {over} (doar raportat)")
    night = gg[gg.index.hour.isin([22, 23, 0, 1, 2, 3])]
    log(f"GG noaptea (22-04): maxim = {np.nanmax(night.to_numpy()):.3f} kWh; "
        f"ore cu > 0.05 kWh = {int((night > 0.05).sum().sum())}  (trebuie sa fie ~0)")

    log("\n== 6. ALEGEREA CASELOR ==")
    miss_gc, miss_gg = gc.isna().mean(), gg.isna().mean()
    train_gg_sum = gg.loc[:TRAIN_END].sum()
    ok = (miss_gc <= MAX_MISSING) & (miss_gg <= MAX_MISSING) & (train_gg_sum > 0)
    log(f"Case pastrate: {int(ok.sum())} din {len(ok)}")
    log(f"  eliminate pt >{MAX_MISSING:.0%} lipsa GC: {int((miss_gc > MAX_MISSING).sum())}, "
        f"GG: {int((miss_gg > MAX_MISSING).sum())}, fara productie in anul de antrenare: {int((train_gg_sum <= 0).sum())}")
    keep = ok[ok].index.tolist()
    gc, gg, cust = gc[keep], gg[keep], cust.loc[keep]
    log(f"Lipsa medie dupa filtrare: GC {gc.isna().mean().mean():.2%}, GG {gg.isna().mean().mean():.2%}")

    log("\n== 7. RADIATIA SI ALINIEREA ==")
    rad = pd.read_csv(rad_files[0], parse_dates=["time"]).set_index("time")["shortwave_radiation"]
    steps = rad.index.to_series().diff().dropna().value_counts()
    log(f"Radiatie: {rad.index.min()} .. {rad.index.max()}, pasi: {steps.to_dict()}, lipsa: {int(rad.isna().sum())}")
    gg_mean = gg.mean(axis=1)
    best, overall, per_month, shifted = find_radiation_shift(gg_mean, rad)
    top = sorted(overall, key=overall.get, reverse=True)[:3]
    log("Cele mai bune decalaje (k ore => aligned[S] = rad[S+k]) si corelatia:")
    for k in top:
        log(f"  k={k:+d}: corr={overall[k]:.4f}")
    month_shifts = sorted(set(per_month.values()))
    log(f"Decalaj ales: k={best:+d}")
    if len(month_shifts) > 1:
        log(f"!! ATENTIE: decalajul optim difera de la o luna la alta: {month_shifts}. "
            f"Probabil ora de vara. Trimiteti-mi tabelul de mai jos.")
        for (y, m), k in per_month.items():
            log(f"   {y}-{m:02d}: k={k:+d}")
    else:
        log("Decalajul e acelasi in toate lunile (nu se vede efect de ora de vara).")
    rad_aligned = shifted[best].rename("radiation_wm2").to_frame()

    log("\n== 8. NET SI SALVARE ==")
    net = gc - gg      # ce ar vedea un contor net; DOAR asta primeste modelul
    save(net, "net_hourly")            # intrare model
    save(rad_aligned, "radiation_hourly")   # intrare model
    save(cust, "customers")            # intrare model (capacitate, cod postal)
    save(gc, "hidden_gc_hourly")       # ascuns: pentru simulator
    save(gg, "hidden_gg_hourly")       # ascuns: adevarul
    (OUT / "cleaning_log.txt").write_text("\n".join(LOG), encoding="utf-8")
    log(f"\nGata. Jurnal: {OUT / 'cleaning_log.txt'}")


if __name__ == "__main__":
    main()