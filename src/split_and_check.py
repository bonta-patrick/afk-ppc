"""
split_and_check.py - pasul 2: imparte datele in timp si face verificarile vizuale.

Ce face:
  1. imparte reteaua neta si radiatia in antrenare / validare / test (cronologic)
  2. salveaza fiecare parte separat in data/processed/splits/
  3. scrie granitele in config/split.json (ca sa le foloseasca toate scripturile)
  4. face 3 grafice de verificare + un raport text in reports/checks/

Disciplina testului: graficele si statisticile folosesc DOAR antrenarea si validarea.
Pentru test se afiseaza doar numarul de randuri si lipsurile.

Rulare:   python src/split_and_check.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # fara fereastra: salveaza direct in fisiere
import matplotlib.pyplot as plt

PROC = Path("data/processed")
SPLIT_DIR = PROC / "splits"
CHK = Path("reports/checks")
CFG = Path("config")
for d in (SPLIT_DIR, CHK, CFG):
    d.mkdir(parents=True, exist_ok=True)

REAL_DECLINE_RATIO = 0.7   # casa e marcata daca randamentul din validare < 70% din cel din antrenare

SPLITS = {
    "train": ("2010-07-01 00:00", "2011-06-30 23:00"),
    "val":   ("2011-07-01 00:00", "2012-06-30 23:00"),
    "test":  ("2012-07-01 00:00", "2013-06-30 23:00"),
}

REPORT = []
def log(msg=""):
    print(msg)
    REPORT.append(str(msg))


def load(name):
    p = PROC / f"{name}.parquet"
    if not p.exists():
        raise SystemExit(f"Lipseste {p}. Rulati intai: python src/prepare.py")
    return pd.read_parquet(p)


def main():
    net = load("net_hourly")
    rad = load("radiation_hourly")["radiation_wm2"]
    cust = load("customers")
    gc, gg = load("hidden_gc_hourly"), load("hidden_gg_hourly")
    cap = cust["capacity_kw"].astype(float)
    cap.index = cap.index.astype(str)           # coloanele din tabele sunt text

    # ---------------------------------------------------------------- 1. impartire
    log("== 1. IMPARTIRE IN TIMP ==")
    for name, (a, b) in SPLITS.items():
        n = net.loc[a:b]
        n.to_parquet(SPLIT_DIR / f"net_{name}.parquet")
        rad.loc[a:b].to_frame().to_parquet(SPLIT_DIR / f"radiation_{name}.parquet")
        log(f"{name:5s} {a[:10]} .. {b[:10]}: {len(n)} ore, {n.shape[1]} case, lipsa {n.isna().mean().mean():.3%}")
    (CFG / "split.json").write_text(json.dumps(SPLITS, indent=2), encoding="utf-8")
    log("Granitele sunt in config/split.json")

    # perioada de lucru pentru verificari: doar antrenare + validare
    tv = slice(SPLITS["train"][0], SPLITS["val"][1])
    gg_tv, gc_tv, rad_tv = gg.loc[tv], gc.loc[tv], rad.loc[tv]
    yld = gg_tv.div(cap, axis=1)               # kWh produsi pe kW instalat, pe ora

    # ---------------------------------------------------------------- 2. randament
    log("\n== 2. RANDAMENT (kWh/kW instalat/zi), doar antrenare si validare ==")
    per_year = {}
    for name in ("train", "val"):
        a, b = SPLITS[name]
        days = pd.date_range(a, b, freq="h").normalize().nunique()
        per_year[name] = gg.loc[a:b].sum().div(cap).div(days)
        q = per_year[name].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        log(f"{name:5s}: mediana {q[0.5]:.2f} | 5%-95%: {q[0.05]:.2f} .. {q[0.95]:.2f}")
    ratio = (per_year["val"] / per_year["train"]).dropna()
    log(f"Raport validare/antrenare: mediana {ratio.median():.3f} (aproape de 1 = ani comparabili)")

    low = per_year["train"].sort_values().head(5)
    log("Cele mai mici randamente in antrenare (posibil sistem slab, sau eroare de capacitate):")
    for h, v in low.items():
        log(f"   casa {h}: {v:.2f}  ({v / per_year['train'].median():.0%} din mediana)")
    drop = ratio.sort_values().head(5)
    log("Cele mai mari scaderi antrenare -> validare (candidate la 'defecte reale probabile'):")
    for h, v in drop.items():
        log(f"   casa {h}: validare = {v:.0%} din antrenare")

    # ---------------------------------------------------------------- 2b. scaderi reale
    flagged = ratio[ratio < REAL_DECLINE_RATIO].sort_values()
    log(f"\n== 2b. CASE CU SCADERE REALA DE PRODUCTIE (validare < {REAL_DECLINE_RATIO:.0%} din antrenare): {len(flagged)} ==")
    monthly = gg_tv.div(cap, axis=1).resample("MS").sum()
    rel = monthly.div(monthly.median(axis=1), axis=0)     # randament lunar fata de casa mediana (anuleaza sezonul)
    if len(flagged):
        for h, v in flagged.items():
            healthy = rel[h].quantile(0.8)
            below = rel[h][rel[h] < 0.6 * healthy]
            start = f"{below.index[0]:%Y-%m}" if len(below) else "-"
            log(f"   casa {h}: validare = {v:.0%} din antrenare; prima luna sub 60% din nivelul propriu sanatos: {start}")
        show = rel[flagged.index[:8]] * 100
        show.index = [f"{d:%Y-%m}" for d in show.index]
        log("Randament lunar relativ (100 = ca mediana caselor); o casa sanatoasa ramane aproximativ constanta:")
        log(show.round(0).astype(int).to_string())
        fig, ax = plt.subplots(figsize=(10, 4))
        for h in flagged.index[:8]:
            ax.plot(rel.index, rel[h] * 100, label=f"casa {h}")
        ax.axhline(100, color="gray", linewidth=0.8)
        ax.set_ylabel("randament lunar relativ (%)")
        ax.set_title("Case cu scadere reala de productie (antrenare + validare)")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout(); fig.savefig(CHK / "4_scaderi_reale.png", dpi=120); plt.close(fig)
    (CFG / "real_decline_homes.json").write_text(
        json.dumps({str(h): round(float(v), 3) for h, v in flagged.items()}, indent=2), encoding="utf-8")
    log("Lista salvata in config/real_decline_homes.json (de exclus din injectare si calibrarea alarmelor false)")

    # ---------------------------------------------------------------- 3. grafice
    # (a) profil orar al productiei pe luni
    fig, ax = plt.subplots(figsize=(8, 4))
    prof = yld.groupby([yld.index.month, yld.index.hour]).mean().mean(axis=1)
    for m, label in ((1, "ian"), (4, "apr"), (7, "iul"), (10, "oct")):
        ax.plot(prof.loc[m].index, prof.loc[m].values, label=label)
    ax.set_xlabel("ora (ceasul din fisierul Ausgrid)")
    ax.set_ylabel("kWh / kW instalat")
    ax.set_title("Productia medie pe ore: trebuie sa fie un deal cu varful catre amiaza")
    ax.legend()
    fig.tight_layout(); fig.savefig(CHK / "1_profil_orar.png", dpi=120); plt.close(fig)

    # (b) o casa tipica, o saptamana insorita si una innorata
    typical = per_year["train"].sort_values().index[len(per_year["train"]) // 2]
    daily_rad = rad_tv.resample("D").mean()
    wk = daily_rad.rolling(7).mean().dropna()
    wk = wk[(wk.index >= wk.index[0] + pd.Timedelta(days=7))]
    ends = {"insorita": wk.idxmax(), "innorata": wk.idxmin()}
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharey=True)
    for ax, (label, end) in zip(axes, ends.items()):
        s, e = end - pd.Timedelta(days=6), end + pd.Timedelta(hours=23)
        ax.plot(gc_tv.loc[s:e, typical], label="consum (GC)")
        ax.plot(gg_tv.loc[s:e, typical], label="productie (GG, ascuns)")
        ax.plot(net.loc[s:e, typical], label="net = consum - productie", linewidth=2)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_title(f"Casa {typical}, saptamana {label} ({s:%Y-%m-%d})")
        ax.set_ylabel("kWh / ora")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(CHK / "2_o_casa_o_saptamana.png", dpi=120); plt.close(fig)

    # (c) radiatie vs productie
    df = pd.DataFrame({"rad": rad_tv, "yld": yld.mean(axis=1)}).dropna()
    df = df[df["rad"] > 10]
    samp = df.sample(min(5000, len(df)), random_state=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(samp["rad"], samp["yld"], s=4, alpha=0.4)
    ax.set_xlabel("radiatie aliniata (W/m2)")
    ax.set_ylabel("productie medie (kWh / kW instalat / ora)")
    ax.set_title("Trebuie sa fie aproape o dreapta")
    fig.tight_layout(); fig.savefig(CHK / "3_radiatie_vs_productie.png", dpi=120); plt.close(fig)
    log(f"\nCorelatie radiatie - productie medie (ore de zi, antrenare+validare): {df['rad'].corr(df['yld']):.3f}")
    log(f"\nGrafice salvate in {CHK}/  (casa tipica folosita: {typical})")

    (CHK / "raport.txt").write_text("\n".join(REPORT), encoding="utf-8")


if __name__ == "__main__":
    main()