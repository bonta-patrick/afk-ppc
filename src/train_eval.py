"""
train_eval.py (v2) - antreneaza modelul (LightGBM), il compara cu baseline-urile si il evalueaza.

Pasi (rulati-i in ordine):
    python src/train_eval.py --period train   # antreneaza si salveaza modelele (anul 2010-11)
    python src/train_eval.py --period val     # acuratete + alegerea pragului CUSUM pe validare
    python src/train_eval.py --period test    # aplica totul INGHETAT pe test (o singura data, la final)

Variante ("ablatie"):
    own     istoricul casei + calendar + consumul de noapte (are si luna)
    rad     + radiatia solara            (fara luna)
    nb      + "productia" din vecini     (fara luna)
    full    radiatie + vecini            (fara luna)
    full_m  ca full, dar CU luna (versiunea v1, pentru comparatie)

De ce am scos luna: cu un singur an de antrenare, luna si vremea sunt amestecate. Daca decembrie
2010 a fost insorit, modelul invata "decembrie = soare mult" in loc de "radiatie mare = soare mult",
si in alt an greseste la toate casele deodata.

Ajustarea fata de vecini (--peer on, implicit):
    in fiecare zi, o parte din reziduu este comuna tuturor caselor (vremea prost prezisa). Estimam
    acea parte ca mediana reziduurilor pe kW instalat si o scadem. Ramane ce e specific casei.
    --peer off dezactiveaza ajustarea (ca in v1), pentru ablatie.

Cum devine un model o alarma:
  1. modelul ghiceste netul "normal" al casei pe orele de zi
  2. reziduul = net real - net ghicit (pozitiv = mai mult net decat se astepta = lipseste solar?)
  3. media zilnica a reziduului, ajustata fata de vecini, standardizata cu zgomotul normal al casei (z)
  4. CUSUM aduna dovezile pe mai multe zile:  s = max(0, s + z - k);  alarma cand s > h,
     dupa care s revine la 0 (o alarma = o verificare)
  5. h se alege pe VALIDARE, pe casele curate: cel mai sensibil prag care respecta tinta de alarme
     false (aceeasi tinta si acelasi criteriu ca la baseline)
"""
from pathlib import Path
import argparse
import json
import pickle
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

import evalkit as ev
import features as F
from baseline import TARGET_FA_PER_1000

MODELS = Path("models")
CFG = Path("config")
PROC = Path("data/processed")
REPORTS = Path("reports")

OWN = ["hour", "month", "dow", "is_dst", "cap", "night_mean", "home"]
NOMONTH = [f for f in OWN if f != "month"]
VARIANTS = {
    "own": OWN,
    "rad": NOMONTH + ["rad", "rad_cap"],
    "nb": NOMONTH + ["nb_yield"],
    "full": NOMONTH + ["rad", "rad_cap", "nb_yield"],
    "full_m": OWN + ["rad", "rad_cap", "nb_yield"],
}
DEFAULT_VARIANTS = "own,rad,nb,full,full_m"
PARAMS = dict(n_estimators=300, learning_rate=0.07, num_leaves=63, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)
K_FOLDS = 4
CUSUM_K = 0.5
Z_CLIP = 8.0
H_GRID = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 60]


# ------------------------------------------------------------------ utilitare
def fit(X, y):
    return lgb.LGBMRegressor(**PARAMS).fit(X, y)


def daily_mean(R, idx, cols):
    return pd.DataFrame(R, index=idx, columns=cols).resample("D").mean()


def peer_adjust(daily, cap):
    """Scade partea comuna tuturor caselor (reziduu pe kW instalat, mediana pe case), pe fiecare zi."""
    c = cap.reindex(daily.columns).to_numpy(float)
    w = daily.div(c, axis=1).median(axis=1)
    common = pd.DataFrame(np.outer(w.to_numpy(), c), index=daily.index, columns=daily.columns)
    return daily - common


def robust_scale(daily):
    bias = daily.median()
    sd = (1.4826 * (daily - bias).abs().median()).clip(lower=0.02)
    return pd.DataFrame({"bias": bias, "sd": sd})


def cusum_alarms(z, h, k=CUSUM_K):
    """CUSUM cu resetare: s = max(0, s + z - k); cand s > h se da alarma si s revine la 0
    (presupunem ca o alarma duce la o verificare, dupa care dovezile se aduna de la zero)."""
    Z = np.clip(np.nan_to_num(z.to_numpy(float), nan=0.0), -Z_CLIP, Z_CLIP)
    s = np.zeros(Z.shape[1])
    out = np.zeros(Z.shape, dtype=bool)
    for i in range(Z.shape[0]):
        s = np.maximum(0.0, s + Z[i] - k)
        fire = s > h
        out[i] = fire
        s[fire] = 0.0
    return pd.DataFrame(out, index=z.index, columns=z.columns)


def calibrate_h(z, labels, lost, clean, target):
    rows = []
    for h in H_GRID:
        sm = ev.summarize(cusum_alarms(z, h), labels, lost, clean)
        rows.append({"param": h, "fa": sm["fa_per_1000"], "recall": ev.overall_recall(sm),
                     "delay": ev.overall_delay(sm), "kwh": ev.overall_kwh(sm)})
    t = pd.DataFrame(rows)
    best, note = ev.pick_threshold(t, target)
    return best, t, note


def load_model(v, mode):
    with open(MODELS / f"lgbm_{v}.pkl", "rb") as f:
        model = pickle.load(f)
    p = MODELS / f"scale_{v}_{mode}.csv"
    if not p.exists():
        raise SystemExit(f"Lipseste {p}. Rulati din nou: python src/train_eval.py --period train")
    sc = pd.read_csv(p, index_col=0)
    sc.index = sc.index.astype(str)
    return model, sc


# ------------------------------------------------------------------ antrenare
def run_train(variants):
    MODELS.mkdir(exist_ok=True)
    base = F.load_base()
    cap = base["cap"]
    X, y, idx, cols = F.build("train", base)
    T, J = y.shape
    yv = y.ravel()
    home = np.tile(np.array(cols, dtype=object), T)
    flagged = set(json.loads((CFG / "real_decline_homes.json").read_text(encoding="utf-8")).keys())
    is_flag = np.isin(home, list(flagged))
    ok = (~np.isnan(yv)) & X["night_mean"].notna().to_numpy()
    use = ok & ~is_flag
    fold = np.repeat((((idx.normalize() - idx[0].normalize()).days.to_numpy() // 7) % K_FOLDS), J)
    print(f"Antrenare: {int(use.sum())} randuri (ore de zi x case), {len(cols) - len(flagged)} case; "
          f"{len(flagged)} case cu scadere reala excluse din antrenare")

    for v in variants:
        feats = VARIANTS[v]
        t0 = time.time()
        oof = np.full(len(yv), np.nan)
        for f in range(K_FOLDS):
            tr, te = use & (fold != f), use & (fold == f)
            oof[te] = fit(X.loc[tr, feats], yv[tr]).predict(X.loc[te, feats])
        final = fit(X.loc[use, feats], yv[use])
        pred = oof.copy()
        ex = ok & is_flag
        if ex.any():
            pred[ex] = final.predict(X.loc[ex, feats])
        daily = daily_mean((yv - pred).reshape(T, J), idx, cols)
        robust_scale(daily).to_csv(MODELS / f"scale_{v}_raw.csv")
        robust_scale(peer_adjust(daily, cap)).to_csv(MODELS / f"scale_{v}_peer.csv")
        with open(MODELS / f"lgbm_{v}.pkl", "wb") as fh:
            pickle.dump(final, fh)
        mae_oof = np.nanmean(np.abs(yv[use] - oof[use]))
        print(f"[{v:6s}] gata in {time.time() - t0:.0f}s | eroare medie out-of-fold pe antrenare: {mae_oof:.4f} kWh/ora")
        if v == "full":
            imp = pd.Series(final.booster_.feature_importance(importance_type="gain"), index=feats)
            print("   importanta trasaturilor (full):", (100 * imp / imp.sum()).sort_values(ascending=False).round(1).to_dict())
    print(f"Modele salvate in {MODELS}/")


# ------------------------------------------------------------------ evaluare
def profile_mae(base, X, T, J, y, clean_idx, cols):
    """Baseline de acuratete: media casei la aceeasi ora si luna in anul de antrenare."""
    Xt, yt, _, tcols = F.build("train", base)
    Tt = yt.shape[0]
    tdf = pd.DataFrame({"home": np.tile(np.array(tcols, dtype=object), Tt),
                        "hour": Xt["hour"].to_numpy(), "month": Xt["month"].to_numpy(), "y": yt.ravel()})
    prof = tdf.groupby(["home", "hour", "month"])["y"].mean().rename("p").reset_index()
    vdf = pd.DataFrame({"home": np.tile(np.array(cols, dtype=object), T),
                        "hour": X["hour"].to_numpy(), "month": X["month"].to_numpy()})
    p = vdf.merge(prof, how="left", on=["home", "hour", "month"])["p"].to_numpy().reshape(T, J)
    return float(np.nanmean(np.abs(y - p)[:, clean_idx]))


def run_eval(period, variants, mode, verbose):
    freeze = (period == "test")
    base = F.load_base()
    cap = base["cap"]
    net, labels, lost, clean, real = ev.load_period(period)
    X, y, idx, cols = F.build(period, base)
    T, J = y.shape
    clean_set = set(clean)
    clean_idx = [i for i, c in enumerate(cols) if c in clean_set]
    thr_path = CFG / "model_thresholds.json"
    all_thr = json.loads(thr_path.read_text(encoding="utf-8")) if thr_path.exists() else {}
    if freeze and mode not in all_thr:
        raise SystemExit(f"Nu exista praguri pentru modul '{mode}'. Rulati intai: "
                         f"python src/train_eval.py --period val --peer {mode}")
    thr = dict(all_thr.get(mode, {})) if freeze else {}
    print(f"Perioada {period}, ajustare fata de vecini: {mode}. {J} case ({len(clean)} curate), {T // 10} zile")
    base_mae = profile_mae(base, X, T, J, y, clean_idx, cols)
    print(f"Acuratete, baseline 'media casei la aceeasi ora si luna': {base_mae:.4f} kWh/ora (case curate)")

    rows, common = [], {}
    capv = cap.reindex(cols)
    for v in variants:
        model, sc = load_model(v, mode)
        pred = model.predict(X[VARIANTS[v]]).reshape(T, J)
        resid = y - pred
        mae = float(np.nanmean(np.abs(resid[:, clean_idx])))
        daily = daily_mean(resid, idx, cols)
        common[v] = daily.div(capv, axis=1).median(axis=1)
        d = peer_adjust(daily, cap) if mode == "peer" else daily
        z = d.sub(sc["bias"].reindex(cols)).div(sc["sd"].reindex(cols))
        if not freeze:
            best, tab, note = calibrate_h(z, labels, lost, clean, TARGET_FA_PER_1000)
            h = float(best["param"])
            thr[v] = h
            if verbose:
                print(f"\n[{v}] alegerea pragului h pe validare:\n{tab.round(3).to_string(index=False)}")
            if note:
                print(f"[{v}] {note}")
        else:
            h = float(thr[v])
        sm = ev.summarize(cusum_alarms(z, h), labels, lost, clean)
        ev.print_summary(f"MODEL '{v}' (h={h}), perioada {period}, MAE {mae:.4f}", sm)
        row = {"variant": v, "MAE_kWh_ora": round(mae, 4), "h": h, "FA_la_1000": round(sm["fa_per_1000"], 2),
               "max_noi_7zile": sm["max_new_week"],
               "kWh_pierduti_medie": round(ev.overall_kwh(sm), 1)}
        for t, r in sm["per_type"].iterrows():
            row[f"{t}_prinse%"] = r["prinse"]
            row[f"{t}_zile"] = r["zile_mediane_pana_la_detectie"]
        for t, r in sm["per_load"].iterrows():
            row[f"{t}_alarma%"] = r["cu_alarma"]
        rows.append(row)
        if v == "full":
            daily.to_parquet(PROC / f"resid_{period}_{v}.parquet")
            z.to_parquet(PROC / f"z_{period}_{v}_{mode}.parquet")

    if not freeze:
        all_thr[mode] = thr
        CFG.mkdir(exist_ok=True)
        thr_path.write_text(json.dumps(all_thr, indent=2), encoding="utf-8")
        print(f"\nPraguri ({mode}) salvate in {thr_path}: {thr}")
    cb = pd.DataFrame(common)
    cb = cb.groupby(cb.index.month).mean().reindex([7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6]).round(3)
    print("\nBias comun pe luna = mediana reziduului pe kW, peste case (aproape de 0 = vremea e bine explicata):")
    print(cb.to_string())
    tab = pd.DataFrame(rows).set_index("variant")
    REPORTS.mkdir(exist_ok=True)
    tab.to_csv(REPORTS / f"model_{period}_{mode}.csv")
    print(f"\n===== COMPARATIE VARIANTE, perioada {period}, ajustare '{mode}' (baseline MAE {base_mae:.4f}) =====")
    print(tab.T.to_string())
    print(f"\nSalvat in {REPORTS / f'model_{period}_{mode}.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=["train", "val", "test"], required=True)
    ap.add_argument("--variants", default=DEFAULT_VARIANTS)
    ap.add_argument("--peer", choices=["on", "off"], default="on")
    ap.add_argument("--verbose", action="store_true", help="afiseaza tabelul de calibrare a pragului")
    args = ap.parse_args()
    variants = [v.strip() for v in args.variants.split(",")]
    mode = "peer" if args.peer == "on" else "raw"
    if args.period == "train":
        run_train(variants)
    else:
        run_eval(args.period, variants, mode, args.verbose)


if __name__ == "__main__":
    main()