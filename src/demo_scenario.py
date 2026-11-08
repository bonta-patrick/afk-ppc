"""
demo_scenario.py - alege scenariul demo RULAND pipeline-ul (src/demo_replay.py), nu prin rezultate scrise de mana.

    python src/demo_scenario.py                 # fereastra de 120 de zile (~2 minute la 1 s/zi)
    python src/demo_scenario.py --days 90 --sec-per-day 1.0

Pasi:
  1. ruleaza replay-ul pe toata perioada de test (aceleasi praguri inghetate, acelasi model);
  2. cauta o fereastra de --days zile in care exista, dupa rezultatele pipeline-ului:
       normal      o casa curata (date reale, nemodificate) fara nicio alarma in fereastra si in cele 28 de zile dinainte;
       temporary   o casa cu o singura alarma in fereastra, pe care verificarea soare/consum NU o trimite
                   la investigatie (MONITORIZEAZA), dupa care nu mai apare nicio alarma in fereastra
                   (afisata in aplicatie ca "abatere pastrata in monitorizare");
       persistent  o casa cu subperformanta cunoscuta (scadere reala de productie sau defect simulat) a carei prima
                   alarma din fereastra duce la CONTACTEAZA (investigatie recomandata); casele curate sunt excluse,
                   pentru ca acolo un CONTACTEAZA este o alarma falsa;
  3. prefera date reale pentru fiecare rol; daca nu exista, foloseste o casa din copia de test cu defect sau consum
     SIMULAT (src/simulate.py, aceeasi simulare ca evaluarea oficiala) si o marcheaza "Anomalie simulata";
  4. scrie config/demo_scenario.json (fereastra, casele, provenienta datelor, regulile de selectie).

Aplicatia nu foloseste rezultatele de aici: in demo, fiecare zi este procesata din nou, live.
Daca nu se gaseste nicio fereastra potrivita, scriptul spune asta si nu forteaza nimic.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_replay as R  # noqa: E402
import decide as dc  # noqa: E402

ROOT = R.ROOT
PRE_QUIET = 28      # zile fara alarme inainte de fereastra (rolurile normal/temporary/persistent)
MIN_WARMUP = 35     # fereastra incepe dupa cel putin 35 de zile de istoric procesat (CUSUM + fereastra de 28 de zile)


def source_of(h, labels, clean, real):
    if h in labels.index:
        r = labels.loc[h]
        return {"data": "simulated", "kind": r["kind"], "type": r["type"], "start": str(pd.Timestamp(r["start"]).date()),
                "params": json.loads(r["params"]), "mean_effect_kwh_per_day": round(float(r["mean_effect_kwh_per_day"]), 2)}
    if h in real:
        return {"data": "real", "kind": "real_decline"}
    return {"data": "real", "kind": "clean" if h in clean else "other"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=120)
    ap.add_argument("--sec-per-day", type=float, default=1.0)
    args = ap.parse_args()
    L = args.days

    core = R.ReplayCore()
    t0 = time.time()
    S = R.ReplaySession(core, start_i=0)
    while S.i_next <= S.end_i:
        S.step()
    print(f"Pipeline rulat pe {len(core.days)} zile in {time.time() - t0:.1f}s: {len(S.decisions)} alarme")
    ev = pd.DataFrame(S.decisions)
    sim = ROOT / "data" / "processed" / "sim" / core.period
    labels = pd.read_csv(sim / "labels.csv", dtype={"home": str}).set_index("home")
    clean = set(json.loads((sim / "clean_homes.json").read_text(encoding="utf-8")))
    real = set(json.loads((sim / "real_decline.json").read_text(encoding="utf-8")))
    by_home = {h: g.sort_values("i") for h, g in ev.groupby("home")}
    nd = len(core.days)
    cap = core.cap

    def events_in(h, a, b):
        g = by_home.get(h)
        return g[(g["i"] >= a) & (g["i"] <= b)] if g is not None else ev.iloc[0:0]

    best = None
    for a in range(MIN_WARMUP, nd - L + 1):
        b = a + L - 1
        cand = {"persistent": [], "temporary": [], "normal": []}
        for h in core.cols:
            pre = events_in(h, a - PRE_QUIET, a - 1)
            win = events_in(h, a, b)
            if len(pre):
                continue
            if not len(win):
                if h in clean:
                    smax = float(np.nanmax(S.sh[a:b + 1, core.cols.index(h)]))
                    cand["normal"].append((smax, -cap[h], h))
                continue
            first = win.iloc[0]
            pos = (first["i"] - a) / L
            if first["action"] == dc.ACTION_CONTACT and 0.45 <= pos <= 0.85:
                is_real = h in real          # scadere reala de productie (date reale); casele curate NU (ar fi alarme false)
                started = (h in labels.index and labels.loc[h, "kind"] == "fault"
                           and a + 14 <= core.days.get_loc(pd.Timestamp(labels.loc[h, "start"]).normalize()) < first["i"])
                if is_real or started:
                    cand["persistent"].append((0 if is_real else 1, abs(pos - 0.65), h))
            if (len(win) == 1 and first["action"] == dc.ACTION_MONITOR and 0.2 <= pos <= 0.6):
                cand["temporary"].append((0 if h in clean else 1, abs(pos - 0.35), h))
        if all(cand.values()):
            pick = {k: sorted(v)[0] for k, v in cand.items()}
            score = (pick["persistent"][0], pick["temporary"][0], a)
            if best is None or score < best[0]:
                best = (score, a, {k: v[-1] for k, v in pick.items()})
    if best is None:
        print(f"!! Nicio fereastra de {L} zile nu contine toate cele trei situatii cu regulile actuale. "
              f"Nu fortez rezultatul: incercati alta durata (--days) sau o anomalie controlata intr-o copie a datelor.")
        sys.exit(1)

    _, a, homes = best
    b = a + L - 1
    out = {
        "period": core.period,
        "start": str(core.days[a].date()),
        "end": str(core.days[b].date()),
        "days": L,
        "sec_per_day": args.sec_per_day,
        "history_from": str(core.days[0].date()),
        "homes": homes,
        "sources": {k: source_of(h, labels, clean, real) for k, h in homes.items()},
        "selection_run": {k: [{"date": str(pd.Timestamp(r["date"]).date()), "action": r["action"],
                               "solar_frac": None if pd.isna(r["solar_frac"]) else round(float(r["solar_frac"]), 3)}
                              for _, r in events_in(h, a, b).iterrows()] for k, h in homes.items()},
        "rules": {
            "normal": "casa curata (date reale), fara alarma in fereastra si in cele 28 de zile dinainte; cea mai mica valoare CUSUM maxima",
            "temporary": "o singura alarma in fereastra, cu actiunea MONITORIZEAZA, intre 20% si 60% din fereastra; preferata o casa curata",
            "persistent": "casa cu scadere reala de productie (date reale) sau cu defect simulat care incepe in fereastra; prima alarma "
                          "din fereastra are actiunea CONTACTEAZA, intre 45% si 85% din fereastra; preferate datele reale",
            "window": f"{L} zile, dupa cel putin {MIN_WARMUP} zile de istoric; se alege prima fereastra cu cele trei roluri "
                      "(preferand date reale)",
        },
        "thresholds": {"cusum_h": core.h, "cusum_k": core.k, "theta": core.theta, "mode": core.mode},
        "generated_by": "python src/demo_scenario.py",
    }
    R.SCENARIO_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Fereastra: {out['start']} -> {out['end']} ({L} zile, ~{L * args.sec_per_day / 60:.1f} min la {args.sec_per_day} s/zi)")
    for k, h in homes.items():
        print(f"  {k:10s} casa {h:>4s} ({cap[h]:.2f} kW) sursa: {out['sources'][k]} | alarme: {out['selection_run'][k]}")
    print(f"Salvat: {R.SCENARIO_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
