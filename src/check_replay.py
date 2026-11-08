"""
check_replay.py - verifica modul demo (src/demo_replay.py). Nu modifica nimic pe disc.

    python src/check_replay.py            # toate verificarile
    python src/check_replay.py --quick    # fara comparatia pe tot anul

1. Replay zi cu zi == rularea obisnuita (features.build pe tot anul -> acelasi model -> decide.build_events
   -> decide.assign), pe aceeasi perioada, cu acelasi istoric: aceleasi z, aceleasi alarme, aceleasi actiuni.
   In plus, comparatie cu demo_data/test/decisions.csv (rezultatul oficial salvat al testului).
2. Start / Pauza / Continuare / Reset, pornire dubla, fara pasi suprapusi.
3. Fara cazuri sau notificari duplicate (un caz deschis pe casa, o notificare pe caz).
4. Date lipsa (o casa fara citiri cateva zile, o zi intreaga fara date) si erori (un pas care arunca exceptie).
5. Textele explicatiilor in romana si engleza.
"""
import argparse
import copy
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

import features as F  # noqa: E402
import decide as dc  # noqa: E402
from train_eval import VARIANTS, daily_mean  # noqa: E402
from messages import explain_alert, customer_message  # noqa: E402
import demo_replay as R  # noqa: E402

OK = []


def check(cond, msg):
    print(("  OK   " if cond else "  FAIL ") + msg)
    OK.append(bool(cond))


def batch_run(core):
    """Rularea obisnuita, pe tot anul deodata, cu functiile existente."""
    base = F.load_base()
    X, y, idx, cols = F.build(core.period, base)
    T, J = y.shape
    pred = core.model.predict(X[VARIANTS[dc.VARIANT]]).reshape(T, J)
    daily = daily_mean(y - pred, idx, cols)
    z = daily.sub(pd.Series(core.bias, index=cols)).div(pd.Series(core.sd, index=cols))
    # ca decide.load_inputs
    net = pd.read_parquet(F.PROC / "sim" / core.period / "net_faulty.parquet")
    rad = base["rad"]
    Rm = daily[cols].sub(pd.Series(core.bias, index=cols)).to_numpy(float)
    rad_day = rad[rad.index.hour.isin(dc.DAY_HOURS)].resample("D").mean().reindex(z.index)
    Xm = rad_day.to_numpy(float)[:, None] / 1000.0 * core.capv[None, :]
    Mm = net[net.index.hour.isin(dc.DAY_HOURS)].isna().resample("D").mean().reindex(z.index)[cols].to_numpy(float)
    ev = dc.build_events(z, Rm, Xm, Mm, cols, core.h)
    ev["action"] = dc.assign(ev, core.theta)
    return z, ev


def full_replay(core):
    S = R.ReplaySession(core, start_i=0)
    t0 = time.time()
    while S.i_next <= S.end_i:
        S.step()
    return S, time.time() - t0


def key(df):
    return df.assign(date=pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d"))[["home", "date", "action"]] \
        .sort_values(["home", "date"]).reset_index(drop=True)


def test_equivalence(core):
    print("\n1. Replay zi cu zi vs. rularea obisnuita (aceeasi perioada, acelasi istoric, aceeasi logica)")
    t0 = time.time()
    zb, evb = batch_run(core)
    tb = time.time() - t0
    S, tr = full_replay(core)
    zr = pd.DataFrame(S.zh, index=core.days, columns=core.cols)
    diff = np.nanmax(np.abs(zr.to_numpy() - zb[core.cols].reindex(core.days).to_numpy()))
    nan_same = np.array_equal(np.isnan(zr.to_numpy()), np.isnan(zb[core.cols].reindex(core.days).to_numpy()))
    check(diff < 1e-9 and nan_same, f"z zilnic identic pe {zr.shape[0]} zile x {zr.shape[1]} case (diferenta maxima {diff:.2e})")
    evr = pd.DataFrame(S.decisions)
    a, b = key(evr), key(evb)
    check(a.equals(b), f"aceleasi alarme si actiuni: replay {len(a)}, rulare obisnuita {len(b)}")
    m = evr.merge(evb, on=["home", "date"], suffixes=("_r", "_b"))
    sf = np.nanmax(np.abs(m["solar_frac_r"] - m["solar_frac_b"])) if len(m) else 0.0
    check(sf < 1e-9, f"aceeasi parte solara la fiecare alarma (diferenta maxima {sf:.2e})")
    off = ROOT / "demo_data" / core.period / "decisions.csv"
    if off.exists():
        o = pd.read_csv(off, dtype={"home": str})
        check(a.equals(key(o)), f"identic cu rezultatul oficial salvat ({off.relative_to(ROOT)}, {len(o)} alarme)")
    print(f"     timp: rulare obisnuita {tb:.1f}s, replay {tr:.1f}s pentru {len(core.days)} zile "
          f"({1000 * tr / len(core.days):.0f} ms/zi; median {np.median(S.step_ms):.0f} ms, maxim {np.max(S.step_ms):.0f} ms)")
    return S


def test_no_future(core):
    print("\n1b. Fara informatie din viitor")
    i = 120
    S = R.ReplaySession(core, start_i=0)
    for _ in range(i + 1):
        S.step()
    z_before = S.zh[i].copy()
    # o copie a sursei in care TOATE datele de dupa ziua i (si noaptea zilei i) sunt stricate
    c2 = copy.copy(core)
    c2._N = core._N.copy()
    c2._N[i + 1:] = 99.0
    c2._night = core._night.copy()
    c2._night.iloc[core._night_pos[core.days[i]]:] = 99.0
    r = c2.predict_day(i)
    same = np.allclose(np.nan_to_num(r["z"], nan=-7), np.nan_to_num(z_before, nan=-7))
    check(same, f"ziua {core.days[i]:%Y-%m-%d} nu depinde de datele de dupa ea (nici de noaptea zilei curente)")


def wait_until(cond, timeout=20.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_controls(core, scen):
    print("\n2. Start / Pauza / Continuare / Reset")
    sc = dict(scen, sec_per_day=0.02)
    run = R.ReplayRunner(core, sc, {"currency": "RON", "price_per_kwh": 1.30})
    # detector de pasi suprapusi
    active, overlap = [0], [False]
    orig = run.session.step

    def guarded(*a, **k):
        active[0] += 1
        if active[0] > 1:
            overlap[0] = True
        try:
            return orig(*a, **k)
        finally:
            active[0] -= 1
    run.session.step = guarded
    check(run.start(), "Start porneste replay-ul")
    check(not run.start(), "un al doilea Start este refuzat (un singur replay)")
    th = [threading.Thread(target=run.start) for _ in range(5)]
    [x.start() for x in th]
    [x.join() for x in th]
    n_threads = sum(1 for x in threading.enumerate() if x.name == "dimmer-replay")
    check(n_threads == 1, f"porniri simultane din 5 fire: {n_threads} fir de replay activ")
    check(wait_until(lambda: run.status == "running" and run.session.i_next > run.session.start_i + 5), "trece de istoric si ruleaza")
    run.pause()
    time.sleep(0.15)
    i1 = run.session.i_next
    time.sleep(0.4)
    check(run.status == "paused" and run.session.i_next == i1, f"Pauza opreste progresul (ramane la ziua {i1})")
    e1 = run.elapsed()
    time.sleep(0.2)
    check(abs(run.elapsed() - e1) < 1e-6, "timpul real scurs nu creste in pauza")
    run.resume()
    check(wait_until(lambda: run.session.i_next > i1 + 3), "Continuare reia de unde a ramas")
    check(wait_until(lambda: run.status == "finished", 60), "replay-ul se termina la capatul ferestrei")
    S = run.session
    check(S.i_next == S.end_i + 1, f"toate cele {S.end_i - S.start_i + 1} zile din fereastra au fost procesate o singura data")
    check(not overlap[0], "niciun pas suprapus")
    test_dedup(S)
    n_cases = len(S.cases)
    data_before = core._N.sum() if np.isfinite(core._N).all() else np.nansum(core._N)
    run.reset()
    S2 = run.session
    check(run.status == "idle" and S2.i_next == 0 and not S2.feed and not S2.cases and not S2.notifications
          and not S2.notified, f"Reset curata starea demo (aveam {n_cases} cazuri)")
    check(np.nansum(core._N) == data_before, "Reset nu atinge datele")
    check(run.start(), "dupa Reset se poate porni din nou")
    check(wait_until(lambda: run.status == "running"), "ruleaza din nou")
    run.reset()
    check(run.status == "idle" and run._thread is None and
          sum(1 for x in threading.enumerate() if x.name == "dimmer-replay") == 0, "Reset in timpul rularii opreste firul")


def test_dedup(S):
    print("\n3. Fara duplicate")
    cid = [n["case_id"] for n in S.notifications]
    check(len(cid) == len(set(cid)), f"o singura notificare pe caz ({len(cid)} notificari, {len(set(cid))} cazuri notificate)")
    homes = [c["home"] for c in S.cases.values()]
    check(len(homes) == len(set(homes)), f"cel mult un caz deschis pe casa ({len(homes)} cazuri)")
    contacts = [r for r in S.decisions if r["i"] >= S.start_i and r["action"] == dc.ACTION_CONTACT]
    upd = sum(sum(1 for r in c["events"] if r["action"] == dc.ACTION_CONTACT) - 1 for c in S.cases.values())
    ded = [e for e in S.feed if e["kind"] == "notif_dedup"]
    upd_feed = [e for e in S.feed if e["kind"] == "case_updated"]
    check(len(contacts) == len(S.cases) + upd and upd == len(ded) == len(upd_feed),
          f"{len(contacts)} alarme CONTACTEAZA in fereastra = {len(S.cases)} cazuri noi + {upd} atasate la un caz deschis "
          f"(fiecare cu notificarea suprimata: {len(ded)})")
    # pierderea acumulata nu numara de doua ori aceleasi zile
    bad = [c for c in S.cases.values() if c["loss_days"] > (c["period"][1] - c["period"][0]).days + 1]
    check(not bad, "pierderea acumulata nu aduna zile suprapuse")


def test_missing_and_errors(core, scen):
    print("\n4. Date lipsa si erori")
    S = R.ReplaySession(core, start_i=0, watch=[scen["homes"]["persistent"]])
    j = core.cols.index(scen["homes"]["persistent"])
    try:
        for i in range(40):
            N = core._N[i].copy()
            if 20 <= i < 26:
                N[:, j] = np.nan                      # o casa fara citiri 6 zile
            if i == 30:
                N[:] = np.nan                         # o zi intreaga fara date
            S.step(N)
        ok = True
    except Exception as e:  # noqa: BLE001
        ok = False
        print("    ", e)
    check(ok, "pasii continua cu citiri lipsa")
    check(np.isnan(S.zh[20:26, j]).all() and np.allclose(S.M[20:26, j], 1.0), "zilele fara citiri au abatere necunoscuta si 100% lipsa")
    check(np.isnan(S.zh[30]).all(), "ziua fara date nu produce valori inventate")
    kinds = {e["kind"] for e in S.feed}
    check("missing" in kinds and "missing_fleet" in kinds, "evenimente vizibile pentru citirile lipsa")
    # o fereastra cu prea multe lipsuri trebuie sa duca la VERIFICA_DATE, nu la CONTACTEAZA
    f = dc.event_features(S.R, S.X, np.where(np.isfinite(S.M), 0.5, S.M), 39, j)
    f.update({"home": "x", "date": core.days[39]})
    check(dc.assign(pd.DataFrame([f]), core.theta).iloc[0] == dc.ACTION_DATA, "peste 20% citiri lipsa -> VERIFICA_DATE")

    run = R.ReplayRunner(core, dict(scen, sec_per_day=0.01))
    boom_at = run.session.start_i + 3
    orig = core.predict_day

    def boom(i, N=None):
        if i == boom_at:
            raise RuntimeError("eroare simulata de test")
        return orig(i, N)
    core.predict_day = boom
    try:
        run.start()
        check(wait_until(lambda: run.status == "error", 30), "o exceptie intr-un pas opreste replay-ul cu stare 'error'")
        check(run.error and "eroare simulata" in run.error and any(e["kind"] == "error" for e in run.session.feed),
              "eroarea apare in jurnalul de activitate")
        check(run.session.i_next == boom_at, "nicio zi nu e sarita sau procesata pe jumatate")
    finally:
        core.predict_day = orig
        run.reset()


def test_texts(S):
    print("\n5. Texte RO / EN")
    rows = S.decisions[:200]
    try:
        for r in rows:
            for lg in ("ro", "en"):
                explain_alert(r, lg)
                if r["action"] == dc.ACTION_CONTACT:
                    customer_message(r, {"currency": "RON", "price_per_kwh": 1.3}, lang=lg)
        ok = True
    except Exception as e:  # noqa: BLE001
        print("    ", e)
        ok = False
    check(ok, f"explicatii si mesaje generate in ambele limbi pentru {len(rows)} alarme")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    core = R.ReplayCore()
    print(f"Date si model incarcate o singura data in {core.load_seconds:.1f}s: {len(core.cols)} case, "
          f"{len(core.days)} zile ({core.days[0]:%Y-%m-%d} -> {core.days[-1]:%Y-%m-%d}), h={core.h}, theta={core.theta}")
    scen = R.load_scenario()
    if scen is None:
        sys.exit("Lipseste config/demo_scenario.json. Rulati: python src/demo_scenario.py")
    S = None
    if not args.quick:
        S = test_equivalence(core)
    test_no_future(core)
    test_controls(core, scen)
    test_missing_and_errors(core, scen)
    if S is None:
        S, _ = full_replay(core)
    test_texts(S)
    print(f"\n{sum(OK)}/{len(OK)} verificari trecute in {time.time() - t0:.0f}s")
    sys.exit(0 if all(OK) else 1)


if __name__ == "__main__":
    main()
