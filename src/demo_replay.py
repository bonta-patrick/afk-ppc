"""
demo_replay.py - modul demo: redare accelerata a datelor istorice, procesate LIVE prin lantul real.

Ce face:
  - incarca O SINGURA DATA modelul antrenat (models/lgbm_full.pkl + scale_full_raw.csv), pragurile
    inghetate (config/model_thresholds.json, config/decision.json) si datele perioadei de test;
  - trimite pipeline-ului, una cate una, ZILE complete (granularitatea reala a modelului: media zilnica
    a reziduului pe orele 8-17 si un CUSUM zilnic); nu exista decizii la 30 de minute;
  - la pasul pentru ziua D foloseste doar ce era disponibil la sfarsitul zilei D:
        citirile nete orare ale zilei D (8-17), consumul de noapte din cele 28 de zile ANTERIOARE
        (exact ca features.build), radiatia orara a zilei D, starea CUSUM si istoricul zilnic pana la D;
  - aplica aceleasi reguli: LightGBM -> abatere zilnica -> z -> CUSUM (k, h) -> alarma (pauza 14 zile)
    -> verificarea soare/consum (decide.event_features + decide.assign) -> actiune;
  - in fereastra demo, o actiune CONTACTEAZA deschide automat un caz DEMO (o casa are cel mult un caz
    deschis, ca in triage.run_queue) si o notificare in inboxul demo (ciorna din messages.customer_message).

Ce NU face: nu reantreneaza, nu schimba praguri, nu scrie nimic pe disc, nu trimite nimic in afara aplicatiei.
Cazurile si notificarile demo traiesc doar in memoria sesiunii (izolate de datele aplicatiei).

Verificarea fata de rularea obisnuita:  python src/check_replay.py
Alegerea scenariului (din rezultatele pipeline-ului):  python src/demo_scenario.py
"""
from pathlib import Path
import json
import threading
import time
import warnings
import numpy as np
import pandas as pd

import features as F
import decide as dc
import pickle
from train_eval import VARIANTS, CUSUM_K, Z_CLIP
from evalkit import COOLDOWN_DAYS

ROOT = Path(__file__).resolve().parents[1]
PROC, CFG = ROOT / "data" / "processed", ROOT / "config"
SCENARIO_PATH = CFG / "demo_scenario.json"
SPEEDS = [2.0, 1.0, 0.5, 0.25, 0.1]        # secunde reale pentru o zi simulata
FEED_MAX = 5000       # evenimente pastrate in memorie (interfata le afiseaza pe ultimele)


# ------------------------------------------------------------------ date + model (o singura data)
class ReplayCore:
    """Datele si modelul, doar pentru citire. Se construieste o singura data pe proces."""

    def __init__(self, period="test", root=ROOT):
        proc, cfg = root / "data" / "processed", root / "config"
        need = [proc / "net_hourly.parquet", proc / "radiation_hourly.parquet", proc / "customers.parquet",
                proc / "sim" / period / "net_faulty.parquet", root / "models" / f"lgbm_{dc.VARIANT}.pkl"]
        missing = [str(p.relative_to(root)) for p in need if not p.exists()]
        if missing:
            raise FileNotFoundError("Lipsesc: " + ", ".join(missing))
        t0 = time.time()
        self.period = period
        splits = json.loads((cfg / "split.json").read_text(encoding="utf-8"))
        a, b = splits[period]
        net_real = pd.read_parquet(proc / "net_hourly.parquet")
        rad = pd.read_parquet(proc / "radiation_hourly.parquet")["radiation_wm2"]
        cust = pd.read_parquet(proc / "customers.parquet")
        cap = cust["capacity_kw"].astype(float)
        cap.index = cap.index.astype(str)
        net_p = pd.read_parquet(proc / "sim" / period / "net_faulty.parquet")
        self.cols = [str(c) for c in net_p.columns]
        self.all_homes = [str(c) for c in net_real.columns]
        J = len(self.cols)

        dec_cfg = json.loads((cfg / "decision.json").read_text(encoding="utf-8"))
        self.mode = dec_cfg.get("mode", "raw")
        self.theta = float(dec_cfg["theta"])
        self.h = float(json.loads((cfg / "model_thresholds.json").read_text(encoding="utf-8"))[self.mode][dc.VARIANT])
        self.k = CUSUM_K
        # o singura incarcare (aceleasi fisiere ca train_eval.load_model, cu cai absolute)
        with open(root / "models" / f"lgbm_{dc.VARIANT}.pkl", "rb") as fh:
            self.model = pickle.load(fh)
        sc = pd.read_csv(root / "models" / f"scale_{dc.VARIANT}_{self.mode}.csv", index_col=0)
        sc.index = sc.index.astype(str)
        self.feats = VARIANTS[dc.VARIANT]
        self.bias = sc["bias"].reindex(self.cols).to_numpy(float)
        self.sd = sc["sd"].reindex(self.cols).to_numpy(float)
        self.capv = cap.reindex(self.cols).to_numpy(float)
        self.cap = pd.Series(self.capv, index=self.cols)

        self.days = pd.date_range(pd.Timestamp(a).normalize(), pd.Timestamp(b).normalize(), freq="D")
        nd, nh = len(self.days), len(F.DAY_HOURS)
        hours = pd.DatetimeIndex([d + pd.Timedelta(hours=x) for d in self.days for x in F.DAY_HOURS])
        # sursa de date ("contorul"): pasul i citeste DOAR felia zilei i
        self._N = net_p.reindex(hours)[self.cols].to_numpy(float).reshape(nd, nh, J)
        self._R = rad.reindex(hours).to_numpy(float).reshape(nd, nh)
        self._dst = F.sydney_dst(hours).reshape(nd, nh)
        # consumul de noapte, pe zile (din netul REAL, ca in features.build); se citesc doar zilele anterioare
        night = net_real[net_real.index.hour.isin(F.NIGHT_HOURS)].resample("D").mean()
        night.columns = [str(c) for c in night.columns]
        self._night = night[self.cols]
        self._night_pos = {d: i for i, d in enumerate(night.index)}
        self.load_seconds = time.time() - t0

    # --- citiri limitate la ziua i (si trecut)
    def day_batch(self, i):
        return self._N[i], self._R[i], self._dst[i]

    def night_baseline(self, day):
        """Media consumului de noapte din cele 28 de zile ANTERIOARE (min. 7 zile cu date), ca features.build.
        Se calculeaza doar pe randurile de dinaintea zilei `day` (aceeasi functie pandas ca rularea obisnuita,
        ca rezultatul sa fie identic bit cu bit; un arbore LightGBM poate avea pragul exact pe aceasta valoare)."""
        pos = self._night_pos.get(day)
        if pos is None or pos == 0:
            return np.full(len(self.cols), np.nan)
        past = self._night.iloc[:pos]
        return past.rolling(F.NIGHT_WINDOW, min_periods=7).mean().iloc[-1].to_numpy(float)

    def predict_day(self, i, N=None):
        """Pasul 1-3 pentru o zi: trasaturi -> LightGBM -> abatere zilnica. Returneaza dict cu valorile zilei."""
        d = self.days[i]
        Nd, Rd, dst = self.day_batch(i)
        if N is not None:
            Nd = N
        T, J = Nd.shape
        NM = self.night_baseline(d)
        Y = (NM[None, :] - Nd) / self.capv[None, :]
        NB = F.loo_median(Y)
        RC = Rd[:, None] / 1000.0 * self.capv[None, :]
        hours = np.array(F.DAY_HOURS)
        X = pd.DataFrame({
            "hour": np.repeat(hours, J),
            "month": np.full(T * J, d.month),
            "dow": np.full(T * J, d.day_of_week),
            "is_dst": np.repeat(dst, J),
            "cap": np.tile(self.capv, T),
            "night_mean": np.broadcast_to(NM, (T, J)).ravel(),
            "rad": np.repeat(Rd, J),
            "rad_cap": RC.ravel(),
            "nb_yield": NB.ravel(),
            "home": pd.Categorical(np.tile(np.array(self.cols, dtype=object), T), categories=self.all_homes),
        })
        pred = self.model.predict(X[self.feats]).reshape(T, J)
        resid = (Nd - NM[None, :]) - pred
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            daily = np.nanmean(resid, axis=0)
            obs = np.nanmean(Nd, axis=0)
            rad_day = np.nanmean(Rd) if np.isfinite(Rd).any() else np.nan
        return {"daily": daily, "obs": obs, "exp": obs - daily, "z": (daily - self.bias) / self.sd,
                "R": daily - self.bias, "X": rad_day / 1000.0 * self.capv, "M": np.isnan(Nd).mean(axis=0),
                "n_readings": int(Nd.size), "n_missing": int(np.isnan(Nd).sum())}


# ------------------------------------------------------------------ starea unei sesiuni demo
def solar_loss_per_day(row):
    """Pierderea solara estimata pe zi = abatere totala x parte solara (aceeasi regula ca messages._split)."""
    tot, sun = row.get("est_pierdere_kwh_zi", np.nan), row.get("solar_frac", np.nan)
    if not (pd.notna(tot) and pd.notna(sun)):
        return np.nan
    return max(float(tot), 0.0) * min(max(float(sun), 0.0), 1.0)


class ReplaySession:
    """Starea pipeline-ului (CUSUM, istoric zilnic) + starea demo (evenimente, cazuri, notificari)."""

    def __init__(self, core, start_i=0, end_i=None, watch=(), econ=None):
        self.core = core
        nd, J = len(core.days), len(core.cols)
        self.start_i = int(start_i)
        self.end_i = int(nd - 1 if end_i is None else end_i)
        self.watch = [str(h) for h in watch]
        self.econ = econ or {}
        self.i_next = 0
        self.s = np.zeros(J)
        self.prev_fire = np.zeros(J, dtype=bool)
        self.last_onset = np.full(J, -10 ** 9)
        nan = lambda: np.full((nd, J), np.nan)
        self.R, self.X, self.M = nan(), nan(), nan()
        self.obs, self.exp, self.zh, self.sh = nan(), nan(), nan(), nan()
        self.decisions = []          # toate alarmele (ca decisions.csv), si din istoricul procesat
        self.feed = []               # evenimente demo (doar in fereastra)
        self.cases = {}              # home -> caz deschis (cel mult unul pe casa)
        self.notifications = []
        self.notified = set()        # deduplicare: id-urile de caz deja notificate
        self.seq = 0
        self.case_seq = 0
        self.step_ms = []

    @property
    def in_window(self):
        return self.i_next > self.start_i

    def _emit(self, kind, sim_day, **kw):
        self.seq += 1
        ev = {"id": self.seq, "kind": kind, "sim_date": None if sim_day is None else pd.Timestamp(sim_day),
              "wall": time.time(), **kw}
        self.feed.append(ev)
        if len(self.feed) > FEED_MAX:
            del self.feed[: len(self.feed) - FEED_MAX]
        return ev

    def step(self, N=None):
        """Proceseaza ziua i_next. N (optional) inlocuieste citirile zilei (folosit doar in teste)."""
        core, i = self.core, self.i_next
        if i > self.end_i:
            return None
        t0 = time.perf_counter()
        d = core.days[i]
        r = core.predict_day(i, N)
        self.R[i], self.X[i], self.M[i] = r["R"], r["X"], r["M"]
        self.obs[i], self.exp[i], self.zh[i] = r["obs"], r["exp"], r["z"]
        # CUSUM cu resetare (aceeasi formula ca train_eval.cusum_alarms)
        Z = np.clip(np.nan_to_num(r["z"], nan=0.0), -Z_CLIP, Z_CLIP)
        s_new = np.maximum(0.0, self.s + Z - core.k)
        fire = s_new > core.h
        s_peak = s_new.copy()
        s_new[fire] = 0.0
        self.s = s_new
        self.sh[i] = s_new
        # alarma noua = aprindere dupa o zi fara alarma, la cel putin 14 zile de precedenta (evalkit.onsets)
        onset = fire & ~self.prev_fire & ((i - self.last_onset) >= COOLDOWN_DAYS)
        self.last_onset[onset] = i
        self.prev_fire = fire
        acting = i >= self.start_i
        watch = set(self.watch)
        if acting:
            for j in np.flatnonzero(r["M"] > 0):
                h = core.cols[j]
                if h in watch:
                    self._emit("missing", d, home=h, frac=float(r["M"][j]))
            if r["n_missing"] > 0.5 * r["n_readings"]:
                self._emit("missing_fleet", d, frac=r["n_missing"] / r["n_readings"])
        for j in np.flatnonzero(onset):
            h = core.cols[j]
            f = dc.event_features(self.R, self.X, self.M, i, j)
            f.update({"home": h, "date": d})
            action = dc.assign(pd.DataFrame([f]), core.theta).iloc[0]
            f["action"] = action
            f["est_pierdere_kwh_zi"] = max(f["excess_kwh_h"], 0) * 10 if pd.notna(f["excess_kwh_h"]) else np.nan
            f["cusum"] = float(s_peak[j])
            f["z_today"] = float(r["z"][j]) if np.isfinite(r["z"][j]) else np.nan
            f["window_lo"] = core.days[max(0, i - dc.WINDOW + 1)]
            f["i"] = i
            self.decisions.append(f)
            if acting:
                self._act(f)
        self.i_next += 1
        ms = 1000 * (time.perf_counter() - t0)
        self.step_ms.append(ms)
        return {"date": d, "ms": ms, "alarms": int(onset.sum()), "missing": r["n_missing"], "readings": r["n_readings"]}

    def _act(self, f):
        """Actiunile automate pentru o alarma din fereastra demo."""
        h, d = f["home"], f["date"]
        self._emit("alarm", d, home=h, cusum=f["cusum"], h=self.core.h, z=f["z_today"])
        self._emit("check", d, home=h, row=dict(f))
        case = self.cases.get(h)
        if f["action"] != dc.ACTION_CONTACT:
            if case is not None:                       # alarma pe o casa cu caz deschis: atasata la caz
                case["events"].append(dict(f))
                self._emit("case_evidence", d, home=h, case_id=case["id"], action=f["action"])
            else:
                self._emit("no_case", d, home=h, action=f["action"])
            return
        loss_day = solar_loss_per_day(f)
        score = (f["est_pierdere_kwh_zi"] if pd.notna(f["est_pierdere_kwh_zi"]) else 0.0) * \
            (min(max(f["solar_frac"], 0.0), 1.0) if pd.notna(f["solar_frac"]) else 0.0)   # = scorul din triage.py
        i, lo = f["i"], max(0, f["i"] - dc.WINDOW + 1)
        if case is None:
            self.case_seq += 1
            case = {"id": f"DEMO-{self.case_seq:03d}", "home": h, "opened": d, "events": [], "priority": score,
                    "loss_kwh": 0.0, "loss_days": 0, "counted_to": lo - 1, "period": [self.core.days[lo], d],
                    "loss_per_day": loss_day}
            self.cases[h] = case
            new = True
        else:
            case["priority"] = max(case["priority"], score)        # ca in triage.run_queue
            new = False
        # pierderea acumulata: doar zilele cu date care nu au fost deja numarate (fara suprapuneri intre alarme)
        a = max(lo, case["counted_to"] + 1)
        ok = np.isfinite(self.R[a:i + 1, self.core.cols.index(h)]) & np.isfinite(self.X[a:i + 1, self.core.cols.index(h)])
        n_new = int(ok.sum())
        if pd.notna(loss_day) and n_new:
            case["loss_kwh"] += loss_day * n_new
            case["loss_days"] += n_new
        case["counted_to"] = i
        case["period"] = [case["period"][0], d]
        case["loss_per_day"] = loss_day
        case["events"].append(dict(f))
        case["updated"] = d
        if new:
            self._emit("case_created", d, home=h, case_id=case["id"], priority=score)
        else:
            self._emit("case_updated", d, home=h, case_id=case["id"], priority=case["priority"])
        if case["id"] in self.notified:
            self._emit("notif_dedup", d, home=h, case_id=case["id"])
        else:
            self.notified.add(case["id"])
            self.notifications.append({"id": f"N-{len(self.notifications) + 1:03d}", "case_id": case["id"], "home": h,
                                       "sim_date": d, "wall": time.time(), "row": dict(f), "channel": "inbox"})
            self._emit("notification", d, home=h, case_id=case["id"])


# ------------------------------------------------------------------ rularea in fundal (un singur fir per sesiune)
class ReplayRunner:
    """Porneste, pune pe pauza, continua si reseteaza replay-ul. Un singur fir de executie, deci pasii nu se
    pot suprapune: daca o zi dureaza mai mult decat intervalul ales, urmatoarea porneste imediat dupa ea."""

    def __init__(self, core, scenario, econ=None):
        self.core = core
        self.scenario = scenario
        self.econ = econ or {}
        self._ctl = threading.Lock()          # pornire/oprire
        self.lock = threading.Lock()          # starea sesiunii (pas vs. citire din interfata)
        self._thread = None
        self._stop = threading.Event()
        self._run = threading.Event()         # setat = ruleaza, sters = pauza
        self.sec_per_day = float(scenario.get("sec_per_day", 1.0))
        self.last_seen = time.time()
        self._new_session()

    def _new_session(self):
        days = self.core.days
        start = days.get_loc(pd.Timestamp(self.scenario["start"]))
        end = min(len(days) - 1, start + int(self.scenario["days"]) - 1)
        watch = [h for h in self.scenario["homes"].values()]
        self.session = ReplaySession(self.core, start, end, watch, self.econ)
        self.status = "idle"
        self.error = None
        self.overruns = 0
        self.real_elapsed = 0.0
        self._t_resume = None
        self.skip = None                      # derulare rapida pana la urmatoarea alarma a unei case

    # --- comenzi
    def start(self):
        with self._ctl:
            if self._thread is not None and self._thread.is_alive():
                return False                  # previne doua replay-uri simultane
            if self.status not in ("idle",):
                return False
            self._stop.clear()
            self._run.set()
            self.status = "warming"
            self._thread = threading.Thread(target=self._loop, name="dimmer-replay", daemon=True)
            self._thread.start()
            return True

    def pause(self):
        if self.status == "running":
            self._run.clear()
            self._account()
            self.status = "paused"
            with self.lock:
                self.session._emit("paused", self.session.core.days[max(0, self.session.i_next - 1)])

    def resume(self):
        if self.status == "paused":
            self.status = "running"
            self._t_resume = time.time()
            with self.lock:
                self.session._emit("resumed", self.session.core.days[max(0, self.session.i_next - 1)])
            self._run.set()

    def reset(self):
        """Opreste executia activa si curata DOAR starea demo (progres, evenimente, cazuri, deduplicare)."""
        with self._ctl:
            self._stop.set()
            self._run.set()
            t = self._thread
            if t is not None:
                t.join(timeout=10)
            self._thread = None
            with self.lock:
                self._new_session()

    def skip_to_next(self, home):
        """Proceseaza zilele fara pauza intre ele pana la urmatoarea alarma a casei `home` (sau finalul ferestrei).
        Zilele trec prin acelasi lant, cu aceleasi actiuni; doar asteptarea dintre pasi dispare."""
        with self._ctl:
            if self.status not in ("running", "paused") or self.skip is not None:
                return False
            was_paused = self.status == "paused"
            with self.lock:
                S = self.session
                self.skip = {"home": str(home), "k": len(S.decisions), "from": S.i_next, "was_paused": was_paused}
                S._emit("skip", S.core.days[max(0, S.i_next - 1)], home=str(home))
            if was_paused:
                self.status = "running"
                self._t_resume = time.time()
                self._run.set()
            return True

    def _check_skip(self, S):
        sk = self.skip
        if sk is None:
            return False
        if any(r["home"] == sk["home"] for r in S.decisions[sk["k"]:]):
            self.skip = None
            S._emit("skip_done", S.core.days[S.i_next - 1], home=sk["home"], n_days=S.i_next - sk["from"])
            return "pause" if sk["was_paused"] else True
        return True

    def set_speed(self, sec):
        self.sec_per_day = float(sec)

    def _account(self):
        if self._t_resume is not None:
            self.real_elapsed += time.time() - self._t_resume
            self._t_resume = None

    def elapsed(self):
        return self.real_elapsed + (time.time() - self._t_resume if self._t_resume is not None else 0.0)

    # --- firul de executie
    def _loop(self):
        S = self.session
        try:
            t0 = time.time()
            n_alarms = 0
            while S.i_next < S.start_i and not self._stop.is_set():      # istoric: aceeasi procesare, fara actiuni
                with self.lock:
                    n_alarms += S.step()["alarms"]
            if self._stop.is_set():
                return
            with self.lock:
                if S.start_i > 0:
                    S._emit("warmup", S.core.days[S.start_i - 1], first=S.core.days[0], n_days=S.start_i,
                            alarms=n_alarms, seconds=time.time() - t0)
                S._emit("started", S.core.days[S.start_i], speed=self.sec_per_day)
            self.status = "running"
            self._t_resume = time.time()
            while S.i_next <= S.end_i and not self._stop.is_set():
                self._run.wait()
                if self._stop.is_set():
                    break
                t = time.perf_counter()
                with self.lock:
                    S.step()
                    sk = self._check_skip(S)
                if sk == "pause":
                    self.pause()
                    continue
                if sk:                                                   # derulare rapida: fara asteptare
                    continue
                dt = time.perf_counter() - t
                if dt > self.sec_per_day:
                    self.overruns += 1
                # asteptare intreruptibila; pauza/reset au efect imediat
                end = time.perf_counter() + max(0.0, self.sec_per_day - dt)
                while not self._stop.is_set() and time.perf_counter() < end:
                    self._stop.wait(min(0.05, max(0.0, end - time.perf_counter())))
                if time.time() - self.last_seen > 180:                 # nimeni nu mai priveste: pauza
                    self.pause()
            if not self._stop.is_set():
                self._account()
                with self.lock:
                    if self.skip is not None:
                        S._emit("skip_none", S.core.days[S.end_i], home=self.skip["home"])
                        self.skip = None
                    S._emit("finished", S.core.days[S.end_i], n_days=S.end_i - S.start_i + 1)
                self.status = "finished"
        except Exception as e:  # noqa: BLE001 - eroarea se afiseaza in interfata, executia se opreste curat
            self._account()
            self.error = f"{type(e).__name__}: {e}"
            with self.lock:
                S._emit("error", S.core.days[min(S.i_next, len(S.core.days) - 1)], message=self.error)
            self.status = "error"


def load_scenario(path=SCENARIO_PATH):
    if not Path(path).exists():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))
