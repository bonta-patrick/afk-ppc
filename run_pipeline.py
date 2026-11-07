#!/usr/bin/env python3
"""
run_pipeline.py - ruleaza tot lantul Dimmer, de la datele brute pana la rezultate.

Folosire (din orice folder; scriptul se muta singur in radacina repo-ului):
    python run_pipeline.py                 # pasii 1-8: pana la validare (fara testul final)
    python run_pipeline.py --test          # 1-8 + testul final + demo_data (O SINGURA DATA)
    python run_pipeline.py --only-test     # doar testul final, cu modelele si pragurile deja existente
    python run_pipeline.py --extras        # adauga pasii optionali (ablatia --peer on, inspect_misses, check_test)
    python run_pipeline.py --list          # arata pasii si iese
    python run_pipeline.py --dry-run       # arata comenzile, fara sa le ruleze
    python run_pipeline.py --from 5        # reia de la pasul 5 (ex. dupa o eroare)
    python run_pipeline.py --only 6,7      # ruleaza doar pasii 6 si 7

Reguli:
  * Se opreste la prima eroare si spune la ce pas si ce comanda a esuat.
  * Iesirea fiecarui pas e salvata in UTF-8 in reports/logs/<pas>.txt (nu mai e nevoie de fix_encoding.py).
  * Testul final e protejat: nu ruleaza daca exista deja rezultate pe test (reports/triage_test.csv)
    sau daca pragurile din config/ difera de cele de la tag-ul git `pre-test`. --force-test trece peste.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_DIR = Path("reports/logs")

RAW_FILES = [
    "data/raw/Solar home 2010-2011.csv",
    "data/raw/Solar home 2011-2012.csv",
    "data/raw/Solar home 2012-2013.csv",
    "data/raw/sydney_shortwave_radiation_2010-2013.csv",
]
FROZEN = ["config/baseline_thresholds.json", "config/model_thresholds.json", "config/decision.json"]
PACKAGES = ["pandas", "numpy", "lightgbm", "sklearn", "pyarrow", "matplotlib"]
FREEZE_TAG = "pre-test"


@dataclass
class Step:
    num: str                  # "1", "3a", "9b" ...
    name: str                 # nume scurt, folosit si pentru fisierul de log
    cmd: list[str]            # argumentele pentru python
    group: str                # val | test | extra | extra_test
    needs: list[str] = field(default_factory=list)   # fisiere care trebuie sa existe inainte

    @property
    def order(self) -> tuple[int, str]:
        digits = "".join(c for c in self.num if c.isdigit())
        return int(digits), self.num[len(digits):]


STEPS = [
    Step("1", "prepare", ["src/prepare.py"], "val", RAW_FILES),
    Step("2", "split_and_check", ["src/split_and_check.py"], "val", ["data/processed/net_hourly.parquet"]),
    Step("3a", "simulate_val", ["src/simulate.py", "--period", "val"], "val", ["config/split.json"]),
    Step("3b", "simulate_test", ["src/simulate.py", "--period", "test"], "val", ["config/split.json"]),
    Step("4", "baseline_val", ["src/baseline.py", "--period", "val"], "val"),
    Step("5", "train", ["src/train_eval.py", "--period", "train"], "val", ["config/real_decline_homes.json"]),
    Step("6", "train_eval_val", ["src/train_eval.py", "--period", "val", "--peer", "off"], "val",
         ["models/lgbm_full.pkl"]),
    Step("6x", "train_eval_val_peer", ["src/train_eval.py", "--period", "val", "--peer", "on"], "extra",
         ["models/lgbm_full.pkl"]),
    Step("6y", "inspect_misses", ["src/inspect_misses.py"], "extra", ["models/lgbm_full.pkl"]),
    Step("7", "decide_val", ["src/decide.py", "--period", "val"], "val",
         ["data/processed/z_val_full_raw.parquet"]),
    Step("8", "triage_val", ["src/triage.py", "--period", "val"], "val", ["data/processed/decisions_val.csv"]),
    # ---- testul final: pragurile sunt doar CITITE din config/, nu rescrise
    Step("9a", "baseline_test", ["src/baseline.py", "--period", "test"], "test",
         ["config/baseline_thresholds.json", "data/processed/sim/test"]),
    Step("9b", "train_eval_test", ["src/train_eval.py", "--period", "test", "--peer", "off"], "test",
         ["config/model_thresholds.json", "models/lgbm_full.pkl"]),
    Step("9c", "decide_test", ["src/decide.py", "--period", "test"], "test",
         ["config/decision.json", "data/processed/z_test_full_raw.parquet"]),
    Step("9d", "triage_test", ["src/triage.py", "--period", "test"], "test",
         ["data/processed/decisions_test.csv"]),
    Step("9x", "check_test", ["src/check_test.py"], "extra_test"),
    Step("10", "make_demo_data", ["src/make_demo_data.py", "--period", "test"], "test",
         ["data/processed/resid_test_full.parquet"]),
]


# ------------------------------------------------------------------ verificari
def check_packages() -> list[str]:
    return [p for p in PACKAGES if importlib.util.find_spec(p) is None]


def git(*args: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True)
    except FileNotFoundError:
        return None


def frozen_changed() -> list[str] | None:
    """Fisierele cu praguri care difera de tag-ul pre-test. None = nu putem verifica (fara git/tag)."""
    r = git("rev-parse", "--verify", "--quiet", f"refs/tags/{FREEZE_TAG}")
    if r is None or r.returncode != 0:
        return None
    r = git("diff", "--name-only", FREEZE_TAG, "--", *FROZEN)
    if r is None or r.returncode != 0:
        return None
    return [l for l in r.stdout.splitlines() if l.strip()]


def guard_test(force: bool) -> None:
    problems = []
    if Path("reports/triage_test.csv").exists():
        problems.append("exista deja rezultate pe test (reports/triage_test.csv): testul final s-a rulat.")
    changed = frozen_changed()
    if changed is None:
        print(f"[atentie] nu pot compara pragurile cu tag-ul git '{FREEZE_TAG}' (lipsa git sau tag).")
    elif changed:
        problems.append(f"pragurile difera de tag-ul '{FREEZE_TAG}': {', '.join(changed)}")
    if problems and not force:
        msg = "\n  - ".join(problems)
        raise SystemExit(f"Testul final NU a fost rulat:\n  - {msg}\n"
                         "Testul se ruleaza o singura data, cu pragurile inghetate. "
                         "Daca stiti ce faceti, adaugati --force-test.")
    for p in problems:
        print(f"[atentie, --force-test] {p}")


# ------------------------------------------------------------------ rulare
def run_step(step: Step, total: int, idx: int) -> float:
    missing = [p for p in step.needs if not Path(p).exists()]
    if missing:
        raise SystemExit(f"\nPasul {step.num} ({step.name}) nu poate porni, lipsesc: {', '.join(missing)}\n"
                         f"Rulati intai pasii anteriori (python run_pipeline.py --list).")
    cmd = [sys.executable, *step.cmd]
    print(f"\n{'=' * 78}\n[{idx}/{total}] Pasul {step.num}: {step.name}\n$ python {' '.join(step.cmd)}\n{'=' * 78}",
          flush=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"}
    t0 = time.time()
    log_path = LOG_DIR / f"{step.num}_{step.name}.txt"
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                text=True, encoding="utf-8", errors="replace")
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = proc.wait()
    dt = time.time() - t0
    if code != 0:
        raise SystemExit(f"\nEROARE la pasul {step.num} ({step.name}), cod {code}, dupa {dt:.0f}s.\n"
                         f"Comanda: python {' '.join(step.cmd)}\nJurnal: {log_path}\n"
                         f"Dupa ce reparati, reluati cu: python run_pipeline.py --from {step.num}"
                         f"{' --test' if step.group in ('test', 'extra_test') else ''}")
    print(f"--> pasul {step.num} gata in {dt:.0f}s (jurnal: {log_path})", flush=True)
    return dt


def select(args) -> list[Step]:
    groups = set()
    if not args.only_test:
        groups.add("val")
    if args.test or args.only_test:
        groups.add("test")
    if args.extras:
        groups.add("extra")
        if "test" in groups:
            groups.add("extra_test")
    steps = [s for s in STEPS if s.group in groups]
    if args.only:
        wanted = {x.strip() for x in args.only.split(",")}
        unknown = wanted - {s.num for s in STEPS}
        if unknown:
            raise SystemExit(f"Pasi necunoscuti: {', '.join(sorted(unknown))}. Vedeti --list.")
        steps = [s for s in STEPS if s.num in wanted]
    if args.from_step:
        start = next((s for s in STEPS if s.num == args.from_step), None)
        if start is None:
            raise SystemExit(f"Pasul '{args.from_step}' nu exista. Vedeti --list.")
        steps = [s for s in steps if s.order >= start.order]
    return steps


def main() -> None:
    ap = argparse.ArgumentParser(description="Ruleaza lantul Dimmer de la date brute la rezultate.")
    ap.add_argument("--test", action="store_true", help="adauga testul final (o singura data!) si demo_data")
    ap.add_argument("--only-test", action="store_true", help="doar testul final, cu modelele/pragurile existente")
    ap.add_argument("--extras", action="store_true", help="pasii optionali: ablatia --peer on, inspect_misses, check_test")
    ap.add_argument("--from", dest="from_step", metavar="PAS", help="reia de la acest pas (ex. 5, 9b)")
    ap.add_argument("--only", metavar="PASI", help="doar acesti pasi, separati prin virgula (ex. 6,7)")
    ap.add_argument("--force-test", action="store_true", help="trece peste protectia testului final")
    ap.add_argument("--list", action="store_true", help="arata pasii si iese")
    ap.add_argument("--dry-run", action="store_true", help="arata comenzile fara sa le ruleze")
    args = ap.parse_args()
    os.chdir(ROOT)

    if args.list:
        for s in STEPS:
            print(f"{s.num:>4}  {s.group:<10} python {' '.join(s.cmd)}")
        return

    steps = select(args)
    if not steps:
        raise SystemExit("Niciun pas selectat.")
    # garda priveste doar pasii care EVALUEAZA pe test (9a-9d); pasul 10 doar citeste rezultatele
    runs_test = any(s.group == "test" and s.num != "10" for s in steps)
    trains_or_calibrates = any(s.group == "val" and s.num not in ("1", "2", "3a", "3b") for s in steps)

    if args.dry_run:
        for s in steps:
            print(f"[{s.num}] python {' '.join(s.cmd)}")
        if runs_test:
            print("(testul final ar fi verificat inainte: rezultate existente + praguri fata de tag-ul pre-test)")
        return

    missing = check_packages()
    if missing:
        raise SystemExit(f"Lipsesc pachete Python: {', '.join(missing)}.\n"
                         f"Activati mediul (.venv) sau rulati: {sys.executable} -m pip install -r requirements.txt")
    if runs_test and not trains_or_calibrates:
        guard_test(args.force_test)     # doar test: verificam inainte de orice
    elif runs_test:
        # se recalibreaza pe validare inainte de test: verificam rezultatele vechi acum, pragurile dupa pasul 8
        if Path("reports/triage_test.csv").exists() and not args.force_test:
            guard_test(False)

    t_all, done = time.time(), []
    for i, s in enumerate(steps, 1):
        if s.group == "test" and trains_or_calibrates and not any(d.group == "test" for d in done):
            guard_test(args.force_test)   # pragurile tocmai recalculate trebuie sa fie cele inghetate
        run_step(s, len(steps), i)
        done.append(s)
    print(f"\nGATA: {len(done)} pasi in {time.time() - t_all:.0f}s. Jurnale in {LOG_DIR}/.")
    if not runs_test:
        print("Testul final nu a fost rulat. Cand sunteti gata (o singura data): python run_pipeline.py --only-test")


if __name__ == "__main__":
    main()
