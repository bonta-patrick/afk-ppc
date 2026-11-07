"""
run_pipeline.py - ruleaza tot lantul, in ordine, de la datele brute la rezultate.

    python run_pipeline.py --dry-run       # arata pasii, FARA sa execute nimic (sigur, rulati-l primul)
    python run_pipeline.py                 # pana la validare (fara testul final)
    python run_pipeline.py --test          # adauga si testul final + pregatirea aplicatiei
    python run_pipeline.py --test --yes    # fara intrebarile de confirmare

ATENTIE: lantul REESCRIE data/processed/, models/, reports/ si pragurile din config/.
Rulati-l doar intr-un clone separat, niciodata in repo-ul de lucru, si nu faceti commit din el.
Testul final trebuie rulat O SINGURA DATA, dupa ce pragurile au fost alese pe validare.
Cerinta: fisierele brute in data/raw/ (vezi README, sectiunea Date).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

COMMON = [
    ("curatarea datelor", ["src/prepare.py"]),
    ("impartire in timp + verificari", ["src/split_and_check.py"]),
    ("simulare defecte, validare", ["src/simulate.py", "--period", "val"]),
    ("simulare defecte, test", ["src/simulate.py", "--period", "test"]),
    ("baseline: calibrare pe validare", ["src/baseline.py", "--period", "val"]),
    ("model: antrenare", ["src/train_eval.py", "--period", "train"]),
    ("model: validare, fara ajustare fata de vecini (varianta finala)", ["src/train_eval.py", "--period", "val", "--peer", "off"]),
    ("model: validare, cu ajustare fata de vecini (ablatie)", ["src/train_eval.py", "--period", "val", "--peer", "on"]),
    ("decizie: validare", ["src/decide.py", "--period", "val"]),
    ("triaj: validare", ["src/triage.py", "--period", "val"]),
]
TEST = [
    ("baseline: TEST (praguri inghetate)", ["src/baseline.py", "--period", "test"]),
    ("model: TEST (praguri inghetate)", ["src/train_eval.py", "--period", "test", "--peer", "off"]),
    ("decizie: TEST", ["src/decide.py", "--period", "test"]),
    ("triaj: TEST", ["src/triage.py", "--period", "test"]),
    ("date pentru aplicatie", ["src/make_demo_data.py", "--period", "test"]),
]


def ask(msg):
    return input(msg).strip().lower() == "y"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="ruleaza si testul final")
    ap.add_argument("--yes", action="store_true", help="nu mai cere confirmari")
    ap.add_argument("--dry-run", action="store_true", help="arata pasii, fara sa ruleze nimic")
    args = ap.parse_args()
    steps = list(COMMON) + (list(TEST) if args.test else [])

    missing = [p for p in ("src/prepare.py", "src/simulate.py", "src/train_eval.py") if not Path(p).exists()]
    raw = list(Path("data/raw").glob("Solar home*.csv")) if Path("data/raw").exists() else []
    if args.dry_run:
        print("Pasi care ar rula (nu se executa nimic):")
        for i, (name, cmd) in enumerate(steps, 1):
            print(f"  [{i}/{len(steps)}] {name}:  python {' '.join(cmd)}")
        print(f"\nVerificari: fisiere src/ lipsa: {missing or 'niciunul'}; fisiere brute Ausgrid gasite in data/raw: {len(raw)} (trebuie 3)")
        return
    if missing:
        sys.exit(f"Lipsesc scripturile {missing}. Rulati din radacina repo-ului.")
    if len(raw) != 3:
        sys.exit(f"Astept 3 fisiere 'Solar home*.csv' in data/raw/, am gasit {len(raw)}.")
    if not args.yes:
        if not ask("Lantul REESCRIE data/processed/, models/, reports/ si pragurile din config/. "
                   "Sunteti intr-un clone separat, nu in repo-ul de lucru? Continuati? [y/N] "):
            print("Oprit. Nu s-a modificat nimic.")
            return
        if args.test and not ask("Testul final trebuie rulat O SINGURA DATA, cu pragurile deja inghetate. Continuati? [y/N] "):
            print("Oprit. Rulati fara --test ca sa ajungeti doar pana la validare.")
            return
    t0 = time.time()
    for i, (name, cmd) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {name}:  python {' '.join(cmd)}", flush=True)
        t = time.time()
        res = subprocess.run([sys.executable, *cmd])
        if res.returncode != 0:
            print(f"\nEROARE la pasul {i} ({name}). Rezolvati problema si rulati din nou.")
            sys.exit(res.returncode)
        print(f"    gata in {time.time() - t:.0f}s", flush=True)
    print(f"\nTot lantul a rulat in {(time.time() - t0) / 60:.1f} minute.")


if __name__ == "__main__":
    main()