from pathlib import Path

for f in sorted(Path("data/raw").glob("*.csv")):
    print("=" * 80)
    print(f.name, f"({f.stat().st_size/1e6:.1f} MB)")
    with open(f, encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= 6:
                break
            print(i, repr(line[:200]))