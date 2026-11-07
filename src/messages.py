"""
messages.py - ciorna mesajului care ar pleca catre prosumator, pentru alarmele CONTACTEAZA.

Este un SABLON: textul il adapteaza si il aproba un operator uman inainte de trimitere.
Sistemul nu trimite nimic singur. Mesajul nu afirma ca panourile sunt defecte (nu putem sti
de la distanta); spune ce s-a observat, cat ar putea costa si ce poate verifica clientul.

Intrare: un rand din decisions.csv (n_days, est_pierdere_kwh_zi, solar_frac, date) si, optional,
config/economics.json (price_per_kwh, currency).
"""
import numpy as np
import pandas as pd


def ro(x, d=1):
    """Numar cu virgula zecimala, ca in romana."""
    return f"{x:.{d}f}".replace(".", ",")


def zile(n):
    """'14 zile', dar '28 de zile': in romana, 'de' apare cand ultimele doua cifre sunt 00 sau >= 20."""
    r = n % 100
    return f"{n} de zile" if r >= 20 or (r == 0 and n > 0) else f"{n} zile"


def customer_message(row, econ=None, sender="Echipa de suport"):
    n_days = int(row.get("n_days", 28)) if pd.notna(row.get("n_days", np.nan)) else 28
    loss = row.get("est_pierdere_kwh_zi", np.nan)
    sun = row.get("solar_frac", np.nan)
    loss = float(loss) if pd.notna(loss) else np.nan
    sun = float(sun) if pd.notna(sun) else np.nan

    if pd.notna(sun) and sun >= 0.7:
        link = "iar diferența este mai mare în zilele însorite și aproape dispare în cele înnorate"
    else:
        link = "iar diferența pare să fie mai mare în zilele însorite (cu o oarecare incertitudine)"

    parts = ["Subiect: Producția panourilor dumneavoastră pare mai mică decât de obicei", "", "Bună ziua,", ""]
    if pd.notna(loss):
        parts.append(f"Analizând datele contorului dumneavoastră din ultimele aproximativ {zile(n_days)}, am observat că, în orele de zi, "
                     f"luați din rețea cu circa {ro(loss)} kWh pe zi mai mult decât ne-am fi așteptat, {link}.")
    else:
        parts.append(f"Analizând datele contorului dumneavoastră din ultimele aproximativ {zile(n_days)}, am observat că, în orele de zi, "
                     f"luați din rețea mai mult decât ne-am fi așteptat, {link}.")
    parts.append("Acesta poate fi un semn că panourile sau invertorul produc mai puțin decât ar trebui.")
    if econ and econ.get("price_per_kwh") and pd.notna(loss):
        cur = "lei" if econ.get("currency", "RON") == "RON" else econ.get("currency")
        cost_day = loss * float(econ["price_per_kwh"])
        parts.append(f"Dacă diferența se menține, ar putea însemna aproximativ {ro(cost_day)} {cur} pe zi "
                     f"(circa {cost_day * 30:.0f} {cur} pe lună) în plus pe factură, la prețul de {ro(float(econ['price_per_kwh']), 2)} {cur}/kWh.")
    parts += ["",
              "Nu putem confirma acest lucru de la distanță: poate fi la fel de bine o schimbare în consumul locuinței (de exemplu un aparat nou).",
              "",
              "Ce puteți verifica rapid:",
              "  • afișajul sau aplicația invertorului: arată erori sau o producție egală cu zero?",
              "  • dacă panourile sunt umbrite sau acoperite (frunze, zăpadă, murdărie);",
              "  • dacă siguranțele sau întrerupătoarele sistemului fotovoltaic sunt pornite.",
              "",
              "Dacă doriți, vă putem ajuta să programați o verificare. Răspundeți pur și simplu la acest mesaj.",
              "",
              "Cu stimă,",
              sender]
    return "\n".join(parts)


if __name__ == "__main__":
    demo = {"n_days": 28, "est_pierdere_kwh_zi": 3.2, "solar_frac": 0.82}
    print(customer_message(demo, {"price_per_kwh": 1.30, "currency": "RON"}))