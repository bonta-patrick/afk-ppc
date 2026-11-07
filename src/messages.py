"""
messages.py - texte pentru alarme, in romana si engleza (parametrul lang="ro" | "en"):
  - explain_alert(row, lang):     explicatia pentru echipa (afisata in aplicatie)
  - customer_message(row, econ, lang): ciorna mesajului catre prosumator, pentru alarmele CONTACTEAZA

Mesajul este un SABLON: textul il adapteaza si il aproba un operator uman inainte de trimitere.
Sistemul nu trimite nimic singur. Mesajul nu afirma ca panourile sunt defecte (nu putem sti
de la distanta); spune ce s-a observat, cat ar putea costa si ce poate verifica clientul.

Pierderea afisata clientului este DOAR partea care urmareste soarele:
    pierdere_solara = abatere_totala x parte_solara   (parte_solara limitata la 0..1)
abaterea totala poate include si consum nou, deci ar supraestima productia lipsa.

Intrare: un rand din decisions.csv (n_days, missing, note, excess_kwh_h, est_pierdere_kwh_zi, solar_frac, action)
si, optional, config/economics.json (price_per_kwh, currency).
"""
import numpy as np
import pandas as pd

MIN_DAYS = 14


def nf(x, d=1, lang="ro"):
    """Numar formatat: virgula zecimala in romana, punct in engleza."""
    s = f"{x:.{d}f}"
    return s.replace(".", ",") if lang == "ro" else s


ro = nf  # compatibilitate cu versiunea veche


def _num(row, key):
    v = row.get(key, np.nan)
    return float(v) if pd.notna(v) else np.nan


def _split(row):
    """(abatere totala kWh/zi, parte solara 0..1, pierdere solara kWh/zi)."""
    total = _num(row, "est_pierdere_kwh_zi")
    sun = _num(row, "solar_frac")
    total = max(total, 0.0) if pd.notna(total) else np.nan
    sun = min(max(sun, 0.0), 1.0) if pd.notna(sun) else np.nan
    solar = total * sun if pd.notna(total) and pd.notna(sun) else np.nan
    return total, sun, solar


def explain_alert(row, lang="ro"):
    en = lang == "en"
    action = row.get("action")
    note = row.get("note") if pd.notna(row.get("note", np.nan)) else ""
    n_days = int(row["n_days"]) if pd.notna(row.get("n_days", np.nan)) else 0
    total, sun, solar = _split(row)
    f = lambda x, d=1: nf(x, d, lang)
    if action == "VERIFICA_DATE":
        if note == "few_days":
            if en:
                return (f"Start of period: only {n_days} days of history, too little to tell a failed panel from a new load. "
                        f"We will reassess once we have at least {MIN_DAYS} days.")
            return (f"Început de perioadă: doar {n_days} zile de istoric, prea puțin pentru a deosebi un panou stricat "
                    f"de un consum nou. Reevaluăm când avem cel puțin {MIN_DAYS} zile.")
        miss = _num(row, "missing")
        if en:
            return f"Unreliable data ({f(100 * miss, 0)}% of daytime readings are missing): we check the data before taking any action."
        return f"Date nesigure ({f(100 * miss, 0)}% din citirile de zi lipsesc): verificăm datele înainte de orice acțiune."
    if note == "flat_sun":
        if en:
            return ("Deviation during the day, but the sun varied too little over the last 28 days to tell a failed panel "
                    "from a new load. Waiting for days with different sun.")
        return ("Abatere la orele de zi, dar soarele a variat prea puțin în ultimele 28 de zile ca să deosebim un panou stricat "
                "de un consum nou. Așteptăm zile cu soare diferit.")
    if note == "small_excess" or pd.isna(sun) or pd.isna(total):
        return "Small, inconclusive deviation; we keep monitoring." if en else "Abatere mică și neconcludentă; continuăm monitorizarea."
    if action == "CONTACTEAZA":
        if en:
            strong = "strong indication" if sun >= 0.7 else "moderate indication (medium confidence)"
            return (f"For ~{n_days} days, net consumption during the day has been {f(total)} kWh/day above normal. "
                    f"{f(100 * sun, 0)}% of the deviation rises with the sun: a {strong} that solar production is missing. "
                    f"Estimated production loss: ~{f(solar)} kWh/day.")
        tare = "indiciu puternic" if sun >= 0.7 else "indiciu moderat (încredere medie)"
        return (f"De ~{n_days} zile, la orele de zi consumul net e cu {f(total)} kWh/zi peste normal. "
                f"{f(100 * sun, 0)}% din abatere crește odată cu soarele: {tare} că lipsește producția solară. "
                f"Pierdere estimată din producție: ~{f(solar)} kWh/zi.")
    if en:
        return (f"Net consumption is {f(total)} kWh/day above normal during the day, but only {f(100 * sun, 0)}% of the deviation "
                f"follows the sun: it looks like a new load, not a failed panel. We are not sending a technician; we will reassess.")
    return (f"Consumul net e cu {f(total)} kWh/zi peste normal la orele de zi, dar doar {f(100 * sun, 0)}% din abatere "
            f"urmărește soarele: seamănă cu un consum nou, nu cu un panou stricat. Nu trimitem tehnician; reevaluăm.")


def customer_message(row, econ=None, sender=None, lang="ro"):
    en = lang == "en"
    sender = sender or ("Support team" if en else "Echipa de suport")
    f = lambda x, d=1: nf(x, d, lang)
    n_days = int(row["n_days"]) if pd.notna(row.get("n_days", np.nan)) else 28
    total, sun, solar = _split(row)
    cur_code = (econ or {}).get("currency", "RON")
    cur = ("lei" if cur_code == "RON" and not en else cur_code)
    if en:
        parts = ["Subject: Your solar panels may be producing less than usual", "", "Hello,", ""]
    else:
        parts = ["Subiect: Producția panourilor dumneavoastră pare mai mică decât de obicei", "", "Bună ziua,", ""]
    if pd.notna(total) and pd.notna(solar):
        if en:
            incert = "" if (pd.notna(sun) and sun >= 0.7) else " (with some uncertainty)"
            parts.append(f"Looking at your meter data for roughly the last {n_days} days, we noticed that during daytime hours your net consumption "
                         f"(what you draw from the grid minus what you send back) is about {f(total)} kWh per day higher than we expected. "
                         f"Of that difference, about {f(solar)} kWh per day rises with the sun{incert}: "
                         f"it is larger on sunny days and smaller on cloudy ones.")
        else:
            incert = "" if (pd.notna(sun) and sun >= 0.7) else " (cu o oarecare incertitudine)"
            parts.append(f"Analizând datele contorului dumneavoastră din ultimele aproximativ {n_days} de zile, am observat că, în orele de zi, "
                         f"consumul dumneavoastră net (ce luați din rețea minus ce trimiteți în rețea) este cu circa {f(total)} kWh pe zi mai mare decât ne-am fi așteptat. "
                         f"Din această diferență, aproximativ {f(solar)} kWh pe zi crește odată cu soarele{incert}: "
                         f"este mai mare în zilele însorite și mai mică în cele înnorate.")
    else:
        if en:
            parts.append(f"Looking at your meter data for roughly the last {n_days} days, we noticed that during daytime hours your net consumption "
                         f"is higher than we expected, and the difference seems related to the sun.")
        else:
            parts.append(f"Analizând datele contorului dumneavoastră din ultimele aproximativ {n_days} de zile, am observat că, în orele de zi, "
                         f"consumul dumneavoastră net este mai mare decât ne-am fi așteptat, iar diferența pare legată de soare.")
    parts.append("This can be a sign that your panels or inverter are producing less than they should." if en
                 else "Acesta poate fi un semn că panourile sau invertorul produc mai puțin decât ar trebui.")
    if econ and econ.get("price_per_kwh") and pd.notna(solar):
        cost_day = solar * float(econ["price_per_kwh"])
        price = f(float(econ["price_per_kwh"]), 2)
        if en:
            parts.append(f"If the difference continues, the missing production could mean roughly {f(cost_day)} {cur} per day "
                         f"(about {cost_day * 30:.0f} {cur} per month) more on your bill, at a price of {price} {cur}/kWh.")
        else:
            parts.append(f"Dacă diferența se menține, producția lipsă ar putea însemna aproximativ {f(cost_day)} {cur} pe zi "
                         f"(circa {cost_day * 30:.0f} {cur} pe lună) în plus pe factură, la prețul de {price} {cur}/kWh.")
    if en:
        parts += ["",
                  "We cannot confirm this remotely: it may just as well be a change in how the home uses power (for example a new appliance).",
                  "",
                  "What you can quickly check:",
                  "  - the inverter display or app: does it show errors or zero production?",
                  "  - whether the panels are shaded or covered (leaves, snow, dirt);",
                  "  - whether the breakers or switches of the solar system are on.",
                  "",
                  "If you like, we can help you schedule a check. Simply reply to this message.",
                  "",
                  "Kind regards,",
                  sender]
    else:
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
    demo = {"n_days": 28, "est_pierdere_kwh_zi": 1.8, "solar_frac": 0.48, "action": "CONTACTEAZA"}
    for lg in ("ro", "en"):
        print(explain_alert(demo, lg)); print()
        print(customer_message(demo, {"price_per_kwh": 1.30, "currency": "RON"}, lang=lg)); print("-" * 60)