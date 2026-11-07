"""
Dimmer - aplicatia demo (Streamlit).

Citeste doar din demo_data/test/ (creat de src/make_demo_data.py), deci ruleaza si dupa `git clone`.
    python -m streamlit run app/app.py        (din radacina repo-ului, ca sa se aplice .streamlit/config.toml)

Design: navigare sus, carduri, grafice Altair interactive (tooltip pe fiecare grafic).
Culori: albastru = Dimmer, portocaliu = B1, turcoaz = B2 (validate pentru daltonism);
actiunile au culori de stare + iconita + text, niciodata doar culoare.
Costuri: chihlimbar = energie pierduta, violet = vizite (validate separat).
"""
import base64
import html as _html
import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from triage import run_queue, metrics  # noqa: E402
from messages import customer_message  # noqa: E402

DEMO = ROOT / "demo_data" / "test"
st.set_page_config(page_title="Dimmer", page_icon="☀️", layout="wide", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ design
# Culorile seriilor sunt tonuri medii, lizibile si pe fundal deschis, si pe fundal inchis; textul graficelor
# (axe, legende) il coloreaza tema Streamlit, deci urmeaza automat light/dark.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # Dimmer, B1, B2
MUTED, NEUTRAL, AXIS = "#898781", "#8a8984", "#c3c2b7"
CRIT, WARN = "#d03b3b", "#fab219"
AMBER, VIOLET = "#eda100", "#6250d6"                           # cost: energie, vizite
TRUTH = "#8070e8"                                              # marcajul „adevarului” pe graficul cazului
FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'

ACT = {  # culoare de stare, iconita, text (fundalul pastilei se calculeaza in CSS din culoare)
    "CONTACTEAZA": (CRIT, "☎", "Contactează"),
    "MONITORIZEAZA": (WARN, "◷", "Monitorizează"),
    "VERIFICA_DATE": (MUTED, "⚠", "Verifică datele"),
}
TYPE_RO = {"total": "defect total", "partial": "defect parțial", "intermittent": "defect intermitent",
           "gradual": "defect treptat", "ac": "aer condiționat nou", "pool": "pompă de piscină nouă",
           "ev": "mașină electrică nouă", "wfh": "muncă de acasă"}
OUTCOME_RO = {"tp": "✅ defect găsit și reparat", "real": "🔎 scădere reală (cauză necunoscută)",
              "fp_clean": "❌ vizită irosită (casă fără problemă)", "fp_load": "❌ vizită irosită (doar consum nou)",
              "fp_pre": "❌ vizită irosită (defectul nu începuse)"}
MONTHS_RO = "['ian','feb','mar','apr','mai','iun','iul','aug','sep','oct','noi','dec']"

# Fundalul paginii si widget-urile native le da tema Streamlit (.streamlit/config.toml); CSS-ul de mai jos
# coloreaza doar elementele noastre, prin variabile cu doua seturi: light (implicit) si dark.
LIGHT_TOKENS = """--ink:#0b0b0b;--ink2:#52514e;--muted:#6f6e69;--hair:rgba(11,11,11,.10);--card:#fff;--card-alt:#fbfaf8;
  --soft:#f3f2ee;--accent:#2a78d6;--accent-ink:#1c5cab;--accent-soft:#eaf2fc;--accent-line:#b7d3f6;
  --note-bg:#fff8e6;--note-line:#f1d48a;--note-ink:#3d3a33;--info-bg:#eef5fd;--info-line:#cde2fb;
  --ok-bg:#eef8ee;--ok-line:#cfe9cf;--flag-ink:#6b4e00;--flag-bg:#fff4d6;--shadow:0 1px 2px rgba(11,11,11,.04)"""
DARK_TOKENS = """--ink:#ecebe6;--ink2:#bdbcb5;--muted:#93928b;--hair:rgba(255,255,255,.10);--card:#1c1c1a;--card-alt:#181817;
  --soft:rgba(255,255,255,.045);--accent:#4a8fe0;--accent-ink:#8cbaf3;--accent-soft:rgba(74,143,224,.20);
  --accent-line:rgba(74,143,224,.50);--note-bg:rgba(237,161,0,.10);--note-line:rgba(237,161,0,.38);--note-ink:#dcd9cf;
  --info-bg:rgba(74,143,224,.12);--info-line:rgba(74,143,224,.38);--ok-bg:rgba(27,175,122,.12);
  --ok-line:rgba(27,175,122,.38);--flag-ink:#f3c969;--flag-bg:rgba(250,178,25,.15);--shadow:none"""

CSS = """
<style>
:root, html[data-dm-theme="light"]{%(light)s}
html[data-dm-theme="dark"]{%(dark)s}
.stApp, .stApp button, .stApp input, .stApp textarea, .stApp select{font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
.block-container,[data-testid="stMainBlockContainer"]{padding-top:4.2rem;padding-bottom:3rem;max-width:1240px}
.stApp [class*="st-key-card"]{background:var(--card);border:1px solid var(--hair);border-radius:14px;
  padding:16px 18px 10px;box-shadow:var(--shadow)}
.dm-brand{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:0 0 4px}
.dm-brand .nm{font-size:30px;font-weight:750;line-height:1.05;letter-spacing:-.02em;color:var(--ink)}
.dm-brand .tg{color:var(--ink2);font-size:15px;margin-top:2px}
.dm-chips{display:flex;gap:6px;flex-wrap:wrap;margin-left:auto}
.dm-chip{display:inline-flex;align-items:center;gap:6px;font-size:12.5px;color:var(--ink2);background:var(--card);
  border:1px solid var(--hair);border-radius:999px;padding:3px 10px;white-space:nowrap}
.dm-chip i{width:8px;height:8px;border-radius:50%%;display:inline-block;background:var(--c,#898781)}
.dm-h{font-size:12.5px;font-weight:650;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin:26px 0 10px}
.dm-h2{font-size:21px;font-weight:700;letter-spacing:-.01em;color:var(--ink);margin:4px 0 2px}
.dm-sub{color:var(--ink2);font-size:14.5px;line-height:1.5;margin:0 0 6px;max-width:900px}
.dm-sub b,.dm-lead b,.dm-note b,.dm-reveal b{color:var(--ink)}
.dm-grid{display:grid;gap:12px}
.dm-g4{grid-template-columns:repeat(4,minmax(0,1fr))}
.dm-g3{grid-template-columns:repeat(3,minmax(0,1fr))}
.dm-g2{grid-template-columns:repeat(2,minmax(0,1fr))}
@media (max-width:980px){.dm-g4,.dm-g3{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:600px){.dm-g4,.dm-g3,.dm-g2{grid-template-columns:1fr}}
.dm-card{background:var(--card);border:1px solid var(--hair);border-radius:14px;padding:16px 18px;box-shadow:var(--shadow)}
.dm-step .n{width:26px;height:26px;border-radius:50%%;background:var(--accent-soft);color:var(--accent-ink);font-weight:700;
  font-size:13px;display:flex;align-items:center;justify-content:center;margin-bottom:10px}
.dm-step b{display:block;font-size:15px;margin-bottom:4px;color:var(--ink)}
.dm-step p,.dm-act p{margin:0;color:var(--ink2);font-size:13.5px;line-height:1.45}
.dm-act{border-left:4px solid var(--c)}
.dm-act .dm-pill{margin-bottom:8px}
.dm-tile .l{font-size:13px;color:var(--ink2);line-height:1.3}
.dm-tile .v{font-size:32px;font-weight:650;line-height:1.15;margin:6px 0 4px;color:var(--ink)}
.dm-tile .v small{font-size:15px;font-weight:500;color:var(--ink2);margin-left:4px}
.dm-tile .s{font-size:12.5px;color:var(--muted);line-height:1.4}
.dm-tile.hl,.dm-cmp.hl{border-color:var(--accent-line);box-shadow:inset 0 3px 0 var(--accent)}
.dm-pill{display:inline-flex;align-items:center;gap:6px;font-size:12.5px;font-weight:600;color:var(--ink);
  background:color-mix(in srgb,var(--c) 16%%,transparent);border-radius:999px;padding:3px 10px;white-space:nowrap}
.dm-pill i{width:8px;height:8px;border-radius:50%%;background:var(--c)}
.dm-note{background:var(--note-bg);border:1px solid var(--note-line);border-radius:12px;padding:14px 16px;color:var(--note-ink);
  font-size:14px;line-height:1.55}
.dm-lead{font-size:17px;line-height:1.55;color:var(--ink);max-width:900px;margin:6px 0 0}
.dm-tl .it{display:grid;grid-template-columns:96px 160px 1fr;gap:12px;align-items:start;padding:11px 0;border-top:1px solid var(--hair)}
.dm-tl .it:first-child{border-top:0}
.dm-tl .d{font-variant-numeric:tabular-nums;color:var(--ink2);font-size:13.5px;padding-top:3px}
.dm-tl .t{font-size:14px;line-height:1.5;color:var(--ink)}
@media (max-width:760px){.dm-tl .it{grid-template-columns:1fr;gap:4px}}
.dm-case{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:2px}
.dm-case .t{font-size:22px;font-weight:700;margin-right:6px;color:var(--ink)}
.dm-reveal{background:var(--info-bg);border:1px solid var(--info-line);border-radius:12px;padding:12px 14px;font-size:14px;
  line-height:1.5;color:var(--ink)}
.dm-neu{background:var(--soft);border:1px solid var(--hair);border-radius:12px;padding:12px 14px;font-size:14px;color:var(--ink2);max-width:820px}
.dm-ok{background:var(--ok-bg);border:1px solid var(--ok-line);border-radius:12px;padding:12px 14px;font-size:14px;color:var(--ink)}
.dm-foot{color:var(--muted);font-size:12.5px;margin-top:34px;padding-top:14px;border-top:1px solid var(--hair)}
.dm-list{margin:0;padding-left:18px;color:var(--ink);font-size:14px;line-height:1.6}
.dm-cmp .hd{display:flex;align-items:center;gap:8px;margin-bottom:2px}
.dm-cmp .hd i{width:10px;height:10px;border-radius:3px;background:var(--c)}
.dm-cmp .hd b{font-size:16px;color:var(--ink)}
.dm-cmp .sb{font-size:12.5px;color:var(--muted);margin-bottom:10px}
.dm-cmp .row{display:flex;justify-content:space-between;align-items:baseline;gap:10px;padding:7px 0;border-top:1px solid var(--hair)}
.dm-cmp .row span{font-size:13px;color:var(--ink2)}
.dm-cmp .row strong{font-size:20px;font-weight:650;color:var(--ink);font-variant-numeric:tabular-nums}
.dm-cmp .row strong.best{color:var(--accent-ink)}
.dm-cmp.alt{background:var(--card-alt);border-style:dashed}
.dm-mail{background:var(--card);border:1px solid var(--hair);border-radius:14px;overflow:hidden;max-width:820px}
.dm-mail .top{background:var(--soft);padding:12px 18px;border-bottom:1px solid var(--hair);font-size:13.5px;color:var(--ink2);line-height:1.7}
.dm-mail .top b{color:var(--ink);font-weight:600}
.dm-mail .body{padding:16px 18px;font-size:14.5px;line-height:1.6;color:var(--ink);white-space:pre-wrap}
.dm-mail .flag{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;color:var(--flag-ink);
  background:var(--flag-bg);border-radius:999px;padding:2px 9px;margin-left:6px}
</style>
""" % {"light": LIGHT_TOKENS, "dark": DARK_TOKENS}

# Streamlit nu reruleaza scriptul cand utilizatorul schimba tema din meniu, deci tema activa o citim in browser:
# luminozitatea fundalului pus de Streamlit -> <html data-dm-theme="light|dark">, verificata continuu.
THEME_JS = """<script>
(function () {
  function bg() {
    var els = [document.querySelector('[data-testid="stApp"]'), document.body, document.documentElement];
    for (var i = 0; i < els.length; i++) {
      if (!els[i]) continue;
      var m = (getComputedStyle(els[i]).backgroundColor.match(/[\\d.]+/g) || []).map(Number);
      if (m.length >= 3 && (m.length < 4 || m[3] > 0)) return m;
    }
    return null;
  }
  function check() {
    var m = bg();
    if (!m) return;
    var t = (0.2126 * m[0] + 0.7152 * m[1] + 0.0722 * m[2]) / 255 < 0.5 ? "dark" : "light";
    if (document.documentElement.getAttribute("data-dm-theme") !== t) document.documentElement.setAttribute("data-dm-theme", t);
  }
  check();
  if (!window.__dmThemeWatch) window.__dmThemeWatch = setInterval(check, 250);
})();
</script>"""


def theme_css():
    """CSS-ul aplicatiei; pana porneste scriptul din browser, folosim tema raportata de Streamlit (evita o clipire)."""
    first = f':root:not([data-dm-theme]){{{DARK_TOKENS}}}' if st.context.theme.type == "dark" else ""
    return CSS.replace("</style>", first + "</style>")


LOGO = """<svg xmlns="http://www.w3.org/2000/svg" width="42" height="42" viewBox="0 0 42 42" aria-hidden="true">
<g stroke="#eda100" stroke-width="2.6" stroke-linecap="round">
<line x1="21" y1="3" x2="21" y2="8"/><line x1="21" y1="34" x2="21" y2="39"/><line x1="3" y1="21" x2="8" y2="21"/>
<line x1="34" y1="21" x2="39" y2="21"/><line x1="8.3" y1="8.3" x2="11.8" y2="11.8"/><line x1="30.2" y1="30.2" x2="33.7" y2="33.7"/>
<line x1="8.3" y1="33.7" x2="11.8" y2="30.2"/><line x1="30.2" y1="11.8" x2="33.7" y2="8.3"/></g>
<circle cx="21" cy="21" r="9.5" fill="#eda100"/><path d="M21 11.5 a9.5 9.5 0 0 1 0 19 z" fill="#0b0b0b" opacity=".22"/></svg>"""
# st.html curata SVG-ul inline, deci il trimitem ca imagine
LOGO_IMG = (f'<img alt="" width="42" height="42" src="data:image/svg+xml;base64,'
            f'{base64.b64encode(LOGO.encode()).decode()}">')


# ------------------------------------------------------------------ mici utilitare
def esc(s):
    return _html.escape(str(s))


def ro(x, d=1):
    """Numar in format romanesc: 1.234,5"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    s = f"{x:,.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def pill(action):
    c, icon, label = ACT[action]
    return f'<span class="dm-pill" style="--c:{c}"><i></i>{icon} {label}</span>'


def tile(label, value, sub="", hl=False):
    return (f'<div class="dm-card dm-tile{" hl" if hl else ""}"><div class="l">{label}</div>'
            f'<div class="v">{value}</div><div class="s">{sub}</div></div>')


def grid(items, n):
    return f'<div class="dm-grid dm-g{n}">' + "".join(items) + "</div>"


def section(title):
    st.html(f'<div class="dm-h">{esc(title)}</div>')


def page_title(title, sub=""):
    st.html(f'<div class="dm-h2">{esc(title)}</div>' + (f'<p class="dm-sub">{sub}</p>' if sub else ""))


def smooth(s, n=7):
    return s.rolling(n, min_periods=3).mean()


def date_axis():
    return alt.Axis(title=None, labelExpr=f"{MONTHS_RO}[month(datum.value)] + ' ' + year(datum.value)",
                    tickCount=7, labelAngle=0, grid=False)


def num_axis(title, thousands=False):
    """Axa numerica in format romanesc: 0,5 si 20.000."""
    if thousands:
        return alt.Axis(title=title, format=",.0f", labelExpr="replace(datum.label, ',', '.')", tickCount=5)
    return alt.Axis(title=title, labelExpr="replace(datum.label, '.', ',')", tickCount=5)


def show(chart, height=330):
    """Fara culori de text fixate: Streamlit coloreaza axele si legenda dupa tema activa (si le redeseneaza la
    schimbarea temei); noi doar le estompam cu opacitate, ca sa arate la fel in light si dark."""
    ch = (chart.properties(height=height, background="transparent")
          .configure_view(stroke=None)
          .configure_axis(labelFont=FONT, titleFont=FONT, labelFontSize=11.5, titleFontSize=12, titleFontWeight="normal",
                          labelPadding=6, labelOpacity=0.68, titleOpacity=0.8, gridOpacity=0.55,
                          domainColor=NEUTRAL, domainOpacity=0.6, tickColor=NEUTRAL, tickOpacity=0.6)
          .configure_legend(orient="top", labelFont=FONT, labelFontSize=12.5, titleFontSize=12, titleFont=FONT,
                            titleFontWeight="normal", labelOpacity=0.85, titleOpacity=0.65, symbolStrokeWidth=2.5,
                            columnPadding=18, labelLimit=320)
          .configure_text(font=FONT))
    st.altair_chart(ch, width="stretch", theme=None)


# ------------------------------------------------------------------ date
if not (DEMO / "meta.json").exists():
    st.error("Lipsește folderul demo_data/test. Rulați: `python src/make_demo_data.py --period test`")
    st.stop()


@st.cache_data(show_spinner=False)
def load():
    d = DEMO
    rd = lambda n: pd.read_parquet(d / n)  # noqa: E731
    return {"net": rd("net_day.parquet"), "exp": rd("expected_day.parquet"), "gg": rd("gg_day.parquet"),
            "lost": rd("lost_day.parquet"), "added": rd("added_day.parquet"),
            "labels": pd.read_csv(d / "labels.csv", dtype={"home": str}, parse_dates=["start"]),
            "dec": pd.read_csv(d / "decisions.csv", parse_dates=["date"], dtype={"home": str}),
            "b1": pd.read_csv(d / "b1_events.csv", parse_dates=["date"], dtype={"home": str}),
            "b2": pd.read_csv(d / "b2_events.csv", parse_dates=["date"], dtype={"home": str}),
            "cust": pd.read_csv(d / "customers.csv", dtype={"home": str}).set_index("home"),
            "bt": pd.read_csv(d / "baseline_test.csv"),
            "mt": pd.read_csv(d / "model_test_raw.csv", index_col=0),
            "tri": pd.read_csv(d / "triage_test.csv", dtype={"K_pe_sapt": str}),
            "meta": json.loads((d / "meta.json").read_text(encoding="utf-8"))}


D = load()
meta, labels, dec = D["meta"], D["labels"], D["dec"]
cap = D["cust"]["capacity_kw"]
lab = labels.set_index("home")
real_homes, clean_homes = meta["real"], meta["clean"]
FAULT_TYPES = ["total", "partial", "intermittent", "gradual"]

# rezultate de pe test (din fisierele salvate; cele ale stratului de decizie sunt din reports/test_decide.txt)
_b1 = D["bt"][D["bt"]["metoda"] == "B1"]
_b2 = D["bt"][D["bt"]["metoda"] == "B2"]
R = {"b1_rec": _b1["prinse"].mean(), "b2_rec": _b2["prinse"].mean(),
     "b1_fa": _b1["fa_la_1000_zile_casa"].iloc[0], "b2_fa": _b2["fa_la_1000_zile_casa"].iloc[0],
     "dm_rec": float(np.mean([D["mt"].loc["full", f"{t}_prinse%"] for t in FAULT_TYPES])),
     "dm_fa": float(D["mt"].loc["full", "FA_la_1000"]), "dm_mae": float(D["mt"].loc["full", "MAE_kWh_ora"]),
     "base_mae": 0.5124, "dec_rec": 62, "dec_fa": 1.00, "pool_raw": 100, "pool_dec": 40,
     "b1_kwh": _b1["kWh_pierduti_pana_la_detectie"].mean(), "b2_kwh": _b2["kWh_pierduti_pana_la_detectie"].mean(),
     "dm_kwh": float(D["mt"].loc["full", "kWh_pierduti_medie"]), "dec_kwh": 223}

# preturi (config/economics.json, copiat de make_demo_data.py); valori implicite doar daca lipseste fisierul
ECON_PATH = DEMO / "economics.json"
ECON = (json.loads(ECON_PATH.read_text(encoding="utf-8")) if ECON_PATH.exists()
        else {"currency": "RON", "price_per_kwh": 1.30, "visit_cost": 100.0,
              "price_source": "valoare implicită; completați config/economics.json", "visit_cost_note": "ipoteză"})
CUR = "lei" if ECON.get("currency", "RON") == "RON" else ECON.get("currency", "")


def compare_cards(with_raw=False):
    """Comparatia principala pe test: B1, B2 si Dimmer (model + verificare), cu valorile cele mai bune evidentiate."""
    rows = [("Defecte simulate prinse", "rec", "%", True), ("Alarme false / 1.000 zile-casă", "fa", "", False),
            ("kWh pierduți pe defect", "kwh", "", False)]
    meths = [("B1", "regulă simplă: fără export la prânz", ORANGE,
              {"rec": R["b1_rec"], "fa": R["b1_fa"], "kwh": R["b1_kwh"]}, ""),
             ("B2", "abatere față de anul trecut", AQUA, {"rec": R["b2_rec"], "fa": R["b2_fa"], "kwh": R["b2_kwh"]}, ""),
             ("Dimmer", "model + verificarea „soare sau consum”", BLUE,
              {"rec": R["dec_rec"], "fa": R["dec_fa"], "kwh": R["dec_kwh"]}, " hl")]
    best = {k: (max if hi else min)(m[3][k] for m in meths) for _, k, _, hi in rows}
    if with_raw:
        meths.append(("Dimmer fără verificare", "mai sensibil, dar cu mai multe vizite false", BLUE,
                      {"rec": R["dm_rec"], "fa": R["dm_fa"], "kwh": R["dm_kwh"]}, " alt"))
    fmt = {"rec": lambda v: f"{ro(v, 1 if v % 1 else 0)}%", "fa": lambda v: ro(v, 2), "kwh": lambda v: ro(v, 0)}
    cards = []
    for name, sub, c, vals, cls in meths:
        body = "".join(f'<div class="row"><span>{lbl}</span><strong class="{"best" if vals[k] == best[k] and "alt" not in cls else ""}">'
                       f'{fmt[k](vals[k])}</strong></div>' for lbl, k, _, _ in rows)
        cards.append(f'<div class="dm-card dm-cmp{cls}" style="--c:{c}"><div class="hd"><i></i><b>{name}</b></div>'
                     f'<div class="sb">{sub}</div>{body}</div>')
    return grid(cards, 4 if with_raw else 3)


def home_label(h):
    if h in lab.index:
        r = lab.loc[h]
        return f"Casa {h} · {TYPE_RO.get(r['type'], r['type'])} · {ro(cap[h])} kW"
    if h in real_homes:
        return f"Casa {h} · scădere reală · {ro(cap[h])} kW"
    return f"Casa {h} · {ro(cap[h])} kW"


# ------------------------------------------------------------------ coada (aceeasi logica ca src/triage.py)
POL_B1, POL_B1C, POL_ALL, POL_DM = ("B1 (regulă simplă), în ordinea sosirii", "B1, după capacitatea instalată",
                                    "Model, toate alarmele, după pierderea estimată", "Dimmer: model + decizie, după scor")
POL_NONE = "Fără nicio verificare"
POL_STYLE = {POL_DM: (BLUE, [1, 0]), POL_ALL: (BLUE, [6, 4]), POL_B1: (ORANGE, [1, 0]),
             POL_B1C: (ORANGE, [6, 4]), POL_NONE: (MUTED, [2, 3])}


def policies():
    dd = dec.copy()
    dd["loss"] = dd["est_pierdere_kwh_zi"].fillna(0.0)
    dd["score"] = dd["loss"] * dd["solar_frac"].fillna(0.0).clip(0, 1)
    contact = dd[dd["action"] == "CONTACTEAZA"]
    b1 = D["b1"]
    return {POL_B1: (b1.assign(priority=0.0), "fifo"),
            POL_B1C: (b1.assign(priority=b1["home"].map(cap).fillna(0.0)), "priority"),
            POL_ALL: (dd.assign(priority=dd["loss"]), "priority"),
            POL_DM: (contact.assign(priority=contact["score"]), "priority")}


@st.cache_data(show_spinner=False)
def queue_result(name, K):
    events, order = policies()[name]
    f_start = {r.home: pd.Timestamp(r.start).normalize() for r in labels[labels["kind"] == "fault"].itertuples()}
    days = pd.date_range(meta["start"], meta["end"], freq="D")
    visits, _ = run_queue(events, K, order, f_start, set(real_homes), set(clean_homes), days)
    m = metrics(visits, f_start, D["lost"], days[-1] + pd.Timedelta(hours=23), len(days) / 7.0)
    cum = D["lost"].cumsum()
    tot = pd.Series(0.0, index=days)
    rep = {r.home: r.date for r in visits[visits["outcome"] == "tp"].itertuples()}
    for h in D["lost"].columns:
        c = cum[h].copy()
        if h in rep:
            c.loc[c.index > rep[h]] = c.loc[rep[h]]
        tot = tot.add(c.reindex(days).ffill().fillna(0.0), fill_value=0.0)
    return m, visits, tot


# ================================================================== pagini
def page_overview():
    st.html(f'<p class="dm-lead">Un prosumator are un contor care vede doar <b>diferența</b> dintre ce ia din rețea și ce '
            f'trimite în rețea. Când panourile se strică, nimeni nu observă: furnizorul vede doar un consum mai mare. '
            f'Dar un consum mai mare poate fi și o piscină nouă, o mașină electrică sau munca de acasă. '
            f'<b>Dimmer</b> le deosebește și spune echipei pe cine merită să verifice primul.</p>')

    section("Cum lucrează agentul")
    steps = [("Ghicește normalul", "Un model LightGBM, antrenat pe un an, prezice contorul fiecărei case pe orele de zi, "
              "din radiația solară, calendar și casele vecine. Nu vede niciodată producția panourilor."),
             ("Adună dovezi", "CUSUM pe abaterea zilnică: o zi ciudată nu declanșează nimic, o abatere care persistă da."),
             ("Soare sau consum?", "Lipsa de producție crește odată cu soarele; o pompă de piscină nu. "
              "Așa separă un panou stricat de un consum nou."),
             ("Prioritizează", "Echipa verifică doar K case pe săptămână. Agentul le ordonează după "
              "pierderea estimată × cât de „solar” arată golul.")]
    st.html(grid([f'<div class="dm-card dm-step"><div class="n">{i}</div><b>{t}</b><p>{p}</p></div>'
                  for i, (t, p) in enumerate(steps, 1)], 4))

    section("Ce face cu o alarmă")
    acts = [("CONTACTEAZA", "Golul urmărește soarele: probabil lipsește producția solară. Cazul intră în coada echipei."),
            ("MONITORIZEAZA", "Golul nu urmărește soarele (probabil consum nou) sau nu putem spune încă. Nu trimitem pe nimeni."),
            ("VERIFICA_DATE", "Prea puține zile cu date sau prea multe citiri lipsă. Agentul spune „nu sunt sigur”.")]
    st.html(grid([f'<div class="dm-card dm-act" style="--c:{ACT[a][0]}">{pill(a)}<p>{t}</p></div>' for a, t in acts], 3))

    section("Testul final (2012-07 → 2013-06, rulat o singură dată): Dimmer prinde mai mult, cu mai puține vizite false")
    st.html(compare_cards())
    st.caption("Pentru Dimmer, o alarmă falsă este o alarmă „Contactează” pe o casă curată, adică o vizită irosită; "
               "pentru B1 și B2, orice alarmă duce la o vizită. Pe validare, la același procent de defecte prinse ca B1 (80%), "
               "Dimmer a făcut jumătate din vizitele false (0,57 față de 1,11), cu ~11% mai mulți kWh pierduți. "
               f"Fără verificarea soare/consum, modelul prinde {ro(R['dm_rec'], 1)}%, dar la {ro(R['dm_fa'], 2)} alarme false.")
    tri = D["tri"]
    k3 = tri[tri["K_pe_sapt"] == "3"].set_index("politica")
    dm3 = k3[k3.index.str.startswith("5.")].iloc[0]
    b13 = k3[k3.index.str.startswith("1a.")].iloc[0]
    st.html(grid([
        tile("Eroarea de predicție", f"−{ro(100 * (1 - R['dm_mae'] / R['base_mae']), 0)}%",
             f"{ro(R['dm_mae'], 3)} față de {ro(R['base_mae'], 3)} kWh/oră (media casei la aceeași oră și lună)"),
        tile("Defecte reparate la 3 verificări/săpt.", f"{dm3['defecte_reparate%']:.0f}%",
             f"B1: {b13['defecte_reparate%']:.0f}% · {ro(dm3['kWh_pierduti_pe_defect'], 0)} față de "
             f"{ro(b13['kWh_pierduti_pe_defect'], 0)} kWh pierduți pe defect", hl=True),
        tile("Piscină nouă luată drept defect", f"{R['pool_dec']}%",
             f"față de {R['pool_raw']}% fără verificarea „soare sau consum”"),
    ], 3))

    section("Ce nu pretindem")
    st.html('<div class="dm-note"><b>Defectele sunt simulate</b> pe producție reală; nu am confirmat defecte fizice. '
            'Nu prindem defectele totale mai repede decât o regulă simplă (22,5 zile față de 9), iar '
            '<b>aerul condiționat rămâne confundat</b> cu un defect (rulează în zilele însorite). '
            'Datele sunt din Australia (Ausgrid, 2010–2013): <b>metoda se transferă, cifrele nu</b>.</div>')


def page_case():
    page_title("Un caz, pe larg",
               "Ce vede sistemul (doar contorul net), ce aștepta modelul și ce a decis agentul. "
               "Apoi apăsați „Arată adevărul” ca să vedeți ce s-a întâmplat de fapt cu panourile.")
    faults = labels[labels["kind"] == "fault"].sort_values("mean_effect_kwh_per_day", ascending=False)["home"].tolist()
    loads = labels[labels["kind"] == "load"].sort_values("mean_effect_kwh_per_day", ascending=False)["home"].tolist()
    n_al = dec.groupby("home").size()
    clean_sorted = sorted(clean_homes, key=lambda h: -int(n_al.get(h, 0)))
    cats = {"Defect simulat": faults, "Consum nou": loads, "Scădere reală": real_homes, "Casă curată": clean_sorted}

    left, right = st.columns([1.25, 3], gap="medium")
    with left.container(key="card_pick"):
        cat = st.pills("Tipul casei", list(cats), default="Defect simulat", required=True, key="cat")
        if not cats[cat]:
            st.info("Nu există case în această categorie în datele curente.")
            return
        h = st.selectbox("Casa", cats[cat], key=f"home_{cat}",
                         format_func=lambda x: home_label(x) + (f" · {int(n_al.get(x, 0))} alarme" if cat == "Casă curată" else ""))
        truth = st.toggle("🔍 Arată adevărul", key="truth", help="Producția reală e ascunsă pentru sistem; o afișăm doar aici.")
        st.caption("Sistemul vede doar netul. Producția panourilor e „adevărul ascuns”, folosit doar la evaluare.")

    ev_h = dec[dec["home"] == h].sort_values("date")
    with right.container(key="card_chart"):
        chips = [f'<span class="t">Casa {esc(h)}</span>',
                 f'<span class="dm-chip">{ro(cap[h])} kW instalați</span>',
                 f'<span class="dm-chip">cod poștal {esc(D["cust"].loc[h, "postcode"])}</span>',
                 f'<span class="dm-chip">{len(ev_h)} alarme</span>']
        if len(ev_h):
            last = ev_h.iloc[-1]
            chips.append(pill(last["action"]))
            if pd.notna(last["est_pierdere_kwh_zi"]) and last["action"] == "CONTACTEAZA":
                chips.append(f'<span class="dm-chip">pierdere estimată ≈ {ro(last["est_pierdere_kwh_zi"])} kWh/zi</span>')
        st.html(f'<div class="dm-case">{"".join(chips)}</div>')

        net, exp = D["net"][h], D["exp"][h]
        df = pd.DataFrame({"date": net.index, "real": smooth(net).to_numpy(), "exp": smooth(exp).to_numpy()})
        df["hi"] = df[["real", "exp"]].max(axis=1)
        df["zi"] = df["date"].dt.strftime("%d.%m.%Y")
        df["real_s"] = df["real"].map(lambda v: f"{ro(v, 2)} kWh/oră")
        df["exp_s"] = df["exp"].map(lambda v: f"{ro(v, 2)} kWh/oră")
        df["gap_s"] = (df["real"] - df["exp"]).map(lambda v: f"{'+' if v > 0 else ''}{ro(v, 2)} kWh/oră")
        L_REAL, L_EXP = "Net real (media pe 7 zile)", "Net așteptat de model"
        long = pd.concat([pd.DataFrame({"date": df["date"], "v": df["real"], "serie": L_REAL}),
                          pd.DataFrame({"date": df["date"], "v": df["exp"], "serie": L_EXP})])
        x = alt.X("date:T", axis=date_axis())
        area = alt.Chart(df).mark_area(color=CRIT, opacity=0.13).encode(x=x, y="exp:Q", y2="hi:Q")
        lines = alt.Chart(long).mark_line(strokeWidth=2, strokeCap="round", strokeJoin="round").encode(
            x=x, y=alt.Y("v:Q", axis=num_axis("kWh pe oră, orele de zi")),
            color=alt.Color("serie:N", scale=alt.Scale(domain=[L_REAL, L_EXP], range=[BLUE, MUTED]),
                            legend=alt.Legend(title=None, symbolType="stroke")),
            strokeDash=alt.StrokeDash("serie:N", scale=alt.Scale(domain=[L_REAL, L_EXP], range=[[1, 0], [5, 4]]),
                                      legend=None))
        nearest = alt.selection_point(nearest=True, on="pointerover", fields=["date"], empty=False, clear="pointerout")
        rule = alt.Chart(df).mark_rule(color=AXIS, strokeWidth=1).encode(
            x=x, opacity=alt.when(nearest).then(alt.value(1)).otherwise(alt.value(0)),
            tooltip=[alt.Tooltip("zi:N", title="Ziua"), alt.Tooltip("real_s:N", title="Net real"),
                     alt.Tooltip("exp_s:N", title="Net așteptat"), alt.Tooltip("gap_s:N", title="Diferență")]
        ).add_params(nearest)
        layers = [area, lines, rule]
        if len(ev_h):
            pts = ev_h[["date", "action", "explicatie"]].copy()
            pts["y"] = df.set_index("date")["real"].reindex(pts["date"]).to_numpy()
            pts["actiune"] = pts["action"].map(lambda a: f"{ACT[a][1]} {ACT[a][2]}")
            pts["zi"] = pts["date"].dt.strftime("%d.%m.%Y")
            pts["expl"] = pts["explicatie"].fillna("").map(lambda s: s if len(s) < 190 else s[:187] + "…")
            dom = [f"{ACT[a][1]} {ACT[a][2]}" for a in ACT]
            layers.append(alt.Chart(pts).mark_point(filled=True, size=85, opacity=1).encode(
                x="date:T", y="y:Q",
                color=alt.Color("actiune:N", scale=alt.Scale(domain=dom, range=[ACT[a][0] for a in ACT]),
                                legend=alt.Legend(title="Alarme", symbolType="circle", orient="bottom")),
                tooltip=[alt.Tooltip("zi:N", title="Ziua"), alt.Tooltip("actiune:N", title="Acțiune"),
                         alt.Tooltip("expl:N", title="De ce")]))
        start = lab.loc[h, "start"] if (truth and h in lab.index) else None
        if start is not None:
            mk = pd.DataFrame({"date": [start], "txt": [f"Începe {TYPE_RO[lab.loc[h, 'type']]}: {start:%d.%m.%Y}"]})
            layers.append(alt.Chart(mk).mark_rule(color=TRUTH, strokeWidth=1.6).encode(
                x="date:T", tooltip=alt.Tooltip("txt:N", title="Adevărul")))
            layers.append(alt.Chart(mk).mark_point(shape="triangle-down", filled=True, size=110, color=TRUTH, opacity=1).encode(
                x="date:T", y=alt.value(0), tooltip=alt.Tooltip("txt:N", title="Adevărul")))
        show(alt.layer(*layers).resolve_scale(color="independent", strokeDash="independent"), height=340)
        st.caption("Zona roșie: contorul arată mai mult consum net decât aștepta modelul (posibil producție lipsă). "
                   "Treceți cu mouse-ul peste grafic pentru valori.")

        if truth:
            g = D["gg"][h]
            kind = lab.loc[h, "kind"] if h in lab.index else ("real" if h in real_homes else "clean")
            if kind == "fault":
                after = g - D["lost"][h].reindex(g.index).fillna(0.0)
                series = {"Producție reală, fără defect": (smooth(g), MUTED, [5, 4]),
                          "Producție după defectul simulat": (smooth(after), BLUE, [1, 0])}
                r = lab.loc[h]
                msg = (f"<b>Adevărul:</b> {TYPE_RO[r['type']]} simulat din {r['start']:%d.%m.%Y} "
                       f"(linia violet din graficul de sus); pierdere medie {ro(r['mean_effect_kwh_per_day'])} kWh/zi.")
            elif kind == "load":
                series = {"Producția (neschimbată: panourile funcționează)": (smooth(g), BLUE, [1, 0]),
                          "Consum nou adăugat": (smooth(D["added"][h]), ORANGE, [1, 0])}
                r = lab.loc[h]
                msg = (f"<b>Adevărul:</b> panourile funcționează; am adăugat consum de tip "
                       f"<i>{TYPE_RO[r['type']]}</i> din {r['start']:%d.%m.%Y} (linia violet din graficul de sus).")
            elif kind == "real":
                series = {"Producția reală măsurată": (smooth(g), BLUE, [1, 0])}
                msg = ("<b>Adevărul:</b> producția reală a scăzut în această perioadă. "
                       "<b>Cauza nu o cunoaștem</b> (poate fi defect, umbră, panou scos).")
            else:
                series = {"Producția reală": (smooth(g), BLUE, [1, 0])}
                msg = "<b>Adevărul:</b> casă fără nicio modificare simulată; alarmele de mai sus sunt alarme false."
            tl = pd.concat([pd.DataFrame({"date": s.index, "v": s.to_numpy(), "serie": n}) for n, (s, _, _) in series.items()])
            tl["vs"] = tl["v"].map(lambda v: f"{ro(v)} kWh/zi")
            tl["zi"] = tl["date"].dt.strftime("%d.%m.%Y")
            names = list(series)
            ch = alt.Chart(tl).mark_line(strokeWidth=2, strokeCap="round").encode(
                x=alt.X("date:T", axis=date_axis()), y=alt.Y("v:Q", axis=num_axis("kWh pe zi")),
                color=alt.Color("serie:N", scale=alt.Scale(domain=names, range=[series[n][1] for n in names]),
                                legend=alt.Legend(title=None, symbolType="stroke") if len(names) > 1 else None),
                strokeDash=alt.StrokeDash("serie:N", scale=alt.Scale(domain=names, range=[series[n][2] for n in names]),
                                          legend=None),
                tooltip=[alt.Tooltip("zi:N", title="Ziua"), alt.Tooltip("serie:N", title="Serie"),
                         alt.Tooltip("vs:N", title="Valoare")])
            st.html('<div class="dm-h" style="margin-top:6px">Ce s-a întâmplat de fapt (ascuns pentru sistem)</div>')
            show(ch, height=230)
            st.html(f'<div class="dm-reveal">{msg}</div>')

    section(f"Alarmele casei {h} ({len(ev_h)})")
    with st.container(key="card_tl"):
        if len(ev_h):
            items = "".join(f'<div class="it"><div class="d">{r.date:%d.%m.%Y}</div><div>{pill(r.action)}</div>'
                            f'<div class="t">{esc(r.explicatie)}</div></div>'
                            for r in ev_h.tail(6).iloc[::-1].itertuples())
            st.html(f'<div class="dm-tl">{items}</div>')
            if len(ev_h) > 6:
                st.caption(f"Ultimele 6 din {len(ev_h)} alarme, cea mai recentă prima.")
        else:
            st.html('<div class="dm-ok">✓ Nicio alarmă în perioada de test.</div>')

    section("Mesajul către client")
    contact_h = ev_h[ev_h["action"] == "CONTACTEAZA"]
    if len(contact_h):
        last = contact_h.iloc[-1]
        msg = customer_message(last, ECON)
        lines = msg.split("\n")
        if lines and lines[0].startswith("Subiect:"):
            subject, body = lines[0][len("Subiect:"):].strip(), "\n".join(lines[1:]).strip("\n")
        else:
            subject, body = "Producția panourilor dumneavoastră", msg
        st.html(f'<div class="dm-mail"><div class="top"><b>Către:</b> prosumatorul casei {esc(h)}'
                f'<span class="flag">✎ ciornă · se trimite doar după aprobarea unui operator</span><br>'
                f'<b>Subiect:</b> {esc(subject)}<br>'
                f'<b>Declanșat de:</b> alarma „Contactează” din {last["date"]:%d.%m.%Y}</div>'
                f'<div class="body">{esc(body)}</div></div>')
        c1, c2 = st.columns([1, 3], vertical_alignment="center")
        c1.download_button("⬇ Descarcă ciorna (.txt)", msg, file_name=f"mesaj_casa_{h}_{last['date']:%Y%m%d}.txt",
                           mime="text/plain", key="dl_msg")
        c2.caption("Șablon de adaptat și de aprobat de un operator uman; sistemul nu trimite nimic singur. "
                   "Mesajul nu afirmă că panourile sunt defecte (nu putem ști de la distanță).")
    else:
        why = ("are doar alarme „Monitorizează” sau „Verifică datele”" if len(ev_h) else "nu are nicio alarmă")
        st.html(f'<div class="dm-neu">Pentru casa {esc(h)} nu pleacă niciun mesaj către client: {why}. '
                f'Mesajul se pregătește doar pentru alarmele „Contactează”.</div>')


def page_queue():
    page_title("Coada de verificări sub capacitate limitată",
               "Echipa poate verifica doar <b>K case pe săptămână</b>. Fiecare alarmă deschide un caz, iar echipa le ia pe "
               "cele mai importante. O vizită la o casă cu defect îl repară; una la o casă fără defect e <b>irosită</b>.")
    names = [POL_B1, POL_B1C, POL_ALL, POL_DM]
    with st.container(key="card_ctrl"):
        c1, c2 = st.columns([2, 3], gap="large")
        unlimited = c1.toggle("Fără limită de capacitate", key="unlim")
        K = None if unlimited else c1.slider("Capacitate: verificări pe săptămână", 1, 10, 3, key="K", disabled=unlimited)
        short = {POL_B1: "B1 · ordinea sosirii", POL_B1C: "B1 · după capacitate",
                 POL_ALL: "Model · toate alarmele", POL_DM: "Dimmer · model + decizie"}
        chosen = c2.multiselect("Politici comparate", names, default=[POL_B1, POL_B1C, POL_DM], key="pols",
                                format_func=short.get)

    res = {n: queue_result(n, K) for n in names}
    dm, b1, b1c = res[POL_DM][0], res[POL_B1][0], res[POL_B1C][0]
    kt = "fără limită" if K is None else f"K = {K} pe săptămână"
    section(f"Dimmer față de regula simplă B1 ({kt})")
    st.html(grid([
        tile("Defecte reparate", f"{dm['defecte_reparate%']:.0f}%",
             f"B1: {b1['defecte_reparate%']:.0f}% (sosire) · {b1c['defecte_reparate%']:.0f}% (capacitate)", hl=True),
        tile("kWh pierduți pe defect", f"{ro(dm['kWh_pierduti_pe_defect'], 0)}<small>kWh</small>",
             f"B1: {ro(b1['kWh_pierduti_pe_defect'], 0)} · {ro(b1c['kWh_pierduti_pe_defect'], 0)}", hl=True),
        tile("Vizite irosite", f"{dm['irosite']}", f"B1: {b1['irosite']} · {b1c['irosite']}"),
        tile("Precizia vizitelor", f"{dm['precizie%']:.0f}%", f"B1: {b1['precizie%']:.0f}% · {b1c['precizie%']:.0f}%"),
    ], 4))

    if not chosen:
        st.info("Alegeți cel puțin o politică.")
        return
    section(f"Energia pierdută pe cele {D['lost'].shape[1]} defecte simulate, cumulat")
    with st.container(key="card_cum"):
        series = {POL_NONE: D["lost"].cumsum().sum(axis=1)}
        series.update({n: res[n][2] for n in names if n in chosen})
        dom = list(series)
        long = pd.concat([pd.DataFrame({"date": s.index, "v": s.to_numpy(), "pol": n}) for n, s in series.items()])
        wide = pd.DataFrame({"date": series[POL_NONE].index})
        tips = [alt.Tooltip("zi:N", title="Ziua")]
        for i, n in enumerate(dom):
            wide[f"p{i}"] = [f"{ro(v, 0)} kWh" for v in series[n].reindex(wide["date"]).to_numpy()]
            tips.append(alt.Tooltip(f"p{i}:N", title=n))
        wide["zi"] = wide["date"].dt.strftime("%d.%m.%Y")
        x = alt.X("date:T", axis=date_axis())
        lines = alt.Chart(long).mark_line(strokeWidth=2, strokeCap="round").encode(
            x=x, y=alt.Y("v:Q", axis=num_axis("kWh pierduți, cumulat", thousands=True)),
            color=alt.Color("pol:N", scale=alt.Scale(domain=dom, range=[POL_STYLE[n][0] for n in dom]),
                            legend=alt.Legend(title=None, symbolType="stroke", columns=2)),
            strokeDash=alt.StrokeDash("pol:N", scale=alt.Scale(domain=dom, range=[POL_STYLE[n][1] for n in dom]),
                                      legend=alt.Legend(title=None, symbolType="stroke", columns=2)))
        nearest = alt.selection_point(nearest=True, on="pointerover", fields=["date"], empty=False, clear="pointerout")
        rule = alt.Chart(wide).mark_rule(color=AXIS).encode(
            x=x, opacity=alt.when(nearest).then(alt.value(1)).otherwise(alt.value(0)), tooltip=tips).add_params(nearest)
        show(alt.layer(lines, rule), height=340)
        st.caption("Linie plină = ordonare proprie, linie întreruptă = variantă. Albastru = model, portocaliu = regula simplă B1.")

    section("Toate politicile")
    rows = [{"Politică": short[n], "Reparate": res[n][0]["defecte_reparate%"], "kWh_med": res[n][0]["kWh_pierduti_pe_defect"],
             "kWh_mdn": res[n][0]["kWh_mediana_pe_defect"], "Vizite": res[n][0]["vizite_total"], "Găsite": res[n][0]["gasite"],
             "Irosite": res[n][0]["irosite"], "Precizie": res[n][0]["precizie%"],
             "Zile": res[n][0]["zile_mediane_pana_la_reparatie"]} for n in names]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Politică": st.column_config.TextColumn(width="medium"),
        "Reparate": st.column_config.ProgressColumn("Reparate", min_value=0, max_value=100, format="%.0f%%"),
        "Precizie": st.column_config.ProgressColumn("Precizie", min_value=0, max_value=100, format="%.0f%%"),
        "kWh_med": st.column_config.NumberColumn("kWh/defect", format="%.0f", help="kWh pierduți pe defect, medie"),
        "kWh_mdn": st.column_config.NumberColumn("kWh/defect (med.)", format="%.0f", help="kWh pierduți pe defect, mediană"),
        "Vizite": st.column_config.NumberColumn("Vizite totale", width="small"),
        "Găsite": st.column_config.NumberColumn(width="small"),
        "Irosite": st.column_config.NumberColumn("Vizite irosite", width="small"),
        "Zile": st.column_config.NumberColumn("Zile până la reparare", format="%.0f", help="mediana")})

    section(f"Costul în {CUR}, cu ipoteze pe care le puteți modifica ({kt})")
    with st.container(key="card_cost"):
        k1, k2, k3 = st.columns([1, 1, 2], gap="medium", vertical_alignment="bottom")
        price = k1.number_input(f"Prețul energiei ({CUR}/kWh)", min_value=0.0, max_value=10.0,
                                value=float(ECON["price_per_kwh"]), step=0.05, key="price")
        visit = k2.number_input(f"Costul unei vizite ({CUR})", min_value=0.0, max_value=5000.0,
                                value=float(ECON["visit_cost"]), step=10.0, key="visit")
        with k3.popover("De unde vin valorile?", width="stretch"):
            st.markdown(f"**Prețul energiei.** {ECON.get('price_source', '')}")
            if ECON.get("price_note"):
                st.markdown(ECON["price_note"])
            st.markdown(f"**Costul unei vizite.** {ECON.get('visit_cost_note', '')}")
        n_f = D["lost"].shape[1]
        crow = []
        for n in names:
            mm = res[n][0]
            e, v = mm["kWh_pierduti_pe_defect"] * n_f * price, mm["vizite_total"] * visit
            crow.append({"pol": short[n], "n": n, "energie": e, "vizite": v, "total": e + v})
        cdf = pd.DataFrame(crow)
        dm_t = cdf.loc[cdf["n"] == POL_DM, "total"].iloc[0]
        b1c_t = cdf.loc[cdf["n"] == POL_B1C, "total"].iloc[0]
        st.html(grid([tile("Cost total Dimmer", f"{ro(dm_t, 0)}<small>{CUR}</small>",
                           f"energie pierdută + vizite, pe cele {n_f} defecte simulate", hl=True),
                      tile("Cost total B1 după capacitate", f"{ro(b1c_t, 0)}<small>{CUR}</small>",
                           "cea mai bună variantă a regulii simple"),
                      tile("Diferență", f"{'−' if dm_t <= b1c_t else '+'}{ro(abs(1 - dm_t / b1c_t) * 100, 0)}%",
                           f"{ro(abs(b1c_t - dm_t), 0)} {CUR} {'economisiți' if dm_t <= b1c_t else 'în plus'} cu Dimmer")], 3))
        long = cdf.melt(id_vars=["pol", "total"], value_vars=["energie", "vizite"], var_name="comp", value_name="lei")
        long["comp"] = long["comp"].map({"energie": "Energie pierdută", "vizite": "Vizite"})
        long["ord"] = long["comp"].map({"Energie pierdută": 0, "Vizite": 1})
        long["lei_s"] = long["lei"].map(lambda v: f"{ro(v, 0)} {CUR}")
        # totalul sta in eticheta axei (colorata de tema), nu ca text desenat peste grafic
        total_lbl = {r.pol: f"{r.pol}  ·  {ro(r.total, 0)} {CUR}" for r in cdf.itertuples()}
        long["pol_t"] = long["pol"].map(total_lbl)
        order = [total_lbl[short[n]] for n in names]
        y = alt.Y("pol_t:N", sort=order, axis=alt.Axis(title=None, labelFontSize=12.5, labelLimit=340, labelOpacity=0.9,
                                                       domain=False, ticks=False))
        bars = alt.Chart(long).mark_bar(size=22).encode(
            y=y, x=alt.X("lei:Q", stack="zero", axis=num_axis(CUR, thousands=True)),
            color=alt.Color("comp:N", scale=alt.Scale(domain=["Energie pierdută", "Vizite"], range=[AMBER, VIOLET]),
                            legend=alt.Legend(title=None, symbolType="square")),
            order=alt.Order("ord:Q"),
            tooltip=[alt.Tooltip("pol:N", title="Politică"), alt.Tooltip("comp:N", title="Componentă"),
                     alt.Tooltip("lei_s:N", title="Cost")])
        show(bars, height=210)
        st.caption(f"Energie = kWh pierduți pe defect × {n_f} defecte × preț (pierderea clientului, la prețul final). "
                   f"Vizite = numărul total de vizite × costul unei vizite. Modificați valorile de mai sus ca să vedeți "
                   f"când merită modelul: cu cât o vizită costă mai mult, cu atât contează mai mult vizitele evitate.")

    with st.expander("Jurnalul verificărilor făcute de Dimmer"):
        vis = res[POL_DM][1]
        if len(vis):
            t = pd.DataFrame({"Data": vis["date"].dt.strftime("%d.%m.%Y"), "Casa": vis["home"],
                              "Rezultat": vis["outcome"].map(OUTCOME_RO), "Zile de așteptare": vis["wait_days"]})
            st.dataframe(t, hide_index=True, width="stretch")
    st.caption("Valori pe 40 de defecte simulate (10 pe tip). Cu atât de puține defecte, diferențele mici dintre politici nu sunt concludente.")


def page_real():
    page_title("Episoade reale de scădere a producției",
               "Pe lângă defectele simulate, datele conțin case a căror producție <b>a scăzut real</b>. Nu știm de ce "
               "(defect, umbră, panou scos), deci le numim <i>episoade probabile</i>. Lista vine dintr-o regulă fixă "
               "(randamentul din perioadă sub 70% din cel din anul de antrenare), nu din vreun model.")
    if not real_homes:
        st.info("Nu există case cu scădere reală în datele curente.")
        return
    b1, b2 = D["b1"], D["b2"]
    first = lambda df, h: df.loc[df["home"] == h, "date"].min()  # noqa: E731
    rows = []
    for h in real_homes:
        e = dec[dec["home"] == h]
        rows.append({"Casa": h, "kW": cap[h], "B1": first(b1, h), "B2": first(b2, h), "Dimmer": first(dec, h),
                     "Dimmer, primul contact": e.loc[e["action"] == "CONTACTEAZA", "date"].min() if len(e) else pd.NaT})
    t = pd.DataFrame(rows).set_index("Casa")
    n = len(real_homes)
    n_contact = int(t["Dimmer, primul contact"].notna().sum())
    st.html(grid([tile("B1 a dat alarmă", f"{t['B1'].notna().sum()}<small>din {n}</small>", "regula „fără export la prânz”"),
                  tile("B2 a dat alarmă", f"{t['B2'].notna().sum()}<small>din {n}</small>", "abatere față de anul trecut"),
                  tile("Dimmer a dat alarmă", f"{t['Dimmer'].notna().sum()}<small>din {n}</small>",
                       f"dintre care {n_contact} au ajuns la „Contactează”", hl=True)], 3))

    with st.container(key="card_rchart"):
        h = st.selectbox("Casa", real_homes, format_func=home_label, key="real_home")
        g = smooth(D["gg"][h])
        prod = pd.DataFrame({"date": g.index, "v": g.to_numpy()})
        prod["vs"] = prod["v"].map(lambda v: f"{ro(v)} kWh/zi")
        prod["zi"] = prod["date"].dt.strftime("%d.%m.%Y")
        # data primei alarme sta in legenda (nu ca text peste grafic, unde se suprapunea cu liniile apropiate)
        marks = []
        for lbl, df_, c in (("B1", b1, ORANGE), ("B2", b2, AQUA), ("Dimmer", dec, BLUE)):
            d0 = first(df_, h)
            marks.append({"date": d0, "met": f"{lbl}: {d0:%d.%m.%Y}" if pd.notna(d0) else f"{lbl}: fără alarmă", "c": c})
        x = alt.X("date:T", axis=date_axis())
        line = alt.Chart(prod).mark_line(color=NEUTRAL, strokeWidth=2).encode(
            x=x, y=alt.Y("v:Q", axis=num_axis("Producția reală, kWh pe zi")),
            tooltip=[alt.Tooltip("zi:N", title="Ziua"), alt.Tooltip("vs:N", title="Producție")])
        mk = pd.DataFrame(marks)
        scale = alt.Scale(domain=mk["met"].tolist(), range=mk["c"].tolist())
        rules = alt.Chart(mk.dropna(subset=["date"])).mark_rule(strokeWidth=2).encode(
            x="date:T", color=alt.Color("met:N", scale=scale, legend=alt.Legend(title="Prima alarmă", symbolType="stroke")),
            tooltip=alt.Tooltip("met:N", title="Prima alarmă"))
        show(alt.layer(line, rules), height=330)
        st.caption("Sistemul nu vede producția, doar netul; aici o afișăm ca să se vadă scăderea.")

    section("Prima alarmă pe fiecare casă")
    show_t = t.copy()
    for c in ["B1", "B2", "Dimmer", "Dimmer, primul contact"]:
        show_t[c] = show_t[c].dt.strftime("%d.%m.%Y").fillna("—")
    st.dataframe(show_t, width="stretch", height=36 * (len(show_t) + 1) + 4, column_config={
        "kW": st.column_config.NumberColumn(format="%.2f"),
        "Dimmer": st.column_config.TextColumn("Dimmer, prima alarmă")})


def page_results():
    page_title("Rezultatele, cinstit",
               "Toate pragurile au fost alese pe anul de validare și <b>înghețate</b> înainte de testul final, rulat o singură dată.")

    section("0 · Comparația principală (test)")
    st.html(compare_cards(with_raw=True))
    st.caption("Ultimul card (chenar punctat) este un alt punct de funcționare: modelul fără verificarea „soare sau consum”, "
               "mai sensibil, dar cu mai multe vizite false. Nu îl comparați direct cu B1.")

    section("1 · Predicția")
    st.html(grid([tile("Baseline: media casei la aceeași oră și lună", f"{ro(R['base_mae'], 3)}<small>kWh/oră</small>",
                       "validare 0,503"),
                  tile("Model LightGBM (radiație + vecini)", f"{ro(R['dm_mae'], 3)}<small>kWh/oră</small>", "validare 0,410", hl=True),
                  tile("Îmbunătățire", f"−{ro(100 * (1 - R['dm_mae'] / R['base_mae']), 1)}%", "validare −18,4%")], 3))
    m = D["mt"]
    abl = m[["MAE_kWh_ora", "FA_la_1000", "kWh_pierduti_medie"]].copy()
    names_v = {"own": "own · doar istoricul casei", "rad": "rad · + radiație", "nb": "nb · + vecini",
               "full": "full · radiație + vecini (final)", "full_m": "full_m · cu luna"}
    abl.index = [names_v.get(v, v) for v in abl.index]
    with st.expander("Ablația: ce aduce fiecare ingredient (test)"):
        st.dataframe(abl, width="stretch", column_config={
            "MAE_kWh_ora": st.column_config.NumberColumn("MAE (kWh/oră)", format="%.4f"),
            "FA_la_1000": st.column_config.NumberColumn("Alarme false / 1.000", format="%.2f"),
            "kWh_pierduti_medie": st.column_config.NumberColumn("kWh pierduți (medie)", format="%.1f")})

    section("2 · Detecția, pe tipul de defect (test, praguri înghețate; Dimmer = modelul fără verificare)")
    b = D["bt"]
    rows = []
    for typ in FAULT_TYPES:
        for meth in ("B1", "B2"):
            x = b[(b["type"] == typ) & (b["metoda"] == meth)].iloc[0]
            rows.append({"tip": TYPE_RO[typ], "metoda": meth, "prinse": x["prinse"], "zile": x["zile_mediane_pana_la_detectie"]})
        rows.append({"tip": TYPE_RO[typ], "metoda": "Dimmer", "prinse": m.loc["full", f"{typ}_prinse%"],
                     "zile": m.loc["full", f"{typ}_zile"]})
    det = pd.DataFrame(rows)
    det["p_s"] = det["prinse"].map(lambda v: f"{v:.0f}%")
    det["z_s"] = det["zile"].map(lambda v: f"{ro(v)} zile")
    c1, c2 = st.columns([3, 2], gap="medium")
    with c1.container(key="card_det"):
        order = [TYPE_RO[t] for t in FAULT_TYPES]
        bars = alt.Chart(det).mark_bar(size=18, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("tip:N", sort=order, axis=alt.Axis(title=None, labelAngle=0, labelFontSize=12.5, labelOpacity=0.85),
                    scale=alt.Scale(paddingInner=0.25)),
            xOffset=alt.XOffset("metoda:N", sort=["B1", "B2", "Dimmer"], scale=alt.Scale(paddingInner=0.12)),
            y=alt.Y("prinse:Q", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(title="Defecte prinse", format=".0f",
                                                                                 labelExpr="datum.label + '%'", tickCount=5)),
            color=alt.Color("metoda:N", scale=alt.Scale(domain=["B1", "B2", "Dimmer"], range=[ORANGE, AQUA, BLUE]),
                            legend=alt.Legend(title=None)),
            tooltip=[alt.Tooltip("tip:N", title="Tip"), alt.Tooltip("metoda:N", title="Metodă"),
                     alt.Tooltip("p_s:N", title="Prinse"), alt.Tooltip("z_s:N", title="Zile mediane până la alarmă")])
        show(bars, height=300)
    with c2:
        st.html(grid([tile("Alarme false / 1.000 zile-casă", f"{ro(R['dm_fa'], 2)}",
                           f"Dimmer fără verificare (ținta era ≤ 2,0) · B1 {ro(R['b1_fa'], 2)} · B2 {ro(R['b2_fa'], 2)}"),
                      tile("Defecte prinse, medie pe 4 tipuri", f"{ro(R['dm_rec'], 1)}%",
                           f"Dimmer fără verificare · B1 {ro(R['b1_rec'], 1)}% · B2 {ro(R['b2_rec'], 0)}%", hl=True)], 2))
        piv = det.pivot(index="tip", columns="metoda", values="zile").reindex(order)[["B1", "B2", "Dimmer"]]
        st.caption("Zile mediane până la alarmă")
        st.dataframe(piv, width="stretch", column_config={c: st.column_config.NumberColumn(format="%.1f") for c in piv.columns})
    st.caption("Pe validare detecția era la paritate cu B1; pe test e mai bună, dar cu 10 defecte pe tip o parte din diferență poate fi întâmplare.")

    section("3 · Stratul de decizie: soare sau consum?")
    dl = pd.DataFrame({"Fără verificare": ["2,12", "78%", "162", "100%", "80%", "60%"],
                       "Cu verificare (prag 0,4)": ["1,00", "62%", "223", "40%", "20%", "60%"]},
                      index=["Alarme „Contactează” false / 1.000", "Defecte prinse", "kWh pierduți (medie)",
                             "Piscină cu alarmă", "Muncă de acasă cu alarmă", "Aer condiționat cu alarmă"])
    c1, c2 = st.columns([3, 2], gap="medium")
    c1.dataframe(dl, width="stretch")
    c2.html('<div class="dm-note">Verificarea reduce vizitele false și confuzia cu consumul constant, dar <b>costă recall</b> '
            '(78% → 62%). Defectele slabe (parțiale, intermitente, treptate) trec prin ea cam ca alarmele false; '
            'cele puternice (totale: 77%) și scăderile reale (73%) sunt păstrate.</div>')

    section("4 · Triajul cu capacitate (aceleași calcule ca în reports/triage_test.csv)")
    rows = []
    for K in (1, 2, 3, 5, None):
        for n in (POL_B1C, POL_DM):
            mm, _, _ = queue_result(n, K)
            rows.append({"K pe săpt.": "fără limită" if K is None else str(K), "Politică": n,
                         "Reparate": mm["defecte_reparate%"], "kWh pe defect": mm["kWh_pierduti_pe_defect"],
                         "Vizite irosite": mm["irosite"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "Reparate": st.column_config.ProgressColumn("Defecte reparate", min_value=0, max_value=100, format="%.0f%%"),
        "kWh pe defect": st.column_config.NumberColumn(format="%.0f")})

    section("Limite")
    st.html('<div class="dm-card"><ul class="dm-list">'
            '<li>Defectele sunt <b>simulate</b> (40 în test, câte 10 pe tip). Cu atât de puține, diferențele mici dintre metode nu sunt concludente.</li>'
            '<li>Alarmele false pe test (2,12 la 1.000) au depășit ținta de 2,0.</li>'
            '<li><b>Defectele totale sunt prinse mai încet</b> decât de o regulă simplă (22,5 zile față de 9).</li>'
            '<li><b>Aerul condiționat rămâne confundat</b> cu un defect: rulează în zilele calde, deci însorite.</li>'
            '<li>Pe episoadele reale, Dimmer alarmează la fel de multe case ca B1 (9 din 12), iar B2 mai multe (11 din 12); '
            'pentru unele (de exemplu casa 234) alarmează mult mai târziu decât baseline-urile.</li>'
            '<li>Datele sunt din Australia (Ausgrid, 2010–2013); metoda se transferă, cifrele nu.</li>'
            '<li>Nu am confirmat defecte fizice; spunem „merită verificat”, nu „este defect”.</li></ul></div>')


# ================================================================== cadru comun
st.html(theme_css())
pg = st.navigation([st.Page(page_overview, title="Prezentare", icon="☀️", url_path="prezentare", default=True),
                    st.Page(page_case, title="Un caz", icon="🔍", url_path="caz"),
                    st.Page(page_queue, title="Coada de verificări", icon="🧰", url_path="coada"),
                    st.Page(page_real, title="Episoade reale", icon="📉", url_path="reale"),
                    st.Page(page_results, title="Rezultate", icon="📊", url_path="rezultate")], position="top")

st.html(f'<div class="dm-brand">{LOGO_IMG}<div><div class="nm">Dimmer</div>'
        f'<div class="tg">Găsim panourile solare stricate pe tăcute, doar din contorul net</div></div>'
        f'<div class="dm-chips"><span class="dm-chip">Ausgrid · Sydney · {len(D["net"].columns)} case</span>'
        f'<span class="dm-chip">Test {meta["start"][:7]} → {meta["end"][:7]}</span>'
        f'<span class="dm-chip" style="--c:{BLUE}"><i></i>praguri înghețate</span>'
        f'<span class="dm-chip" style="--c:{WARN}"><i></i>defecte simulate pe producție reală</span></div></div>'
        + THEME_JS, unsafe_allow_javascript=True)
pg.run()
st.html('<div class="dm-foot">Date: Ausgrid Solar Home Electricity Data (CC BY) · radiație: Open-Meteo · '
        'model LightGBM antrenat de echipă, rulat local · cod și rezultate reproductibile în repo.</div>')
