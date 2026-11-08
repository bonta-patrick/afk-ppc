"""
Dimmer - aplicatia demo (Streamlit), in romana si engleza.

Citeste doar din demo_data/test/ (creat de src/make_demo_data.py), deci ruleaza si dupa `git clone`.
    python -m streamlit run app/app.py

Limba se schimba din selectorul RO / EN din dreapta sus. Toate textele sunt in dictionarul TX, ca perechi (ro, en).
"""
import html
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import altair as alt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from triage import run_queue, metrics  # noqa: E402
from messages import customer_message, explain_alert  # noqa: E402
import demo_replay as DR  # noqa: E402

DEMO = ROOT / "demo_data" / "test"
st.set_page_config(page_title="Dimmer", layout="wide", initial_sidebar_state="collapsed")

st.session_state.setdefault("lang", "ro")
LANG = st.session_state["lang"] if st.session_state["lang"] in ("ro", "en") else "ro"
LI = 0 if LANG == "ro" else 1

INK, MUTED, LINE = "#1d1d1f", "#86868b", "#e5e5ea"
GOLD, GOLD_D = "#ff9f0a", "#bf6f00"
C_CONTACT, C_MONITOR, C_DATA = "#d70015", "#c77700", "#8e8e93"
ACTION = {"CONTACTEAZA": ("CONTACTEAZĂ", "CONTACT", C_CONTACT, "b-contact"),
          "MONITORIZEAZA": ("MONITORIZEAZĂ", "MONITOR", C_MONITOR, "b-monitor"),
          "VERIFICA_DATE": ("VERIFICĂ DATELE", "CHECK DATA", C_DATA, "b-data")}
KINDS = {"total": ("defect total", "total fault"), "partial": ("defect parțial", "partial fault"),
         "intermittent": ("defect intermitent", "intermittent fault"), "gradual": ("defect treptat", "gradual fault"),
         "ac": ("aer condiționat nou", "new air conditioning"), "pool": ("pompă de piscină nouă", "new pool pump"),
         "ev": ("mașină electrică nouă", "new electric car"), "wfh": ("muncă de acasă", "working from home")}
OUTCOMES = {"tp": ("Defect găsit și reparat", "Fault found and repaired"),
            "real": ("Scădere reală (cauză necunoscută)", "Real drop (unknown cause)"),
            "fp_clean": ("Vizită irosită (casă fără problemă)", "Wasted visit (home with no problem)"),
            "fp_load": ("Vizită irosită (doar consum nou)", "Wasted visit (only a new load)"),
            "fp_pre": ("Vizită irosită (defectul nu începuse)", "Wasted visit (the fault had not started)")}
POLICIES = {"b1_fifo": ("B1 (regulă simplă), în ordinea sosirii", "B1 (simple rule), in order of arrival"),
            "b1_cap": ("B1, după capacitatea instalată", "B1, by installed capacity"),
            "model_all": ("Model, toate alarmele, după pierderea estimată", "Model, all alerts, by estimated loss"),
            "dimmer": ("Dimmer: model + decizie, după scor", "Dimmer: model + decision, by score")}
PAGE_KEYS = ["overview", "demo", "queue", "results"]


def t(key, **kw):
    s = TX[key][LI]
    return s.format(**kw) if kw else s


def kind(k):
    return KINDS.get(k, (k, k))[LI]


def fmt(x, d=0):
    s = f"{x:,.{d}f}"
    if LANG == "ro":
        s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return s


def L(ro_text, en_text):
    return ro_text if LANG == "ro" else en_text


CSS = """
<style>
:root { --ink:#1d1d1f; --mut:#6e6e73; --line:#d2d2d7; --bg:#fbfbfd; --tint:#f5f5f7; --gold:#ff9f0a; --ease:cubic-bezier(.2,.8,.2,1); }
.stApp { background:var(--bg); color:var(--ink); -webkit-font-smoothing:antialiased;
  font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text","Segoe UI",Inter,Roboto,"Helvetica Neue",Arial,sans-serif; }
header[data-testid="stHeader"] { display:none; }
#MainMenu, footer { visibility:hidden; }
.block-container { max-width:1160px; padding-top:1.1rem; padding-bottom:6rem; }
h1,h2,h3,h4 { font-family:inherit; letter-spacing:-0.03em; color:var(--ink); }

@keyframes rise { from { opacity:0; transform:translateY(26px); } to { opacity:1; transform:none; } }
@keyframes breathe { 0%,100% { opacity:.35; transform:scale(.9); } 50% { opacity:1; transform:scale(1.06); } }
@keyframes pulse { 0%,100% { box-shadow:0 0 0 0 rgba(255,159,10,.38); } 50% { box-shadow:0 0 0 9px rgba(255,159,10,0); } }
@keyframes grow { from { transform:scaleX(0); } to { transform:scaleX(1); } }
@keyframes fill { 0%,100% { width:12%; } 50% { width:100%; } }
@keyframes knob { 0%,100% { left:calc(12% - 9px); } 50% { left:calc(100% - 9px); } }

.topbar { display:flex; align-items:center; gap:.7rem; padding:.5rem 0 .9rem 0; }
.logo { width:24px; height:24px; border-radius:50%; flex:none; animation:pulse 3.6s ease-in-out infinite;
  background:radial-gradient(circle at 35% 30%,#fff6d6 0%,#ffd166 38%,#ff9f0a 78%); }
.brand-n { font-size:1.3rem; font-weight:700; letter-spacing:-.04em; }
.brand-t { color:#86868b; font-size:.85rem; margin-left:.2rem; }
.divider { height:1px; background:rgba(0,0,0,.08); margin:1rem 0 1.6rem 0; }

.st-key-nav div[role="radiogroup"] { display:inline-flex; flex-wrap:wrap; gap:.15rem; padding:.28rem; border-radius:999px;
  background:rgba(245,245,247,.85); border:1px solid rgba(0,0,0,.06); backdrop-filter:saturate(180%) blur(16px); }
.st-key-nav label[data-baseweb="radio"] { margin:0; padding:.42rem 1.1rem; border-radius:999px; cursor:pointer; transition:background .35s var(--ease), box-shadow .35s var(--ease); }
.st-key-nav label[data-baseweb="radio"] > div:first-child { display:none; }
.st-key-nav label[data-baseweb="radio"] p { font-size:.88rem; font-weight:600; color:#6e6e73; margin:0; transition:color .3s; }
.st-key-nav label[data-baseweb="radio"]:hover p { color:#1d1d1f; }
.st-key-nav label[data-baseweb="radio"]:has(input:checked) { background:#fff; box-shadow:0 1px 3px rgba(0,0,0,.14), 0 6px 18px rgba(0,0,0,.07); }
.st-key-nav label[data-baseweb="radio"]:has(input:checked) p { color:#1d1d1f; }

.hero { position:relative; overflow:hidden; isolation:isolate; background:#000; color:#f5f5f7; border-radius:32px; padding:5.2rem 4rem 4.4rem 4rem; }
.hero:before { content:""; position:absolute; inset:0; z-index:-1;
  background:radial-gradient(1100px 500px at 88% -12%,rgba(255,159,10,.28),transparent 62%), radial-gradient(760px 380px at -4% 112%,rgba(255,209,102,.10),transparent 62%); }
.orb { position:absolute; right:-130px; top:-170px; width:580px; height:580px; border-radius:50%; z-index:-1; filter:blur(6px); animation:breathe 7s ease-in-out infinite;
  background:radial-gradient(circle,rgba(255,196,64,.92) 0%,rgba(255,159,10,.34) 38%,rgba(255,159,10,0) 68%); }
.pill { display:inline-block; padding:.34rem .9rem; border-radius:999px; border:1px solid rgba(255,255,255,.18); background:rgba(255,255,255,.06);
  font-size:.78rem; letter-spacing:.05em; color:#d9d9de; backdrop-filter:blur(8px); animation:rise .9s var(--ease) both; animation-delay:.05s; }
.hero-t { font-size:clamp(2.4rem,5.4vw,4.3rem); line-height:1.04; font-weight:700; letter-spacing:-.045em; margin:1.25rem 0 1.1rem 0; max-width:900px;
  animation:rise .9s var(--ease) both; animation-delay:.18s; }
.grad { background:linear-gradient(90deg,#ffe29a,#ff9f0a); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero-s { font-size:1.18rem; line-height:1.55; color:#a1a1a6; max-width:720px; animation:rise .9s var(--ease) both; animation-delay:.32s; }
.fader { margin-top:2.5rem; width:260px; height:4px; border-radius:4px; background:rgba(255,255,255,.14); position:relative; animation:rise .9s var(--ease) both; animation-delay:.46s; }
.fader i { position:absolute; left:0; top:0; height:100%; border-radius:4px; background:linear-gradient(90deg,#ffd166,#ff9f0a); animation:fill 7s ease-in-out infinite; }
.fader b { position:absolute; top:50%; margin-top:-9px; width:18px; height:18px; border-radius:50%; background:#fff; box-shadow:0 0 22px rgba(255,170,30,.9); animation:knob 7s ease-in-out infinite; }

.reveal { animation:rise .85s var(--ease) both; animation-delay:var(--d,0s); }
@supports (animation-timeline:view()) {
  .reveal { animation:rise linear both; animation-timeline:view(); animation-range:entry 0% entry 55%; animation-delay:0s; }
}

.eyebrow { text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; color:#bf6f00; margin:4.2rem 0 .55rem 0; }
.sec-t { font-size:clamp(1.8rem,3.4vw,2.7rem); font-weight:700; letter-spacing:-.04em; line-height:1.08; margin:0 0 .7rem 0; }
.sec-s { font-size:1.08rem; color:#6e6e73; max-width:760px; line-height:1.55; margin:0 0 1.6rem 0; }

.card, .kpi { background:#fff; border-radius:22px; padding:1.5rem 1.6rem; height:100%; box-sizing:border-box; border:1px solid rgba(0,0,0,.04);
  box-shadow:0 1px 2px rgba(0,0,0,.04), 0 10px 30px rgba(0,0,0,.05); transition:transform .45s var(--ease), box-shadow .45s var(--ease); }
.card:hover, .kpi:hover { transform:translateY(-5px); box-shadow:0 2px 4px rgba(0,0,0,.05), 0 24px 54px rgba(0,0,0,.10); }
.card-t { font-size:1.08rem; font-weight:650; letter-spacing:-.015em; margin-bottom:.45rem; line-height:1.3; }
.card-p { color:#6e6e73; font-size:.95rem; line-height:1.6; }
.card-tag { display:inline-block; margin-top:.8rem; font-size:.68rem; letter-spacing:.12em; text-transform:uppercase; color:#bf6f00; font-weight:700; }
.step-num { font-size:3.1rem; font-weight:700; letter-spacing:-.05em; line-height:1; margin-bottom:.8rem;
  background:linear-gradient(180deg,#ffd166,#ff9f0a); -webkit-background-clip:text; background-clip:text; color:transparent; }
.kpi-v { font-size:2.3rem; font-weight:700; letter-spacing:-.04em; font-variant-numeric:tabular-nums; line-height:1.1; }
.kpi-l { font-size:.92rem; font-weight:600; margin-top:.45rem; }
.kpi-n { font-size:.8rem; color:#86868b; margin-top:.35rem; line-height:1.45; }

.badge { display:inline-block; padding:.16rem .65rem; border-radius:999px; font-size:.68rem; font-weight:700; letter-spacing:.07em; color:#fff; white-space:nowrap; }
.b-contact { background:#d70015; }
.b-monitor { background:#c77700; }
.b-data { background:#8e8e93; }
.alert-row { display:flex; gap:.9rem; align-items:flex-start; padding:.85rem .6rem; border-top:1px solid rgba(0,0,0,.07); border-radius:10px; transition:background .3s; }
.alert-row:hover { background:rgba(0,0,0,.025); }
.alert-d { width:84px; flex:none; color:#86868b; font-size:.82rem; font-variant-numeric:tabular-nums; padding-top:.15rem; }
.alert-b { width:128px; flex:none; padding-top:.05rem; }
.alert-t { font-size:.92rem; line-height:1.55; color:#2c2c2e; }

.bar-row { display:grid; grid-template-columns:minmax(150px,250px) 1fr 150px; align-items:center; gap:1rem; margin:1rem 0; }
.bar-l { font-size:.92rem; }
.bar-t { height:12px; border-radius:99px; background:#ececf0; overflow:hidden; }
.bar-f { height:100%; border-radius:99px; background:linear-gradient(90deg,#ffd166,#ff9f0a); transform-origin:left; animation:grow 1.4s var(--ease) both; animation-delay:var(--d,.15s); }
.bar-f.alt { background:linear-gradient(90deg,#c7c7cc,#8e8e93); }
.bar-v { text-align:right; font-size:.88rem; font-variant-numeric:tabular-nums; color:#1d1d1f; font-weight:600; }
@supports (animation-timeline:view()) {
  .bar-f { animation:grow linear both; animation-timeline:view(); animation-range:entry 10% entry 75%; animation-delay:0s; }
}
.note { border-radius:18px; background:#f5f5f7; padding:1.1rem 1.3rem; font-size:.93rem; color:#3a3a3c; line-height:1.6; }
.footnote { color:#86868b; font-size:.8rem; margin-top:.8rem; line-height:1.55; }
.hl-num { font-weight:700; color:#1d1d1f; }
.dark-card { background:#1d1d1f; color:#f5f5f7; border-radius:22px; padding:1.7rem 1.8rem; }
.dark-card .card-t { color:#fff; }
.dark-card .card-p { color:#a1a1a6; }

[data-testid="stImage"] img { border-radius:22px; background:#fff; box-shadow:0 1px 2px rgba(0,0,0,.04), 0 10px 30px rgba(0,0,0,.05); }
[data-testid="stDataFrame"] { border-radius:18px; overflow:hidden; box-shadow:0 1px 2px rgba(0,0,0,.04), 0 8px 24px rgba(0,0,0,.05); }
[data-testid="stExpander"] { border-radius:18px; border:1px solid rgba(0,0,0,.06); background:#fff; }
div[data-baseweb="select"] > div { border-radius:12px; }

@media (max-width:820px) {
  .hero { padding:3.2rem 1.6rem 3rem 1.6rem; border-radius:24px; }
  .bar-row { grid-template-columns:1fr; gap:.4rem; }
  .bar-v { text-align:left; }
}
@media (prefers-reduced-motion:reduce) {
  *, *:before, *:after { animation:none !important; transition:none !important; }
}
.st-key-lang div[role="radiogroup"] { display:inline-flex; gap:.1rem; padding:.2rem; border-radius:999px; background:rgba(245,245,247,.85); border:1px solid rgba(0,0,0,.06); float:right; }
.st-key-lang label[data-baseweb="radio"] { margin:0; padding:.25rem .8rem; border-radius:999px; cursor:pointer; transition:background .35s var(--ease), box-shadow .35s var(--ease); }
.st-key-lang label[data-baseweb="radio"] > div:first-child { display:none; }
.st-key-lang label[data-baseweb="radio"] p { font-size:.8rem; font-weight:700; letter-spacing:.06em; color:#6e6e73; margin:0; }
.st-key-lang label[data-baseweb="radio"]:has(input:checked) { background:#fff; box-shadow:0 1px 3px rgba(0,0,0,.14); }
.st-key-lang label[data-baseweb="radio"]:has(input:checked) p { color:#1d1d1f; }

.dm-banner { display:inline-flex; align-items:center; gap:.55rem; padding:.36rem .95rem; border-radius:999px; background:#1d1d1f; color:#f5f5f7;
  font-size:.78rem; font-weight:600; letter-spacing:.04em; margin:.2rem 0 1.1rem 0; }
.dm-dot { width:8px; height:8px; border-radius:50%; background:#8e8e93; flex:none; }
.dm-dot.on { background:#ff9f0a; animation:pulse 1.8s ease-in-out infinite; }
.dm-big { font-size:1.9rem; font-weight:700; letter-spacing:-.04em; font-variant-numeric:tabular-nums; line-height:1.1; }
.dm-lbl { font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; color:#86868b; font-weight:700; margin-bottom:.25rem; }
.dm-sm { font-size:.8rem; color:#86868b; margin-top:.25rem; line-height:1.45; }
.dm-track { height:6px; border-radius:99px; background:#ececf0; overflow:hidden; margin:.55rem 0 .2rem 0; }
.dm-fill { height:100%; border-radius:99px; background:linear-gradient(90deg,#ffd166,#ff9f0a); transition:width .4s var(--ease); }
.dm-fill.over { background:#d70015; }
.dm-state { display:inline-block; padding:.3rem .85rem; border-radius:999px; font-size:.82rem; font-weight:700; color:#fff; margin:.15rem 0 .5rem 0; }
.s-wait { background:#c7c7cc; } .s-monitor { background:#1a9e5c; } .s-almon { background:#c77700; } .s-aldata { background:#8e8e93; } .s-invest { background:#d70015; }
.tag { display:inline-block; padding:.12rem .55rem; border-radius:6px; font-size:.7rem; font-weight:700; letter-spacing:.03em; white-space:nowrap; }
.tag-real { background:#eef7f1; color:#1a7a48; } .tag-sim { background:#fff1e0; color:#a35a00; border:1px dashed #e0a050; }
.feed { max-height:560px; overflow-y:auto; padding-right:.3rem; }
.ev { display:grid; grid-template-columns:12px 92px 1fr; gap:.55rem; padding:.55rem .2rem; border-top:1px solid rgba(0,0,0,.06); font-size:.84rem; line-height:1.5; }
.ev-dot { width:8px; height:8px; border-radius:50%; margin-top:.45rem; background:#c7c7cc; }
.ev-d { color:#86868b; font-variant-numeric:tabular-nums; font-size:.76rem; line-height:1.35; }
.ev-t { color:#2c2c2e; }
.k-alarm { background:#ff9f0a; } .k-contact { background:#d70015; } .k-monitor { background:#c77700; } .k-data { background:#8e8e93; }
.k-case { background:#d70015; box-shadow:0 0 0 3px rgba(215,0,21,.15); } .k-notif { background:#1d1d1f; } .k-err { background:#d70015; }
.ev-new { animation:rise .6s var(--ease) both; }
.ev-tbl { width:100%; border-collapse:collapse; font-size:.86rem; }
.ev-tbl td { padding:.42rem .3rem; border-top:1px solid rgba(0,0,0,.06); vertical-align:top; }
.ev-tbl td:first-child { color:#6e6e73; width:44%; }
.fl-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(15px,1fr)); gap:4px; background:#fff; border-radius:22px; padding:1.2rem 1.3rem;
  border:1px solid rgba(0,0,0,.04); box-shadow:0 1px 2px rgba(0,0,0,.04), 0 10px 30px rgba(0,0,0,.05); }
.fl-c { display:inline-block; width:100%; aspect-ratio:1; min-width:12px; border-radius:4px; background:#e5e5ea; transition:background .5s var(--ease); }
.fl-s-wait { background:#f2f2f5; } .fl-s-almon { background:#c77700; border-radius:50%; } .fl-s-aldata { background:#8e8e93; border-radius:50%; }
.fl-s-invest { background:#d70015; }
.fl-w { box-shadow:0 0 0 2px #fff, 0 0 0 3.5px #1d1d1f; }
.fl-legend { display:flex; flex-wrap:wrap; gap:.5rem 1.4rem; margin-top:.8rem; font-size:.8rem; color:#3a3a3c; }
.fl-leg { display:inline-flex; align-items:center; gap:.45rem; }
.fl-leg .fl-c { width:12px; min-width:12px; }
</style>
"""

# ------------------------------------------------------------------ texte (ro, en)
TX = {
    # --- schelet
    "tagline": ("agent de triaj pentru panouri solare", "triage agent for solar panels"),
    "nav_overview": ("Prezentare", "Overview"),
    "nav_cases": ("Cazuri", "Cases"),
    "nav_queue": ("Triaj", "Triage"),
    "nav_real": ("Episoade reale", "Real episodes"),
    "nav_results": ("Rezultate", "Results"),
    "missing_demo": ("Lipsește folderul demo_data/test. Rulați: `python src/make_demo_data.py --period test`",
                     "The demo_data/test folder is missing. Run: `python src/make_demo_data.py --period test`"),

    # --- prezentare: hero
    "pill": ("Agent de triaj pentru prosumatori", "Triage agent for solar prosumers"),
    "hero_a": ("Contorul vede doar ce rămâne.", "The meter only sees what's left."),
    "hero_b": ("Dimmer găsește ce lipsește.", "Dimmer finds what's missing."),
    "hero_sub": ("Un sistem solar care nu mai produce (panou, cablu sau invertor defect) nu dă nicio alarmă, iar contorul vede doar diferența dintre consum și producție. "
                 "Dimmer ghicește ce ar trebui să arate contorul, deosebește o defecțiune a sistemului solar (panouri, cabluri, invertor etc.) de un consum nou și spune cine merită verificat primul.",
                 "A solar system that stops producing (a faulty panel, cable or inverter) raises no alarm, and the meter only sees the difference between consumption and production. "
                 "Dimmer estimates what the meter should read, tells a fault in the solar system (panels, wiring, inverter, etc.) from a new load, and says who is worth checking first."),

    # --- prezentare: avantaje
    "s1_eye": ("Avantaje măsurate", "Measured results"),
    "s1_title": ("Ce câștigați", "What you gain"),
    "s1_sub": ("Testul final: un an pe care nu l-am atins la reglare, rulat o singură dată, cu praguri înghețate. Defectele sunt simulate pe producție reală (40, câte 10 pe tip).",
               "Final test: a year we never touched while tuning, run once, with frozen thresholds. Faults are simulated on real production (40, ten per type)."),
    "kpi1_l": ("Eroare de predicție mai mică", "Lower prediction error"),
    "kpi1_n": ("față de media istorică a casei · 225 de case, un an întreg", "vs. the home's historical average · 225 homes, a full year"),
    "kpi2_l": ("Defecte simulate găsite", "Simulated faults found"),
    "kpi2_n": ("regula simplă B1: 42,5% · B2: 55%", "simple rule B1: 42.5% · B2: 55%"),
    "kpi3_l": ("Alarme false la 1.000 de zile-casă", "False alarms per 1,000 home-days"),
    "kpi3_n": ("scade de la 2,12 fără verificare · B1: 1,40 · B2: 2,00", "down from 2.12 without verification · B1: 1.40 · B2: 2.00"),
    "kpi4_l": ("Piscina luată drept defect", "Pool pump mistaken for a fault"),
    "kpi4_n": ("100% fără verificare, 40% cu verificarea soare/consum", "100% without verification, 40% with the sun/consumption check"),
    "c1_t": ("Triaj sub capacitate limitată", "Triage under limited capacity"),
    "c1_p": ("Dacă echipa poate verifica 3 case pe săptămână, Dimmer repară 62% din defecte, față de 35% pentru o regulă ordonată după mărimea sistemului. La o verificare pe săptămână: 38% față de 18%.",
             "If the team can check 3 homes a week, Dimmer repairs 62% of faults, versus 35% for a rule ordered by system size. At one check a week: 38% versus 18%."),
    "c1_tag": ("Vezi: Triaj", "See: Triage"),
    "c2_t": ("Mai puține vizite degeaba", "Fewer wasted visits"),
    "c2_p": ("Fără limită de capacitate, Dimmer face 80 de vizite irosite, față de 96 pentru B1 ordonat după capacitate, cu mai puțină energie pierdută (223 față de 295 kWh pe defect). Pe anul de validare, la același procent găsit ca B1, vizitele false au fost la jumătate.",
             "With no capacity limit, Dimmer makes 80 wasted visits versus 96 for B1 ordered by capacity, and loses less energy (223 vs. 295 kWh per fault). On the validation year, at the same share of faults found as B1, false visits were halved."),
    "c2_tag": ("Vezi: Rezultate", "See: Results"),
    "c3_t": ("Fiecare alarmă are explicație", "Every alert comes with an explanation"),
    "c3_p": ("Sistemul spune cât de mare e abaterea, ce parte urmărește soarele și ce acțiune propune. Pentru cazurile CONTACTEAZĂ pregătește și un mesaj către client, pe care îl aprobă un om.",
             "The system states how large the deviation is, how much of it follows the sun and what action it proposes. For CONTACT cases it also drafts a customer message that a person approves."),
    "c3_tag": ("Vezi: Demo live", "See: Live demo"),

    # --- prezentare: de ce atat
    "s2_eye": ("Limitele datelor", "Limits of the data"),
    "s2_title": ("De ce 62% și nu 90%, pe acest set de date", "Why 62% and not 90%, on this dataset"),
    "s2_sub": ("Nu e o limită a modelului, ci a informației disponibile: contorul vede doar diferența dintre consum și producție, nu fiecare în parte.",
               "It is a limit of the available information, not of the model: the meter sees only the difference between consumption and production, not each one separately."),
    "bars_t": ("Cu cât semnalul e mai mare față de zgomot, cu atât se găsește mai mult", "The larger the signal relative to the noise, the more faults are found"),
    "bars_p": ("Raportul semnal/zgomot = cât net în plus produce defectul pe zi, împărțit la cât variază normal consumul casei. Pe anul de validare, la pragul ales:",
               "Signal-to-noise ratio = how much extra net a fault produces per day, divided by how much the home's consumption normally varies. On the validation year, at the chosen threshold:"),
    "bar1": ("Semnal peste 2 ori zgomotul", "Signal above 2x the noise"),
    "bar2": ("Între 1 și 2 ori zgomotul", "Between 1x and 2x the noise"),
    "bar3": ("Între 0,5 și 1 ori zgomotul", "Between 0.5x and 1x the noise"),
    "bar1v": ("100% · 8 din 8", "100% · 8 of 8"),
    "bar2v": ("83% · 5 din 6", "83% · 5 of 6"),
    "bar3v": ("50% · 3 din 6", "50% · 3 of 6"),
    "bars_foot": ("La un prag foarte permisiv, doar 1 din 20 de defecte nu e găsit: pierderea nu vine din date imposibile, ci din compromisul dintre sensibilitate și alarme false.",
                  "At a very permissive threshold only 1 of 20 faults is missed: the loss does not come from impossible data but from the trade-off between sensitivity and false alarms."),
    "dark_t": ("Ce se întâmplă în date", "What happens in the data"),
    "dark_html": ('<b style="color:#f5f5f7">Sisteme mici.</b> Majoritatea au între 1 și 3 kW. Pe validare, defectele parțiale simulate au raportul semnal/zgomot între 0,5 și 1,4: semnalul e de același ordin cu zgomotul.<br><br>'
                  '<b style="color:#f5f5f7">Doar netul.</b> Contorul nu separă consumul de producție, deci modelul trebuie să ghicească consumul, iar o parte din variație rămâne neexplicată.<br><br>'
                  '<b style="color:#f5f5f7">Un singur an de istoric.</b> Fiecare lună apare o singură dată la antrenare, iar defectele sunt simulate: 40 pe test.',
                  '<b style="color:#f5f5f7">Small systems.</b> Most are between 1 and 3 kW. On validation, simulated partial faults have a signal-to-noise ratio between 0.5 and 1.4: the signal is of the same order as the noise.<br><br>'
                  '<b style="color:#f5f5f7">Net only.</b> The meter does not separate consumption from production, so the model has to guess consumption and part of the variation stays unexplained.<br><br>'
                  '<b style="color:#f5f5f7">A single year of history.</b> Each month appears only once in training, and the faults are simulated: 40 in the test.'),
    "cost_t": ("De ce nu urcăm procentul scoțând verificarea", "Why we don't raise the percentage by removing the verification"),
    "cost_p": ("Fără verificarea soare/consum, modelul găsește 77,5% din defecte, dar face de 2,3 ori mai multe vizite irosite. Costul, cu ipotezele din config/economics.json (1,30 lei/kWh, 100 lei pe vizită):",
               "Without the sun/consumption check the model finds 77.5% of faults but makes 2.3 times more wasted visits. Cost, using the assumptions in config/economics.json (1.30 lei/kWh, 100 lei per visit):"),
    "cost_r1a": ("Dimmer, cu verificare", "Dimmer, with verification"),
    "cost_r1b": ("62,5% găsite", "62.5% found"),
    "cost_r1v": ("80 vizite irosite · ≈ 23.000 lei", "80 wasted visits · ≈ 23,000 lei"),
    "cost_r2a": ("Modelul singur", "Model alone"),
    "cost_r2b": ("77,5% găsite", "77.5% found"),
    "cost_r2v": ("188 vizite irosite · ≈ 31.300 lei", "188 wasted visits · ≈ 31,300 lei"),
    "cost_foot": ("Test, capacitate nelimitată. Costul unei vizite este o ipoteză fără sursă publică; varianta cu verificare iese mai ieftină dacă o vizită costă mai mult de aproximativ 27 de lei.",
                  "Test, unlimited capacity. The cost of a visit is an assumption without a public source; the verified variant comes out cheaper if a visit costs more than about 27 lei."),

    # --- prezentare: metoda
    "s3_eye": ("Metoda", "Method"),
    "s3_title": ("Cum funcționează, în patru pași", "How it works, in four steps"),
    "st1_t": ("Ghicește", "Estimate"),
    "st1_p": ("Un model învață, dintr-un an, ce net arată fiecare casă la fiecare oră de zi, ținând cont de soare, de casele vecine și de obiceiurile ei.",
              "A model learns, from one year of data, what net each home shows at every daytime hour, accounting for the sun, neighbouring homes and the home's own habits."),
    "st2_t": ("Adună dovezi", "Accumulate evidence"),
    "st2_p": ("O singură zi ciudată nu produce alarmă. Abaterile se adună zi de zi; alarma apare doar când ele persistă.",
              "A single odd day raises no alert. Deviations are added up day by day; an alert appears only when they persist."),
    "st3_t": ("Soare sau consum?", "Sun or consumption?"),
    "st3_p": ("Lipsa producției crește odată cu soarele, un consum nou (piscină, muncă de acasă) nu. Dimmer măsoară ce parte din abatere urmărește soarele.",
              "Missing production grows with the sun; a new load (pool pump, working from home) does not. Dimmer measures how much of the deviation follows the sun."),
    "st4_t": ("Ordonează și explică", "Rank and explain"),
    "st4_p": ("Cazurile se ordonează după pierderea estimată, sub o capacitate de K verificări pe săptămână, fiecare cu explicație.",
              "Cases are ranked by estimated loss, under a capacity of K checks per week, each with an explanation."),
    "s4_eye": ("Decizia", "Decision"),
    "s4_title": ("Cele trei acțiuni", "The three actions"),
    "act1_t": ("Abaterea urmărește soarele", "The deviation follows the sun"),
    "act1_p": ("Probabil lipsește producția solară. Se pregătește un mesaj către client.", "Solar production is probably missing. A customer message is drafted."),
    "act2_t": ("Abaterea nu urmărește soarele", "The deviation does not follow the sun"),
    "act2_p": ("Probabil un consum nou, sau nu putem spune încă. Nu trimitem pe nimeni; reevaluăm.", "Probably a new load, or we cannot tell yet. Nobody is sent; we reassess."),
    "act3_t": ("Date insuficiente", "Insufficient data"),
    "act3_p": ("Prea puține zile cu date sau prea multe citiri lipsă. Verificăm datele înainte de orice acțiune.", "Too few days of data or too many missing readings. We check the data before any action."),

    # --- prezentare: continut
    "s5_eye": ("Conținut", "Contents"),
    "s5_title": ("Tot ce există în proiect", "Everything in the project"),
    "it1_t": ("Curățarea datelor", "Data cleaning"),
    "it1_p": ("297 din 300 de case păstrate; ceasul Ausgrid (cu ora de vară) aliniat cu radiația.", "297 of 300 homes kept; the Ausgrid clock (with daylight saving) aligned with the radiation data."),
    "it1_g": ("", ""),
    "it2_t": ("Simulator de defecte", "Fault simulator"),
    "it2_p": ("Patru tipuri de defect și patru de consum nou, aplicate pe producția reală.", "Four fault types and four new-load types, applied to real production."),
    "it2_g": ("Rezultate", "Results"),
    "it3_t": ("Model de predicție", "Prediction model"),
    "it3_p": ("LightGBM, cinci variante comparate pentru a vedea ce contribuie.", "LightGBM, five variants compared to see what contributes."),
    "it3_g": ("Rezultate", "Results"),
    "it4_t": ("Dovezi pe mai multe zile", "Evidence over several days"),
    "it4_p": ("CUSUM, cu pragul ales pe validare la o țintă de alarme false.", "CUSUM, with the threshold chosen on validation for a target false-alarm rate."),
    "it4_g": ("", ""),
    "it5_t": ("Verificare soare/consum", "Sun/consumption check"),
    "it5_p": ("Separă o defecțiune a sistemului solar (panouri, cabluri, invertor) de un consum constant.", "Separates a solar-system fault (panels, wiring, inverter) from a constant load."),
    "it5_g": ("Demo live", "Live demo"),
    "it6_t": ("Baseline-uri", "Baselines"),
    "it6_p": ("Două reguli simple și o a treia, ordonată după capacitate, ca comparația să fie corectă.", "Two simple rules and a third, ordered by capacity, so the comparison is fair."),
    "it6_g": ("Rezultate", "Results"),
    "it7_t": ("Coadă cu capacitate", "Capacity-limited queue"),
    "it7_p": ("K verificări pe săptămână, politici comparate zi cu zi.", "K checks per week, policies compared day by day."),
    "it7_g": ("Triaj", "Triage"),
    "it8_t": ("Explicații și mesaj către client", "Explanations and customer message"),
    "it8_p": ("Text pentru fiecare alarmă și o ciornă pentru clientul unui caz CONTACTEAZĂ.", "Text for every alert and a draft for the customer of a CONTACT case."),
    "it8_g": ("Demo live", "Live demo"),
    "it9_t": ("Calculator de cost", "Cost calculator"),
    "it9_p": ("Prețul energiei și costul unei vizite, modificabile.", "Energy price and cost of a visit, adjustable."),
    "it9_g": ("Triaj", "Triage"),
    "it10_t": ("Episoade reale", "Real episodes"),
    "it10_p": ("12 case a căror producție a scăzut real, comparate cu baseline-urile.", "12 homes whose production really dropped, compared with the baselines."),
    "it10_g": ("Rezultate", "Results"),
    "it11_t": ("Evaluare onestă", "Honest evaluation"),
    "it11_p": ("Praguri înghețate, test rulat o singură dată, erori de evaluare corectate și documentate.", "Frozen thresholds, test run once, evaluation errors corrected and documented."),
    "it11_g": ("Rezultate", "Results"),
    "it12_t": ("Reproductibilitate", "Reproducibility"),
    "it12_p": ("Lanț complet într-un script, date brute în repo, aplicație cu date mici.", "The full chain in one script, raw data in the repo, an app with small data."),
    "it12_g": ("", ""),
    "s6_eye": ("Ce nu pretindem", "What we do not claim"),
    "s6_title": ("Limite", "Limitations"),
    "limits_note": ("Defectele sunt simulate pe producție reală, nu confirmate în teren, iar la 10 defecte pe tip diferențele mici nu sunt concludente. "
                    "Dimmer găsește defectele totale mai lent decât o regulă simplă (22,5 zile față de 9). Aerul condiționat rămâne confundat cu un defect, pentru că rulează în zilele însorite. "
                    "Alarmele false ale modelului singur au depășit ținta de 2,0 la 1.000. Datele sunt din Australia: metoda se transferă, cifrele nu. "
                    "Modelul are nevoie de un an de istoric pe casă; nu l-am testat pe clienți noi sau pe alt set de date. Spunem „merită verificat”, nu „este defect”.",
                    "Faults are simulated on real production, not confirmed in the field, and with 10 faults per type small differences are not conclusive. "
                    "Dimmer finds total faults more slowly than a simple rule (22.5 days vs. 9). Air conditioning stays confused with a fault, because it runs on sunny days. "
                    "The model alone exceeded the target of 2.0 false alarms per 1,000. The data is from Australia: the method transfers, the numbers do not. "
                    "The model needs one year of history per home; we have not tested it on new customers or on another dataset. We say “worth checking”, not “is faulty”."),
    "footer": ("Date: Ausgrid, Solar Home Electricity Data (Sydney, 2010-2013, CC BY), {n} de case · Radiație: Open-Meteo · Autori: Bonta Patrick, Gilca Thomas, Matei Ioana, Holhos Alexandru",
               "Data: Ausgrid, Solar Home Electricity Data (Sydney, 2010-2013, CC BY), {n} homes · Radiation: Open-Meteo · Authors: Bonta Patrick, Gilca Thomas, Matei Ioana, Holhos Alexandru"),

    # --- cazuri
    "cs_eye": ("Explorator", "Explorer"),
    "cs_title": ("Un caz, pe larg", "One case, in depth"),
    "cs_sub": ("Linia neagră e netul real al casei la orele de zi (media pe 7 zile), cea punctată e ce ar fi trebuit să arate. Diferența dintre ele se colorează, iar alarmele apar ca puncte.",
               "The black line is the home's real net during daytime hours (7-day average); the dotted line is what it should have shown. The difference between them is shaded and alerts appear as dots."),
    "cat_fault": ("Defect simulat", "Simulated fault"),
    "cat_load": ("Consum nou simulat (panourile sunt bune)", "Simulated new load (panels are fine)"),
    "cat_real": ("Scădere reală de producție (probabil defect)", "Real production drop (probable fault)"),
    "cat_clean": ("Casă curată", "Clean home"),
    "w_type": ("Tipul casei", "Home type"),
    "w_home": ("Casa", "Home"),
    "w_truth": ("Arată adevărul (ascuns pentru sistem)", "Show the truth (hidden from the system)"),
    "home_k": ("casa {h} · {k} · {kw} kW", "home {h} · {k} · {kw} kW"),
    "home_real": ("casa {h} · scădere reală · {kw} kW", "home {h} · real drop · {kw} kW"),
    "home_plain": ("casa {h} · {kw} kW", "home {h} · {kw} kW"),
    "al_suffix": (" · {n} alarme", " · {n} alerts"),
    "no_cat": ("Nu există case în această categorie în datele curente.", "There are no homes in this category in the current data."),
    "lg_net": ("net real, orele de zi", "real net, daytime hours"),
    "lg_exp": ("net așteptat de model", "net expected by the model"),
    "lg_gap": ("mai mult net decât se aștepta", "more net than expected"),
    "lg_start": ("  început defect simulat", "  simulated fault starts"),
    "y_net": ("kWh pe oră (net)", "kWh per hour (net)"),
    "y_day": ("kWh pe zi", "kWh per day"),
    "lg_p_nofault": ("producție reală, fără defect", "real production, no fault"),
    "lg_p_fault": ("producție după defectul simulat", "production after the simulated fault"),
    "lg_p_same": ("producția (neschimbată: panourile funcționează)", "production (unchanged: the panels work)"),
    "lg_added": ("consum nou adăugat (kWh/zi)", "added new load (kWh/day)"),
    "lg_p_meas": ("producția reală măsurată (kWh/zi)", "measured real production (kWh/day)"),
    "lg_p_real": ("producția reală (kWh/zi)", "real production (kWh/day)"),
    "tr_fault": ("Adevărul: {k} simulat din {d}; pierdere medie {x} kWh/zi.", "The truth: simulated {k} from {d}; average loss {x} kWh/day."),
    "tr_load": ("Adevărul: panourile funcționează; am adăugat consum de tip {k} din {d}.", "The truth: the panels work; we added a new load of type {k} from {d}."),
    "tr_real": ("Adevărul: producția reală a scăzut în această perioadă. Cauza nu o cunoaștem (poate fi un defect la panouri, cabluri sau invertor, umbră, panou scos).",
                "The truth: real production dropped in this period. We do not know the cause (a fault in the panels, wiring or inverter, shade, or a removed panel)."),
    "tr_clean": ("Adevărul: casă fără nicio modificare simulată; alarmele de mai sus sunt alarme false.", "The truth: a home with no simulated change; the alerts above are false alarms."),
    "al_head": ("Alarme · casa {h} · {n}", "Alerts · home {h} · {n}"),
    "al_last": ("Sunt afișate ultimele 6 din {n} alarme.", "Showing the last 6 of {n} alerts."),
    "al_none": ("Nicio alarmă în perioada de test.", "No alerts in the test period."),
    "msg_eye": ("Mesaj către client", "Customer message"),
    "msg_exp": ("Ciornă pentru ultima alarmă CONTACTEAZĂ", "Draft for the latest CONTACT alert"),
    "msg_cap": ("Șablon de adaptat și de aprobat de un operator uman; sistemul nu trimite nimic singur. Mesajul nu afirmă că sistemul e defect și nici ce componentă (panouri, cabluri, invertor): nu putem ști de la distanță.",
                "A template to be adapted and approved by a human operator; the system sends nothing on its own. The message does not claim the system is faulty, nor which component (panels, wiring, inverter): we cannot know remotely."),
    "msg_none": ("Pentru această casă nu pleacă niciun mesaj către client: nicio alarmă de tip CONTACTEAZĂ.", "No customer message is sent for this home: there is no CONTACT alert."),

    # --- coada
    "q_eye": ("Triaj", "Triage"),
    "q_title": ("Coada de verificări sub capacitate limitată", "Inspection queue under limited capacity"),
    "q_sub": ("Echipa poate verifica doar K case pe săptămână. Fiecare alarmă deschide un caz, iar echipa le ia pe cele mai importante. O vizită la o casă cu defect îl repară; una la o casă fără defect e irosită.",
              "The team can check only K homes per week. Each alert opens a case, and the team takes the most important ones. A visit to a home with a fault repairs it; a visit to a home without one is wasted."),
    "q_unl": ("Fără limită de capacitate", "No capacity limit"),
    "q_slider": ("Capacitate: verificări pe săptămână", "Capacity: checks per week"),
    "q_pol": ("Politici comparate", "Policies compared"),
    "q_pick": ("Alegeți cel puțin o politică.", "Pick at least one policy."),
    "col_pol": ("Politică", "Policy"),
    "col_rep": ("Defecte reparate (%)", "Faults repaired (%)"),
    "col_kwhm": ("kWh pierduți pe defect (medie)", "kWh lost per fault (mean)"),
    "col_kwhmed": ("kWh pierduți pe defect (mediană)", "kWh lost per fault (median)"),
    "col_vt": ("Vizite totale", "Total visits"),
    "col_vw": ("Vizite irosite", "Wasted visits"),
    "col_ff": ("Defecte găsite", "Faults found"),
    "col_prec": ("Precizia vizitelor (%)", "Visit precision (%)"),
    "col_days": ("Zile mediane până la reparație", "Median days to repair"),
    "q_none": ("fără nicio verificare", "no inspections"),
    "q_ylab": ("kWh de producție pierduți, cumulat", "kWh of lost production, cumulative"),
    "q_ct": ("Energia pierdută pe cele {n} defecte simulate", "Energy lost across the {n} simulated faults"),
    "q_ctk": (" · K = {k} pe săptămână", " · K = {k} per week"),
    "cost_eye": ("Cost", "Cost"),
    "cost_title": ("Cu ipoteze pe care le puteți modifica", "With assumptions you can change"),
    "in_price": ("Prețul energiei (lei/kWh)", "Energy price (lei/kWh)"),
    "in_visit": ("Costul unei vizite (lei)", "Cost of one visit (lei)"),
    "col_en": ("Energie pierdută (lei)", "Energy lost (lei)"),
    "col_vis": ("Vizite (lei)", "Visits (lei)"),
    "col_tot": ("Cost total (lei)", "Total cost (lei)"),
    "cost_note": ("Preț: ofertele publice pentru casnici din 2026 (1,06-1,45 lei/kWh), sursele sunt în config/economics.json; valoarea implicită e 1,30 lei/kWh. Cost vizită: ipoteză a autorilor, fără sursă publică.",
                  "Price: public 2026 household offers (1.06-1.45 lei/kWh), sources are in config/economics.json; the default is 1.30 lei/kWh. Cost of a visit: an assumption by the authors, with no public source."),
    "log_exp": ("Jurnalul verificărilor (Dimmer)", "Inspection log (Dimmer)"),
    "log_date": ("Data", "Date"),
    "log_home": ("Casa", "Home"),
    "log_out": ("Rezultat", "Outcome"),
    "log_wait": ("Zile de așteptare", "Days waiting"),
    "q_foot": ("Valorile sunt pe 40 de defecte simulate (10 pe tip). Cu atât de puține, diferențele mici dintre politici nu sunt concludente.",
               "Values are over 40 simulated faults (10 per type). With so few, small differences between policies are not conclusive."),

    # --- episoade reale
    "r_eye": ("Date reale", "Real data"),
    "r_title": ("Episoade reale de scădere a producției", "Real production-drop episodes"),
    "r_sub": ("Pe lângă defectele simulate, datele conțin case a căror producție a scăzut real. Nu știm de ce (defect la panouri, cabluri sau invertor, umbră, panou scos), deci le numim episoade probabile. "
              "Lista s-a stabilit după o regulă fixă (randamentul din perioadă sub 70% din cel din anul de antrenare), nu după rezultatele vreunui model, iar casele nu au primit defecte simulate.",
              "Besides the simulated faults, the data contains homes whose production really dropped. We do not know why (a fault in the panels, wiring or inverter, shade, a removed panel), so we call them probable episodes. "
              "The list was set by a fixed rule (yield in the period below 70% of the training year), not by the results of any model, and these homes received no simulated faults."),
    "r_none": ("Nu există case cu scădere reală în datele curente.", "There are no real-drop homes in the current data."),
    "r_of": ("{a} din {b}", "{a} of {b}"),
    "r_k1l": ("B1 a alarmat", "B1 alerted"),
    "r_k1n": ("regulă simplă", "simple rule"),
    "r_k2l": ("B2 a alarmat", "B2 alerted"),
    "r_k2n": ("față de anul trecut", "vs. last year"),
    "r_k3l": ("Dimmer a alarmat", "Dimmer alerted"),
    "r_k3n": ("model + verificare", "model + verification"),
    "r_home": ("Casa", "Home"),
    "r_dfirst": ("Dimmer, prima alarmă", "Dimmer, first alert"),
    "r_dcontact": ("Dimmer, primul CONTACT", "Dimmer, first CONTACT"),
    "r_lg": ("producția reală (kWh/zi, media pe 7 zile)", "real production (kWh/day, 7-day average)"),
    "r_first": ("prima alarmă {m}: {d}", "first alert {m}: {d}"),
    "r_foot": ("Sistemul nu vede producția, doar netul. Aici o afișăm doar ca să vedem scăderea.", "The system does not see production, only the net. We show it here only so the drop is visible."),

    # --- rezultate
    "rs_eye": ("Evaluare", "Evaluation"),
    "rs_title": ("Rezultatele, cinstit", "The results, honestly"),
    "rs_sub": ("Toate pragurile au fost alese pe anul de validare și înghețate înainte de testul final, rulat o singură dată.",
               "All thresholds were chosen on the validation year and frozen before the final test, which was run once."),
    "rs_m_eye": ("Comparația principală", "Main comparison"),
    "rs_m_title": ("Test, 2012-07 → 2013-06", "Test, 2012-07 → 2013-06"),
    "lab_found": ("Defecte simulate prinse", "Simulated faults found"),
    "lab_fa": ("Alarme false la 1.000 zile-casă", "False alarms per 1,000 home-days"),
    "lab_kwh": ("kWh pierduți pe defect", "kWh lost per fault"),
    "col_dim": ("Dimmer (model + verificare)", "Dimmer (model + verification)"),
    "col_dimn": ("Dimmer fără verificare", "Dimmer without verification"),
    "rs_m_foot": ("Pentru Dimmer, o alarmă falsă este o alarmă CONTACTEAZĂ pe o casă curată, adică o vizită irosită. Ultima coloană este un alt punct de funcționare: mai sensibil, dar cu mai multe vizite false; nu o comparați direct cu B1.",
                  "For Dimmer, a false alarm is a CONTACT alert on a clean home, that is, a wasted visit. The last column is a different operating point: more sensitive, but with more false visits; do not compare it directly with B1."),
    "rs_p_eye": ("Predicția", "Prediction"),
    "rs_p_title": ("Eroarea medie, în kWh pe oră", "Mean error, in kWh per hour"),
    "p_col": ("Eroare medie (kWh/oră)", "Mean error (kWh/hour)"),
    "p_r1": ("Baseline: media casei la aceeași oră și lună", "Baseline: the home's average for the same hour and month"),
    "p_r2": ("Model LightGBM (radiație + vecini)", "LightGBM model (radiation + neighbours)"),
    "p_r3": ("Îmbunătățire", "Improvement"),
    "p_axis": ("validare / test", "validation / test"),
    "p_abl": ("Ablația pe test (ce contribuie fiecare ingredient), din reports/model_test_raw.csv:", "Ablation on the test (what each ingredient contributes), from reports/model_test_raw.csv:"),
    "abl_fa": ("Alarme false la 1.000", "False alarms per 1,000"),
    "abl_kwh": ("kWh pierduți (medie)", "kWh lost (mean)"),
    "rs_d_eye": ("Detecția", "Detection"),
    "rs_d_title": ("Pe tipul de defect (test)", "By fault type (test)"),
    "d_type": ("Tip defect", "Fault type"),
    "d_found": ("prinse %", "found %"),
    "d_days": ("zile", "days"),
    "d_model": ("Model fără verificare", "Model without verification"),
    "rs_d_foot": ("Alarme false la 1.000 zile-casă: B1 1,40 · B2 2,00 · model fără verificare 2,12 (peste ținta de 2,0). Pe validare, detecția era la paritate cu B1; pe test e mai bună, dar cu 10 defecte pe tip o parte din diferență poate fi întâmplare.",
                  "False alarms per 1,000 home-days: B1 1.40 · B2 2.00 · model without verification 2.12 (above the target of 2.0). On validation, detection was on par with B1; on test it is better, but with 10 faults per type part of the difference may be chance."),
    "rs_x_eye": ("Stratul de decizie", "Decision layer"),
    "rs_x_title": ("Soare sau consum?", "Sun or consumption?"),
    "x_c1": ("Fără verificare", "Without verification"),
    "x_c2": ("Cu verificare (prag 0,4)", "With verification (threshold 0.4)"),
    "x_r1": ("Alarme CONTACT false la 1.000", "False CONTACT alerts per 1,000"),
    "x_r2": ("Defecte prinse", "Faults found"),
    "x_r3": ("kWh pierduți (medie)", "kWh lost (mean)"),
    "x_r4": ("Piscină cu alarmă", "Pool pump alerted"),
    "x_r5": ("Muncă de acasă cu alarmă", "Working from home alerted"),
    "x_r6": ("Aer condiționat cu alarmă", "Air conditioning alerted"),
    "rs_x_foot": ("Verificarea reduce vizitele false și confuzia cu consumul constant, dar costă recall (78% → 62%). Defectele slabe (parțiale, intermitente, treptate) trec prin verificare cam ca alarmele false (40-51% față de 47%); cele puternice (totale: 77%) și scăderile reale (73%) sunt păstrate.",
                  "The verification reduces false visits and confusion with constant loads, but it costs recall (78% → 62%). Weak faults (partial, intermittent, gradual) pass the check about as often as false alarms (40-51% vs. 47%); strong ones (total: 77%) and real drops (73%) are kept."),
    "rs_t_eye": ("Triaj", "Triage"),
    "rs_t_title": ("Cu capacitate limitată (aceleași calcule ca în reports/triage_test.csv)", "With limited capacity (the same calculations as reports/triage_test.csv)"),
    "t_k": ("K pe săptămână", "K per week"),
    "t_pol": ("Politică", "Policy"),
    "t_rep": ("Reparate %", "Repaired %"),
    "t_kwh": ("kWh pe defect", "kWh per fault"),
    "t_w": ("Vizite irosite", "Wasted visits"),
    "t_nolimit": ("fără limită", "no limit"),
    "rs_l_eye": ("Limite", "Limitations"),
    "rs_l_title": ("Ce trebuie să știți", "What you should know"),
    "rs_l_note": ("• Defectele sunt simulate (40 în test, câte 10 pe tip). Cu atât de puține, diferențele mici dintre metode nu sunt concludente.<br>"
                  "• Alarmele false pe test (2,12 la 1.000) au depășit ținta de 2,0.<br>"
                  "• Defectele totale sunt prinse mai încet decât de o regulă simplă (22,5 zile față de 9).<br>"
                  "• Aerul condiționat rămâne confundat cu un defect: rulează în zilele calde, deci însorite.<br>"
                  "• Pe episoadele reale, Dimmer alarmează la fel de multe case ca B1 (9 din 12), iar B2 mai multe (11 din 12); pentru unele (de exemplu casa 234) alarmează mult mai târziu decât baseline-urile.<br>"
                  "• Modelul are nevoie de un an de istoric pe casă; nu l-am testat pe clienți noi sau pe alt set de date.<br>"
                  "• Datele sunt din Australia (Ausgrid, 2010-2013); metoda se transferă, cifrele nu.<br>"
                  "• Nu am confirmat defecte fizice; spunem „merită verificat”, nu „este defect”.",
                  "• Faults are simulated (40 in the test, ten per type). With so few, small differences between methods are not conclusive.<br>"
                  "• False alarms on the test (2.12 per 1,000) exceeded the target of 2.0.<br>"
                  "• Total faults are caught more slowly than by a simple rule (22.5 days vs. 9).<br>"
                  "• Air conditioning remains confused with a fault: it runs on hot, therefore sunny, days.<br>"
                  "• On the real episodes, Dimmer alerts as many homes as B1 (9 of 12) and B2 more (11 of 12); for some (for example home 234) it alerts much later than the baselines.<br>"
                  "• The model needs one year of history per home; we have not tested it on new customers or on another dataset.<br>"
                  "• The data is from Australia (Ausgrid, 2010-2013); the method transfers, the numbers do not.<br>"
                  "• We have not confirmed physical faults; we say “worth checking”, not “is faulty”."),
}


# --- demo live (redare accelerata a datelor istorice, procesare prin lantul real)
TX.update({
    "nav_demo": ("Demo live", "Live demo"),
    "dm_eye": ("Demo", "Demo"),
    "dm_title": ("Dimmer, zi cu zi, pe date istorice", "Dimmer, day by day, on historical data"),
    "dm_sub": ("Fiecare zi din anul de test trece, pe rând, prin același lanț ca rularea obișnuită: modelul LightGBM antrenat (încărcat o singură dată), "
               "dovezile CUSUM, verificarea soare/consum, triajul și mesajul. Nimic nu e precalculat: ce vedeți mai jos se calculează acum.",
               "Each day of the test year goes, in turn, through the same chain as the regular run: the trained LightGBM model (loaded once), "
               "the CUSUM evidence, the sun/consumption check, triage and the message. Nothing is precomputed: what you see below is being computed now."),
    "dm_banner": ("Date istorice · redare accelerată · procesare live", "Historical data · accelerated replay · live processing"),
    "dm_missing": ("Modul demo are nevoie de datele procesate și de modelul antrenat ({m}). Rulați lanțul (README, secțiunea 4.4): "
                   "`python src/prepare.py`, `python src/simulate.py --period test`, apoi `python src/demo_scenario.py`.",
                   "The demo mode needs the processed data and the trained model ({m}). Run the chain (README, section 4.4): "
                   "`python src/prepare.py`, `python src/simulate.py --period test`, then `python src/demo_scenario.py`."),
    "dm_noscen": ("Lipsește config/demo_scenario.json. Rulați: `python src/demo_scenario.py`",
                  "config/demo_scenario.json is missing. Run: `python src/demo_scenario.py`"),
    "dm_start": ("Start", "Start"), "dm_pause": ("Pauză", "Pause"), "dm_resume": ("Continuare", "Resume"), "dm_reset": ("Reset", "Reset"),
    "dm_speed": ("Viteză", "Speed"), "dm_speed_f": ("{s} s / zi simulată", "{s} s / simulated day"),
    "dm_st_idle": ("Gata de pornire", "Ready"), "dm_st_warming": ("Se procesează istoricul anterior", "Processing the earlier history"),
    "dm_st_running": ("Rulează", "Running"), "dm_st_paused": ("În pauză", "Paused"), "dm_st_finished": ("Terminat", "Finished"),
    "dm_st_error": ("Eroare", "Error"),
    "dm_simdate": ("Data simulată", "Simulated date"), "dm_simnote": ("ultima zi completă procesată", "last full day processed"),
    "dm_real": ("Timp real", "Real time"), "dm_realnote": ("scurs de la Start, fără pauze", "since Start, excluding pauses"),
    "dm_prog": ("Ziua {a} din {b}", "Day {a} of {b}"),
    "dm_stepms": ("ultimul pas: {ms} ms pentru o zi", "last step: {ms} ms for one day"),
    "dm_overrun": (" · {n} pași mai lungi decât intervalul, rulați unul după altul", " · {n} steps longer than the interval, run one after another"),
    "dm_window": ("Fereastra demo: {a} → {b} ({n} zile). Înainte, istoricul {c} → {d} trece prin același lanț, fără acțiuni, ca CUSUM-ul să pornească din aceeași stare ca rularea obișnuită.",
                  "Demo window: {a} → {b} ({n} days). Before it, the history {c} → {d} goes through the same chain, without actions, so CUSUM starts from the same state as in the regular run."),
    "dm_k_days": ("Zile procesate", "Days processed"), "dm_k_al": ("Alarme CUSUM", "CUSUM alerts"),
    "dm_k_cases": ("Cazuri create", "Cases created"), "dm_k_notif": ("Notificări", "Notifications"),
    "dm_k_fleet": ("în fereastră, pe toate cele {n} instalații", "in the window, across all {n} installations"),
    "dm_scen": ("Scenariul", "Scenario"),
    "dm_scen_p": ("Ales de src/demo_scenario.py după rezultatele pipeline-ului pe anul de test (reguli în config/demo_scenario.json). În demo, totul se recalculează.",
                  "Chosen by src/demo_scenario.py from the pipeline's results on the test year (rules in config/demo_scenario.json). In the demo, everything is recomputed."),
    "role_normal": ("Funcționare normală", "Normal operation"),
    "role_temporary": ("Abatere păstrată în monitorizare", "Deviation kept under monitoring"),
    "role_persistent": ("Subperformanță persistentă", "Persistent underperformance"),
    "src_real": ("Date reale, nemodificate", "Real data, unmodified"),
    "src_real_decline": ("Date reale · producția a scăzut real (cauză necunoscută)", "Real data · production really dropped (unknown cause)"),
    "src_sim_fault": ("Anomalie simulată · {k} din {d}", "Simulated anomaly · {k} from {d}"),
    "src_sim_load": ("Anomalie simulată · {k} din {d} (panourile funcționează)", "Simulated anomaly · {k} from {d} (panels work)"),
    "src_hidden": ("Proveniența datelor e afișată pentru prezentare; pipeline-ul nu o vede.", "Data provenance is shown for the presentation; the pipeline does not see it."),
    "dm_track": ("Instalația urmărită", "Tracked installation"),
    "dm_home_opt": ("casa {h} · {r}", "home {h} · {r}"),
    "dm_home_case": ("casa {h} · caz {c}", "home {h} · case {c}"),
    "dm_state": ("Starea instalației", "Installation state"),
    "st_wait": ("În așteptare", "Waiting"),
    "st_monitor": ("Monitorizare", "Monitoring"),
    "st_al_monitor": ("Abatere detectată · păstrată în monitorizare", "Deviation detected · kept under monitoring"),
    "st_al_data": ("Abatere detectată · date de verificat", "Deviation detected · data to check"),
    "st_invest": ("Investigație recomandată", "Investigation recommended"),
    "st_n_wait": ("Fereastra demo nu a început încă.", "The demo window has not started yet."),
    "st_n_monitor": ("Nicio alarmă în ultimele 14 zile; dovezile CUSUM sunt sub prag.", "No alert in the last 14 days; the CUSUM evidence is below the threshold."),
    "st_n_after": ("Alarmă pe {d}; verificarea a decis: {a}. Nicio alarmă nouă timp de 14 zile.", "Alert on {d}; the check decided: {a}. No new alert for 14 days."),
    "st_n_case": ("Cazul {c} este deschis din {d}. Merită verificat; nu e un defect confirmat.", "Case {c} has been open since {d}. Worth checking; not a confirmed fault."),
    "dm_cusum": ("Dovezi acumulate (CUSUM)", "Accumulated evidence (CUSUM)"),
    "dm_cusum_n": ("{s} din pragul {h}. Nu este o probabilitate: suma abaterilor zilnice standardizate, minus {k} pe zi. La depășirea pragului apare o alarmă, apoi suma revine la 0.",
                   "{s} of the threshold {h}. Not a probability: the sum of standardised daily deviations, minus {k} per day. Crossing the threshold raises an alert, then the sum resets to 0."),
    "dm_ztoday": ("Abaterea standardizată a zilei: z = {z}", "Standardised deviation of the day: z = {z}"),
    "dm_ch_obs": ("net observat", "observed net"), "dm_ch_exp": ("net estimat de model", "model-estimated net"),
    "dm_ch_y": ("kWh pe oră (net, orele 8-17)", "kWh per hour (net, hours 8-17)"),
    "dm_ch_note": ("Valori zilnice până la data simulată. Contorul vede doar netul (consum − producție), nu producția: un net mai mare decât cel estimat poate însemna producție lipsă sau consum nou. Punctele sunt alarmele, colorate după acțiune.",
                   "Daily values up to the simulated date. The meter only sees net (consumption − production), not production: a net higher than estimated may mean missing production or a new load. Dots are alerts, coloured by action."),
    "dm_feed": ("Activitate", "Activity"), "dm_feed_all": ("Toate instalațiile", "All installations"),
    "dm_feed_none": ("Încă niciun eveniment. Apăsați Start.", "No events yet. Press Start."),
    "ev_warmup": ("Istoric procesat {a} → {b}: {n} zile prin același lanț, {al} alarme, fără acțiuni ({s} s).",
                  "History processed {a} → {b}: {n} days through the same chain, {al} alerts, no actions ({s} s)."),
    "ev_started": ("Începe fereastra demo. Acțiunile automate sunt active.", "The demo window starts. Automatic actions are on."),
    "ev_paused": ("Pauză.", "Paused."), "ev_resumed": ("Continuare.", "Resumed."),
    "ev_finished": ("Fereastra demo s-a încheiat: {n} zile procesate.", "The demo window has ended: {n} days processed."),
    "ev_error": ("Eroare: {m}. Replay-ul s-a oprit; nimic nu a fost scris pe disc.", "Error: {m}. The replay stopped; nothing was written to disk."),
    "ev_missing": ("Casa {h}: {p}% din citirile de zi lipsesc.", "Home {h}: {p}% of daytime readings are missing."),
    "ev_missing_fleet": ("{p}% din citirile zilei lipsesc în toată rețeaua.", "{p}% of the day's readings are missing across the network."),
    "ev_alarm": ("Casa {h}: dovezile CUSUM au trecut de prag ({s} > {hh}). Abatere detectată; urmează verificarea soare/consum.",
                 "Home {h}: CUSUM evidence crossed the threshold ({s} > {hh}). Deviation detected; the sun/consumption check follows."),
    "ev_check": ("Verificare, casa {h} → {a}. {x}", "Check, home {h} → {a}. {x}"),
    "ev_no_case": ("Casa {h}: {a}. Nu se deschide caz, nu pleacă nicio notificare.", "Home {h}: {a}. No case is opened, no notification is sent."),
    "ev_case_created": ("Caz {c} creat automat pentru casa {h} · prioritate {p} kWh/zi.", "Case {c} created automatically for home {h} · priority {p} kWh/day."),
    "ev_case_updated": ("Casa {h} are deja cazul {c} deschis: alarma nouă e atașată, prioritatea devine {p} kWh/zi.",
                        "Home {h} already has case {c} open: the new alert is attached, priority becomes {p} kWh/day."),
    "ev_case_evidence": ("Casa {h}: alarma ({a}) e atașată la cazul deschis {c}.", "Home {h}: the alert ({a}) is attached to open case {c}."),
    "ev_notification": ("Notificare pentru {c} în inboxul demo (netrimisă).", "Notification for {c} in the demo inbox (not sent)."),
    "ev_notif_dedup": ("Casa {h}: notificarea pentru {c} există deja; nu se trimite din nou.", "Home {h}: the notification for {c} already exists; it is not sent again."),
    "dm_cases": ("Cazuri create în demo", "Cases created in the demo"),
    "dm_cases_sub": ("Izolate de datele aplicației: există doar în memoria acestei sesiuni, iar Reset le șterge. Ordonate după prioritatea din triaj "
                     "(pierdere estimată × parte solară, în kWh/zi; nu este o probabilitate). O casă are cel mult un caz deschis; alarmele noi se atașează la el.",
                     "Isolated from the app's data: they exist only in this session's memory, and Reset deletes them. Ordered by the triage priority "
                     "(estimated loss × solar share, in kWh/day; not a probability). A home has at most one open case; new alerts are attached to it."),
    "dm_cases_none": ("Niciun caz încă.", "No cases yet."),
    "c_id": ("Caz", "Case"), "c_home": ("Casa", "Home"), "c_open": ("Deschis", "Opened"), "c_src": ("Proveniența datelor", "Data provenance"),
    "c_prio": ("Prioritate (kWh/zi)", "Priority (kWh/day)"), "c_loss": ("Pierdere estimată acumulată (kWh)", "Estimated accumulated loss (kWh)"),
    "c_cost": ("≈ valoare ({cur})", "≈ value ({cur})"), "c_n": ("Alarme", "Alerts"),
    "c_pick": ("Explicația și raportul pentru", "Explanation and report for"),
    "c_expl": ("Explicație", "Explanation"),
    "c_evid": ("Dovezi calculate", "Computed evidence"),
    "e_date": ("Data alarmei", "Alert date"), "e_cusum": ("CUSUM la alarmă / prag", "CUSUM at alert / threshold"),
    "e_days": ("Zile cu date în fereastra de 28", "Days with data in the 28-day window"), "e_miss": ("Citiri de zi lipsă", "Missing daytime readings"),
    "e_exc": ("Abatere medie la orele de zi", "Mean daytime deviation"), "e_sun": ("Parte solară (cât din abatere crește cu soarele)", "Solar share (how much of the deviation rises with the sun)"),
    "e_rule": ("Motivul deciziei", "Reason for the decision"),
    "e_exc_v": ("{x} kWh/oră · {y} kWh/zi", "{x} kWh/hour · {y} kWh/day"),
    "rule_data_miss": ("{p}% citiri lipsă > 20% → VERIFICĂ DATELE", "{p}% missing readings > 20% → CHECK DATA"),
    "rule_data_few": ("doar {n} zile cu date < 14 → VERIFICĂ DATELE", "only {n} days with data < 14 → CHECK DATA"),
    "rule_flat": ("soarele a variat prea puțin în fereastră → MONITORIZEAZĂ", "the sun varied too little in the window → MONITOR"),
    "rule_small": ("abatere medie prea mică pentru a calcula partea solară → MONITORIZEAZĂ", "mean deviation too small to compute the solar share → MONITOR"),
    "rule_contact": ("parte solară {s} ≥ prag {t} → CONTACTEAZĂ (investigație recomandată)", "solar share {s} ≥ threshold {t} → CONTACT (investigation recommended)"),
    "rule_monitor": ("parte solară {s} < prag {t} → MONITORIZEAZĂ", "solar share {s} < threshold {t} → MONITOR"),
    "c_money": ("Estimare financiară", "Financial estimate"),
    "c_acc": ("Pierdere acumulată estimată: {k} kWh ≈ {v} {cur}", "Estimated accumulated loss: {k} kWh ≈ {v} {cur}"),
    "c_acc_n": ("Perioada: {a} → {b}, {n} zile cu date, fiecare zi numărată o singură dată chiar dacă alarmele se suprapun. Tarif: {p} {cur}/kWh (config/economics.json; prețul plătit de client).",
                "Period: {a} → {b}, {n} days with data, each day counted once even when alerts overlap. Tariff: {p} {cur}/kWh (config/economics.json; the price paid by the customer)."),
    "c_proj": ("Proiecție, neinclusă mai sus: ~{k} kWh/zi → ~{v} {cur} în următoarele 30 de zile, dacă abaterea se menține.",
               "Projection, not included above: ~{k} kWh/day → ~{v} {cur} over the next 30 days, if the deviation persists."),
    "c_noloss": ("Pierderea solară nu se poate estima pentru acest caz.", "The solar loss cannot be estimated for this case."),
    "c_disc": ("Indiciu că lipsește producția solară, nu un defect confirmat. Cauza nu e neapărat la panouri: pot fi și cablurile, invertorul sau altă componentă; doar o verificare la fața locului o poate stabili.",
               "An indication that solar production is missing, not a confirmed fault. The cause is not necessarily the panels: it may also be the wiring, the inverter or another component; only an on-site check can tell."),
    "c_report": ("Descarcă raportul (.md)", "Download the report (.md)"),
    "c_timeline": ("Alarmele cazului", "The case's alerts"),
    "dm_inbox": ("Inbox demo", "Demo inbox"),
    "dm_inbox_sub": ("Notificările sunt generate de mecanismul existent (ciorna mesajului către client din src/messages.py), una pe caz. "
                     "Rămân aici: proiectul nu are niciun destinatar de test configurat, deci nimic nu pleacă din aplicație. Un operator le adaptează și le aprobă.",
                     "Notifications are generated by the existing mechanism (the customer message draft from src/messages.py), one per case. "
                     "They stay here: the project has no test recipient configured, so nothing leaves the app. An operator adapts and approves them."),
    "dm_inbox_none": ("Inboxul e gol.", "The inbox is empty."),
    "dm_notif_h": ("{n} · {c} · casa {h} · {d}", "{n} · {c} · home {h} · {d}"),
    "dm_unsent": ("DEMO · netrimisă · de aprobat de un operator", "DEMO · not sent · to be approved by an operator"),
    "dm_inbox_more": ("Sunt afișate ultimele {a} din {b} notificări.", "Showing the latest {a} of {b} notifications."),
    "dm_err": ("Replay-ul s-a oprit cu o eroare: {m}. Apăsați Reset.", "The replay stopped with an error: {m}. Press Reset."),
    "rep_title": ("Raport de caz (DEMO)", "Case report (DEMO)"),
    "dm_truth_h": ("Adevărul, ascuns pentru sistem", "The truth, hidden from the system"),
    "dm_truth_note": ("Producția reală (kWh/zi) vine din datele Ausgrid și e afișată doar până la data simulată. Pipeline-ul nu o vede.",
                      "Real production (kWh/day) comes from the Ausgrid data and is shown only up to the simulated date. The pipeline does not see it."),
    "dm_base": ("Aceeași casă, alte metode, până la data simulată", "Same home, other methods, up to the simulated date"),
    "dm_base_b1": ("B1 · regulă simplă (zile la rând fără export la prânz)", "B1 · simple rule (days in a row with no midday export)"),
    "dm_base_b2": ("B2 · față de același interval din anul de antrenare", "B2 · vs. the same window in the training year"),
    "dm_base_dim": ("Dimmer · alarme din replay", "Dimmer · alerts from the replay"),
    "dm_base_note": ("B1 și B2 au pragurile înghețate și vin din rularea obișnuită (demo_data/test); sunt afișate doar alarmele de până la data simulată. Alarmele Dimmer includ și istoricul procesat înainte de fereastră.",
                     "B1 and B2 use frozen thresholds and come from the regular run (demo_data/test); only alerts up to the simulated date are shown. Dimmer's alerts include the history processed before the window."),
    "dm_none": ("nicio alarmă", "no alerts"),
    "dm_fleet": ("Rețeaua, la data simulată", "The network, at the simulated date"),
    "dm_fleet_sub": ("Toate cele {n} instalații, câte un pătrat. Culoarea arată starea; intensitatea portocalie arată dovezile CUSUM acumulate față de prag (nu o probabilitate). Conturul negru marchează instalațiile din scenariu; treceți cu mouse-ul peste un pătrat pentru detalii.",
                     "All {n} installations, one square each. Colour shows the state; orange intensity shows the accumulated CUSUM evidence relative to the threshold (not a probability). The black outline marks the scenario installations; hover over a square for details."),
    "fl_monitor": ("Monitorizare: {n}, dintre care {m} cu dovezi în acumulare", "Monitoring: {n}, of which {m} with evidence building up"),
    "fl_count": ("{l}: {n}", "{l}: {n}"),
    "fl_title": ("casa {h} · {s} · CUSUM {c} / {t} · {src}", "home {h} · {s} · CUSUM {c} / {t} · {src}"),
    "dm_inbox_all": ("Toate instalațiile", "All installations"),
    "dm_inbox_scope": ("Implicit sunt afișate doar instalațiile din scenariu și cea urmărită.", "By default only the scenario installations and the tracked one are shown."),
    "dm_inbox_hidden": ("Nicio notificare pentru instalațiile urmărite; {n} în total pe rețea (activați „Toate instalațiile”).",
                        "No notifications for the tracked installations; {n} in total across the network (turn on “All installations”)."),
    "rep_banner": ("Date istorice · redare accelerată · procesare live. Caz demonstrativ, nu un caz real al aplicației.",
                   "Historical data · accelerated replay · live processing. Demonstration case, not a real case of the app."),
})


def check_tx():
    for k, v in TX.items():
        assert isinstance(v, tuple) and len(v) == 2, f"TX[{k!r}] trebuie sa fie (ro, en)"


check_tx()

KPI_TEMPLATE = """
<!doctype html><html><head><meta charset="utf-8"><style>
html,body { margin:0; background:transparent; font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","Segoe UI",Inter,Roboto,Arial,sans-serif; color:#1d1d1f; -webkit-font-smoothing:antialiased; }
.row { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; padding:6px 4px 18px 4px; }
.k { background:#fff; border-radius:22px; padding:22px 24px; border:1px solid rgba(0,0,0,.04); box-sizing:border-box;
     box-shadow:0 1px 2px rgba(0,0,0,.04),0 10px 30px rgba(0,0,0,.05); transition:transform .45s cubic-bezier(.2,.8,.2,1), box-shadow .45s cubic-bezier(.2,.8,.2,1);
     opacity:0; transform:translateY(24px); animation:in .9s cubic-bezier(.2,.8,.2,1) forwards; }
.k:hover { transform:translateY(-5px); box-shadow:0 2px 4px rgba(0,0,0,.05),0 24px 54px rgba(0,0,0,.10); }
.k:nth-child(1){animation-delay:.05s} .k:nth-child(2){animation-delay:.15s} .k:nth-child(3){animation-delay:.25s} .k:nth-child(4){animation-delay:.35s}
@keyframes in { to { opacity:1; transform:none; } }
.v { font-size:40px; font-weight:700; letter-spacing:-.04em; font-variant-numeric:tabular-nums; line-height:1.05;
     background:linear-gradient(180deg,#1d1d1f 55%,#6e6e73); -webkit-background-clip:text; background-clip:text; color:transparent; }
.l { font-size:14.5px; font-weight:600; margin-top:10px; }
.n { font-size:12.5px; color:#86868b; margin-top:6px; line-height:1.45; }
@media (prefers-reduced-motion:reduce){ .k{animation:none;opacity:1;transform:none} }
</style></head><body><div class="row">
<div class="k"><div class="v num" data-from="0" data-to="19" data-d="0" data-pre="−" data-suf="%">−19%</div><div class="l">__L1__</div><div class="n">__N1__</div></div>
<div class="k"><div class="v num" data-from="0" data-to="62" data-d="0" data-suf="%">62%</div><div class="l">__L2__</div><div class="n">__N2__</div></div>
<div class="k"><div class="v num" data-from="2.12" data-to="1" data-d="2">__ONE__</div><div class="l">__L3__</div><div class="n">__N3__</div></div>
<div class="k"><div class="v num" data-from="100" data-to="40" data-d="0" data-suf="%">40%</div><div class="l">__L4__</div><div class="n">__N4__</div></div>
</div><script>
(function(){
  var DEC = "__DEC__";
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var els = document.querySelectorAll('.num');
  function fmt(v,d){ return v.toFixed(d).replace('.',DEC); }
  els.forEach(function(el,i){
    var from=parseFloat(el.dataset.from), to=parseFloat(el.dataset.to), d=parseInt(el.dataset.d||'0',10);
    var pre=el.dataset.pre||'', suf=el.dataset.suf||'';
    if(reduce){ el.textContent = pre+fmt(to,d)+suf; return; }
    el.textContent = pre+fmt(from,d)+suf;
    var dur=1600, t0=null;
    function ease(t){ return 1-Math.pow(1-t,4); }
    function tick(ts){
      if(t0===null) t0=ts;
      var p=Math.min(1,(ts-t0)/dur);
      el.textContent = pre+fmt(from+(to-from)*ease(p),d)+suf;
      if(p<1) requestAnimationFrame(tick); else el.textContent = pre+fmt(to,d)+suf;
    }
    setTimeout(function(){ requestAnimationFrame(tick); }, 350+i*130);
  });
})();
</script></body></html>
"""


def kpi_html():
    e = html.escape
    out = KPI_TEMPLATE
    for i in range(1, 5):
        out = out.replace(f"__L{i}__", e(t(f"kpi{i}_l"))).replace(f"__N{i}__", e(t(f"kpi{i}_n")))
    return out.replace("__DEC__", "," if LANG == "ro" else ".").replace("__ONE__", "1,00" if LANG == "ro" else "1.00")


# ------------------------------------------------------------------ animatia „contorul net” (problema intr-o imagine)
METER_TX = {
    "eye": ("Problema într-o singură imagine", "The problem in one picture"),
    "hint": ("Mișcă soarele și consumul, apoi bifează cele două situații.", "Move the sun and the consumption, then tick the two situations."),
    "sky": ("panouri (GG)", "panels (GG)"), "house": ("casa (consum GC)", "home (consumption GC)"),
    "meter": ("contor net", "net meter"), "grid": ("rețea", "grid"), "pool": ("pompă piscină", "pool pump"),
    "gg": ("Producția panourilor", "Panel production"), "gc": ("Consumul casei", "Home consumption"),
    "hidden": ("ascuns", "hidden"),
    "seen": ("Furnizorul vede doar: net = consum − producție", "The utility only sees: net = consumption − production"),
    "gap": ("Față de o zi normală cu același soare", "Versus a normal day with the same sun"),
    "sun": ("Soare", "Sun"), "cons": ("Consumul de bază", "Base consumption"),
    "broken": ("Defect: producție la jumătate (panouri, cabluri, invertor)", "Fault: half production (panels, wiring, inverter)"),
    "poolc": ("Piscină nouă (+0,8 kWh/oră)", "New pool (+0.8 kWh/hour)"),
    "unit": ("kWh/oră", "kWh/hour"),
    "exp": ("casa trimite curent în rețea (export)", "the home sends power to the grid (export)"),
    "imp": ("casa ia curent din rețea (import)", "the home draws power from the grid (import)"),
    "bal": ("echilibru", "balanced"),
    "t_both": ("Ambele deodată. Contorul vede doar suma: netul e cu {gap} kWh/oră peste o zi normală. Din numărul ăsta singur nu ai cum să le desparți.",
               "Both at once. The meter only sees the sum: net is {gap} kWh/hour above a normal day. From this number alone you cannot tell them apart."),
    "t_broken": ("Sistemul solar pierde {loss} kWh în ora asta, iar netul crește cu {gap}. Coboară soarele: diferența scade odată cu el. La 0% soare, defectul nu se mai vede deloc, oricare ar fi componenta.",
                 "The solar system loses {loss} kWh this hour and net rises by {gap}. Lower the sun: the gap shrinks with it. At 0% sun, the fault is invisible, whatever the component."),
    "t_pool": ("Piscina adaugă 0,80 kWh în fiecare oră, indiferent de soare. Pentru contor arată ca un sistem solar defect: netul crește cu {gap}. Coboară soarele: diferența rămâne la fel.",
               "The pool adds 0.80 kWh every hour, whatever the sun. To the meter it looks like a faulty solar system: net rises by {gap}. Lower the sun: the gap stays the same."),
    "t_norm_exp": ("Casă normală. Produce mai mult decât consumă, deci netul e negativ: trimite curent în rețea. Bifează una dintre situații.",
                   "Normal home. It produces more than it uses, so net is negative: it sends power to the grid. Tick one of the situations."),
    "t_norm_imp": ("Casă normală. Consumă mai mult decât produce, deci ia curent din rețea. Bifează una dintre situații.",
                   "Normal home. It uses more than it produces, so it draws power from the grid. Tick one of the situations."),
}

METER_TEMPLATE = """
<!doctype html><html><head><meta charset="utf-8"><style>
:root { --ink:#1d1d1f; --mut:#86868b; --ink2:#3a3a3c; --line:rgba(0,0,0,.08); --tint:#f5f5f7; --sky:#eef4fc;
  --blue:#0a66d6; --sun:#ff9f0a; --sun-soft:#fff4e0; --imp:#d70015; --exp:#1a9e5c; --ease:cubic-bezier(.2,.8,.2,1); }
* { box-sizing:border-box; }
html,body { margin:0; background:transparent; color:var(--ink); -webkit-font-smoothing:antialiased;
  font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","Segoe UI",Inter,Roboto,Arial,sans-serif; }
.card { background:#fff; border-radius:22px; padding:22px 24px 20px; border:1px solid rgba(0,0,0,.04); margin:4px;
  box-shadow:0 1px 2px rgba(0,0,0,.04),0 10px 30px rgba(0,0,0,.05); display:grid; gap:16px;
  opacity:0; transform:translateY(24px); animation:in .9s var(--ease) .1s forwards; }
@keyframes in { to { opacity:1; transform:none; } }
.hd { display:flex; justify-content:space-between; align-items:baseline; gap:12px; flex-wrap:wrap; }
.hd b { font-size:19px; letter-spacing:-.02em; }
.hd span { font-size:13px; color:var(--mut); }
.grid { display:grid; grid-template-columns:minmax(0,1.5fr) minmax(0,1fr); gap:22px; align-items:center; }
@media (max-width:760px) { .grid { grid-template-columns:1fr; } }
svg { width:100%; height:auto; display:block; }
.sky { fill:var(--sky); } .ground { fill:var(--tint); }
.house { fill:#fff; stroke:var(--ink2); stroke-width:1.6; }
.roof { fill:var(--tint); stroke:var(--ink2); stroke-width:1.6; stroke-linejoin:round; }
.panel { fill:var(--blue); stroke:#fff; stroke-width:1.2; transition:fill .3s, opacity .3s; }
.panel.bad { fill:var(--mut); opacity:.55; }
.sunc { fill:var(--sun); } .ray { stroke:var(--sun); stroke-width:2.4; stroke-linecap:round; }
#mSun { transition:opacity .3s; }
#mSun .rays { transform-origin:0 0; animation:spin 24s linear infinite; }
@keyframes spin { to { transform:rotate(360deg); } }
.poolr { fill:#e3edfa; stroke:var(--blue); stroke-width:1.4; }
.box { fill:#fff; stroke:var(--ink2); stroke-width:1.6; } .lcd { fill:var(--ink); }
.lcdt { fill:#fbfbfd; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; font-weight:600; }
.pylon { fill:none; stroke:var(--mut); stroke-width:2; stroke-linejoin:round; }
.wire { stroke:rgba(0,0,0,.12); stroke-width:2; }
.lbl { fill:var(--mut); font-size:11px; }
.flow { fill:none; stroke-linecap:round; stroke-dasharray:4 8; animation:flow .9s linear infinite; transition:opacity .3s; }
.flow.prod { stroke:var(--sun); } .flow.imp { stroke:var(--imp); } .flow.exp { stroke:var(--exp); }
.flow.rev { animation-direction:reverse; }
@keyframes flow { to { stroke-dashoffset:-12; } }
#mPool { transition:opacity .35s; }
.side { display:grid; gap:10px; font-size:14px; }
.row { display:flex; justify-content:space-between; gap:10px; border-bottom:1px solid var(--line); padding-bottom:8px; }
.row b, .big { font-variant-numeric:tabular-nums; }
.hid { font-size:10.5px; color:var(--mut); border:1px dashed rgba(0,0,0,.18); border-radius:5px; padding:0 5px; margin-left:6px; }
.seen { background:var(--tint); border-radius:16px; padding:12px 14px; }
.seen .s { font-size:12.5px; color:var(--mut); }
.big { font-size:32px; font-weight:700; letter-spacing:-.04em; line-height:1.15; }
.dir { font-size:13px; color:var(--ink2); }
.ctrls { display:flex; flex-wrap:wrap; gap:12px 26px; align-items:center; }
.ctl { display:grid; gap:4px; font-size:13px; color:var(--ink2); min-width:180px; }
.ctl .v { font-weight:600; color:var(--ink); font-variant-numeric:tabular-nums; }
input[type=range] { width:100%; accent-color:var(--sun); }
.tog { display:inline-flex; gap:8px; align-items:center; font-size:13.5px; color:var(--ink2); cursor:pointer; user-select:none; }
.tog input { accent-color:var(--sun); width:16px; height:16px; }
.insight { font-size:14px; line-height:1.55; background:var(--sun-soft); border-radius:14px; padding:12px 14px; min-height:3.4em; }
@media (prefers-reduced-motion:reduce) { .card { animation:none; opacity:1; transform:none; } .flow, #mSun .rays { animation:none; } }
</style></head><body><div class="card">
<div class="hd"><b data-t="eye"></b><span data-t="hint"></span></div>
<div class="grid">
<svg viewBox="0 0 600 300" role="img">
  <rect x="0" y="0" width="600" height="268" class="sky" rx="14"/>
  <rect x="0" y="262" width="600" height="38" class="ground"/>
  <g id="mSun" transform="translate(74,64)">
    <g class="rays"><line class="ray" x1="0" y1="-38" x2="0" y2="-31"/><line class="ray" x1="0" y1="31" x2="0" y2="38"/><line class="ray" x1="-38" y1="0" x2="-31" y2="0"/><line class="ray" x1="31" y1="0" x2="38" y2="0"/><line class="ray" x1="-27" y1="-27" x2="-22" y2="-22"/><line class="ray" x1="22" y1="22" x2="27" y2="27"/><line class="ray" x1="-27" y1="27" x2="-22" y2="22"/><line class="ray" x1="22" y1="-22" x2="27" y2="-27"/></g>
    <circle r="22" class="sunc"/>
  </g>
  <path id="mBeam" d="M100 84 L214 120" class="flow prod" stroke-width="2"/>
  <polygon points="196,152 300,84 404,152" class="roof"/>
  <g id="mPanels"><polygon class="panel" points="214,140 238,124 262,124 238,140"/><polygon class="panel" points="240,140 264,124 288,124 264,140"/><polygon class="panel" points="232,128 252,114 276,114 256,128"/></g>
  <rect x="210" y="152" width="180" height="110" class="house"/>
  <rect x="286" y="210" width="28" height="52" class="roof"/>
  <rect x="228" y="172" width="34" height="26" class="roof"/>
  <rect x="338" y="172" width="34" height="26" class="roof"/>
  <path id="fProd" d="M250 134 L250 166" class="flow prod" stroke-width="4"/>
  <g id="mPool" opacity="0"><rect x="96" y="232" width="92" height="26" rx="8" class="poolr"/><text x="142" y="250" text-anchor="middle" class="lbl" data-t="pool"></text><path d="M188 245 L210 245" class="wire"/></g>
  <path d="M390 205 L560 205" class="wire"/>
  <path id="fNet" d="M392 205 L438 205 M502 205 L556 205" class="flow imp" stroke-width="4"/>
  <rect x="438" y="176" width="64" height="58" rx="7" class="box"/>
  <rect x="445" y="185" width="50" height="22" rx="3" class="lcd"/>
  <text id="mLcd" x="470" y="200.5" text-anchor="middle" class="lcdt">+0</text>
  <text x="470" y="225" text-anchor="middle" class="lbl" data-t="meter"></text>
  <path d="M560 262 L572 140 L584 262 M562 236 L582 236 M566 198 L578 198 M552 156 L592 156" class="pylon"/>
  <text x="572" y="284" text-anchor="middle" class="lbl" data-t="grid"></text>
  <text x="300" y="284" text-anchor="middle" class="lbl" data-t="house"></text>
  <text x="250" y="108" text-anchor="middle" class="lbl" data-t="sky"></text>
</svg>
<div class="side">
  <div class="row"><span><span data-t="gg"></span><span class="hid" data-t="hidden"></span></span><b id="mGG"></b></div>
  <div class="row"><span><span data-t="gc"></span><span class="hid" data-t="hidden"></span></span><b id="mGC"></b></div>
  <div class="seen"><div class="s" data-t="seen"></div><div class="big" id="mNet"></div><div class="dir" id="mDir"></div></div>
  <div class="row"><span data-t="gap"></span><b id="mGap"></b></div>
</div>
</div>
<div class="ctrls">
  <label class="ctl"><span><span data-t="sun"></span> <span class="v" id="vSun"></span></span><input type="range" id="sSun" min="0" max="100" value="80"></label>
  <label class="ctl"><span><span data-t="cons"></span> <span class="v" id="vCons"></span></span><input type="range" id="sCons" min="20" max="250" value="60"></label>
  <label class="tog"><input type="checkbox" id="cBroken"> <span data-t="broken"></span></label>
  <label class="tog"><input type="checkbox" id="cPool"> <span data-t="poolc"></span></label>
</div>
<div class="insight" id="mInsight" aria-live="polite"></div>
</div><script>
(function(){
  var T = __TX__, DEC = "__DEC__", CAP = 2.0;
  function $(s){ return document.querySelector(s); }
  document.querySelectorAll("[data-t]").forEach(function(el){ el.textContent = T[el.dataset.t]; });
  function f(v,d){ return Math.abs(v).toFixed(d).replace(".", DEC); }
  function fs(v,d){ var r = Number(v.toFixed(d)); return (r > 0 ? "+" : r < 0 ? "\\u2212" : "") + f(v,d); }
  function sub(s,o){ return s.replace(/\\{(\\w+)\\}/g, function(_,k){ return o[k]; }); }
  var sSun=$("#sSun"), sCons=$("#sCons"), cB=$("#cBroken"), cP=$("#cPool");
  function upd(){
    var s = sSun.value/100, base = sCons.value/100;
    var ggN = CAP*0.8*s, gg = ggN*(cB.checked?0.5:1), gc = base+(cP.checked?0.8:0);
    var net = gc-gg, netN = base-ggN, gap = net-netN;
    $("#vSun").textContent = Math.round(s*100)+"%";
    $("#vCons").textContent = f(base,2)+" "+T.unit;
    $("#mGG").textContent = f(gg,2)+" "+T.unit;
    $("#mGC").textContent = f(gc,2)+" "+T.unit;
    $("#mNet").textContent = fs(net,2)+" kWh";
    $("#mDir").textContent = net < -0.005 ? T.exp : (net > 0.005 ? T.imp : T.bal);
    $("#mLcd").textContent = fs(net,2);
    $("#mGap").textContent = fs(gap,2)+" "+T.unit;
    $("#mSun").style.opacity = String(0.2+0.8*s);
    var beam=$("#mBeam"); beam.style.opacity = s>0.02?"1":"0"; beam.setAttribute("stroke-width", String(1+3*s));
    document.querySelectorAll("#mPanels .panel").forEach(function(p,i){ p.classList.toggle("bad", cB.checked && i!==2); });
    $("#mPool").setAttribute("opacity", cP.checked?"1":"0");
    var fp=$("#fProd"); fp.style.opacity = gg>0.01?"1":"0"; fp.setAttribute("stroke-width", String(1.5+5*Math.min(1,gg/1.6)));
    var fn=$("#fNet");
    fn.setAttribute("class", "flow "+(net>=0?"imp rev":"exp"));
    fn.style.opacity = Math.abs(net)>0.01?"1":"0";
    fn.setAttribute("stroke-width", String(1.5+5*Math.min(1,Math.abs(net)/1.6)));
    var o = {gap: fs(gap,2), loss: f(ggN*0.5,2)}, t;
    if (cB.checked && cP.checked) t = sub(T.t_both,o);
    else if (cB.checked) t = sub(T.t_broken,o);
    else if (cP.checked) t = sub(T.t_pool,o);
    else t = net<0 ? T.t_norm_exp : T.t_norm_imp;
    $("#mInsight").textContent = t;
  }
  [sSun,sCons,cB,cP].forEach(function(x){ x.addEventListener("input", upd); x.addEventListener("change", upd); });
  upd();
})();
</script></body></html>
"""


def meter_html():
    tx = {k: v[LI] for k, v in METER_TX.items()}
    js = json.dumps(tx, ensure_ascii=False).replace("</", "<\\/")
    return METER_TEMPLATE.replace("__TX__", js).replace("__DEC__", "," if LANG == "ro" else ".")


if not (DEMO / "meta.json").exists():
    st.markdown(CSS, unsafe_allow_html=True)
    st.error(t("missing_demo"))
    st.stop()


# ------------------------------------------------------------------ utilitare de afisare
def esc(x):
    return html.escape(str(x), quote=True)


def md(s):
    st.markdown(s, unsafe_allow_html=True)


def section(eyebrow_key, title_key, sub_key=None, reveal=False):
    cls = ' class="reveal"' if reveal else ""
    md(f'<div{cls}><div class="eyebrow">{esc(t(eyebrow_key))}</div><div class="sec-t">{esc(t(title_key))}</div>'
       + (f'<div class="sec-s">{esc(t(sub_key))}</div>' if sub_key else "") + "</div>")


def kpi(value, label, note=""):
    return f'<div class="kpi"><div class="kpi-v">{esc(value)}</div><div class="kpi-l">{esc(label)}</div><div class="kpi-n">{esc(note)}</div></div>'


def card(title, body, tag="", delay=0.0, anim=True):
    tg = f'<div class="card-tag">{esc(tag)}</div>' if tag else ""
    cls = "card reveal" if anim else "card"
    return f'<div class="{cls}" style="--d:{delay}s"><div class="card-t">{esc(title)}</div><div class="card-p">{esc(body)}</div>{tg}</div>'


def action_label(a):
    return ACTION[a][LI]


def badge(a):
    return f'<span class="badge {ACTION[a][3]}">{esc(action_label(a))}</span>'


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.label.set_color(MUTED)
    ax.grid(axis="y", color="#efeff4", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor("white")


def new_fig(w=11, h=3.8):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("white")
    style_ax(ax)
    return fig, ax


def legend(ax, **kw):
    leg = ax.legend(frameon=False, fontsize=8, **kw)
    for tx in leg.get_texts():
        tx.set_color(INK)


def show_fig(container, fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight", pad_inches=0.3, facecolor="white")
    plt.close(fig)
    data = buf.getvalue()
    for kw in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            container.image(data, **kw)
            return
        except Exception:
            continue


def embed_html(doc, height):
    doc = doc.strip()
    if hasattr(st, "iframe"):
        try:
            st.iframe(doc, height=height)
            return
        except Exception:
            pass
    components.html(doc, height=height)


# ------------------------------------------------------------------ date
@st.cache_data(show_spinner=False)
def load():
    d = DEMO
    rd = lambda n: pd.read_parquet(d / n)
    return {"net": rd("net_day.parquet"), "exp": rd("expected_day.parquet"), "gg": rd("gg_day.parquet"),
            "lost": rd("lost_day.parquet"), "added": rd("added_day.parquet"),
            "labels": pd.read_csv(d / "labels.csv", dtype={"home": str}, parse_dates=["start"]),
            "dec": pd.read_csv(d / "decisions.csv", parse_dates=["date"], dtype={"home": str}),
            "b1": pd.read_csv(d / "b1_events.csv", parse_dates=["date"], dtype={"home": str}),
            "b2": pd.read_csv(d / "b2_events.csv", parse_dates=["date"], dtype={"home": str}),
            "cust": pd.read_csv(d / "customers.csv", dtype={"home": str}).set_index("home"),
            "meta": json.loads((d / "meta.json").read_text(encoding="utf-8"))}


ECON_PATH = DEMO / "economics.json"
ECON = (json.loads(ECON_PATH.read_text(encoding="utf-8")) if ECON_PATH.exists()
        else {"currency": "RON", "price_per_kwh": 1.30, "visit_cost": 100.0})

D = load()
meta, labels, dec = D["meta"], D["labels"], D["dec"]
cap = D["cust"]["capacity_kw"]
lab = labels.set_index("home")
real_homes, clean_homes = meta["real"], meta["clean"]


def home_label(h):
    kw = fmt(cap[h], 1)
    if h in lab.index:
        return t("home_k", h=h, k=kind(lab.loc[h, "type"]), kw=kw)
    if h in real_homes:
        return t("home_real", h=h, kw=kw)
    return t("home_plain", h=h, kw=kw)


def smooth(s, n=7):
    return s.rolling(n, min_periods=3).mean()


# ------------------------------------------------------------------ coada (aceeasi logica ca src/triage.py)
def policies():
    dd = dec.copy()
    dd["loss"] = dd["est_pierdere_kwh_zi"].fillna(0.0)
    dd["score"] = dd["loss"] * dd["solar_frac"].fillna(0.0).clip(0, 1)
    contact = dd[dd["action"] == "CONTACTEAZA"]
    b1 = D["b1"]
    return {"b1_fifo": (b1.assign(priority=0.0), "fifo"),
            "b1_cap": (b1.assign(priority=b1["home"].map(cap).fillna(0.0)), "priority"),
            "model_all": (dd.assign(priority=dd["loss"]), "priority"),
            "dimmer": (contact.assign(priority=contact["score"]), "priority")}


@st.cache_data(show_spinner=False)
def queue_result(pid, K):
    events, order = policies()[pid]
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


# ================================================================== PAGINI
def page_overview():
    md('<div class="hero"><div class="orb"></div>'
       f'<div class="pill">{esc(t("pill"))}</div>'
       f'<div class="hero-t">{esc(t("hero_a"))}<br><span class="grad">{esc(t("hero_b"))}</span></div>'
       f'<div class="hero-s">{esc(t("hero_sub"))}</div>'
       '<div class="fader"><i></i><b></b></div></div>')

    md('<div style="height:1.6rem"></div>')
    embed_html(meter_html(), 640)

    section("s1_eye", "s1_title", "s1_sub", reveal=True)
    embed_html(kpi_html(), 206)
    c = st.columns(3, gap="medium")
    for i, col in enumerate(c, 1):
        col.markdown(card(t(f"c{i}_t"), t(f"c{i}_p"), t(f"c{i}_tag"), (i - 1) * 0.1), unsafe_allow_html=True)

    section("s2_eye", "s2_title", "s2_sub", reveal=True)
    left, right = st.columns([6, 5], gap="large")
    with left:
        md(f'<div class="reveal"><div class="card-t">{esc(t("bars_t"))}</div>'
           f'<div class="card-p" style="margin-bottom:.8rem">{esc(t("bars_p"))}</div>'
           f'<div class="bar-row"><div class="bar-l">{esc(t("bar1"))}</div><div class="bar-t"><div class="bar-f" style="width:100%;--d:.1s"></div></div><div class="bar-v">{esc(t("bar1v"))}</div></div>'
           f'<div class="bar-row"><div class="bar-l">{esc(t("bar2"))}</div><div class="bar-t"><div class="bar-f" style="width:83%;--d:.25s"></div></div><div class="bar-v">{esc(t("bar2v"))}</div></div>'
           f'<div class="bar-row"><div class="bar-l">{esc(t("bar3"))}</div><div class="bar-t"><div class="bar-f" style="width:50%;--d:.4s"></div></div><div class="bar-v">{esc(t("bar3v"))}</div></div>'
           f'<div class="footnote">{esc(t("bars_foot"))}</div></div>')
    with right:
        md(f'<div class="dark-card reveal"><div class="card-t">{esc(t("dark_t"))}</div><div class="card-p">{t("dark_html")}</div></div>')
    md(f'<div class="reveal" style="margin-top:2.2rem"><div class="card-t">{esc(t("cost_t"))}</div>'
       f'<div class="card-p" style="margin-bottom:.8rem">{esc(t("cost_p"))}</div>'
       f'<div class="bar-row"><div class="bar-l"><span class="hl-num">{esc(t("cost_r1a"))}</span><br><span style="color:#86868b;font-size:.82rem">{esc(t("cost_r1b"))}</span></div><div class="bar-t"><div class="bar-f" style="width:43%;--d:.1s"></div></div><div class="bar-v">{esc(t("cost_r1v"))}</div></div>'
       f'<div class="bar-row"><div class="bar-l"><span class="hl-num">{esc(t("cost_r2a"))}</span><br><span style="color:#86868b;font-size:.82rem">{esc(t("cost_r2b"))}</span></div><div class="bar-t"><div class="bar-f alt" style="width:100%;--d:.25s"></div></div><div class="bar-v">{esc(t("cost_r2v"))}</div></div>'
       f'<div class="footnote">{esc(t("cost_foot"))}</div></div>')

    section("s3_eye", "s3_title", reveal=True)
    c = st.columns(4, gap="medium")
    for i, col in enumerate(c, 1):
        col.markdown(f'<div class="card reveal" style="--d:{(i - 1) * 0.1}s"><div class="step-num">{i}</div><div class="card-t">{esc(t(f"st{i}_t"))}</div><div class="card-p">{esc(t(f"st{i}_p"))}</div></div>', unsafe_allow_html=True)

    section("s4_eye", "s4_title", reveal=True)
    c = st.columns(3, gap="medium")
    for i, (col, a) in enumerate(zip(c, ["CONTACTEAZA", "MONITORIZEAZA", "VERIFICA_DATE"]), 1):
        col.markdown(f'<div class="card reveal" style="--d:{(i - 1) * 0.1}s">{badge(a)}<div class="card-t" style="margin-top:.9rem">{esc(t(f"act{i}_t"))}</div><div class="card-p">{esc(t(f"act{i}_p"))}</div></div>', unsafe_allow_html=True)

    section("s5_eye", "s5_title", reveal=True)
    for i0 in range(0, 12, 3):
        cols = st.columns(3, gap="medium")
        for j, col in enumerate(cols):
            i = i0 + j + 1
            col.markdown(card(t(f"it{i}_t"), t(f"it{i}_p"), t(f"it{i}_g"), j * 0.08), unsafe_allow_html=True)
            col.markdown('<div style="height:.9rem"></div>', unsafe_allow_html=True)

    section("s6_eye", "s6_title", reveal=True)
    md(f'<div class="note reveal">{esc(t("limits_note"))}</div>')
    md(f'<div class="footnote" style="margin-top:2rem">{esc(t("footer", n=len(D["net"].columns)))}</div>')


def page_cases():
    section("cs_eye", "cs_title", "cs_sub")
    faults = labels[labels["kind"] == "fault"].sort_values("mean_effect_kwh_per_day", ascending=False)["home"].tolist()
    loads = labels[labels["kind"] == "load"].sort_values("mean_effect_kwh_per_day", ascending=False)["home"].tolist()
    n_al = dec.groupby("home").size()
    clean_sorted = sorted(clean_homes, key=lambda x: -int(n_al.get(x, 0)))
    cats = {"fault": faults, "load": loads, "real": real_homes, "clean": clean_sorted}
    left, right = st.columns([1, 3], gap="large")
    cat = left.radio(t("w_type"), list(cats), format_func=lambda k: t("cat_" + k), key="cs_cat")
    options = cats[cat]
    if not options:
        st.info(t("no_cat"))
        return
    h = left.selectbox(t("w_home"), options, key="cs_home",
                       format_func=lambda x: home_label(x) + (t("al_suffix", n=int(n_al.get(x, 0))) if cat == "clean" else ""))
    truth = left.checkbox(t("w_truth"), key="cs_truth")
    ev_h = dec[dec["home"] == h].sort_values("date")

    net, exp = D["net"][h], D["exp"][h]
    fig, ax = new_fig(11, 3.9)
    ys, es = smooth(net), smooth(exp)
    ax.plot(ys.index, ys, color=INK, lw=1.6, label=t("lg_net"))
    ax.plot(es.index, es, color=MUTED, lw=1.3, ls="--", label=t("lg_exp"))
    ax.fill_between(ys.index, es, ys, where=(ys > es), color=GOLD, alpha=0.22, label=t("lg_gap"))
    for a, g in ev_h.groupby("action"):
        ax.scatter(g["date"], ys.reindex(g["date"]).values, color=ACTION[a][2], s=58, zorder=5,
                   label=action_label(a), edgecolor="white", linewidth=1.3)
    if truth and h in lab.index and lab.loc[h, "kind"] == "fault":
        s0 = lab.loc[h, "start"]
        ax.axvline(s0, color=INK, ls=":", lw=1.2)
        ax.text(s0, ax.get_ylim()[1], t("lg_start"), va="top", fontsize=8, color=INK)
    ax.set_ylabel(t("y_net"))
    legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5)
    show_fig(right, fig)
    if truth:
        fig2, ax2 = new_fig(11, 3.0)
        g = D["gg"][h]
        if h in lab.index and lab.loc[h, "kind"] == "fault":
            after = g - D["lost"][h].reindex(g.index).fillna(0.0)
            ax2.plot(smooth(g).index, smooth(g), color=MUTED, ls="--", label=t("lg_p_nofault"))
            ax2.plot(smooth(after).index, smooth(after), color=INK, label=t("lg_p_fault"))
            r = lab.loc[h]
            msg = t("tr_fault", k=kind(r["type"]), d=f"{r['start']:%Y-%m-%d}", x=fmt(r["mean_effect_kwh_per_day"], 1))
        elif h in lab.index and lab.loc[h, "kind"] == "load":
            ad = D["added"][h]
            ax2.plot(smooth(g).index, smooth(g), color=INK, label=t("lg_p_same"))
            ax2.plot(smooth(ad).index, smooth(ad), color=GOLD, label=t("lg_added"))
            r = lab.loc[h]
            msg = t("tr_load", k=kind(r["type"]), d=f"{r['start']:%Y-%m-%d}")
        elif h in real_homes:
            ax2.plot(smooth(g).index, smooth(g), color=INK, label=t("lg_p_meas"))
            msg = t("tr_real")
        else:
            ax2.plot(smooth(g).index, smooth(g), color=INK, label=t("lg_p_real"))
            msg = t("tr_clean")
        ax2.set_ylabel(t("y_day"))
        legend(ax2, loc="upper left")
        show_fig(right, fig2)
        right.markdown(f'<div class="note">{esc(msg)}</div>', unsafe_allow_html=True)

    md(f'<div class="eyebrow" style="margin-top:2.6rem">{esc(t("al_head", h=h, n=len(ev_h)))}</div>')
    if len(ev_h):
        rows = "".join(
            f'<div class="alert-row"><div class="alert-d">{r.date:%Y-%m-%d}</div><div class="alert-b">{badge(r.action)}</div>'
            f'<div class="alert-t">{esc(explain_alert(r._asdict(), LANG))}</div></div>'
            for r in ev_h.tail(6).itertuples(index=False))
        md(rows)
        if len(ev_h) > 6:
            md(f'<div class="footnote">{esc(t("al_last", n=len(ev_h)))}</div>')
    else:
        md(f'<div class="note">{esc(t("al_none"))}</div>')
    contact_h = ev_h[ev_h["action"] == "CONTACTEAZA"]
    if len(contact_h):
        md(f'<div class="eyebrow" style="margin-top:2.6rem">{esc(t("msg_eye"))}</div>')
        with st.expander(t("msg_exp"), expanded=True):
            st.code(customer_message(contact_h.iloc[-1], ECON, lang=LANG), language=None)
            st.caption(t("msg_cap"))
    else:
        md(f'<div class="footnote">{esc(t("msg_none"))}</div>')


def page_queue():
    section("q_eye", "q_title", "q_sub")
    c1, c2 = st.columns([2, 3], gap="large")
    unlimited = c1.checkbox(t("q_unl"))
    K = None if unlimited else c1.slider(t("q_slider"), 1, 10, 3)
    pids = list(POLICIES)
    chosen = c2.multiselect(t("q_pol"), pids, default=[pids[0], pids[1], pids[3]], format_func=lambda p: POLICIES[p][LI])
    if not chosen:
        md(f'<div class="note">{esc(t("q_pick"))}</div>')
        return
    rows, cums = [], {}
    for p in chosen:
        m, visits, cum = queue_result(p, K)
        cums[p] = cum
        rows.append({t("col_pol"): POLICIES[p][LI], t("col_rep"): m["defecte_reparate%"],
                     t("col_kwhm"): m["kWh_pierduti_pe_defect"], t("col_kwhmed"): m["kWh_mediana_pe_defect"],
                     t("col_vt"): m["vizite_total"], t("col_vw"): m["irosite"], t("col_ff"): m["gasite"],
                     t("col_prec"): m["precizie%"], t("col_days"): m["zile_mediane_pana_la_reparatie"]})
    st.dataframe(pd.DataFrame(rows).set_index(t("col_pol")))
    md('<div style="height:1rem"></div>')
    fig, ax = new_fig(11, 3.8)
    none = D["lost"].cumsum().sum(axis=1)
    ax.plot(none.index, none, color=LINE, ls="--", lw=1.6, label=t("q_none"))
    palette = [MUTED, "#aeaeb2", "#c9a24a"]
    pi = 0
    for p, c in cums.items():
        if p == "dimmer":
            ax.plot(c.index, c, lw=2.8, color=GOLD, label=POLICIES[p][LI])
        else:
            ax.plot(c.index, c, lw=1.5, color=palette[pi % len(palette)], label=POLICIES[p][LI])
            pi += 1
    ax.set_ylabel(t("q_ylab"))
    ax.set_title(t("q_ct", n=D["lost"].shape[1]) + ("" if K is None else t("q_ctk", k=K)), fontsize=10, color=INK, loc="left")
    legend(ax, loc="upper left")
    show_fig(st, fig)

    section("cost_eye", "cost_title")
    n_f = D["lost"].shape[1]
    k1, k2 = st.columns(2, gap="large")
    price = k1.number_input(t("in_price"), min_value=0.0, max_value=10.0, value=float(ECON["price_per_kwh"]), step=0.05)
    visit = k2.number_input(t("in_visit"), min_value=0.0, max_value=5000.0, value=float(ECON["visit_cost"]), step=10.0)
    cost = pd.DataFrame([{t("col_pol"): r[t("col_pol")],
                          t("col_en"): round(r[t("col_kwhm")] * n_f * price),
                          t("col_vis"): round(r[t("col_vt")] * visit)} for r in rows]).set_index(t("col_pol"))
    cost[t("col_tot")] = cost[t("col_en")] + cost[t("col_vis")]
    st.dataframe(cost)
    md(f'<div class="footnote">{esc(t("cost_note"))}</div>')

    with st.expander(t("log_exp")):
        _, vis, _ = queue_result("dimmer", K)
        if len(vis):
            tb = vis.assign(**{t("log_date"): vis["date"].dt.strftime("%Y-%m-%d"), t("log_home"): vis["home"],
                               t("log_out"): vis["outcome"].map(lambda o: OUTCOMES[o][LI]), t("log_wait"): vis["wait_days"]})
            st.dataframe(tb[[t("log_date"), t("log_home"), t("log_out"), t("log_wait")]].set_index(t("log_date")))
    md(f'<div class="footnote">{esc(t("q_foot"))}</div>')


def page_real():
    section("r_eye", "r_title", "r_sub")
    if not real_homes:
        md(f'<div class="note">{esc(t("r_none"))}</div>')
        return
    b1, b2 = D["b1"], D["b2"]
    first = lambda df, x: df.loc[df["home"] == x, "date"].min()
    rows = []
    for x in real_homes:
        e = dec[dec["home"] == x]
        rows.append({t("r_home"): x, "kW": cap[x], "B1": first(b1, x), "B2": first(b2, x),
                     t("r_dfirst"): first(dec, x),
                     t("r_dcontact"): e.loc[e["action"] == "CONTACTEAZA", "date"].min() if len(e) else pd.NaT})
    tb = pd.DataFrame(rows).set_index(t("r_home"))
    n_real = len(real_homes)
    c = st.columns(3, gap="medium")
    c[0].markdown(kpi(t("r_of", a=int(tb["B1"].notna().sum()), b=n_real), t("r_k1l"), t("r_k1n")), unsafe_allow_html=True)
    c[1].markdown(kpi(t("r_of", a=int(tb["B2"].notna().sum()), b=n_real), t("r_k2l"), t("r_k2n")), unsafe_allow_html=True)
    c[2].markdown(kpi(t("r_of", a=int(tb[t("r_dfirst")].notna().sum()), b=n_real), t("r_k3l"), t("r_k3n")), unsafe_allow_html=True)
    md('<div style="height:1.2rem"></div>')
    st.dataframe(tb.apply(lambda col: col.dt.strftime("%Y-%m-%d").fillna("—") if col.dtype.kind == "M" else col))
    x = st.selectbox(t("w_home"), real_homes, format_func=home_label, key="real_home")
    g = D["gg"][x]
    fig, ax = new_fig(11, 3.6)
    ax.plot(smooth(g).index, smooth(g), color=INK, lw=1.8, label=t("r_lg"))
    for lbl, df, colr in (("B1", b1, "#0a60d6"), ("B2", b2, "#2e9e4f"), ("Dimmer", dec, GOLD)):
        d0 = first(df, x)
        if pd.notna(d0):
            ax.axvline(d0, color=colr, ls="--", lw=1.5, label=t("r_first", m=lbl, d=f"{d0:%Y-%m-%d}"))
    ax.set_ylabel(t("y_day"))
    legend(ax, loc="upper right")
    show_fig(st, fig)
    md(f'<div class="footnote">{esc(t("r_foot"))}</div>')


def page_results():
    section("rs_eye", "rs_title", "rs_sub")
    section("rs_m_eye", "rs_m_title")
    idx = [t("lab_found"), t("lab_fa"), t("lab_kwh")]
    main = pd.DataFrame({
        "B1": [L("42,5%", "42.5%"), L("1,40", "1.40"), "295"],
        "B2": [L("55%", "55%"), L("2,00", "2.00"), "257"],
        t("col_dim"): [L("62%", "62%"), L("1,00", "1.00"), "223"],
        t("col_dimn"): [L("77,5%", "77.5%"), L("2,12", "2.12"), "162"]}, index=idx)
    st.dataframe(main)
    md(f'<div class="footnote">{esc(t("rs_m_foot"))}</div>')

    section("rs_p_eye", "rs_p_title")
    pred = pd.DataFrame({t("p_col"): {t("p_r1"): L("0,5030 / 0,5124", "0.5030 / 0.5124"),
                                       t("p_r2"): L("0,4104 / 0,4162", "0.4104 / 0.4162"),
                                       t("p_r3"): L("−18,4% / −18,8%", "−18.4% / −18.8%")}}).rename_axis(t("p_axis"))
    st.dataframe(pred)
    mt = DEMO / "model_test_raw.csv"
    if mt.exists():
        md(f'<div class="footnote">{esc(t("p_abl"))}</div>')
        m = pd.read_csv(mt, index_col=0)
        keep = [c for c in ["MAE_kWh_ora", "FA_la_1000", "kWh_pierduti_medie"] if c in m.columns]
        st.dataframe(m[keep].rename(columns={"MAE_kWh_ora": "MAE", "FA_la_1000": t("abl_fa"), "kWh_pierduti_medie": t("abl_kwh")}))

    section("rs_d_eye", "rs_d_title")
    bt = DEMO / "baseline_test.csv"
    if bt.exists() and mt.exists():
        b = pd.read_csv(bt)
        m = pd.read_csv(mt, index_col=0)
        rows = []
        for typ in ["total", "partial", "intermittent", "gradual"]:
            r = {t("d_type"): kind(typ)}
            for meth in ("B1", "B2"):
                x = b[(b["type"] == typ) & (b["metoda"] == meth)]
                if len(x):
                    r[f"{meth}: {t('d_found')}"] = x["prinse"].iloc[0]
                    r[f"{meth}: {t('d_days')}"] = x["zile_mediane_pana_la_detectie"].iloc[0]
            if "full" in m.index and f"{typ}_prinse%" in m.columns:
                r[f"{t('d_model')}: {t('d_found')}"] = m.loc["full", f"{typ}_prinse%"]
                r[f"{t('d_model')}: {t('d_days')}"] = m.loc["full", f"{typ}_zile"]
            rows.append(r)
        st.dataframe(pd.DataFrame(rows).set_index(t("d_type")))
    md(f'<div class="footnote">{esc(t("rs_d_foot"))}</div>')

    section("rs_x_eye", "rs_x_title")
    dec_tb = pd.DataFrame({
        t("x_c1"): [L("2,12", "2.12"), "78%", "162", "100%", "80%", "60%"],
        t("x_c2"): [L("1,00", "1.00"), "62%", "223", "40%", "20%", "60%"]},
        index=[t("x_r1"), t("x_r2"), t("x_r3"), t("x_r4"), t("x_r5"), t("x_r6")])
    st.dataframe(dec_tb)
    md(f'<div class="footnote">{esc(t("rs_x_foot"))}</div>')

    section("rs_t_eye", "rs_t_title")
    rows = []
    for K in (1, 2, 3, 5, None):
        for p in ("b1_cap", "dimmer"):
            m, _, _ = queue_result(p, K)
            rows.append({t("t_k"): t("t_nolimit") if K is None else str(K), t("t_pol"): POLICIES[p][LI],
                         t("t_rep"): m["defecte_reparate%"], t("t_kwh"): m["kWh_pierduti_pe_defect"], t("t_w"): m["irosite"]})
    st.dataframe(pd.DataFrame(rows).set_index([t("t_k"), t("t_pol")]))

    page_real()
    page_cases()

    section("rs_l_eye", "rs_l_title")
    md(f'<div class="note">{t("rs_l_note")}</div>')


# ================================================================== DEMO LIVE (src/demo_replay.py)
DEMO_LIVE = ("warming", "running")
ROLES = ["normal", "temporary", "persistent"]
STATE_CLS = {"st_wait": "s-wait", "st_monitor": "s-monitor", "st_al_monitor": "s-almon", "st_al_data": "s-aldata", "st_invest": "s-invest"}


@st.cache_resource(show_spinner=False)
def demo_core():
    """Datele si modelul, incarcate o singura data (pentru tot serverul)."""
    return DR.ReplayCore()


def demo_runner(core, scen):
    r = st.session_state.get("demo_runner")
    if r is None or r.core is not core or r.scenario != scen:
        if r is not None:
            r.reset()
        r = DR.ReplayRunner(core, scen, ECON)
        st.session_state["demo_runner"] = r
    return r


def cur_label():
    c = ECON.get("currency", "RON")
    return "lei" if (c == "RON" and LANG == "ro") else c


def src_tag(h):
    """Proveniența datelor casei (pentru prezentare; pipeline-ul nu o vede)."""
    if h in lab.index:
        r = lab.loc[h]
        k = "src_sim_fault" if r["kind"] == "fault" else "src_sim_load"
        return t(k, k=kind(r["type"]), d=f"{pd.Timestamp(r['start']):%Y-%m-%d}"), "tag-sim"
    if h in real_homes:
        return t("src_real_decline"), "tag-real"
    return t("src_real"), "tag-real"


def tag_html(h):
    txt, cls = src_tag(h)
    return f'<span class="tag {cls}">{esc(txt)}</span>'


def pct(x):
    return fmt(100 * x, 0) if pd.notna(x) else "—"


def decision_reason(r, theta):
    """Motivul deciziei, cu aceleasi reguli ca decide.assign."""
    miss, note, sf = r.get("missing", np.nan), r.get("note", ""), r.get("solar_frac", np.nan)
    if pd.notna(miss) and miss > DR.dc.MISSING_MAX:
        return t("rule_data_miss", p=pct(miss))
    if note == "few_days":
        return t("rule_data_few", n=int(r["n_days"]))
    if note == "flat_sun":
        return t("rule_flat")
    if note == "small_excess" or pd.isna(sf):
        return t("rule_small")
    key = "rule_contact" if sf >= theta else "rule_monitor"
    return t(key, s=fmt(min(max(sf, 0.0), 1.0), 2), t=fmt(theta, 1))


def ev_class(e):
    k = e["kind"]
    if k == "alarm":
        return "k-alarm"
    if k == "check":
        return {"CONTACTEAZA": "k-contact", "MONITORIZEAZA": "k-monitor"}.get(e["row"]["action"], "k-data")
    if k in ("case_created", "case_updated", "case_evidence"):
        return "k-case"
    if k in ("notification", "notif_dedup"):
        return "k-notif"
    if k == "error":
        return "k-err"
    return ""


def ev_text(e):
    k, h = e["kind"], e.get("home")
    d = lambda x: f"{pd.Timestamp(x):%Y-%m-%d}"
    if k == "warmup":
        return t("ev_warmup", a=d(e["first"]), b=d(e["sim_date"]), n=e["n_days"], al=e["alarms"], s=fmt(e["seconds"], 1))
    if k in ("started", "paused", "resumed"):
        return t("ev_" + k)
    if k == "finished":
        return t("ev_finished", n=e["n_days"])
    if k == "error":
        return t("ev_error", m=e["message"])
    if k == "missing":
        return t("ev_missing", h=h, p=pct(e["frac"]))
    if k == "missing_fleet":
        return t("ev_missing_fleet", p=pct(e["frac"]))
    if k == "alarm":
        return t("ev_alarm", h=h, s=fmt(e["cusum"], 1), hh=fmt(e["h"], 0))
    if k == "check":
        r = e["row"]
        return t("ev_check", h=h, a=action_label(r["action"]), x=explain_alert(r, LANG))
    if k == "no_case":
        return t("ev_no_case", h=h, a=action_label(e["action"]))
    if k == "case_created":
        return t("ev_case_created", c=e["case_id"], h=h, p=fmt(e["priority"], 2))
    if k == "case_updated":
        return t("ev_case_updated", c=e["case_id"], h=h, p=fmt(e["priority"], 2))
    if k == "case_evidence":
        return t("ev_case_evidence", c=e["case_id"], h=h, a=action_label(e["action"]))
    if k == "notification":
        return t("ev_notification", c=e["case_id"])
    if k == "notif_dedup":
        return t("ev_notif_dedup", c=e["case_id"], h=h)
    return k


def home_state(snap, h):
    """Starea curenta, derivata doar din ce a produs pipeline-ul pana la data simulata."""
    if not snap["in_window"]:
        return "st_wait", t("st_n_wait")
    case = snap["cases"].get(h)
    if case is not None:
        return "st_invest", t("st_n_case", c=case["id"], d=f"{case['opened']:%Y-%m-%d}")
    evs = [r for r in snap["decisions"] if r["home"] == h and r["i"] >= snap["start_i"]]
    if evs and snap["i_last"] - evs[-1]["i"] < DR.COOLDOWN_DAYS:
        r = evs[-1]
        key = "st_al_data" if r["action"] == "VERIFICA_DATE" else "st_al_monitor"
        return key, t("st_n_after", d=f"{r['date']:%Y-%m-%d}", a=action_label(r["action"]))
    return "st_monitor", t("st_n_monitor")


def demo_snapshot(runner):
    with runner.lock:
        S = runner.session
        j0, j1 = S.start_i, max(S.start_i, S.i_next)
        return {"status": runner.status, "error": runner.error, "i_next": S.i_next, "i_last": S.i_next - 1,
                "start_i": S.start_i, "end_i": S.end_i, "in_window": S.i_next > S.start_i,
                "feed": list(S.feed), "decisions": list(S.decisions),
                "cases": {h: dict(c, events=list(c["events"])) for h, c in S.cases.items()},
                "notifications": list(S.notifications),
                "obs": S.obs[j0:j1].copy(), "exp": S.exp[j0:j1].copy(), "s": S.sh[max(0, S.i_next - 1)].copy(),
                "z": S.zh[max(0, S.i_next - 1)].copy(), "step_ms": S.step_ms[-1] if S.step_ms else None,
                "elapsed": runner.elapsed(), "overruns": runner.overruns, "speed": runner.sec_per_day}


def demo_chart(core, snap, h, scen):
    j = core.cols.index(h)
    days = core.days[snap["start_i"]:snap["start_i"] + len(snap["obs"])]
    lo, hi = core.days[snap["start_i"]], core.days[snap["end_i"]]
    obs, exp = snap["obs"][:, j], snap["exp"][:, j]
    n_obs, n_exp = t("dm_ch_obs"), t("dm_ch_exp")
    long = pd.concat([pd.DataFrame({"date": days, "v": obs, "s": n_obs}), pd.DataFrame({"date": days, "v": exp, "s": n_exp})])
    x = alt.X("date:T", scale=alt.Scale(domain=[lo.isoformat(), hi.isoformat()]), title=None,
              axis=alt.Axis(format="%Y-%m-%d", labelColor=MUTED, tickColor=LINE, domainColor=LINE, grid=False))
    vals = np.concatenate([obs[np.isfinite(obs)], exp[np.isfinite(exp)]]) if len(days) else np.array([])
    ydom = [float(np.min(vals)) - 0.1, float(np.max(vals)) + 0.1] if len(vals) else [-1.0, 1.0]
    y = alt.Y("v:Q", title=t("dm_ch_y"), scale=alt.Scale(domain=ydom),
              axis=alt.Axis(labelColor=MUTED, titleColor=MUTED, gridColor="#efeff4", domainColor=LINE, tickColor=LINE))
    gap = pd.DataFrame({"date": days, "lo": exp, "hi": np.where(obs > exp, obs, exp)})
    area = alt.Chart(gap).mark_area(color=GOLD, opacity=0.22).encode(x=x, y=alt.Y("lo:Q", scale=alt.Scale(domain=ydom)), y2="hi:Q")
    lines = alt.Chart(long).mark_line(strokeWidth=1.7).encode(
        x=x, y=y,
        color=alt.Color("s:N", scale=alt.Scale(domain=[n_obs, n_exp], range=[INK, MUTED]),
                        legend=alt.Legend(title=None, orient="top", labelColor=INK, symbolStrokeWidth=2)),
        strokeDash=alt.StrokeDash("s:N", scale=alt.Scale(domain=[n_obs, n_exp], range=[[1, 0], [5, 4]]), legend=None))
    layers = [area, lines]
    evs = [r for r in snap["decisions"] if r["home"] == h and r["i"] >= snap["start_i"]]
    if evs:
        pts = pd.DataFrame({"date": [r["date"] for r in evs],
                            "v": [obs[r["i"] - snap["start_i"]] for r in evs],
                            "a": [action_label(r["action"]) for r in evs]})
        dom = [action_label(a) for a in ACTION]
        layers.append(alt.Chart(pts).mark_point(filled=True, size=120, stroke="white", strokeWidth=1.5, opacity=1).encode(
            x=x, y=alt.Y("v:Q", scale=alt.Scale(domain=ydom)),
            color=alt.Color("a:N", scale=alt.Scale(domain=dom, range=[ACTION[a][2] for a in ACTION]),
                            legend=alt.Legend(title=None, orient="bottom", labelColor=INK, symbolType="circle")),
            tooltip=[alt.Tooltip("date:T", format="%Y-%m-%d"), "a:N"]))
    ch = alt.layer(*layers).resolve_scale(color="independent", strokeDash="independent").properties(height=300, background="white", padding={"left": 8, "right": 14, "top": 14, "bottom": 6}) \
        .configure_view(strokeWidth=0)
    st.altair_chart(ch, width="stretch")


def demo_truth(core, snap, h):
    """Adevarul ascuns (productia reala) si comparatia cu B1/B2, doar pana la data simulata."""
    n = len(snap["obs"])
    if not n:
        md(f'<div class="note">{esc(t("st_n_wait"))}</div>')
        return
    i0 = snap["start_i"]
    days = core.days[i0:i0 + n]
    sim = days[-1]
    g = D["gg"][h].reindex(days) if h in D["gg"].columns else pd.Series(np.nan, index=days)
    start = None
    if h in lab.index and lab.loc[h, "kind"] == "fault":
        r = lab.loc[h]
        after = g - D["lost"][h].reindex(days).fillna(0.0)
        series = [(t("lg_p_nofault"), g, MUTED, [5, 4]), (t("lg_p_fault"), after, INK, [1, 0])]
        msg = t("tr_fault", k=kind(r["type"]), d=f"{pd.Timestamp(r['start']):%Y-%m-%d}", x=fmt(r["mean_effect_kwh_per_day"], 1))
        start = pd.Timestamp(r["start"]).normalize()
    elif h in lab.index and lab.loc[h, "kind"] == "load":
        r = lab.loc[h]
        series = [(t("lg_p_same"), g, INK, [1, 0]), (t("lg_added"), D["added"][h].reindex(days).fillna(0.0), GOLD, [1, 0])]
        msg = t("tr_load", k=kind(r["type"]), d=f"{pd.Timestamp(r['start']):%Y-%m-%d}")
        start = pd.Timestamp(r["start"]).normalize()
    elif h in real_homes:
        series = [(t("lg_p_meas"), g, INK, [1, 0])]
        msg = t("tr_real")
    else:
        series = [(t("lg_p_real"), g, INK, [1, 0])]
        msg = t("tr_clean")
    names = [s_[0] for s_ in series]
    long = pd.concat([pd.DataFrame({"date": days, "v": s_[1].to_numpy(float), "s": s_[0]}) for s_ in series])
    lo, hi = core.days[i0], core.days[snap["end_i"]]
    x = alt.X("date:T", scale=alt.Scale(domain=[lo.isoformat(), hi.isoformat()]), title=None,
              axis=alt.Axis(format="%Y-%m-%d", labelColor=MUTED, tickColor=LINE, domainColor=LINE, grid=False))
    layers = [alt.Chart(long).mark_line(strokeWidth=1.6).encode(
        x=x, y=alt.Y("v:Q", title=t("y_day"), axis=alt.Axis(labelColor=MUTED, titleColor=MUTED, gridColor="#efeff4", domainColor=LINE)),
        color=alt.Color("s:N", scale=alt.Scale(domain=names, range=[s_[2] for s_ in series]),
                        legend=alt.Legend(title=None, orient="top", labelColor=INK, labelLimit=400)),
        strokeDash=alt.StrokeDash("s:N", scale=alt.Scale(domain=names, range=[s_[3] for s_ in series]), legend=None))]
    if start is not None and lo <= start <= sim:
        layers.append(alt.Chart(pd.DataFrame({"date": [start]})).mark_rule(color=INK, strokeDash=[2, 3]).encode(x="date:T"))
    st.altair_chart(alt.layer(*layers).properties(height=200, background="white",
                                                  padding={"left": 8, "right": 14, "top": 10, "bottom": 6}).configure_view(strokeWidth=0),
                    width="stretch")
    md(f'<div class="note" style="margin-top:.4rem">{esc(msg)}</div><div class="footnote">{esc(t("dm_truth_note"))}</div>')
    upto = lambda df: [d_ for d_ in df.loc[(df["home"] == h) & (df["date"] <= sim), "date"].sort_values()]
    dim = [r_ for r_ in snap["decisions"] if r_["home"] == h]
    fmt_d = lambda ds: ", ".join(f"{d_:%Y-%m-%d}" for d_ in ds) if ds else t("dm_none")
    rows = [(t("dm_base_b1"), fmt_d(upto(D["b1"]))), (t("dm_base_b2"), fmt_d(upto(D["b2"]))),
            (t("dm_base_dim"), ", ".join(f"{r_['date']:%Y-%m-%d} ({action_label(r_['action'])})" for r_ in dim) or t("dm_none"))]
    md(f'<div class="dm-lbl" style="margin-top:1rem">{esc(t("dm_base"))}</div><table class="ev-tbl">'
       + "".join(f"<tr><td>{esc(a)}</td><td>{esc(b)}</td></tr>" for a, b in rows)
       + f'</table><div class="footnote">{esc(t("dm_base_note"))}</div>')


def fleet_grid(core, snap, focus):
    """Toate instalatiile, colorate dupa starea calculata de pipeline pana la data simulata."""
    section("dm_eye", "dm_fleet")
    md(f'<div class="sec-s" style="margin-top:-.4rem">{esc(t("dm_fleet_sub", n=len(core.cols)))}</div>')
    last = {}
    for r in snap["decisions"]:
        if r["i"] >= snap["start_i"]:
            last[r["home"]] = r
    counts = {k_: 0 for k_ in STATE_CLS}
    building = 0
    cells = []
    for j, h in enumerate(core.cols):
        sv = float(snap["s"][j]) if snap["i_last"] >= 0 and np.isfinite(snap["s"][j]) else 0.0
        frac = min(1.0, sv / core.h)
        if not snap["in_window"]:
            key = "st_wait"
        elif h in snap["cases"]:
            key = "st_invest"
        elif h in last and snap["i_last"] - last[h]["i"] < DR.COOLDOWN_DAYS:
            key = "st_al_data" if last[h]["action"] == "VERIFICA_DATE" else "st_al_monitor"
        else:
            key = "st_monitor"
        counts[key] += 1
        style = ""
        if key == "st_monitor":
            if sv > 0:
                building += 1
                style = f' style="background:rgba(255,159,10,{0.12 + 0.83 * frac:.2f})"'
        title = t("fl_title", h=h, s=t(key), c=fmt(sv, 1), t=fmt(core.h, 0), src=src_tag(h)[0])
        cls = f"fl-c fl-{STATE_CLS[key]}" + (" fl-w" if h in focus else "")
        cells.append(f'<span class="{cls}"{style} title="{esc(title)}"></span>')
    leg = []
    for key in ("st_monitor", "st_al_monitor", "st_al_data", "st_invest", "st_wait"):
        if not counts[key] and key in ("st_wait", "st_al_data"):
            continue
        lbl = t("fl_monitor", n=counts[key], m=building) if key == "st_monitor" else t("fl_count", l=t(key), n=counts[key])
        leg.append(f'<span class="fl-leg"><span class="fl-c fl-{STATE_CLS[key]}"></span>{esc(lbl)}</span>')
    md(f'<div class="fl-grid">{"".join(cells)}</div><div class="fl-legend">{"".join(leg)}</div>')


def case_report(case, core):
    """Raport text (Markdown) din dovezile calculate efectiv."""
    last = [r for r in case["events"] if r["action"] == "CONTACTEAZA"][-1]
    cur, price = cur_label(), float(ECON.get("price_per_kwh") or 0)
    lines = [f"# {t('rep_title')}: {case['id']}", "", f"_{t('rep_banner')}_", "",
             f"- {t('c_home')}: {case['home']} ({fmt(core.cap[case['home']], 2)} kW)",
             f"- {t('c_open')}: {case['opened']:%Y-%m-%d}",
             f"- {t('c_src')}: {src_tag(case['home'])[0]}",
             f"- {t('c_prio')}: {fmt(case['priority'], 2)}", "",
             f"## {t('c_expl')}", "", explain_alert(last, LANG), "", f"## {t('c_evid')}", ""]
    for r in case["events"]:
        lines += [f"### {r['date']:%Y-%m-%d} · {action_label(r['action'])}",
                  f"- {t('e_cusum')}: {fmt(r['cusum'], 1)} / {fmt(core.h, 0)}",
                  f"- {t('e_days')}: {r['n_days']}",
                  f"- {t('e_miss')}: {pct(r['missing'])}%",
                  f"- {t('e_exc')}: " + (t("e_exc_v", x=fmt(r["excess_kwh_h"], 2), y=fmt(r["est_pierdere_kwh_zi"], 1)) if pd.notna(r["excess_kwh_h"]) else "—"),
                  f"- {t('e_sun')}: " + (fmt(min(max(r["solar_frac"], 0), 1), 2) if pd.notna(r["solar_frac"]) else "—"),
                  f"- {t('e_rule')}: {decision_reason(r, core.theta)}", ""]
    lines += [f"## {t('c_money')}", ""]
    if case["loss_days"]:
        lines += [t("c_acc", k=fmt(case["loss_kwh"], 1), v=fmt(case["loss_kwh"] * price, 0), cur=cur), "",
                  t("c_acc_n", a=f"{case['period'][0]:%Y-%m-%d}", b=f"{case['period'][1]:%Y-%m-%d}", n=case["loss_days"], p=fmt(price, 2), cur=cur), ""]
        if pd.notna(case["loss_per_day"]):
            lines += [t("c_proj", k=fmt(case["loss_per_day"], 1), v=fmt(case["loss_per_day"] * 30 * price, 0), cur=cur), ""]
    else:
        lines += [t("c_noloss"), ""]
    lines += [f"_{t('c_disc')}_", ""]
    return "\n".join(lines)


def demo_live(runner, core, scen, was_live):
    runner.last_seen = time.time()
    snap = demo_snapshot(runner)
    live = snap["status"] in DEMO_LIVE
    if live != was_live:
        st.rerun()
    # --- controale
    c = st.columns([1, 1.25, 1, 2.2], gap="small", vertical_alignment="bottom")
    if c[0].button(t("dm_start"), type="primary", disabled=snap["status"] != "idle", width="stretch"):
        runner.start()
        st.rerun()
    if snap["status"] == "paused":
        if c[1].button(t("dm_resume"), width="stretch"):
            runner.resume()
            st.rerun()
    elif c[1].button(t("dm_pause"), disabled=snap["status"] != "running", width="stretch"):
        runner.pause()
        st.rerun()
    if c[2].button(t("dm_reset"), disabled=snap["status"] == "idle" and snap["i_next"] == 0, width="stretch"):
        runner.reset()
        st.rerun()
    sp = c[3].select_slider(t("dm_speed"), options=DR.SPEEDS, value=runner.sec_per_day if runner.sec_per_day in DR.SPEEDS else 1.0,
                            format_func=lambda s: t("dm_speed_f", s=fmt(s, 2 if s < 0.5 else 1)), key="dm_speed")
    if sp != runner.sec_per_day:
        runner.set_speed(sp)
    if snap["status"] == "error":
        st.error(t("dm_err", m=snap["error"]))

    # --- progres: timp simulat separat de timpul real
    n_win = snap["end_i"] - snap["start_i"] + 1
    done = max(0, min(n_win, snap["i_next"] - snap["start_i"]))
    sim_d = core.days[snap["i_last"]] if snap["i_last"] >= 0 else None
    k = st.columns([1.3, 1, 1, 1, 1], gap="medium")
    mins, secs = divmod(int(snap["elapsed"]), 60)
    dot = "on" if live else ""
    k[0].markdown(f'<div class="dm-lbl"><span class="dm-dot {dot}" style="display:inline-block;margin-right:.4rem"></span>{esc(t("dm_simdate"))}</div>'
                  f'<div class="dm-big">{esc(f"{sim_d:%Y-%m-%d}" if sim_d is not None else "—")}</div>'
                  f'<div class="dm-track"><div class="dm-fill" style="width:{100 * done / n_win:.1f}%"></div></div>'
                  f'<div class="dm-sm">{esc(t("dm_prog", a=done, b=n_win))} · {esc(t("dm_simnote"))} · {esc(t("dm_st_" + snap["status"]))}</div>',
                  unsafe_allow_html=True)
    k[1].markdown(f'<div class="dm-lbl">{esc(t("dm_real"))}</div><div class="dm-big">{mins:02d}:{secs:02d}</div>'
                  f'<div class="dm-sm">{esc(t("dm_realnote"))}<br>{esc(t("dm_stepms", ms=fmt(snap["step_ms"], 0)) if snap["step_ms"] else "")}'
                  f'{esc(t("dm_overrun", n=snap["overruns"]) if snap["overruns"] else "")}</div>', unsafe_allow_html=True)
    win_dec = [r for r in snap["decisions"] if r["i"] >= snap["start_i"]]
    for col, key, v in ((k[2], "dm_k_al", len(win_dec)), (k[3], "dm_k_cases", len(snap["cases"])), (k[4], "dm_k_notif", len(snap["notifications"]))):
        col.markdown(f'<div class="dm-lbl">{esc(t(key))}</div><div class="dm-big">{v}</div>'
                     f'<div class="dm-sm">{esc(t("dm_k_fleet", n=len(core.cols)))}</div>', unsafe_allow_html=True)
    md('<div style="height:1.4rem"></div>')

    # --- instalatia urmarita
    homes = {r: scen["homes"][r] for r in ROLES if r in scen["homes"]}
    opts = list(homes.values()) + sorted([h for h in snap["cases"] if h not in homes.values()],
                                         key=lambda x: -snap["cases"][x]["priority"])
    role_of = {h: r for r, h in homes.items()}

    def opt_label(h):
        if h in role_of:
            return t("dm_home_opt", h=h, r=t("role_" + role_of[h]))
        return t("dm_home_case", h=h, c=snap["cases"][h]["id"])
    left, right = st.columns([2.1, 1], gap="large")
    with left:
        a1, a2 = st.columns([1.2, 1], vertical_alignment="bottom")
        default = homes.get("persistent", opts[0])
        if st.session_state.get("dm_home") not in opts:
            st.session_state["dm_home"] = default
        h = a1.selectbox(t("dm_track"), opts, format_func=opt_label, key="dm_home")
        a2.markdown(f'<div style="padding-bottom:.55rem">{tag_html(h)} <span class="dm-sm">{esc(fmt(core.cap[h], 2))} kW</span></div>',
                    unsafe_allow_html=True)
        key, note = home_state(snap, h)
        j = core.cols.index(h)
        sv = float(snap["s"][j]) if snap["i_last"] >= 0 else 0.0
        zv = snap["z"][j] if snap["i_last"] >= 0 else np.nan
        frac = min(1.0, sv / core.h)
        z_txt = t("dm_ztoday", z=fmt(zv, 2)) if pd.notna(zv) else t("dm_ztoday", z="—")
        b1, b2 = st.columns([1, 1.25], gap="large")
        b1.markdown(f'<div class="dm-lbl">{esc(t("dm_state"))}</div><span class="dm-state {STATE_CLS[key]}">{esc(t(key))}</span>'
                    f'<div class="dm-sm">{esc(note)}</div>', unsafe_allow_html=True)
        b2.markdown(f'<div class="dm-lbl">{esc(t("dm_cusum"))}</div>'
                    f'<div class="dm-track" style="height:10px"><div class="dm-fill" style="width:{100 * frac:.1f}%"></div></div>'
                    f'<div class="dm-sm">{esc(t("dm_cusum_n", s=fmt(sv, 1), h=fmt(core.h, 0), k=fmt(core.k, 1)))}<br>{esc(z_txt)}</div>',
                    unsafe_allow_html=True)
        demo_chart(core, snap, h, scen)
        md(f'<div class="footnote" style="margin-top:0">{esc(t("dm_ch_note"))}</div>')
        if st.toggle(t("w_truth"), key="dm_truth"):
            demo_truth(core, snap, h)
    with right:
        f1, f2 = st.columns([1, 1.3], vertical_alignment="center")
        f1.markdown(f'<div class="dm-lbl" style="margin:0">{esc(t("dm_feed"))}</div>', unsafe_allow_html=True)
        show_all = f2.toggle(t("dm_feed_all"), value=False, key="dm_feed_all")
        watch = set(homes.values()) | {h}
        feed = [e for e in snap["feed"] if show_all or e.get("home") is None or e.get("home") in watch]
        last_seen = st.session_state.get("dm_feed_last", 0)
        if feed:
            rows = []
            for e in reversed(feed[-150:]):
                when = f"{e['sim_date']:%Y-%m-%d}" if e["sim_date"] is not None else ""
                wall = time.strftime("%H:%M:%S", time.localtime(e["wall"]))
                new = " ev-new" if e["id"] > last_seen else ""
                rows.append(f'<div class="ev{new}"><div class="ev-dot {ev_class(e)}"></div>'
                            f'<div class="ev-d">{esc(when)}<br>{esc(wall)}</div><div class="ev-t">{esc(ev_text(e))}</div></div>')
            md(f'<div class="feed">{"".join(rows)}</div>')
            st.session_state["dm_feed_last"] = feed[-1]["id"]
        else:
            md(f'<div class="note">{esc(t("dm_feed_none"))}</div>')

    # --- reteaua
    fleet_grid(core, snap, set(homes.values()) | {h})

    # --- cazuri
    section("dm_eye", "dm_cases", "dm_cases_sub")
    cases = sorted(snap["cases"].values(), key=lambda c_: -c_["priority"])
    if not cases:
        md(f'<div class="note">{esc(t("dm_cases_none"))}</div>')
    else:
        price = float(ECON.get("price_per_kwh") or 0)
        tb = pd.DataFrame([{t("c_id"): c_["id"], t("c_home"): c_["home"], t("c_open"): f"{c_['opened']:%Y-%m-%d}",
                            t("c_src"): src_tag(c_["home"])[0], t("c_prio"): round(c_["priority"], 2),
                            t("c_loss"): round(c_["loss_kwh"], 1), t("c_cost", cur=cur_label()): round(c_["loss_kwh"] * price),
                            t("c_n"): len(c_["events"])} for c_ in cases]).set_index(t("c_id"))
        st.dataframe(tb, width="stretch", height=min(38 * len(tb) + 40, 330))
        ids = [c_["id"] for c_ in cases]
        pref = snap["cases"].get(h, {}).get("id")
        if pref and st.session_state.get("dm_case_follow") != pref:
            st.session_state["dm_case"] = pref
            st.session_state["dm_case_follow"] = pref
        if st.session_state.get("dm_case") not in ids:
            st.session_state["dm_case"] = ids[0]
        cid = st.selectbox(t("c_pick"), ids, key="dm_case",
                           format_func=lambda x: next(f"{x} · {t('c_home').lower()} {c_['home']}" for c_ in cases if c_["id"] == x))
        case = next(c_ for c_ in cases if c_["id"] == cid)
        last = [r for r in case["events"] if r["action"] == "CONTACTEAZA"][-1]
        p1, p2 = st.columns([1.2, 1], gap="large")
        with p1:
            md(f'<div class="card" style="height:auto"><div class="dm-lbl">{esc(t("c_expl"))}</div>{tag_html(case["home"])}'
               f'<div class="card-p" style="margin-top:.6rem;color:#2c2c2e">{esc(explain_alert(last, LANG))}</div>'
               f'<div class="footnote">{esc(t("c_disc"))}</div></div>')
            md('<div style="height:.8rem"></div>')
            cur = cur_label()
            if case["loss_days"]:
                pa, pb = case["period"][0].strftime("%Y-%m-%d"), case["period"][1].strftime("%Y-%m-%d")
                money = (f'<div class="card-t">{esc(t("c_acc", k=fmt(case["loss_kwh"], 1), v=fmt(case["loss_kwh"] * price, 0), cur=cur))}</div>'
                         f'<div class="card-p">{esc(t("c_acc_n", a=pa, b=pb, n=case["loss_days"], p=fmt(price, 2), cur=cur))}</div>')
                if pd.notna(case["loss_per_day"]):
                    money += f'<div class="footnote">{esc(t("c_proj", k=fmt(case["loss_per_day"], 1), v=fmt(case["loss_per_day"] * 30 * price, 0), cur=cur))}</div>'
            else:
                money = f'<div class="card-p">{esc(t("c_noloss"))}</div>'
            md(f'<div class="card" style="height:auto"><div class="dm-lbl">{esc(t("c_money"))}</div>{money}</div>')
        with p2:
            r = case["events"][-1]
            rows = [(t("e_date"), f"{r['date']:%Y-%m-%d} · {action_label(r['action'])}"),
                    (t("e_cusum"), f"{fmt(r['cusum'], 1)} / {fmt(core.h, 0)}"),
                    (t("e_days"), str(r["n_days"])), (t("e_miss"), f"{pct(r['missing'])}%"),
                    (t("e_exc"), t("e_exc_v", x=fmt(r["excess_kwh_h"], 2), y=fmt(r["est_pierdere_kwh_zi"], 1)) if pd.notna(r["excess_kwh_h"]) else "—"),
                    (t("e_sun"), fmt(min(max(r["solar_frac"], 0), 1), 2) if pd.notna(r["solar_frac"]) else "—"),
                    (t("e_rule"), decision_reason(r, core.theta))]
            tl = " · ".join(f"{x['date']:%Y-%m-%d} {action_label(x['action'])}" for x in case["events"])
            md(f'<div class="card" style="height:auto"><div class="dm-lbl">{esc(t("c_evid"))}</div><table class="ev-tbl">'
               + "".join(f"<tr><td>{esc(a)}</td><td>{esc(b)}</td></tr>" for a, b in rows)
               + f'</table><div class="footnote">{esc(t("c_timeline"))}: {esc(tl)}</div></div>')
            md('<div style="height:.6rem"></div>')
            st.download_button(t("c_report"), case_report(case, core), file_name=f"{case['id']}_{LANG}.md", mime="text/markdown",
                               on_click="ignore", key=f"dm_rep_{case['id']}")

    # --- inbox
    section("dm_eye", "dm_inbox", "dm_inbox_sub")
    all_notes = snap["notifications"]
    inbox_all = st.toggle(t("dm_inbox_all"), value=False, key="dm_inbox_all")
    focus = set(homes.values()) | {h}
    notes = all_notes if inbox_all else [n for n in all_notes if n["home"] in focus]
    if not inbox_all:
        md(f'<div class="footnote" style="margin:0 0 .8rem 0">{esc(t("dm_inbox_scope"))}</div>')
    if not notes:
        md(f'<div class="note">{esc(t("dm_inbox_hidden", n=len(all_notes)) if all_notes else t("dm_inbox_none"))}</div>')
    else:
        show = list(reversed(notes))[:6]
        for i, n in enumerate(show):
            with st.expander(t("dm_notif_h", n=n["id"], c=n["case_id"], h=n["home"], d=f"{n['sim_date']:%Y-%m-%d}"), expanded=(i == 0)):
                md(f'<span class="tag tag-sim" style="border-style:solid">{esc(t("dm_unsent"))}</span> {tag_html(n["home"])}')
                st.code(customer_message(n["row"], ECON, lang=LANG), language=None, wrap_lines=True)
        if len(notes) > len(show):
            md(f'<div class="footnote">{esc(t("dm_inbox_more", a=len(show), b=len(notes)))}</div>')


def page_demo():
    section("dm_eye", "dm_title", "dm_sub")
    try:
        core = demo_core()
    except FileNotFoundError as e:
        st.warning(t("dm_missing", m=str(e)))
        return
    scen = DR.load_scenario()
    if scen is None:
        st.warning(t("dm_noscen"))
        return
    runner = demo_runner(core, scen)
    live = runner.status in DEMO_LIVE
    md(f'<div class="dm-banner"><span class="dm-dot {"on" if live else ""}"></span>{esc(t("dm_banner"))}</div>')
    d0 = lambda s: f"{pd.Timestamp(s):%Y-%m-%d}"
    first = core.days[0]
    pre_end = pd.Timestamp(scen["start"]) - pd.Timedelta(days=1)
    roles = " · ".join(f"<b>{esc(t('role_' + r))}</b>: {esc(t('home_plain', h=scen['homes'][r], kw=fmt(core.cap[scen['homes'][r]], 2)))} {tag_html(scen['homes'][r])}"
                       for r in ROLES if r in scen["homes"])
    md(f'<div class="note" style="margin-bottom:1.4rem"><b>{esc(t("dm_scen"))}.</b> '
       f'{esc(t("dm_window", a=d0(scen["start"]), b=d0(scen["end"]), n=scen["days"], c=f"{first:%Y-%m-%d}", d=f"{pre_end:%Y-%m-%d}"))}<br>'
       f'{roles}<br><span style="color:#86868b;font-size:.82rem">{esc(t("dm_scen_p"))} {esc(t("src_hidden"))}</span></div>')
    st.fragment(run_every=0.5 if live else None)(demo_live)(runner, core, scen, live)


# ================================================================== schelet
md(CSS)
bar_l, bar_r = st.columns([5, 1])
with bar_l:
    md(f'<div class="topbar"><div class="logo"></div><span class="brand-n">Dimmer</span><span class="brand-t">{esc(t("tagline"))}</span></div>')
with bar_r:
    st.radio("Language", ["ro", "en"], horizontal=True, key="lang", format_func=lambda x: x.upper(), label_visibility="collapsed")
PAGES = {"overview": page_overview, "demo": page_demo, "queue": page_queue, "results": page_results}
if st.session_state.get("page") not in PAGE_KEYS:
    st.session_state["page"] = "overview"
try:
    nav_box = st.container(key="nav")
except TypeError:
    nav_box = st.container()
with nav_box:
    page = st.radio("Navigation", PAGE_KEYS, horizontal=True, label_visibility="collapsed", key="page", format_func=lambda k: t("nav_" + k))
md('<div class="divider"></div>')
PAGES[page]()