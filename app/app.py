"""
Dimmer - aplicatia demo (Streamlit).

Citeste doar din demo_data/test/ (creat de src/make_demo_data.py), deci ruleaza si dupa `git clone`.
    streamlit run app/app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from triage import run_queue, metrics  # noqa: E402
from messages import customer_message  # noqa: E402

DEMO = ROOT / "demo_data" / "test"
st.set_page_config(page_title="Dimmer", page_icon="☀️", layout="wide")

ACTION = {"CONTACTEAZA": ("#d62728", "🔴 CONTACTEAZĂ"),
          "MONITORIZEAZA": ("#f59e0b", "🟠 MONITORIZEAZĂ"),
          "VERIFICA_DATE": ("#6b7280", "⚪ VERIFICĂ DATELE")}
TYPE_RO = {"total": "defect total", "partial": "defect parțial", "intermittent": "defect intermitent",
           "gradual": "defect treptat", "ac": "aer condiționat nou", "pool": "pompă de piscină nouă",
           "ev": "mașină electrică nouă", "wfh": "muncă de acasă"}
ACTION_PLAIN = {"CONTACTEAZA": "CONTACTEAZĂ", "MONITORIZEAZA": "MONITORIZEAZĂ", "VERIFICA_DATE": "VERIFICĂ DATELE"}
OUTCOME_RO = {"tp": "✅ defect găsit și reparat", "real": "🔎 scădere reală (cauză necunoscută)",
              "fp_clean": "❌ vizită irosită (casă fără problemă)", "fp_load": "❌ vizită irosită (doar consum nou)",
              "fp_pre": "❌ vizită irosită (defectul nu începuse)"}

if not (DEMO / "meta.json").exists():
    st.error("Lipsește folderul demo_data/test. Rulați: `python src/make_demo_data.py --period test`")
    st.stop()


# ------------------------------------------------------------------ date
@st.cache_data(show_spinner=False)
def load():
    d = DEMO
    rd = lambda n: pd.read_parquet(d / n)
    D = {"net": rd("net_day.parquet"), "exp": rd("expected_day.parquet"), "gg": rd("gg_day.parquet"),
         "lost": rd("lost_day.parquet"), "added": rd("added_day.parquet"),
         "labels": pd.read_csv(d / "labels.csv", dtype={"home": str}, parse_dates=["start"]),
         "dec": pd.read_csv(d / "decisions.csv", parse_dates=["date"], dtype={"home": str}),
         "b1": pd.read_csv(d / "b1_events.csv", parse_dates=["date"], dtype={"home": str}),
         "b2": pd.read_csv(d / "b2_events.csv", parse_dates=["date"], dtype={"home": str}),
         "cust": pd.read_csv(d / "customers.csv", dtype={"home": str}).set_index("home"),
         "meta": json.loads((d / "meta.json").read_text(encoding="utf-8"))}
    return D


ECON_PATH = DEMO / "economics.json"
ECON = (json.loads(ECON_PATH.read_text(encoding="utf-8")) if ECON_PATH.exists()
        else {"currency": "RON", "price_per_kwh": 1.30, "visit_cost": 100.0,
              "price_source": "valoare implicită; completați config/economics.json", "visit_cost_note": "ipoteză"})

D = load()
meta, labels, dec = D["meta"], D["labels"], D["dec"]
cap = D["cust"]["capacity_kw"]
lab = labels.set_index("home")
real_homes, clean_homes = meta["real"], meta["clean"]


def home_label(h):
    if h in lab.index:
        r = lab.loc[h]
        return f"casa {h} · {TYPE_RO.get(r['type'], r['type'])} · {cap[h]:.1f} kW"
    if h in real_homes:
        return f"casa {h} · scădere reală · {cap[h]:.1f} kW"
    return f"casa {h} · {cap[h]:.1f} kW"


def smooth(s, n=7):
    return s.rolling(n, min_periods=3).mean()


# ------------------------------------------------------------------ coada (aceeasi logica ca src/triage.py)
def policies():
    dd = dec.copy()
    dd["loss"] = dd["est_pierdere_kwh_zi"].fillna(0.0)
    dd["score"] = dd["loss"] * dd["solar_frac"].fillna(0.0).clip(0, 1)
    contact = dd[dd["action"] == "CONTACTEAZA"]
    b1 = D["b1"]
    return {"B1 (regulă simplă), în ordinea sosirii": (b1.assign(priority=0.0), "fifo"),
            "B1, după capacitatea instalată": (b1.assign(priority=b1["home"].map(cap).fillna(0.0)), "priority"),
            "Model, toate alarmele, după pierderea estimată": (dd.assign(priority=dd["loss"]), "priority"),
            "Dimmer: model + decizie, după scor": (contact.assign(priority=contact["score"]), "priority")}


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


# ------------------------------------------------------------------ bara laterala
PAGES = ["1. Ce face Dimmer", "2. Un caz, pe larg", "3. Coada de verificări",
         "4. Episoade reale", "5. Rezultatele, cinstit"]
page = st.sidebar.radio("Secțiune", PAGES)
st.sidebar.caption(f"Date: Ausgrid (Sydney), {len(D['net'].columns)} case. Test: {meta['start']} → {meta['end']}, "
                   "rulat o singură dată, cu praguri înghețate. Defectele sunt **simulate** pe producție reală.")

# ================================================================== 1
if page == PAGES[0]:
    st.title("☀️ Dimmer")
    st.subheader("Găsim panourile solare care s-au stricat pe tăcute, fără să trimitem tehnicieni degeaba")
    st.markdown("""
Un prosumator are un contor care vede doar **diferența** dintre ce ia din rețea și ce trimite în rețea.
Dacă panourile se strică, nimeni nu observă imediat: furnizorul vede doar că omul consumă mai mult.
Dar un consum mai mare poate însemna și o piscină nouă, o mașină electrică sau munca de acasă.

**Dimmer** face trei lucruri:
1. **Ghicește** ce ar trebui să arate contorul fiecărei case (un model LightGBM, antrenat pe un an, care folosește radiația solară și casele vecine).
2. **Adună dovezi pe mai multe zile** (CUSUM) și dă alarmă doar când abaterea persistă.
3. **Decide ce se întâmplă cu alarma**: lipsa de producție solară *crește odată cu soarele*, un consum nou nu. În funcție de asta, alarma devine o acțiune.
""")
    c1, c2, c3 = st.columns(3)
    c1.success("🔴 **CONTACTEAZĂ**\n\nGolul urmărește soarele: probabil lipsește producția solară.")
    c2.warning("🟠 **MONITORIZEAZĂ**\n\nGolul nu urmărește soarele (probabil consum nou) sau nu putem spune încă. Nu trimitem pe nimeni.")
    c3.info("⚪ **VERIFICĂ DATELE**\n\nPrea puține zile sau prea multe citiri lipsă.")
    st.markdown("#### Testul final: aceeași comparație, la același nivel de vizite false")
    ca, cb, cc = st.columns(3)
    ca.markdown("**B1** · regulă simplă")
    ca.metric("Defecte simulate prinse", "42,5%")
    ca.metric("Alarme false la 1.000 zile-casă", "1,40")
    ca.metric("kWh pierduți pe defect", "295")
    cb.markdown("**B2** · față de anul trecut")
    cb.metric("Defecte simulate prinse", "55%")
    cb.metric("Alarme false la 1.000 zile-casă", "2,00")
    cb.metric("kWh pierduți pe defect", "257")
    cc.markdown("**Dimmer** · model + verificare soare/consum")
    cc.metric("Defecte simulate prinse", "62%")
    cc.metric("Alarme false la 1.000 zile-casă", "1,00")
    cc.metric("kWh pierduți pe defect", "223")
    st.caption("Pentru Dimmer, o alarmă falsă este o alarmă CONTACTEAZĂ pe o casă curată, adică o vizită irosită; pentru B1 și B2, orice alarmă duce la o vizită. "
               "Pe test, Dimmer prinde mai mult, cu mai puține vizite false și mai puțină energie pierdută. "
               "Pe validare, la același procent de defecte prinse ca B1 (80%), Dimmer a făcut jumătate din vizitele false (0,57 față de 1,11), cu ~11% mai mulți kWh pierduți. "
               "Fără verificarea soare/consum, modelul e mai sensibil (78% prinse), dar la 2,12 alarme false la 1.000.")
    d1, d2 = st.columns(2)
    d1.metric("Eroarea de predicție", "−19%", "față de istoricul casei", delta_color="off")
    d2.metric("Piscină luată drept defect", "40%", "față de 100% fără verificare", delta_color="off")
    st.caption("Cifre din testul final (secțiunea 5 are tabelele și limitele). Defectele sunt simulate, câte 10 pe tip.")
    st.warning("**Ce nu pretindem.** Defectele sunt simulate pe date reale; nu am confirmat defecte fizice. "
               "Nu detectăm mai repede defectele totale decât o regulă simplă (22,5 zile față de 9), "
               "iar aerul condiționat rămâne confundat cu un defect. Datele sunt din Australia; metoda se transferă, cifrele nu.")

# ================================================================== 2
elif page == PAGES[1]:
    st.header("Un caz, pe larg")
    faults = labels[labels["kind"] == "fault"].sort_values("mean_effect_kwh_per_day", ascending=False)["home"].tolist()
    loads = labels[labels["kind"] == "load"].sort_values("mean_effect_kwh_per_day", ascending=False)["home"].tolist()
    n_al = dec.groupby("home").size()
    clean_sorted = sorted(clean_homes, key=lambda h: -int(n_al.get(h, 0)))
    cats = {"Defect simulat": faults, "Consum nou simulat (panourile sunt bune)": loads,
            "Scădere reală de producție (probabil defect)": real_homes, "Casă curată": clean_sorted}
    left, right = st.columns([1, 3])
    cat = left.radio("Tipul casei", list(cats))
    options = cats[cat]
    if not options:
        st.info("Nu există case în această categorie în datele curente.")
        st.stop()
    h = left.selectbox("Casa", options, format_func=lambda x: home_label(x) + (f" · {int(n_al.get(x, 0))} alarme" if cat == "Casă curată" else ""))
    truth = left.checkbox("🔍 Arată adevărul (ascuns pentru sistem)")
    ev_h = dec[dec["home"] == h].sort_values("date")

    net, exp = D["net"][h], D["exp"][h]
    fig, ax = plt.subplots(figsize=(11, 3.9))
    ys, es = smooth(net), smooth(exp)
    ax.plot(ys.index, ys, color="#1f77b4", lw=1.4, label="net real, orele de zi (media pe 7 zile)")
    ax.plot(es.index, es, color="#6b7280", lw=1.3, ls="--", label="net așteptat de model")
    ax.fill_between(ys.index, es, ys, where=(ys > es), color="#d62728", alpha=0.16, label="mai mult net decât se aștepta")
    for a, g in ev_h.groupby("action"):
        ax.scatter(g["date"], ys.reindex(g["date"]).values, color=ACTION[a][0], s=60, zorder=5, label=ACTION_PLAIN[a], edgecolor="white")
    if truth and h in lab.index and lab.loc[h, "kind"] == "fault":
        s0 = lab.loc[h, "start"]
        ax.axvline(s0, color="black", ls=":", lw=1.3)
        ax.text(s0, ax.get_ylim()[1], " început defect simulat", va="top", fontsize=9)
    ax.set_ylabel("kWh pe oră (net)")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    right.pyplot(fig)
    plt.close(fig)
    if truth:
        fig2, ax2 = plt.subplots(figsize=(11, 3.0))
        g = D["gg"][h]
        if h in lab.index and lab.loc[h, "kind"] == "fault":
            after = g - D["lost"][h].reindex(g.index).fillna(0.0)
            ax2.plot(smooth(g).index, smooth(g), color="#6b7280", ls="--", label="producție reală, fără defect")
            ax2.plot(smooth(after).index, smooth(after), color="#1f77b4", label="producție după defectul simulat")
            r = lab.loc[h]
            msg = f"**Adevărul:** {TYPE_RO[r['type']]} simulat din {r['start']:%Y-%m-%d}; pierdere medie {r['mean_effect_kwh_per_day']:.1f} kWh/zi."
        elif h in lab.index and lab.loc[h, "kind"] == "load":
            ad = D["added"][h]
            ax2.plot(smooth(g).index, smooth(g), color="#1f77b4", label="producția (neschimbată: panourile funcționează)")
            ax2.plot(smooth(ad).index, smooth(ad), color="#f59e0b", label="consum nou adăugat (kWh/zi)")
            r = lab.loc[h]
            msg = f"**Adevărul:** panourile funcționează; am adăugat consum de tip *{TYPE_RO[r['type']]}* din {r['start']:%Y-%m-%d}."
        elif h in real_homes:
            ax2.plot(smooth(g).index, smooth(g), color="#1f77b4", label="producția reală măsurată (kWh/zi)")
            msg = "**Adevărul:** producția reală a scăzut în această perioadă. **Cauza nu o cunoaștem** (poate fi defect, umbră, panou scos)."
        else:
            ax2.plot(smooth(g).index, smooth(g), color="#1f77b4", label="producția reală (kWh/zi)")
            msg = "**Adevărul:** casă fără nicio modificare simulată; alarmele de mai sus sunt alarme false."
        ax2.set_ylabel("kWh pe zi")
        ax2.grid(alpha=0.25)
        ax2.legend(loc="upper left", fontsize=8)
        right.pyplot(fig2)
        plt.close(fig2)
        right.info(msg)
    st.markdown(f"##### Alarmele casei {h} ({len(ev_h)})")
    if len(ev_h):
        show = pd.DataFrame({"Data": ev_h["date"].dt.strftime("%Y-%m-%d"),
                             "Acțiune": ev_h["action"].map(lambda a: ACTION[a][1]),
                             "Explicație": ev_h["explicatie"]})
        st.table(show.tail(6).set_index("Data"))
        if len(ev_h) > 6:
            st.caption(f"Sunt afișate ultimele 6 din {len(ev_h)} alarme.")
    else:
        st.success("Nicio alarmă în perioada de test.")
    contact_h = ev_h[ev_h["action"] == "CONTACTEAZA"]
    if len(contact_h):
        with st.expander("✉️ Mesajul către client (ciornă, pentru ultima alarmă CONTACTEAZĂ)", expanded=True):
            st.code(customer_message(contact_h.iloc[-1], ECON), language=None)
            st.caption("Șablon de adaptat și de aprobat de un operator uman; sistemul nu trimite nimic singur. "
                       "Mesajul nu afirmă că panourile sunt defecte (nu putem ști de la distanță).")
    else:
        st.caption("Pentru această casă nu pleacă niciun mesaj către client: nicio alarmă de tip CONTACTEAZĂ.")

# ================================================================== 3
elif page == PAGES[2]:
    st.header("Coada de verificări sub capacitate limitată")
    st.markdown("Echipa poate verifica doar **K case pe săptămână**. Fiecare alarmă deschide un caz, iar echipa le ia pe cele mai importante. "
                "O vizită la o casă cu defect îl repară; una la o casă fără defect e **irosită**.")
    c1, c2 = st.columns([2, 3])
    unlimited = c1.checkbox("Fără limită de capacitate")
    K = None if unlimited else c1.slider("Capacitate: verificări pe săptămână", 1, 10, 3)
    names = list(policies())
    chosen = c2.multiselect("Politici comparate", names, default=[names[0], names[1], names[3]])
    if chosen:
        rows, cums = [], {}
        for n in chosen:
            m, visits, cum = queue_result(n, K)
            cums[n] = cum
            rows.append({"Politică": n, "Defecte reparate (%)": m["defecte_reparate%"],
                         "kWh pierduți pe defect (medie)": m["kWh_pierduti_pe_defect"],
                         "kWh pierduți pe defect (mediană)": m["kWh_mediana_pe_defect"],
                         "Vizite totale": m["vizite_total"], "Vizite irosite": m["irosite"], "Defecte găsite": m["gasite"],
                         "Precizia vizitelor (%)": m["precizie%"],
                         "Zile mediane până la reparație": m["zile_mediane_pana_la_reparatie"]})
        st.dataframe(pd.DataFrame(rows).set_index("Politică"))
        n_f = D["lost"].shape[1]
        st.markdown("#### Cost, cu ipoteze pe care le puteți modifica")
        k1, k2 = st.columns(2)
        price = k1.number_input("Prețul energiei (lei/kWh)", min_value=0.0, max_value=10.0, value=float(ECON["price_per_kwh"]), step=0.05)
        visit = k2.number_input("Costul unei vizite (lei)", min_value=0.0, max_value=5000.0, value=float(ECON["visit_cost"]), step=10.0)
        cost = pd.DataFrame([{"Politică": r["Politică"],
                              "Energie pierdută (lei)": round(r["kWh pierduți pe defect (medie)"] * n_f * price),
                              "Vizite (lei)": round(r["Vizite totale"] * visit)} for r in rows]).set_index("Politică")
        cost["Cost total (lei)"] = cost["Energie pierdută (lei)"] + cost["Vizite (lei)"]
        st.dataframe(cost)
        st.caption(f"Preț: {ECON.get('price_source', '')} {ECON.get('price_note', '')} "
                   f"Cost vizită: {ECON.get('visit_cost_note', '')}")
        fig, ax = plt.subplots(figsize=(11, 3.8))
        none = D["lost"].cumsum().sum(axis=1)
        ax.plot(none.index, none, color="#9ca3af", ls="--", label="fără nicio verificare")
        for n, c in cums.items():
            ax.plot(c.index, c, lw=2 if "Dimmer" in n else 1.4, label=n)
        ax.set_ylabel("kWh de producție pierduți, cumulat")
        ax.set_title(f"Energia pierdută pe cele {D['lost'].shape[1]} defecte simulate" + ("" if K is None else f", K = {K} pe săptămână"))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
        st.pyplot(fig)
        plt.close(fig)
        with st.expander("Jurnalul verificărilor (Dimmer)"):
            if names[3] in chosen or True:
                _, vis, _ = queue_result(names[3], K)
                if len(vis):
                    t = vis.assign(Data=vis["date"].dt.strftime("%Y-%m-%d"), Casa=vis["home"],
                                   Rezultat=vis["outcome"].map(OUTCOME_RO), **{"Zile de așteptare": vis["wait_days"]})
                    st.dataframe(t[["Data", "Casa", "Rezultat", "Zile de așteptare"]].set_index("Data"))
    st.caption("Valorile sunt pe 40 de defecte simulate (10 pe tip). Cu atât de puține defecte, diferențele mici dintre politici nu sunt concludente.")

# ================================================================== 4
elif page == PAGES[3]:
    st.header("Episoade reale de scădere a producției")
    st.markdown("Pe lângă defectele simulate, datele conțin case a căror producție **a scăzut real**. "
                "Nu știm de ce (defect, umbră, panou scos), deci le numim *episoade probabile*. Lista s-a stabilit după o regulă fixă "
                "(randamentul din perioadă sub 70% din cel din anul de antrenare), nu după rezultatele vreunui model, iar casele nu au primit defecte simulate.")
    if not real_homes:
        st.info("Nu există case cu scădere reală în datele curente.")
        st.stop()
    b1, b2 = D["b1"], D["b2"]
    first = lambda df, h: df.loc[df["home"] == h, "date"].min()
    rows = []
    for h in real_homes:
        e = dec[dec["home"] == h]
        rows.append({"Casa": h, "kW": cap[h], "B1": first(b1, h), "B2": first(b2, h),
                     "Dimmer, prima alarmă": first(dec, h),
                     "Dimmer, primul CONTACT": e.loc[e["action"] == "CONTACTEAZA", "date"].min() if len(e) else pd.NaT})
    t = pd.DataFrame(rows).set_index("Casa")
    n_real = len(real_homes)
    c1, c2, c3 = st.columns(3)
    c1.metric("B1 a alarmat", f"{t['B1'].notna().sum()} din {n_real}")
    c2.metric("B2 a alarmat", f"{t['B2'].notna().sum()} din {n_real}")
    c3.metric("Dimmer a alarmat", f"{t['Dimmer, prima alarmă'].notna().sum()} din {n_real}")
    st.dataframe(t.apply(lambda c: c.dt.strftime("%Y-%m-%d").fillna("—") if c.dtype.kind == "M" else c))
    h = st.selectbox("Casa", real_homes, format_func=home_label)
    g = D["gg"][h]
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(smooth(g).index, smooth(g), color="#374151", lw=1.6, label="producția reală (kWh/zi, media pe 7 zile)")
    for lbl, df, colr in (("B1", b1, "#2563eb"), ("B2", D["b2"], "#16a34a"), ("Dimmer", dec, "#d62728")):
        d0 = first(df, h)
        if pd.notna(d0):
            ax.axvline(d0, color=colr, ls="--", lw=1.4, label=f"prima alarmă {lbl}: {d0:%Y-%m-%d}")
    ax.set_ylabel("kWh pe zi")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="upper right")
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Atenție: sistemul nu vede producția, doar netul. Aici o afișăm doar ca să vedem scăderea.")

# ================================================================== 5
else:
    st.header("Rezultatele, cinstit")
    st.markdown("Toate pragurile au fost alese pe anul de validare și **înghețate** înainte de testul final, rulat o singură dată.")
    st.subheader("0. Comparația principală (test, la același nivel de vizite false)")
    st.dataframe(pd.DataFrame({"B1": {"Defecte simulate prinse": "42,5%", "Alarme false la 1.000 zile-casă": "1,40", "kWh pierduți pe defect": "295"},
                               "B2": {"Defecte simulate prinse": "55%", "Alarme false la 1.000 zile-casă": "2,00", "kWh pierduți pe defect": "257"},
                               "Dimmer (model + verificare)": {"Defecte simulate prinse": "62%", "Alarme false la 1.000 zile-casă": "1,00", "kWh pierduți pe defect": "223"},
                               "Dimmer fără verificare": {"Defecte simulate prinse": "77,5%", "Alarme false la 1.000 zile-casă": "2,12", "kWh pierduți pe defect": "162"}}))
    st.caption("Ultima coloană este un alt punct de funcționare: mai sensibil, dar cu mai multe vizite false; nu o comparați direct cu B1.")
    st.subheader("1. Predicția")
    st.dataframe(pd.DataFrame({"Eroare medie (kWh/oră)": {"Baseline: media casei la aceeași oră și lună": "0,5030 / 0,5124",
                                                           "Model LightGBM (radiație + vecini)": "0,4104 / 0,4162",
                                                           "Îmbunătățire": "−18,4% / −18,8%"}}).rename_axis("validare / test"))
    mt = DEMO / "model_test_raw.csv"
    if mt.exists():
        st.caption("Ablația pe test (ce contribuie fiecare ingredient), din reports/model_test_raw.csv:")
        m = pd.read_csv(mt, index_col=0)
        keep = [c for c in ["MAE_kWh_ora", "FA_la_1000", "kWh_pierduti_medie"] if c in m.columns]
        st.dataframe(m[keep].rename(columns={"MAE_kWh_ora": "MAE", "FA_la_1000": "Alarme false la 1.000", "kWh_pierduti_medie": "kWh pierduți (medie)"}))
    st.subheader("2. Detecția, pe tipul de defect (test)")
    bt = DEMO / "baseline_test.csv"
    if bt.exists() and mt.exists():
        b = pd.read_csv(bt)
        m = pd.read_csv(mt, index_col=0)
        rows = []
        for typ in ["total", "partial", "intermittent", "gradual"]:
            r = {"Tip defect": TYPE_RO[typ]}
            for meth in ("B1", "B2"):
                x = b[(b["type"] == typ) & (b["metoda"] == meth)]
                if len(x):
                    r[f"{meth}: prinse %"] = x["prinse"].iloc[0]
                    r[f"{meth}: zile"] = x["zile_mediane_pana_la_detectie"].iloc[0]
            if "full" in m.index and f"{typ}_prinse%" in m.columns:
                r["Dimmer: prinse %"] = m.loc["full", f"{typ}_prinse%"]
                r["Dimmer: zile"] = m.loc["full", f"{typ}_zile"]
            rows.append(r)
        st.dataframe(pd.DataFrame(rows).set_index("Tip defect"))
    st.caption("Alarme false la 1.000 zile-casă: B1 1,40 · B2 2,00 · Dimmer fără verificare 2,12 (peste ținta de 2,0). "
               "Pe validare, detecția era la paritate cu B1; pe test e mai bună, dar cu 10 defecte pe tip o parte din diferență poate fi întâmplare.")
    st.subheader("3. Stratul de decizie (soare sau consum?)")
    st.dataframe(pd.DataFrame({"Fără verificare": {"Alarme CONTACT false la 1.000": "2,12", "Defecte prinse": "78%", "kWh pierduți (medie)": "162",
                                                    "Piscină cu alarmă": "100%", "Muncă de acasă cu alarmă": "80%", "Aer condiționat cu alarmă": "60%"},
                               "Cu verificare (prag 0,4)": {"Alarme CONTACT false la 1.000": "1,00", "Defecte prinse": "62%", "kWh pierduți (medie)": "223",
                                                             "Piscină cu alarmă": "40%", "Muncă de acasă cu alarmă": "20%", "Aer condiționat cu alarmă": "60%"}}))
    st.caption("Verificarea reduce vizitele false și confuzia cu consumul constant, dar costă recall (78% → 62%). "
               "Defectele slabe (parțiale, intermitente, treptate) trec prin verificare cam ca alarmele false (40–51% față de 47%); "
               "cele puternice (totale: 77%) și scăderile reale (73%) sunt păstrate.")
    st.subheader("4. Triajul cu capacitate (aceleași calcule ca în reports/triage_test.csv)")
    pol = [names_ for names_ in policies()]
    rows = []
    for K in (1, 2, 3, 5, None):
        for n in (pol[1], pol[3]):
            m, _, _ = queue_result(n, K)
            rows.append({"K pe săptămână": "fără limită" if K is None else str(K), "Politică": n,
                         "Reparate %": m["defecte_reparate%"], "kWh pe defect": m["kWh_pierduti_pe_defect"],
                         "Vizite irosite": m["irosite"]})
    st.dataframe(pd.DataFrame(rows).set_index(["K pe săptămână", "Politică"]))
    st.subheader("Limite")
    st.markdown("""
- Defectele sunt **simulate** (40 în test, câte 10 pe tip). Cu atât de puține, diferențele mici dintre metode nu sunt concludente.
- Alarmele false pe test (2,12 la 1.000) au depășit ținta de 2,0.
- **Defectele totale sunt prinse mai încet** decât de o regulă simplă (22,5 zile față de 9).
- **Aerul condiționat rămâne confundat** cu un defect: rulează în zilele calde, deci însorite.
- Pe episoadele reale, Dimmer alarmează la fel de multe case ca B1 (9 din 12), iar B2 mai multe (11 din 12); pentru unele (de exemplu casa 234) alarmează mult mai târziu decât baseline-urile.
- Datele sunt din Australia (Ausgrid, 2010-2013); metoda se transferă, cifrele nu.
- Nu am confirmat defecte fizice; spunem „merită verificat", nu „este defect".
""")