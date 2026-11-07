# afk-ppc · Dimmer

> Un agent care găsește panourile solare stricate „pe tăcute" doar din citirile contorului net și decide **cine merită verificat primul**, fără să trimită tehnicieni degeaba.

**English summary.** Rooftop-solar customers have a net meter: the supplier sees only import minus export, so a failed panel looks like "the customer uses more power". Dimmer (1) predicts each home's expected net reading with a LightGBM model trained on one clean year, (2) accumulates evidence over days with CUSUM, (3) checks whether the gap *grows with the sun* (a missing panel) or is roughly constant (a new load such as a pool pump), and (4) ranks the resulting cases under a weekly inspection capacity. Everything is evaluated on a held-out year, run once with frozen thresholds, on real data with **simulated** faults. Detection is better than simple rules on harder fault types at a somewhat higher false-alarm rate; the main value is triage. Limitations are listed in section 8.

---

## 1. Problema

Un prosumator are un contor care măsoară doar **diferența** dintre ce ia din rețea și ce trimite în rețea. Dacă panourile se strică, nimeni nu observă imediat: furnizorul vede doar un consum net mai mare. Dar un consum mai mare poate însemna și o piscină nouă, o mașină electrică sau munca de acasă. Dacă fiecare creștere devine o alarmă, echipele de teren sunt inundate cu vizite irosite.

## 2. Ce face Dimmer

1. **Ghicește** netul „normal" al fiecărei case pe orele de zi (LightGBM, antrenat pe un an, cu radiația solară și casele vecine).
2. **Adună dovezi pe mai multe zile** (CUSUM). O singură zi ciudată nu produce alarmă.
3. **Decide ce se întâmplă cu alarma**: lipsa de producție solară crește odată cu soarele, un consum nou nu. Alarma devine o acțiune:

| Acțiune | Când |
|---|---|
| **CONTACTEAZĂ** | golul urmărește soarele (`parte_solară` ≥ 0,4): probabil lipsește producția solară |
| **MONITORIZEAZĂ** | golul nu urmărește soarele (probabil consum nou) sau soarele a variat prea puțin ca să știm |
| **VERIFICĂ DATELE** | prea puține zile cu date sau prea multe citiri lipsă |

4. **Ordonează cazurile** sub o capacitate de K verificări pe săptămână, după un scor (pierdere estimată × parte solară), și generează o explicație în cuvinte pentru fiecare alarmă.

Sistemul **nu** pretinde că dovedește un defect fizic. Spune „merită verificat".

## 3. Rezultate pe scurt (testul final, 2012-07 → 2013-06)

| | Valoare |
|---|---|
| Eroarea de predicție (MAE) față de baseline-ul „media casei la aceeași oră și lună" | **−18,8%** (0,4162 față de 0,5124); pe validare −18,4% |
| Defecte simulate prinse (medie pe 4 tipuri) | Dimmer **77,5%** · B1 42,5% · B2 55% |
| Alarme false la 1.000 zile-casă | Dimmer 2,12 · B1 1,40 · B2 2,00 (ținta era ≤ 2,0) |
| Stratul de decizie: alarme CONTACT false la 1.000 | 2,12 → **1,00** (dar defectele prinse 78% → 62%) |
| Piscină / muncă de acasă luate drept defect | 100% → **40%** / 80% → **20%** (aerul condiționat rămâne 60%) |
| Episoade reale de scădere a producției (12 case) | Dimmer 9 · B1 9 · B2 11 |
| Triaj la capacitate mică (K = 1–5 pe săptămână) | repară 38–62% din defecte, față de 18–42% la B1 ordonat după capacitate |

Pe validare, detecția era la paritate cu B1; pe test e mai bună, dar cu 10 defecte pe tip o parte din diferență poate fi întâmplare (secțiunile 7 și 8).

## 4. Cum rulezi

### 4.1 Aplicația demo (fără datele mari)

```bash
pip install -r requirements.txt
python -m streamlit run app/app.py
```

Aplicația citește doar din `demo_data/test/` (inclus în repo, câțiva MB). Are cinci secțiuni: ce face Dimmer, un caz pe larg (cu butonul „Arată adevărul"), coada de verificări cu slider de capacitate, episoadele reale și tabelele de rezultate.

### 4.2 Lanțul complet, de la date brute

1. Puneți datele în `data/raw/` (secțiunea 5).
2. Rulați:

```bash
python run_pipeline.py            # până la validare (fără testul final)
python run_pipeline.py --test     # adaugă testul final, rulat O SINGURĂ DATĂ
```

Antrenarea durează aproximativ 40 de secunde pe variantă pe laptopul autorilor (5 variante). Scriptul se oprește la prima eroare și spune la ce pas.

### 4.3 Pașii pe rând (din rădăcina repo-ului)

| Pas | Comandă | Produce |
|---|---|---|
| 1 | `python src/prepare.py` | date curate orare în `data/processed/`, jurnalul curățării |
| 2 | `python src/split_and_check.py` | împărțirea în timp, grafice în `reports/checks/`, `config/real_decline_homes.json` |
| 3 | `python src/simulate.py --period val` și `--period test` | defecte și consum nou simulate, cu răspunsul corect |
| 4 | `python src/baseline.py --period val` | baseline-uri B1 și B2, praguri în `config/baseline_thresholds.json` |
| 5 | `python src/train_eval.py --period train` | modelele LightGBM în `models/` |
| 6 | `python src/train_eval.py --period val --peer off` | acuratețea, pragul CUSUM în `config/model_thresholds.json` |
| 7 | `python src/decide.py --period val` | pragul stratului de decizie în `config/decision.json` |
| 8 | `python src/triage.py --period val` | coada cu capacitate |
| 9 | aceleași scripturi cu `--period test` | testul final (pragurile sunt înghețate, nu se modifică) |
| 10 | `python src/make_demo_data.py --period test` | `demo_data/test/` pentru aplicație |

Opțional: `python src/train_eval.py --period val --peer on` (ablația „ajustare față de vecini"), `python src/inspect_misses.py` (analiza erorilor pe SNR) și `python src/check_test.py` (verificări pe test, fără efect asupra rezultatelor).

**Notă.** `config/faults.json` (creat automat) presupune cel puțin ~100 de case eligibile (10 defecte × 4 tipuri + 5 consumuri × 4 tipuri). Pentru un set mai mic, reduceți `n_per_type` și `n_load_per_type`.

Dacă `streamlit` nu e recunoscut în PowerShell, folosiți `python -m streamlit`.

## 5. Date

- **Consum și producție:** *Ausgrid Solar Home Electricity Data* (Sydney, 300 case cu panouri, citiri la 30 de minute, 2010-07-01 → 2013-06-30; canale GC consum, GG producție, CL circuit controlat). Licență Creative Commons Attribution. Descărcările oficiale au fost retrase de pe site-ul Ausgrid; am folosit copia publicată de Pierre Haessig (<https://pierreh.eu/downloads/Ausgrid_solar_home_data.zip>, linkată din <https://github.com/pierre-haessig/ausgrid-solar-data>). Fișierele folosite: `Solar home 2010-2011.csv`, `Solar home 2011-2012.csv`, `Solar home 2012-2013.csv`.
- **Radiație solară:** Open-Meteo, API-ul istoric, radiație pe unde scurte, orară, Sydney, 2010-2013, în `data/raw/sydney_shortwave_radiation_2010-2013.csv`. **TODO (autori):** completați coordonatele exacte, fusul orar și parametrii cererii.
- **Datele brute nu sunt în repo** (`data/` și `models/` sunt în `.gitignore`).

Citare: E. L. Ratnam, S. R. Weller, C. M. Kellett, A. T. Murray, „Residential load and rooftop PV generation: an Australian distribution network dataset", *International Journal of Sustainable Energy*, 2017, DOI 10.1080/14786451.2015.1100196.

**Curățarea** (detalii în `data/processed/cleaning_log.txt`):
- formatele de dată diferă între fișiere (`1-Jul-10` și `1/07/2011`); lipsa de citire rămâne lipsă, nu zero;
- citirile se agregă la oră; **297 din 300 de case păstrate** (una pentru valori lipsă, două, casele 97 și 146, pentru producție sistematică noaptea);
- după profilul orar, ceasul Ausgrid urmărește ora de vară (vârful producției se mută cu ~1 oră între iulie și ianuarie), iar fișierul de radiație nu are același decalaj; alinierea se face empiric, separat pe zile cu și fără ora de vară (corelație radiație–producție 0,963 după aliniere);
- `net = consum − producție`; modelul vede doar `net`, producția reală rămâne „adevărul ascuns", folosit doar la evaluare.

**Împărțirea în timp:** antrenare 2010-07 → 2011-06, validare 2011-07 → 2012-06, test 2012-07 → 2013-06.

**Case cu scădere reală.** Casele al căror randament din perioadă scade sub 70% din cel din anul de antrenare sunt marcate (5 pe validare, 12 pe test), excluse din simulare și din numărarea alarmelor false și tratate ca *episoade probabile* (cauza nu se cunoaște). Cele 5 din validare sunt excluse și din antrenare.

## 6. Metodă

**Simulatorul de defecte** (`src/simulate.py`, scenarii în `config/faults.json`, scrise înainte de reglare). Se aplică pe producția reală și se recalculează netul:
- *total* (producția devine 0), *parțial* (40–70% din producție), *intermitent* (0 între orele 12–17 în 30% din zile), *treptat* (−0,3% pe zi, până la −30…−50%);
- *consum nou* pe case sănătoase: aer condiționat (zile calde și însorite), pompă de piscină (ziua), mașină electrică (seara), muncă de acasă (zilele lucrătoare);
- câte 10 defecte pe tip și 5 case pe tip de consum; pe validare doar tipurile *total* și *parțial*, pe test toate cele patru.

**Model** (`src/features.py`, `src/train_eval.py`). LightGBM prezice `net − consumul de bază` pe orele 8–17, unde consumul de bază e media netului de noapte din ultimele 28 de zile. Trăsături: ora, ziua săptămânii, ora de vară, capacitatea, consumul de bază, radiația și radiația × capacitatea, „producția" estimată din vecini (mediana celorlalte case, fără casa curentă) și identitatea casei. Nu se folosesc niciodată citiri recente de zi ale casei sau producția reală. Cinci variante (ablație): `own`, `rad`, `nb`, `full` (varianta finală, fără lună) și `full_m` (cu lună). Hiperparametri: 300 de arbori, rată de învățare 0,07, 63 de frunze, `random_state=42`.

**CUSUM.** Reziduul zilnic, standardizat cu zgomotul normal al casei (măsurat pe predicții out-of-fold pe anul de antrenare): `s = max(0, s + z − 0,5)`; alarmă la `s > h`, apoi `s` revine la 0. Pragul `h = 40` a fost ales pe validare: cel mai mic număr de kWh pierduți cu cel mult 2 alarme false la 1.000 zile-casă.

**Stratul de decizie** (`src/decide.py`). La fiecare alarmă, pe ultimele 28 de zile se potrivește `abatere_zilnică = a + b · soare_așteptat`; `parte_solară = b · media(soare_așteptat) / media(abatere)`. Prag ales pe validare: 0,4, cu cel mult 1 alarmă CONTACT falsă la 1.000.

**Triajul** (`src/triage.py`). Simulare zi cu zi a unei cozi: o alarmă deschide un caz, echipa verifică K cazuri pe săptămână după prioritate; o vizită la o casă cu defect început îl repară, una irosită scoate casa din coadă 30 de zile.

**Baseline-uri** (`src/baseline.py`), fără învățare automată: **B1** = alarmă după N zile la rând fără export la prânz (N = 10); **B2** = media pe 7 zile a abaterii netului de prânz față de același interval din anul de antrenare (h = 2,0). Pragurile se aleg pe validare cu aceeași limită de 2 alarme false la 1.000.

## 7. Evaluare

**Protocol.** Pragurile s-au ales exclusiv pe validare și au fost înghețate (`config/*.json`); testul s-a rulat o singură dată (tag `pre-test` înainte, `test-final` după). Alarme false = alarme noi pe casele curate (pauză de 14 zile între alarme la aceeași casă), la 1.000 zile-casă. *kWh pierduți* = producția pierdută din ziua începerii defectului până la alarmă (nedetectat = până la sfârșitul perioadei).

### 7.1 Predicția

| | Validare | Test |
|---|---|---|
| Baseline „media casei la aceeași oră și lună" | 0,5030 | 0,5124 |
| Model `full` | 0,4104 | 0,4162 |

Ablație (test; kWh pierduți = media pe cele 4 tipuri de defect; alarme false la 1.000):

| Variantă | MAE | kWh pierduți | Alarme false |
|---|---|---|---|
| `own` (doar istoricul casei) | 0,4952 | 202,5 | 1,81 |
| `rad` (+ radiație) | 0,4417 | 178,6 | 1,90 |
| `nb` (+ vecini) | 0,4167 | 164,1 | 2,26 |
| **`full`** | **0,4162** | **162,0** | 2,12 |
| `full_m` (cu lună) | 0,4157 | 169,6 | 2,12 |

Varianta finală (`full`, fără ajustare față de vecini) a fost aleasă înainte de test.

### 7.2 Detecția (test, praguri înghețate)

| | B1 | B2 | **Dimmer** |
|---|---|---|---|
| Alarme false la 1.000 | 1,40 | 2,00 | 2,12 |
| Total: prinse · zile mediane | 100% · **9** | 60% · 25 | 100% · 22,5 |
| Parțial | 30% · 45 | 70% · 43 | 70% · 32 |
| Intermitent | 30% · 112 | 60% · 59 | 80% · 81,5 |
| Treptat | 10% · 142 | 30% · 74 | 60% · 121,5 |
| Prinse, medie | 42,5% | 55% | **77,5%** |
| kWh pierduți, medie | 295 | 257 | **162** |

Pe **validare** (doar *total* și *parțial*): B1 1,11 alarme false, total 100% · 9 zile, parțial 60% · 51,5, 215 kWh; B2 1,62, 70% · 6 și 30% · 18, 537 kWh; Dimmer 1,69, 90% · 16 și 70% · 54, 221 kWh. Deci pe validare Dimmer era la paritate cu B1. Baseline-urile și-au schimbat mult rezultatele între perioade (B1 parțial 60% → 30%, B2 parțial 30% → 70%), iar Dimmer a rămas stabil (70% → 70%); cu 10 defecte pe tip, asta nu e o dovadă.

### 7.3 Stratul de decizie

| | Validare, fără | Validare, cu | Test, fără | Test, cu |
|---|---|---|---|---|
| Alarme CONTACT false la 1.000 | 1,69 | **0,57** | 2,12 | **1,00** |
| Defecte prinse | 80% | 80% | 78% | 62% |
| kWh pierduți (medie) | 221 | 239 | 162 | 223 |
| Piscină cu alarmă | 100% | 40% | 100% | 40% |
| Muncă de acasă cu alarmă | 100% | 40% | 80% | 20% |
| Aer condiționat cu alarmă | 60% | 60% | 60% | 60% |

Ponderea alarmelor clasificate CONTACTEAZĂ, pe tipul real al casei (test): defect total 77%, scădere reală 73%, defect parțial 51%, intermitent 48%, **casă curată 47%**, treptat 40%; piscină 10%, muncă de acasă 14%, aer condiționat 73%. Verificarea păstrează defectele puternice și elimină consumul constant, dar defectele slabe trec prin ea cam ca alarmele false: pentru ele funcționează ca o subțiere, nu ca un filtru selectiv.

### 7.4 Triajul cu capacitate

Test (kWh pierduți pe defect · defecte reparate · vizite irosite):

| K pe săpt. | B1, ordinea sosirii | B1, după capacitate | **Dimmer, după scor** |
|---|---|---|---|
| 1 | 419 · 25% · 32 | 403 · 18% · 37 | **339 · 38% · 30** |
| 2 | 327 · 42% · 58 | 317 · 35% · 62 | **266 · 48% · 60** |
| 3 | 313 · 42% · 68 | 309 · 35% · 70 | **232 · 62% · 73** |
| 5 | 302 · 42% · 82 | 298 · 42% · 81 | **225 · 62% · 76** |
| fără limită | 295 · 42% · 96 | 295 · 42% · 96 | **223 · 62% · 80** |

Pe **validare** avantajul de energie nu apărea față de B1 ordonat după capacitate: K = 3 → 223 kWh (B1 după capacitate) față de 244 (Dimmer); fără limită 215 față de 239. Câștigul de acolo era în vizitele irosite (56 față de 82 la K = 3; 58 față de 104 fără limită). Diferența dintre cele două seturi vine în mare parte din tipurile de defect pe care B1 nu le vede (parțial, intermitent, treptat), prezente doar pe test.

Observații: ordonarea după mărimea sistemului singură (B1 după capacitate) reduce energia pierdută cu mai mult de jumătate față de ordinea sosirii la K = 1 pe validare (822 → 372). În interiorul modelului cu decizie, scorul propriu (pierdere estimată × parte solară) e aproximativ egal cu ordonarea după capacitate la K ≥ 3 și puțin mai bun la K = 1–2 (de exemplu 339 față de 369 kWh la K = 1, pe test). Față de „model, toate alarmele, după pierdere", stratul de decizie pierde energie, dar evită vizite irosite; punctul de echilibru este de ordinul a 10–22 kWh de energie pe vizită evitată. **`config/economics.json` (prețul energiei, costul unei vizite) nu e completat**, deci nu publicăm un cost în monedă.

### 7.5 Analiza erorilor

Detecția depinde de raportul semnal/zgomot (SNR = pierderea medie pe oră de zi / zgomotul zilnic normal al casei). Pe validare, la pragul ales: SNR peste 2 → 8 din 8 prinse; 1–2 → 5 din 6; 0,5–1 → 3 din 6. La un prag foarte permisiv, doar 1 din 20 defecte e ratat; deci pierderea nu vine din date imposibile, ci din compromisul dintre alarme false și sensibilitate. Sistemele sunt mici (1–2 kW), iar consumul casei e zgomotos.

### 7.6 Episoade reale (test)

Pe cele 12 case cu scădere reală, Dimmer a alarmat 9 (aceleași ca B1), B1 9, B2 11. Dimmer a ratat casele 220, 239 și 91; toate cele 9 case alarmate au primit apoi CONTACTEAZĂ, de regulă după 0–102 de zile de la prima alarmă. Casa 234 (producție zero tot anul) a fost alarmată de B1 din prima zi (iulie 2012) și de Dimmer abia în ianuarie 2013.

## 8. Limite

- Defectele sunt **simulate** pe producție reală (40 pe test, 10 pe tip); nu avem defecte confirmate în teren. Cu atât de puține, diferențele mici dintre metode nu sunt concludente.
- Alarmele false pe test (2,12 la 1.000) au depășit ținta de 2,0; B1 are 1,40. O parte din avantajul de detecție poate veni din faptul că Dimmer alarmează mai des.
- **Defectele totale sunt prinse mai lent** decât de B1 (22,5 față de 9 zile). Defectele parțiale se prind după 32–54 de zile.
- Precizia vizitelor e de ~22–26%. Depinde de cât de frecvente sunt defectele; în date am simulat ~13% din case pe an (40 din 297), într-o rețea reală pot fi mai rare.
- Verificarea „soare sau consum" pierde recall (78% → 62% pe test) și nu separă defectele slabe de alarmele false; **aerul condiționat rămâne confundat** cu un defect (rulează în zilele însorite).
- Modelul are nevoie de un an de istoric pe fiecare casă; nu funcționează pe case noi. Casele excluse din antrenare au predicții mai slabe.
- Radiația vine dintr-un singur punct; orientarea și umbrirea acoperișurilor sunt învățate doar din anul de antrenare.
- Datele sunt australiene (2010–2013), cu un singur an de antrenare; metoda se transferă, cifrele nu.
- Costul unei vizite și prețul energiei nu sunt completate.

## 9. Relevanță pentru PPC

Premisa se transferă: la prosumatori, contorul vede importul și exportul, nu producția panourilor. Argumente de business (**de confirmat cu surse oficiale înainte de prezentare**):
1. furnizorul compensează și trebuie să prognozeze energia livrată de prosumatori; panourile defecte strică prognoza;
2. clientul cu panouri stricate vede doar o factură mai mare și sună la call center;
3. un mesaj proactiv înseamnă client păstrat și, eventual, un serviciu de mentenanță.

**TODO (autori):** verificați și descrieți cum sunt măsurați și compensați prosumatorii în România (surse oficiale: ANRE, operatorul de rețea) și cine are în responsabilitate contorul și verificările în teren.

## 10. Direcții de continuare

- citiri separate de **export** și date de la invertor (ar elimina zgomotul consumului); temperatura (pentru aerul condiționat și zilele calde);
- mai mulți ani de istoric și date la 15–30 de minute;
- un strat de decizie care înțelege defectele intermitente și treptate;
- validare pe defecte confirmate în teren;
- calibrarea costurilor (vizită, energie) și a capacității reale a echipelor;
- peer-uri topologice (același transformator) în loc de peer-uri după consum.

## 11. Structura repo-ului

```
afk-ppc/
  README.md
  requirements.txt
  run_pipeline.py        # rulează tot lanțul
  app/app.py             # aplicația demo (Streamlit)
  demo_data/test/        # date mici pentru aplicație (în repo)
  config/                # praguri înghețate, scenarii de defecte, împărțirea în timp
  reports/               # rezultate (validare și test)
  src/
    prepare.py           # curățarea datelor
    split_and_check.py   # împărțirea în timp, verificări, case cu scădere reală
    simulate.py          # simulatorul de defecte și de consum nou
    features.py          # trăsăturile modelului
    baseline.py          # B1 și B2
    train_eval.py        # antrenare LightGBM, ablație, CUSUM
    decide.py            # stratul de decizie (soare sau consum?)
    triage.py            # coada cu capacitate
    evalkit.py           # metrici comune
    inspect_misses.py    # analiza erorilor pe SNR
    make_demo_data.py    # pregătește demo_data/
  data/  models/         # generate local, nu sunt în repo
```

## 12. Reproductibilitate și corectări

- Seed-uri fixe: `random_state=42` la LightGBM; `seed` din `config/faults.json` (+1 pentru validare, +2 pentru test).
- Praguri înghețate în `config/` (`baseline_thresholds.json`, `model_thresholds.json`, `decision.json`); tag-urile `pre-test` și `test-final` marchează înghețarea și rezultatele de pe test.
- Corectări făcute în timpul lucrului, înainte de testul final: (1) evaluarea cerea ca alarma să *pornească* după începutul defectului, deci un CUSUM deja aprins făcea defectul să pară ratat — acum alarma se resetează după fiecare verificare și detecția folosește aceleași reguli ca alarmele false; (2) comparația inițială cu B1 în ordinea sosirii era prea favorabilă, așa că am adăugat B1 și modelul ordonate după capacitate; (3) metrica „case în alarmă simultan" nu e comparabilă între metode și a fost înlocuită cu „alarme noi pe 7 zile"; (4) pragul CUSUM se alege acum prin minimizarea kWh pierduți, nu a numărului de defecte prinse.
- Corespondența cifrelor din acest README cu fișierele din `reports/`: secțiunea 7 a fost scrisă din rezultatele salvate ale rulărilor pe validare și pe test.

## 13. Autori și licență

**Autori:** Bonta Patrick, Gilca Thomas, Matei Ioana, Holhos Alexandru.

**Date:** Ausgrid, CC BY (vezi secțiunea 5). Radiație: Open-Meteo (citați conform site-ului open-meteo.com).

**Licența codului:** **TODO (autori):** alegeți o licență (de exemplu MIT) și adăugați fișierul `LICENSE`.