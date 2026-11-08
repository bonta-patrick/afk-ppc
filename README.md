# afk-ppc · Dimmer

> Un agent care găsește panourile solare stricate „pe tăcute" doar din citirile contorului net și decide **cine merită verificat primul**, fără să trimită tehnicieni degeaba.

**English summary.** Rooftop-solar customers have a net meter: the supplier sees only import minus export, so a failed panel looks like "the customer uses more power". Dimmer (1) predicts each home's expected net reading with a LightGBM model trained on one clean year, (2) accumulates evidence over days with CUSUM, (3) checks whether the gap *grows with the sun* (a missing panel) or is roughly constant (a new load such as a pool pump), and (4) ranks the resulting cases under a weekly inspection capacity. Everything is evaluated on a held-out year, run once with frozen thresholds, on real data with **simulated** faults. On the test year, model + decision catches 62% of simulated faults at 1.00 false dispatches per 1,000 home-days, versus 42.5% at 1.40 for the best simple rule (B1); on validation it matched B1's recall at half the false dispatches. The main value is triage. Limitations are listed in section 8.

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

**Comparația principală, la același nivel de vizite false.** Pentru Dimmer, o „alarmă falsă" este o alarmă CONTACTEAZĂ pe o casă curată (o vizită irosită); pentru B1 și B2, orice alarmă duce la o vizită.

| | B1 | B2 | **Dimmer** (model + verificare) |
|---|---|---|---|
| Defecte simulate prinse (medie pe 4 tipuri) | 42,5% | 55% | **62%** |
| Alarme false la 1.000 zile-casă | 1,40 | 2,00 | **1,00** |
| kWh pierduți pe defect (medie) | 295 | 257 | **223** |

Pe test, Dimmer prinde mai mult, cu mai puține vizite false și mai puțină energie pierdută. Pe **validare**, la același procent de defecte prinse ca B1 (80%), Dimmer a făcut jumătate din vizitele false (0,57 față de 1,11), cu ~11% mai mulți kWh pierduți (239 față de 215). Cu 10 defecte pe tip, diferențele mici nu sunt concludente.

Fără verificarea soare/consum, modelul funcționează la alt punct: **77,5%** defecte prinse la **2,12** alarme false la 1.000 (peste ținta de 2,0). Nu se compară direct cu B1 (1,40).

**Alte rezultate:**

| | Valoare |
|---|---|
| Eroarea de predicție (MAE) față de baseline-ul „media casei la aceeași oră și lună" | **−18,8%** (0,4162 față de 0,5124); pe validare −18,4% |
| Piscină / muncă de acasă luate drept defect (efectul verificării) | 100% → **40%** / 80% → **20%** (aerul condiționat rămâne 60%) |
| Episoade reale de scădere a producției (12 case) | Dimmer 9 · B1 9 · B2 11 |
| Triaj la capacitate mică (K = 1–5 pe săptămână) | repară 38–62% din defecte, față de 18–42% la B1 ordonat după capacitate |

Detalii și limite în secțiunile 7 și 8.

## 4. Cum rulezi

### 4.1 Aplicația demo (fără datele mari)

```bash
pip install -r requirements.txt
python -m streamlit run app/app.py
```

Testat cu Python 3.14. Aplicația citește din `demo_data/test/` (inclus în repo, câțiva MB; doar pagina „Demo live” are nevoie în plus de `data/processed/` și `models/`, vezi 4.4) și poate fi folosită în română sau engleză (selectorul RO / EN din dreapta sus). Are șase secțiuni: ce face Dimmer, demo-ul live (secțiunea 4.4), un caz pe larg (cu butonul „Arată adevărul"), coada de verificări cu slider de capacitate, episoadele reale și tabelele de rezultate.

### 4.2 Lanțul complet, de la date brute

1. Puneți cele trei fișiere Ausgrid în `data/raw/` și descărcați radiația cu `python src/download_solar_radiation.py` (secțiunea 5).
2. Rulați, din rădăcina repo-ului (de preferat într-un clone separat: lanțul rescrie `data/processed/`, `models/`, `reports/` și pragurile din `config/`):

```bash
python run_pipeline.py --dry-run  # arată pașii și verifică fișierele, fără să ruleze nimic
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

Opțional: `python src/train_eval.py --period val --peer on` (ablația „ajustare față de vecini"), `python src/inspect_misses.py` (analiza erorilor pe SNR), `python src/check_test.py` (verificări pe test, fără efect asupra rezultatelor), `python src/check_zero.py` (case cu producție aproape zero, pe luni) și `python src/inspect_raw.py` (primele rânduri din fiecare fișier brut).

**Notă.** `config/faults.json` (creat automat) presupune cel puțin ~100 de case eligibile (10 defecte × 4 tipuri + 5 consumuri × 4 tipuri). Pentru un set mai mic, reduceți `n_per_type` și `n_load_per_type`.

Dacă `streamlit` nu e recunoscut în PowerShell, folosiți `python -m streamlit`.

### 4.4 Demo live: date istorice, redare accelerată, procesare live

Pagina **Demo live** din aplicație trimite, zi cu zi, anul de test prin lanțul real: modelul LightGBM `full` deja antrenat (încărcat o singură dată), CUSUM-ul cu pragul înghețat (h = 40), verificarea soare/consum (θ = 0,4), prioritatea din triaj și ciorna mesajului din `src/messages.py`. Nu se reantrenează nimic, nu se schimbă praguri și nu se scrie nimic pe disc.

**Pornire** (are nevoie de datele brute din `data/raw/` și de `models/`, vezi secțiunea 5):

```bash
python src/prepare.py                    # data/processed/ (scrie doar acolo)
python src/simulate.py --period test     # copia de test cu defectele simulate ale evaluării oficiale
python src/demo_scenario.py              # alege scenariul rulând pipeline-ul -> config/demo_scenario.json
python src/check_replay.py               # opțional: verificările de mai jos (~15 s)
python -m streamlit run app/app.py       # apoi pagina „Demo live” → Start
```

**Cum funcționează.** Un pas = o zi completă, pentru că modelul decide pe zile (media zilnică a abaterii pe orele 8–17, CUSUM zilnic); nu există decizii la 30 de minute. La ziua D se folosesc doar citirile nete ale zilei D, consumul de noapte din cele 28 de zile **anterioare**, radiația zilei D și starea acumulată până atunci. Data simulată și timpul real sunt afișate separat. Controale: Start, Pauză / Continuare, Reset și viteza (2 s … 0,1 s pe zi simulată). Un singur fir de execuție pe sesiune: un al doilea Start este refuzat, iar dacă o zi durează mai mult decât intervalul ales, următoarea pornește după ea (pașii nu se suprapun). Reset oprește execuția și golește doar starea demo (progres, evenimente, cazuri, deduplicare).

**Acțiuni automate.** O alarmă CONTACTEAZĂ din fereastra demo deschide un caz `DEMO-nnn` cu dovezile calculate (CUSUM la alarmă, zile cu date, citiri lipsă, abaterea medie, partea solară, motivul deciziei), prioritatea din triaj (pierdere estimată × parte solară, kWh/zi; nu e o probabilitate) și o notificare în **inboxul demo**. Gruparea incidentelor e regula existentă din `triage.py`: o casă are cel mult un caz deschis, iar alarmele noi se atașează la el fără o notificare nouă. Proiectul nu are niciun destinatar de test configurat, deci nimic nu pleacă din aplicație. Cazurile demo trăiesc doar în memoria sesiunii, separat de datele aplicației.

**Estimarea financiară** (în RON, la 1,30 lei/kWh din `config/economics.json`). *Pierderea acumulată* = pierderea solară estimată pe zi (abatere × parte solară, regula din `messages.py`) × zilele cu date din fereastra de dovezi. Fiecare zi se numără o singură dată, chiar dacă alarmele se suprapun. *Proiecția* pe 30 de zile e afișată separat și nu se adună la pierderea acumulată. Graficul arată netul observat și netul estimat de model, nu producția: contorul nu vede producția.

**Datele și separarea.** Modelul e antrenat pe 2010-07 → 2011-06 (`config/split.json`, `src/train_eval.py --period train`). Replay-ul folosește anul de test 2012-07 → 2013-06, adică aceeași copie `data/processed/sim/test/net_faulty.parquet` ca evaluarea oficială: date Ausgrid reale, plus defectele și consumurile simulate de `src/simulate.py` pe 60 de case. Aceste case sunt etichetate vizibil „Anomalie simulată” în aplicație. Datele originale nu se modifică. Înainte de fereastra demo, istoricul de la 2012-07-01 trece prin același lanț fără acțiuni, ca CUSUM-ul să pornească din aceeași stare ca rularea obișnuită.

**Scenariul ales** (`python src/demo_scenario.py`, 120 de zile ≈ 2 minute la 1 s/zi; se schimbă cu `--days` și `--sec-per-day`): fereastra 2012-09-26 → 2013-01-23, toate trei casele cu **date reale, nemodificate**:
1. *funcționare normală*: casa 251, nicio alarmă;
2. *abatere păstrată în monitorizare*: casa 149, alarmă pe 2012-11-06, dar abaterea nu urmărește soarele, deci acțiunea e MONITORIZEAZĂ: fără caz, fără notificare, fără alte alarme în fereastră. Abaterea nu dispare din date (netul rămâne peste estimare), deci e o abatere respinsă de verificare, nu una care se stinge singură;
3. *subperformanță persistentă*: casa 234, a cărei producție a scăzut real (cauză necunoscută); alarmă pe 2013-01-05 → CONTACTEAZĂ → caz și notificare.

Scriptul alege după rezultatele pipeline-ului, nu după rezultate scrise de mână, iar în aplicație totul se recalculează live. Pentru rolul 3 sunt excluse casele curate, pentru că acolo un CONTACTEAZĂ e o alarmă falsă. Nu a fost nevoie de nicio anomalie injectată suplimentar. În aceeași fereastră, pe toate cele 297 de case apar și alte cazuri, inclusiv pe case cu defecte simulate și alarme false pe case curate: sunt afișate așa cum ies din pipeline.

**Verificări** (`python src/check_replay.py`, 33 de verificări):
- replay-ul zi cu zi dă aceleași valori z, alarme, acțiuni și părți solare ca rularea obișnuită pe tot anul și ca `demo_data/test/decisions.csv` (454 de alarme);
- o zi nu depinde de datele de după ea;
- Start / Pauză / Continuare / Reset, inclusiv porniri simultane, funcționează fără pași suprapuși;
- nu apar cazuri sau notificări duplicate, iar pierderea nu numără aceleași zile de două ori;
- zilele cu citiri lipsă și o eroare într-un pas sunt tratate fără să oprească aplicația;
- explicațiile se generează în română și engleză.

**Limite.** Demo-ul are nevoie de `data/processed/` și `models/` (nu sunt în repo), deci nu rulează direct după `git clone`, ca restul aplicației. Starea demo e per sesiune de browser și se pierde la reîncărcarea serverului. Inspecțiile din teren nu sunt simulate, așa că un caz rămâne deschis până la Reset.

## 5. Date

- **Consum și producție:** *Ausgrid Solar Home Electricity Data* (Sydney, 300 case cu panouri, citiri la 30 de minute, 2010-07-01 → 2013-06-30; canale GC consum, GG producție, CL circuit controlat). Licență Creative Commons Attribution. Descărcările oficiale au fost retrase de pe site-ul Ausgrid; am folosit copia publicată de Pierre Haessig (<https://pierreh.eu/downloads/Ausgrid_solar_home_data.zip>, linkată din <https://github.com/pierre-haessig/ausgrid-solar-data>). Fișierele folosite: `Solar home 2010-2011.csv`, `Solar home 2011-2012.csv`, `Solar home 2012-2013.csv`.
- **Radiație solară:** Open-Meteo, API-ul istoric (`https://archive-api.open-meteo.com/v1/archive`), variabila `shortwave_radiation` (W/m², medie pe ora precedentă), orară, pentru centrul Sydney (latitudine −33,8688, longitudine 151,2093), fus orar `Australia/Sydney`, 2010-01-01 → 2013-12-31. Se descarcă cu `python src/download_solar_radiation.py` în `data/raw/sydney_shortwave_radiation_2010-2013.csv`.
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

**Mesajul către prosumator** (`src/messages.py`, afișat în aplicație la fiecare alarmă CONTACTEAZĂ). O ciornă în română care spune ce s-a observat (cu cât consumă casa mai mult din rețea la orele de zi și că diferența urmărește soarele), ce ar putea costa (dacă `config/economics.json` are un preț), ce poate verifica clientul (afișajul invertorului, umbrirea, siguranțele) și că nu putem confirma de la distanță. Este un șablon de adaptat și de aprobat de un operator uman; sistemul nu trimite nimic singur. Pentru alarmele MONITORIZEAZĂ nu pleacă niciun mesaj.

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

Observații: ordonarea după mărimea sistemului singură (B1 după capacitate) reduce energia pierdută cu mai mult de jumătate față de ordinea sosirii la K = 1 pe validare (822 → 372). În interiorul modelului cu decizie, scorul propriu (pierdere estimată × parte solară) e aproximativ egal cu ordonarea după capacitate la K ≥ 3 și puțin mai bun la K = 1–2 (de exemplu 339 față de 369 kWh la K = 1, pe test). Față de „model, toate alarmele, după pierdere", stratul de decizie pierde energie, dar evită vizite irosite; punctul de echilibru este de ordinul a 10–22 kWh de energie pe vizită evitată. **Costuri** (`config/economics.json`). Prețul energiei: **1,30 lei/kWh**, preț final cu TVA al ofertelor standard pentru casnici în 2026 (interval 1,06–1,45; surse și dată în fișier; căutare din 7 noiembrie 2026, de reconfirmat pe comparatorul ANRE). Energia pierdută e valorizată la prețul plătit de client, deci e pierderea clientului, nu venitul furnizorului. Costul unei vizite (100 lei) este o **ipoteză conservativă, fără sursă publică**. Punctul de echilibru e de **13–29 lei pe vizită evitată** (10–22 kWh × 1,30): dacă o vizită costă mai mult, modelul câștigă. Cu aceste valori, la capacitate nelimitată: pe test, B1 după capacitate ≈ 27.500 lei și Dimmer ≈ 23.000 lei (−17%); pe validare ≈ 18.100 și ≈ 14.100 (−22%). Diferența vine în principal din vizitele evitate. Aplicația permite modificarea ambelor valori.

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
- Costul unei vizite este o ipoteză fără sursă publică; prețul energiei vine din surse publice, de reconfirmat.

## 9. Relevanță pentru PPC

Premisa se transferă: la prosumatori, contorul vede importul și exportul, nu producția panourilor. Argumente de business (**de confirmat cu surse oficiale înainte de prezentare**):
1. furnizorul compensează și trebuie să prognozeze energia livrată de prosumatori; panourile defecte strică prognoza;
2. clientul cu panouri stricate vede doar o factură mai mare și sună la call center;
3. un mesaj proactiv înseamnă client păstrat și, eventual, un serviciu de mentenanță.

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
  LICENSE                # MIT
  requirements.txt
  run_pipeline.py        # rulează tot lanțul
  app/app.py             # aplicația demo (Streamlit, RO / EN)
  app/.streamlit/        # tema aplicației
  demo_data/test/        # date mici pentru aplicație (în repo)
  config/                # praguri înghețate, scenarii de defecte, împărțirea în timp, economics.json
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
    messages.py          # ciorna mesajului către client (RO / EN)
    make_demo_data.py    # pregătește demo_data/
    demo_replay.py       # demo live: replay zi cu zi prin lanțul real, cazuri și inbox demo
    demo_scenario.py     # alege scenariul demo rulând pipeline-ul -> config/demo_scenario.json
    check_replay.py      # verificările modului demo
    download_solar_radiation.py  # descarcă radiația de la Open-Meteo
    check_test.py, check_zero.py, inspect_raw.py  # verificări ajutătoare
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

**Licența codului:** MIT, vezi [LICENSE](LICENSE). Licența acoperă codul; datele Ausgrid și Open-Meteo își păstrează licențele proprii.