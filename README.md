# # Cost-aware PSP Routing für Online-Kreditkartenzahlungen

Dieses Projekt entwickelt ein datengetriebenes Entscheidungsmodell zur **Optimierung von Kreditkartenzahlungen** im Onlinehandel.  
Ziel ist es, für jede Transaktion denjenigen Payment Service Provider (PSP) auszuwählen, der

- die **höchste Erfolgswahrscheinlichkeit** bietet und  
- gleichzeitig die **geringsten erwarteten Transaktionskosten** verursacht.

Das Projekt orientiert sich methodisch am **CRISP-DM-Prozess** und umfasst den kompletten End-to-End-Workflow von der Datenanalyse bis zur modellbasierten Entscheidungsempfehlung.

---

## 1. Problemstellung & Business-Kontext

Ein international agierendes Einzelhandelsunternehmen verzeichnet eine **hohe Ausfallsrate von Online-Kreditkartenzahlungen** (rund 70–80 % Fehlversuche).  
Die Zahlungen werden über mehrere **Payment Service Provider (PSPs)** abgewickelt, für die jeweils transaktionsabhängige Gebühren anfallen.

Aktuell erfolgt die Auswahl des PSPs über ein **statisches, manuell gepflegtes Regelwerk**.  
Die Fachabteilung vermutet, dass ein **datengetriebenes Prognosemodell** bessere Entscheidungen treffen und damit:

- die **Erfolgsrate von Transaktionen erhöhen** und  
- die **Transaktionskosten senken** kann.

---

## 2. Zielsetzung

Das Projekt zielt darauf ab, den Fachbereich *Online-Zahlungen* durch ein ML-Modell zu unterstützen, das:

- Kreditkartentransaktionen automatisch einem von vier angebundenen PSPs zuweist,
- die **Erfolgsquote maximiert** und
- die **erwarteten Gebühren minimiert**.

Die Lösung adressiert damit sowohl:

- **betriebswirtschaftliche Ziele** (Kostenoptimierung) als auch  
- **Kundenzufriedenheit** (weniger fehlgeschlagene Zahlungen).

---

## 3. Datenbasis

Der Datensatz umfasst Kreditkartentransaktionen aus der **DACH-Region (Deutschland, Österreich, Schweiz)** mit u. a. folgenden Merkmalen:

- Zeitstempel der Transaktion  
- Land  
- Betrag  
- verwendeter PSP  
- Kartenmarke (Visa, MasterCard, Diners)  
- 3D-Secure-Status  
- Erfolgskennzeichen (`success` ∈ {0, 1})  

Zusätzlich liegen **gebührenbezogene Daten** je PSP und Transaktionsausgang (erfolgreich / fehlgeschlagen) vor.

Ein wichtiges fachliches Detail ist die Behandlung von **Mehrfachversuchen**:
- Gleicher Betrag, gleiches Land, gleiche Minute → ein Kaufversuch mit mehreren Zahlungsversuchen.

---

## 4. Projektorganisation nach CRISP-DM

### 4.1 Business Understanding

- Hohe Fehlerrate bei Online-Kartenzahlungen
- Kostenstruktur je PSP (erfolgreiche vs. fehlgeschlagene Transaktion)
- Ziel: datengetriebenes Routing statt statischem Regelwerk

### 4.2 Data Understanding

- Transaktionen mit Zeitstempeln Januar/Februar 2019  
- Hoher Anteil fehlgeschlagener Transaktionen (~70–80 %)  
- Uneinheitliche Verteilung des Volumens über die PSPs  
- Mehrfachversuche als fachlich relevantes Muster (Kunden wiederholen fehlgeschlagene Zahlungen)

### 4.3 Data Preparation

Wesentliche Schritte:

- **Datenbereinigung**
  - Prüfung auf fehlende Werte, Ausreißer, Konsistenz
- **Feature Engineering**
  - Stunde des Tages, Wochentag
  - Identifikation von Kaufversuchen (`attempt_id`)
  - Merkmale wie `attempt_no` (Versuchsnummer) und `is_retry` (Wiederholungsversuch)
  - Log-Transformation von Beträgen
- **Kodierung**
  - Kategorische Variablen → numerische Form (z. B. One-Hot-Encoding)
- **Train-/Test-Split**
  - Gruppiert nach `attempt_id` mittels `GroupShuffleSplit`, um Datenleckagen zu vermeiden

---

## 5. Analyse der Datenqualität

- **Vollständigkeit**  
  - Kaum fehlende Werte, vollständige Zeitstempel im Beobachtungszeitraum  
- **Korrektheit & Konsistenz**  
  - Plausible Werte für Länder (DACH), Kartenmarken und PSPs  
  - Transaktionsbeträge positiv und realistisch verteilt  
- **Eindeutigkeit**  
  - Mehrfach vorkommende Transaktionen sind fachlich erklärbare Wiederholungsversuche, kein Datenfehler
- **Repräsentativität**  
  - Hohe Fehlerrate spiegelt das geschäftliche Problem wider  
  - Unausgewogene Verteilung auf PSPs → in der Modellierung zu berücksichtigen (Class Imbalance, Sampling, Gewichtung)

---

## 6. Explorative Visualisierungen

In den Notebooks werden u. a. diese Fragestellungen visualisiert:

- **Gesamterfolgsrate**  
  - Balkendiagramm: Anteil erfolgreicher vs. fehlgeschlagener Zahlungen

- **Erfolgsrate je PSP**  
  - Vergleich der Erfolgsquoten der PSPs → direkte Hebel für Kosten- und Erfolgsoptimierung

- **Einfluss von 3D Secure**  
  - Vergleich von Transaktionen mit/ohne 3D Secure → höhere Erfolgswahrscheinlichkeit mit zusätzlicher Sicherheit

- **Kartenmarkenvergleich**  
  - Erfolgsquoten nach Kartenmarke (Visa, MasterCard, Diners)

- **Tageszeitliche Muster**  
  - Erfolgsraten über den Tag (0–23 Uhr) → mögliche technische Einflüsse (Last, Wartung etc.)

- **Erst- vs. Wiederholungsversuch**  
  - Erfolgswahrscheinlichkeit bei Erst- und Wiederholungsversuchen

---

## 7. Modellierung

Die Umsetzung erfolgt in **Python** mit:

- `pandas`, `numpy`
- `scikit-learn`

### 7.1 Kostenmodell

Für jeden PSP werden unterschiedliche Kosten für:

- erfolgreiche Transaktionen (`C_success`)  
- fehlgeschlagene Transaktionen (`C_fail`)

hinterlegt.

Die erwarteten Kosten berechnen sich als:

\[
E(C) = p_{\text{success}} \cdot C_{\text{success}} + (1 - p_{\text{success}}) \cdot C_{\text{fail}}
\]

Diese Kostenfunktion wird sowohl für die **Baseline** als auch für das **Vorhersagemodell** verwendet.

### 7.2 Train-/Test-Aufteilung

- Verwendung von `GroupShuffleSplit` mit Gruppierung nach `attempt_id`,  
  um Leckagen zwischen Train- und Testset zu vermeiden.

### 7.3 Baseline-Modell

- **Globale Variante**:  
  Auswahl des PSP mit minimalen erwarteten Kosten über alle Transaktionen hinweg.
- **Segmentierte Variante**:  
  Für jede Kombination aus  
  `country`, `card`, `3D_secured` wird der PSP mit minimalen erwarteten Kosten gewählt.

Dieses Basismodell dient als Referenz für die Bewertung des ML-Modells.

### 7.4 Vorhersagemodell (Model Router)

- Für **jeden PSP** wird ein eigenes **Klassifikationsmodell** (z. B. Logistische Regression) trainiert.
- Das Modell schätzt die **Erfolgswahrscheinlichkeit** pro PSP:

\[
p_i = P(\text{success} \mid \text{features}, \text{PSP} = i)
\]

- Für jeden PSP werden erwartete Kosten berechnet:

\[
\text{PSP}^\* = \arg\min_i \left( p_i \cdot C_{\text{success},i} + (1 - p_i) \cdot C_{\text{fail},i} \right)
\]

- Vorverarbeitung:
  - One-Hot-Encoding für kategoriale Variablen
  - Standardisierung für numerische Variablen
  - `LogisticRegression` mit `class_weight="balanced"` für unbalancierte Klassen

---

## 8. Evaluation & Fehleranalyse

Bewertungskriterien u. a.:

- **Erfolgsrate**  
  - Anteil erfolgreicher Zahlungen im Vergleich zur Baseline

- **Erwartete Kosten**  
  - Durchschnittliche Gebühren pro Transaktion

- **Segmentanalyse (z. B. nach Land und Kartenmarke)**  
  - Vergleich von:
    - empirischer Erfolgsrate,
    - beobachteten Durchschnittskosten,
    - modellbasiert erwarteten Kosten.

Die Ergebnisse zeigen:

- In allen analysierten Segmenten **senkt das Modell die erwarteten Kosten** gegenüber dem Status quo.
- Erfolgsraten liegen zwar insgesamt auf einem niedrigen Niveau (ca. 18–24 %),  
  variieren aber deutlich nach Segment (z. B. Land–Karte-Kombination).
- Das Modell tendiert dazu, die Kosten **optimistischer** zu schätzen als beobachtet,  
  was Hinweise auf mögliche Verbesserungen (z. B. feinere Segmentierung, weitere Features, länderspezifische Modelle) liefert.

---

## 9. Deployment & Integration

### 9.1 Technische Einbindung

Geplante Integration als:

- **API-Service** oder
- Bestandteil eines bestehenden **Payment-Gateways**.

Pro Transaktion werden u. a. folgende Eingaben genutzt:

- `country`, `card`, `amount`, `3D_secured`, `is_retry`, `hour`

Der Model Router gibt für alle PSPs:

- Erfolgswahrscheinlichkeiten und
- erwartete Kosten aus

und wählt den PSP mit den geringsten erwarteten Kosten.

### 9.2 GUI-Prototyp

Ein GUI-Prototyp (z. B. mit **Streamlit**) ermöglicht:

- Eingabe relevanter Parameter durch Fachanwender
- Anzeige:
  - empfohlener PSP
  - erwartete Erfolgswahrscheinlichkeit
  - erwartete Kosten
- Monitoring von Kennzahlen in (nahezu) Echtzeit
- Möglichkeit, PSPs temporär zu deaktivieren oder Schwellenwerte anzupassen

### 9.3 Monitoring und Wartung

- **Kontinuierliches Monitoring** der Erfolgsraten und Kosten
- Überwachung von **Daten- und Konzeptdrift**
- **Retraining-Pipeline**, wenn sich Datenverteilungen ändern
- Optional: **A/B-Tests** gegen den bisherigen Routing-Mechanismus

---

## 10. Ordnerstruktur des Repositories

```text
data/
  raw/          # Originaldaten (z. B. PSP_Jan_Feb_2019.xlsx, unverändert)
 

notebooks/
  01_eda.ipynb          # Explorative Datenanalyse
  03_modeling.ipynb     # Modelltraining
  04_evaluation.ipynb   # Modellbewertung

src/
  data_preparation.py   # Skripte zur Bereinigung
  feature_engineering.py


figures/                # Abbildungen

models/
  saved_models/         # gespeicherte ML-Modelle

requirements.txt        # Liste benötigter Python-Bibliotheken
README.md               # Projektbeschreibung, Ziele, Setup
