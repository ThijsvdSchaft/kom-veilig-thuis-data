# Data-driven UX – Kom Veilig Thuis

Deze documentatie bevat de volledige technische en conceptuele uitwerking van het
project **Kom Veilig Thuis**, zoals ingeleverd voor de minor
**Big Data & Design aan de (Hogeschool Utrecht)**.

De inhoud van dit document is inhoudelijk gelijk aan de ingeleverde PDF-versie
die ik bij individuele contributie blok 3 heb ingleverd
en is hier opgenomen voor transparantie en controleerbaarheid.
De bijbehorende Python-scripts zijn te vinden in de `src/` map van deze repository.

---

## 1. Data collection, wrangling & visualisation (UX-perspectief)

### 1.1 Databronnen
Voor Kom Veilig Thuis wordt de veiligheid in de app opgebouwd uit meerdere databronnen:

- Openbare incidentdata (bijv. P2000-meldingen / meldingen van hulpdiensten)
- Gebruikersmeldingen (door appgebruikers gemelde onveilige situaties)
- Contextdata (tijdstip, locatie, type melding)

Het combineren van meerdere bronnen is belangrijk: één dataset kan incompleet zijn,
terwijl meerdere signalen samen een betrouwbaarder beeld geven van de veiligheidssituatie.

---

### 1.2 Datamodel (conceptueel)
Voor de verwerking wordt uitgegaan van een eenvoudige tabelstructuur (DataFrame) met
onder andere de volgende velden:

- `timestamp` – datum en tijd van de melding  
- `lat`, `lon` – locatie  
- `category` – type melding (bijv. overlast, agressie, onveilig gevoel)  
- `source` – herkomst van de melding (bijv. p2000, user)  
- `severity` – ernst van de melding (1–5 of afgeleid)

---

### 1.3 Opschonen & normaliseren
Ruwe data bevat vaak duplicaten, ontbrekende waarden of inconsistenties.
Daarom worden deze stappen uitgevoerd vóór visualisatie en routing.

```python
import pandas as pd
import numpy as np

df = pd.read_csv("incidents.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

df = (
    df.dropna(subset=["timestamp", "lat", "lon"])
      .drop_duplicates(subset=["timestamp", "lat", "lon", "category", "source"])
)

df["category"] = df["category"].astype(str).str.strip().str.lower()
```

UX-relevantie:
Deze stappen verhogen de betrouwbaarheid van de data, waardoor de gebruiker minder
ruis ziet op de kaart en routeadviezen beter te vertrouwen zijn.

1.4 Relevantie: tijdvenster & recency weighting

Voor veiligheidsinformatie is “recent” vaak belangrijker dan “ooit gebeurd”.
Daarom krijgt data een recency-gewicht (exponentiële afname).

```python
now = pd.Timestamp.utcnow()
cutoff = now - pd.Timedelta(days=30)
df_recent = df[df["timestamp"] >= cutoff].copy()

half_life_days = 7
age_days = (now - df_recent["timestamp"]).dt.total_seconds() / (3600 * 24)
df_recent["w_recency"] = 0.5 ** (age_days / half_life_days)
```

UX-relevantie:
Heatmaps en waarschuwingen blijven actueel en voorkomen onnodige paniek door oude incidenten.

1.5 Clusteren naar kaartcellen

In plaats van elke melding exact te plotten (druk, onrustig en privacy-gevoelig),
worden meldingen geaggregeerd per kaartcel (±150 meter).

```python
cell_size = 0.0015

df_recent["cell_x"] = np.floor(df_recent["lat"] / cell_size).astype(int)
df_recent["cell_y"] = np.floor(df_recent["lon"] / cell_size).astype(int)

df_recent["severity"] = pd.to_numeric(
    df_recent.get("severity", 1), errors="coerce"
).fillna(1)

cell_risk = (
    df_recent.assign(risk=lambda d: d["severity"] * d["w_recency"])
    .groupby(["cell_x", "cell_y"], as_index=False)
    .agg(
        risk_sum=("risk", "sum"),
        n_reports=("risk", "size"),
        last_seen=("timestamp", "max")
    )
)
```

UX-relevantie:

Overzichtelijke kaart

Privacybescherming (geen pinpointing)

Robuustheid door aggregatie van meldingen

1.6 Visualisatie-outputs (UX-behoeften)

Voor de interface zijn stabiele en eenvoudige outputs nodig:

Heatmap-intensiteit per cel

Hotspot-indicatoren

Recente meldingen

```python
cell_risk["risk_norm"] = (
    (cell_risk["risk_sum"] - cell_risk["risk_sum"].min()) /
    (cell_risk["risk_sum"].max() - cell_risk["risk_sum"].min() + 1e-9)
)

threshold = cell_risk["risk_sum"].quantile(0.90)
hotspots = cell_risk[cell_risk["risk_sum"] >= threshold].copy()

recent_2h = df[df["timestamp"] >= (now - pd.Timedelta(hours=2))]
```

1.7 Privacy-by-design

Privacy-by-design wordt praktisch toegepast door:

Dataminimalisatie

Geen ruwe incidentlogs in de UI

Alleen geaggregeerde risico-output

Heldere instellingen voor delen en waarschuwingen

```python
client_payload = hotspots[
    ["cell_x", "cell_y", "risk_norm", "n_reports", "last_seen"]
].copy()
```

2. Machine learning techniques – conceptuele toepassing
2.1 Van data naar risicoscore (feature engineering)

In plaats van ML als black box wordt eerst een uitlegbare risicoscore opgebouwd.

```python
local_hour = now.tz_convert("Europe/Amsterdam").hour
night_factor = 1.3 if (local_hour >= 22 or local_hour <= 6) else 1.0
cell_risk["risk_score"] = cell_risk["risk_sum"] * night_factor
```

Dit is een expliciete feature-engineering stap die later door een ML-model kan worden geleerd
of bijgesteld.

2.2 (Licht) learning via calibratie op feedback

Gebruikersfeedback kan worden gebruikt om risicoscores te kalibreren.

```python
feedback_df = pd.read_csv("feedback.csv")

cell = cell_risk.merge(
    feedback_df, on=["cell_x", "cell_y"], how="left"
).fillna({"votes_safe": 0, "votes_unsafe": 0})

cell["feedback_factor"] = (
    1 + cell["votes_unsafe"]
) / (1 + cell["votes_safe"])

cell["risk_score_calibrated"] = (
    cell["risk_score"] * cell["feedback_factor"]
)
```

2.3 Routekeuze op basis van risicolagen

De app biedt meerdere route-opties (snel, veilig, gebalanceerd).


```python
def score_route(route_cells, cell_score_map, alpha=0.7):
    risk = sum(cell_score_map.get(c, 0.0) for c in route_cells)
    distance_proxy = len(route_cells)
    return alpha * risk + (1 - alpha) * distance_proxy
```

2.4 Transparantie (“why this route?”)

Routes worden uitlegbaar gemaakt om black-box gedrag te voorkomen.

```python
def explain_route(route_cells, cell_score_map):
    scores = [(c, cell_score_map.get(c, 0.0)) for c in route_cells]
    top = sorted(scores, key=lambda x: x[1], reverse=True)[:3]
    return {
        "top_risk_cells": top,
        "total_risk": sum(s for _, s in scores)
    }
```

3. Learn new skills through inquisitive learning
3.1 Nieuwe vaardigheden

Tijdgebaseerde analyse (recency, tijdvensters)

Aggregatie en clustering voor visualisatie

Ontwerpen van interpreteerbare scoremodellen

Feedback gebruiken voor calibratie

Privacy-by-design toepassen in data-output

Vertalen van data naar UX-componenten

3.2 Toepassing in het ontwerp

Deze principes zijn direct vertaald naar UX-keuzes:

Heatmaps op basis van gewogen data

Hotspots geselecteerd op drempels i.p.v. gevoel

Routeadvies met keuzevrijheid

Uitlegfunctie verhoogt vertrouwen

Conclusie

Binnen Kom Veilig Thuis is data geen doel op zich, maar een middel om het
veiligheidsgevoel te vergroten. Door data te verzamelen, op te schonen en te
aggregeren ontstaat een bruikbaar veiligheidsbeeld dat in de UX wordt vertaald
naar heatmaps, hotspots en routeadvies. Door een interpreteerbaar scoremodel en
feedback-integratie blijft het systeem transparant, uitlegbaar en privacybewust.

Bronnen (APA-stijl)

Autoriteit Persoonsgegevens. (2023). Privacy by design en privacy by default.
https://autoriteitpersoonsgegevens.nl

Centraal Bureau voor de Statistiek. (2023). Veiligheid en criminaliteit – open data.
https://www.cbs.nl

European Commission. (2018). General Data Protection Regulation (GDPR).
https://gdpr.eu

Gemeente Utrecht. (2023). Veiligheidsmonitor Utrecht.
https://www.utrecht.nl

IBM. (2021). What is machine learning?
https://www.ibm.com/topics/machine-learning

Norman, D. A. (2013). The design of everyday things (Revised ed.). Basic Books.

Open State Foundation. (2021). Open data in Nederland: kansen en toepassingen.
https://openstate.eu

P2000 Nederland. (z.d.). Openbare hulpdienstmeldingen.
https://www.p2000.nl

Tufte, E. R. (2001). The visual display of quantitative information (2nd ed.). Graphics Press.
