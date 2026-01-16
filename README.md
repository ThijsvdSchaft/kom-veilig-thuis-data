# kom-veilig-thuis-data
Data-driven UX-experimenten voor het project "Kom Veilig Thuis"

# Kom Veilig Thuis – Data-driven UX (Thijs)

Deze repository bevat de technische Python-uitwerking van het data-driven UX-concept **Kom Veilig Thuis**.  
Het project laat zien hoe data wordt verzameld, opgeschoond, geanalyseerd en vertaald naar UX-relevante output zoals heatmaps, hotspots en routeadvies.

De nadruk ligt op:
- uitlegbare en transparante datamodellen
- privacy-by-design
- het ondersteunen van UX-beslissingen met data (geen black-box AI)

Deze code vormt de technische onderbouwing van de bijbehorende documentatie.

---

## Context
Dit project is uitgevoerd binnen de opleiding **Communication and Multimedia Design (Hogeschool Utrecht)**.  
De scripts zijn bedoeld als demonstratie en technische verantwoording, niet als productiecode.

Er wordt uitsluitend gewerkt met **dummy/sample data**. Er is geen echte incident- of gebruikersdata opgenomen.

---

## Structuur van de repository

### `src/`
Bevat de Python-scripts waarin de data-driven UX-stappen zijn uitgewerkt.

- `kom_veilig_thuis_pipeline.py`  
  Complete pipeline voor:
  - data-opschoning en normalisatie  
  - recency-weighting en aggregatie naar kaartcellen  
  - feature engineering (o.a. nachtfactor)  
  - berekening van interpreteerbare risicoscores  
  - eenvoudige ML-validatie (logistische regressie)

### `data/`
Bevat uitsluitend voorbeelddata.

- `sample_incidents.csv`  
  Dummy incidentmeldingen die worden gebruikt om de pipeline te demonstreren.

---

## Werkwijze (samenvatting)
Binnen dit project zijn de volgende stappen toegepast:

1. Inladen en opschonen van incidentdata
2. Normaliseren en verrijken van data (tijd, severity, recency)
3. Aggregatie naar grid-cellen voor privacy en leesbaarheid
4. Opbouw van een interpreteerbare risicoscore (feature engineering)
5. Conceptuele validatie met een eenvoudig machine learning-model
6. Vertaling van data-output naar UX-toepassingen (heatmap, hotspots, routeadvies)

---

## Technologie
- Python
- pandas
- numpy
- matplotlib
- scikit-learn

Installeren van dependencies:
```bash
pip install -r requirements.txt
